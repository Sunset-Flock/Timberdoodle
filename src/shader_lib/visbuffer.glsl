#pragma once

#include "../shader_shared/visbuffer.inl"
#include "../shader_shared/asset.inl"

/**
 * DESCRIPTION:
 * Visbuffer is used as an indirection into all relevant rendering data.
 * As the visbuffer is only 32 bit per pixel, it SIGNIFICANTLY reduces used bandwidth when drawing and shading.
 * In effect it is a form of bandwidth compression.
 * To get rendering data from the visbuffer we need to follow several indirections, examples:
 * * albedo texture: vis id -> meshlet instance -> material, contains albedo image id
 * * vertex positions: vis id -> meshlet instance -> mesh -> mesh buffer id, mesh vertex positions offset
 * * transform: vis id -> meshlet instance -> entity transform array
 * 
 * As you can see the meshlet instance is storing all kinds of indices into relevant parts of the data.
 * With the triangle index and the meshlet index in triangle id and meshlet instance, 
 * we also know exactly what part of each mesh we are from the vis id.
*/

void encode_triangle_id(uint instantiated_meshlet_index, uint triangle_index, out uint id)
{
    id = (instantiated_meshlet_index << 7) | (triangle_index);
}

uint meshlet_instance_index_from_triangle_id(uint id)
{
    return id >> 7;
}

uint triangle_index_from_triangle_id(uint id)
{
    return id & 0x7F;
}

void decode_triangle_id(uint id, out uint instantiated_meshlet_index, out uint triangle_index)
{
    instantiated_meshlet_index = meshlet_instance_index_from_triangle_id(id);
    triangle_index = triangle_index_from_triangle_id(id);
}

// Credit: http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
struct BarycentricDeriv
{
    vec3 m_lambda;
    vec3 m_ddx;
    vec3 m_ddy;
};
// Credit: http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
BarycentricDeriv calc_bary_and_deriv(vec4 pt0, vec4 pt1, vec4 pt2, vec2 pixelNdc, vec2 winSize)
{
    BarycentricDeriv ret;   
    vec3 invW = 1.0f / (vec3(pt0.w, pt1.w, pt2.w)); 
    vec2 ndc0 = pt0.xy * invW.x;
    vec2 ndc1 = pt1.xy * invW.y;
    vec2 ndc2 = pt2.xy * invW.z;  
    float invDet = 1.0f / (determinant(mat2x2(ndc2 - ndc1, ndc0 - ndc1)));
    ret.m_ddx = vec3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * invDet * invW;
    ret.m_ddy = vec3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * invDet * invW;
    float ddxSum = dot(ret.m_ddx, vec3(1,1,1));
    float ddySum = dot(ret.m_ddy, vec3(1,1,1));   
    vec2 deltaVec = pixelNdc - ndc0;
    float interpInvW = invW.x + deltaVec.x*ddxSum + deltaVec.y*ddySum;
    float interpW = 1.0f / (interpInvW);    
    ret.m_lambda.x = interpW * (invW[0] + deltaVec.x*ret.m_ddx.x + deltaVec.y*ret.m_ddy.x);
    ret.m_lambda.y = interpW * (0.0f    + deltaVec.x*ret.m_ddx.y + deltaVec.y*ret.m_ddy.y);
    ret.m_lambda.z = interpW * (0.0f    + deltaVec.x*ret.m_ddx.z + deltaVec.y*ret.m_ddy.z); 
    ret.m_ddx *= (2.0f/winSize.x);
    ret.m_ddy *= (2.0f/winSize.y);
    ddxSum    *= (2.0f/winSize.x);
    ddySum    *= (2.0f/winSize.y);  
    ret.m_ddy *= -1.0f;
    ddySum    *= -1.0f; 
    float interpW_ddx = 1.0f / (interpInvW + ddxSum);
    float interpW_ddy = 1.0f / (interpInvW + ddySum);   
    ret.m_ddx = interpW_ddx*(ret.m_lambda*interpInvW + ret.m_ddx) - ret.m_lambda;
    ret.m_ddy = interpW_ddy*(ret.m_lambda*interpInvW + ret.m_ddy) - ret.m_lambda;   
    return ret;
}
// Credit: http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
vec3 interpolate_with_deriv(BarycentricDeriv deriv, float v0, float v1, float v2)
{
  vec3 mergedV = vec3(v0, v1, v2);
  vec3 ret;
  ret.x = dot(mergedV, deriv.m_lambda);
  ret.y = dot(mergedV, deriv.m_ddx);
  ret.z = dot(mergedV, deriv.m_ddy);
  return ret;
}

struct VisbufferTriangleData
{
    uint meshlet_instance_index;
    uint triangle_index;
    MeshletInstance meshlet_instance;
    BarycentricDeriv bari_deriv;
    vec3 world_position;
    vec3 world_position_ddx;
    vec3 world_position_ddy;
    uvec3 vertex_indices;
};

VisbufferTriangleData visgeo_triangle_data(
    uint triangle_id, 
    vec2 xy, 
    vec2 screen_size,
    vec2 inv_screen_size,
    mat4x4 view_proj, 
    daxa_BufferPtr(MeshletInstances) meshlet_instances,
    daxa_BufferPtr(GPUMesh) meshes,
    daxa_BufferPtr(daxa_f32mat4x3) combined_transforms)
{
    VisbufferTriangleData ret;
    decode_triangle_id(triangle_id, ret.meshlet_instance_index, ret.triangle_index);

    ret.meshlet_instance = unpack_meshlet_instance(deref(meshlet_instances).meshlets[ret.meshlet_instance_index]);

    GPUMesh mesh = deref(meshes + ret.meshlet_instance.mesh_index);
    Meshlet meshlet = deref(mesh.meshlets + ret.meshlet_instance.meshlet_index);

    const uvec3 micro_indices = uvec3(
        get_micro_index(mesh.micro_indices, meshlet.micro_indices_offset + ret.triangle_index * 3 + 0),
        get_micro_index(mesh.micro_indices, meshlet.micro_indices_offset + ret.triangle_index * 3 + 1),
        get_micro_index(mesh.micro_indices, meshlet.micro_indices_offset + ret.triangle_index * 3 + 2)
    );

    ret.vertex_indices = uvec3(
        deref(mesh.indirect_vertices + meshlet.indirect_vertex_offset + micro_indices.x),
        deref(mesh.indirect_vertices + meshlet.indirect_vertex_offset + micro_indices.y),
        deref(mesh.indirect_vertices + meshlet.indirect_vertex_offset + micro_indices.z)
    );

    const vec3[] vertex_positions = vec3[](
        deref(mesh.vertex_positions + ret.vertex_indices.x),
        deref(mesh.vertex_positions + ret.vertex_indices.y),
        deref(mesh.vertex_positions + ret.vertex_indices.z)
    );

    mat4x4 model_matrix = mat_4x3_to_4x4(deref(combined_transforms + ret.meshlet_instance.entity_index));

    const vec4[] world_vertex_positions = vec4[](
        model_matrix * vec4(vertex_positions[0],1),
        model_matrix * vec4(vertex_positions[1],1),
        model_matrix * vec4(vertex_positions[2],1)
    );

    const vec4[] clipspace_vertex_positions = vec4[](
        view_proj * world_vertex_positions[0],
        view_proj * world_vertex_positions[1],
        view_proj * world_vertex_positions[2]
    );

    const vec3[] ndc_vertex_positions = vec3[](
        clipspace_vertex_positions[0].xyz / clipspace_vertex_positions[0].z,
        clipspace_vertex_positions[1].xyz / clipspace_vertex_positions[1].z,
        clipspace_vertex_positions[2].xyz / clipspace_vertex_positions[2].z
    );

    vec2 ndc_xy = ((xy + 0.5f) * inv_screen_size) * 2.0f - 1.0f;

    ret.bari_deriv = calc_bary_and_deriv(
        clipspace_vertex_positions[0],
        clipspace_vertex_positions[1],
        clipspace_vertex_positions[2],
        ndc_xy,
        screen_size);
        
    vec3 x_deriv = interpolate_with_deriv(
        ret.bari_deriv, 
        world_vertex_positions[0].x, 
        world_vertex_positions[1].x, 
        world_vertex_positions[2].x);
    ret.world_position.x = x_deriv.x;
    ret.world_position_ddx.x = x_deriv.y;
    ret.world_position_ddy.x = x_deriv.z;

    vec3 y_deriv = interpolate_with_deriv(
        ret.bari_deriv, 
        world_vertex_positions[0].y, 
        world_vertex_positions[1].y, 
        world_vertex_positions[2].y);
    ret.world_position.y = y_deriv.x;
    ret.world_position_ddx.y = y_deriv.y;
    ret.world_position_ddy.y = y_deriv.z;

    vec3 z_deriv = interpolate_with_deriv(
        ret.bari_deriv, 
        world_vertex_positions[0].z, 
        world_vertex_positions[1].z, 
        world_vertex_positions[2].z);
    ret.world_position.z = z_deriv.x;
    ret.world_position_ddx.z = z_deriv.y;
    ret.world_position_ddy.z = z_deriv.z;

    return ret;
}

struct VisbufferTriangleUv
{
    vec2 uv;
    vec2 uv_ddx;
    vec2 uv_ddy;
};

VisbufferTriangleUv visgeo_interpolated_uv(VisbufferTriangleData tri_data, daxa_BufferPtr(GPUMesh) meshes)
{
    GPUMesh mesh = deref(meshes + tri_data.meshlet_instance.mesh_index);

    vec2[] vertex_uvs = vec2[](
        deref(mesh.vertex_uvs + tri_data.vertex_indices[0]),
        deref(mesh.vertex_uvs + tri_data.vertex_indices[1]),
        deref(mesh.vertex_uvs + tri_data.vertex_indices[2])
    );

    VisbufferTriangleUv ret;

    vec3 u_deriv = interpolate_with_deriv(
        tri_data.bari_deriv, 
        vertex_uvs[0].x, 
        vertex_uvs[1].x, 
        vertex_uvs[2].x);
    ret.uv.x = u_deriv.x;
    ret.uv_ddx.x = u_deriv.y;
    ret.uv_ddy.x = u_deriv.z;

    vec3 v_deriv = interpolate_with_deriv(
        tri_data.bari_deriv, 
        vertex_uvs[0].y, 
        vertex_uvs[1].y, 
        vertex_uvs[2].y);
    ret.uv.y = v_deriv.x;
    ret.uv_ddx.y = v_deriv.y;
    ret.uv_ddy.y = v_deriv.z;

    return ret;
}

vec3 visgeo_interpolated_normal(VisbufferTriangleData tri_data, daxa_BufferPtr(GPUMesh) meshes)
{
    GPUMesh mesh = deref(meshes + tri_data.meshlet_instance.mesh_index);

    vec3[] vertex_normals = vec3[](
        deref(mesh.vertex_normals + tri_data.vertex_indices[0]),
        deref(mesh.vertex_normals + tri_data.vertex_indices[1]),
        deref(mesh.vertex_normals + tri_data.vertex_indices[2])
    );

    vec3 ret;
    ret.x = interpolate_with_deriv(
        tri_data.bari_deriv, 
        vertex_normals[0].x, 
        vertex_normals[1].x, 
        vertex_normals[2].x).x;
    ret.y = interpolate_with_deriv(
        tri_data.bari_deriv, 
        vertex_normals[0].y, 
        vertex_normals[1].y, 
        vertex_normals[2].y).x;
    ret.z = interpolate_with_deriv(
        tri_data.bari_deriv, 
        vertex_normals[0].z, 
        vertex_normals[1].z, 
        vertex_normals[2].z).x;
    ret = normalize(ret);
    return ret;
}

mat3x3 visgeo_tbn(VisbufferTriangleData tri_data, VisbufferTriangleUv uv_data, vec3 normal)
{
    // Credit: https://stackoverflow.com/questions/5255806/how-to-calculate-tangent-and-binormal
    /// derivations of the fragment position
    vec3 pos_dx = tri_data.world_position_ddx;
    vec3 pos_dy = tri_data.world_position_ddy;
    // derivations of the texture coordinate
    vec2 texC_dx = uv_data.uv_ddx;
    vec2 texC_dy = uv_data.uv_ddy;
    vec3 t = texC_dy.y * pos_dx - texC_dx.y * pos_dy;
    vec3 b = texC_dx.x * pos_dy - texC_dy.x * pos_dx;
    vec3 n = normal;
    // https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    t = t - n * dot( t, n ); // orthonormalization ot the tangent vectors
    t = normalize(t);
    b = normalize(b);
    b = b - n * dot( b, n ); // orthonormalization of the binormal vectors to the normal vector 
    b = b - t * dot( b, t ); // orthonormalization of the binormal vectors to the tangent vector
    mat3 tbn = mat3( normalize(t), normalize(b), n );
    return tbn;
}