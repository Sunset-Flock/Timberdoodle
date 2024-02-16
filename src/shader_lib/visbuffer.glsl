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

float visgeo_interpolate_float(vec3 derivator, float v0, float v1, float v2)
{
    vec3 mergedV = vec3(v0, v1, v2);
    float ret;
    ret = dot(mergedV, derivator);
    return ret;
}

vec2 visgeo_interpolate_vec2(vec3 derivator, vec2 v0, vec2 v1, vec2 v2)
{
    vec3 merged_x = vec3(v0.x, v1.x, v2.x);
    vec3 merged_y = vec3(v0.y, v1.y, v2.y);
    vec2 ret;
    ret.x = dot(merged_x, derivator);
    ret.y = dot(merged_y, derivator);
    return ret;
}

vec3 visgeo_interpolate_vec3(vec3 derivator, vec3 v0, vec3 v1, vec3 v2)
{
    vec3 merged_x = vec3(v0.x, v1.x, v2.x);
    vec3 merged_y = vec3(v0.y, v1.y, v2.y);
    vec3 merged_z = vec3(v0.z, v1.z, v2.z);
    vec3 ret;
    ret.x = dot(merged_x, derivator);
    ret.y = dot(merged_y, derivator);
    ret.z = dot(merged_z, derivator);
    return ret;
}

vec4 visgeo_interpolate_vec4(vec3 derivator, vec4 v0, vec4 v1, vec4 v2)
{
    vec3 merged_x = vec3(v0.x, v1.x, v2.x);
    vec3 merged_y = vec3(v0.y, v1.y, v2.y);
    vec3 merged_z = vec3(v0.z, v1.z, v2.z);
    vec3 merged_w = vec3(v0.w, v1.w, v2.w);
    vec4 ret;
    ret.x = dot(merged_x, derivator);
    ret.y = dot(merged_y, derivator);
    ret.z = dot(merged_z, derivator);
    ret.w = dot(merged_w, derivator);
    return ret;
}

struct VisbufferTriangleData
{
    uint meshlet_instance_index;
    uint triangle_index;
    MeshletInstance meshlet_instance;
    BarycentricDeriv bari_deriv;
    uvec3 vertex_indices;
    vec3 world_position;
    vec3 world_normal;
    vec3 world_tangent;
    vec2 uv;
    vec2 uv_ddx;
    vec2 uv_ddy;
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

    mat4x4 model_matrix = mat_4x3_to_4x4(deref(combined_transforms + ret.meshlet_instance.entity_index));

    const vec2 ndc_xy = ((xy + 0.5f) * inv_screen_size) * 2.0f - 1.0f;

    const vec3[] vertex_positions = vec3[](
        deref(mesh.vertex_positions + ret.vertex_indices.x),
        deref(mesh.vertex_positions + ret.vertex_indices.y),
        deref(mesh.vertex_positions + ret.vertex_indices.z)
    );

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

    ret.bari_deriv = calc_bary_and_deriv(
        clipspace_vertex_positions[0],
        clipspace_vertex_positions[1],
        clipspace_vertex_positions[2],
        ndc_xy,
        screen_size
    );

    ret.world_position = visgeo_interpolate_vec3(
        ret.bari_deriv.m_lambda, 
        world_vertex_positions[0].xyz,
        world_vertex_positions[1].xyz,
        world_vertex_positions[2].xyz
    );

    const vec3[] vertex_normals = vec3[](
        deref(mesh.vertex_normals + ret.vertex_indices.x),
        deref(mesh.vertex_normals + ret.vertex_indices.y),
        deref(mesh.vertex_normals + ret.vertex_indices.z)
    );

    // WARNING: WE ACTUALLY NEED THE TRANSPOSE INVERSE HERE
    const vec4[] worldspace_vertex_normals = vec4[](
        model_matrix * vec4(vertex_normals[0], 0),
        model_matrix * vec4(vertex_normals[1], 0),
        model_matrix * vec4(vertex_normals[2], 0)
    );

    ret.world_normal = normalize(visgeo_interpolate_vec3(
        ret.bari_deriv.m_lambda, 
        worldspace_vertex_normals[0].xyz,
        worldspace_vertex_normals[1].xyz,
        worldspace_vertex_normals[2].xyz
    ));

    if (uint64_t(mesh.vertex_tangents) != 0)
    {
        const vec3[] vertex_tangents = vec3[](
            deref(mesh.vertex_tangents + ret.vertex_indices.x).xyz,
            deref(mesh.vertex_tangents + ret.vertex_indices.y).xyz,
            deref(mesh.vertex_tangents + ret.vertex_indices.z).xyz
        );

        // WARNING: WE ACTUALLY NEED THE TRANSPOSE INVERSE HERE
        const vec4[] worldspace_vertex_tangents = vec4[](
            model_matrix * vec4(vertex_tangents[0], 0),
            model_matrix * vec4(vertex_tangents[1], 0),
            model_matrix * vec4(vertex_tangents[2], 0)
        );

        ret.world_tangent = normalize(visgeo_interpolate_vec3(
            ret.bari_deriv.m_lambda, 
            worldspace_vertex_tangents[0].xyz,
            worldspace_vertex_tangents[1].xyz,
            worldspace_vertex_tangents[2].xyz
        ));
    }
    else
    {
        ret.world_tangent = vec3(0,0,0);
    }

    const vec2[] vertex_uvs = vec2[](
        deref(mesh.vertex_uvs + ret.vertex_indices.x),
        deref(mesh.vertex_uvs + ret.vertex_indices.y),
        deref(mesh.vertex_uvs + ret.vertex_indices.z)
    );

    ret.uv = visgeo_interpolate_vec2(
        ret.bari_deriv.m_lambda, 
        vertex_uvs[0],
        vertex_uvs[1],
        vertex_uvs[2]
    );

    ret.uv_ddx = visgeo_interpolate_vec2(
        ret.bari_deriv.m_ddx, 
        vertex_uvs[0],
        vertex_uvs[1],
        vertex_uvs[2]
    );

    ret.uv_ddy = visgeo_interpolate_vec2(
        ret.bari_deriv.m_ddy, 
        vertex_uvs[0],
        vertex_uvs[1],
        vertex_uvs[2]
    );

    return ret;
}

// mat3x3 visgeo_tbn(VisbufferTriangleData tri_data, VisbufferTriangleUv uv_data, vec3 normal)
// {
//     // Credit: https://stackoverflow.com/questions/5255806/how-to-calculate-tangent-and-binormal
//     /// derivations of the fragment position
//     vec3 p_dx = tri_data.world_position_ddx;
//     vec3 p_dy = tri_data.world_position_ddy;
//     // derivations of the texture coordinate
//     vec2 tc_dx = uv_data.uv_ddx;
//     vec2 tc_dy = uv_data.uv_ddy;
//     // compute initial tangent and bi-tangent
//     vec3 t = normalize( tc_dy.y * p_dx - tc_dx.y * p_dy );
//     vec3 b = normalize( tc_dy.x * p_dx - tc_dx.x * p_dy ); // sign inversion
//     // get new tangent from a given mesh normal
//     vec3 n = normal;
//     vec3 x = cross(n, t);
//     t = cross(x, n);
//     t = normalize(t);
//     // get updated bi-tangent
//     x = cross(b, n);
//     b = cross(n, x);
//     b = normalize(b);
//     mat3 tbn = mat3(t, b, n);
//     return tbn;
// }