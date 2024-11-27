#pragma once

#include "../shader_shared/visbuffer.inl"
#include "../shader_shared/geometry.inl"
#include "../shader_lib/debug.glsl"

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

uint meshlet_instance_index_from_triangle_id(uint id)
{
    return TRIANGLE_ID_GET_MESHLET_INSTANCE_INDEX(id);
}

uint triangle_index_from_triangle_id(uint id)
{
    return TRIANGLE_ID_GET_MESHLET_TRIANGLE_INDEX(id);
}

// Credit: http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
struct BarycentricDeriv
{
    daxa_f32vec3 m_lambda;
    daxa_f32vec3 m_ddx;
    daxa_f32vec3 m_ddy;
};
// Credit: http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
BarycentricDeriv calc_bary_and_deriv(daxa_f32vec4 pt0, daxa_f32vec4 pt1, daxa_f32vec4 pt2, daxa_f32vec2 pixelNdc, daxa_f32vec2 winSize)
{
    BarycentricDeriv ret;   
    daxa_f32vec3 invW = 1.0f / (daxa_f32vec3(pt0.w, pt1.w, pt2.w)); 
    daxa_f32vec2 ndc0 = pt0.xy * invW.x;
    daxa_f32vec2 ndc1 = pt1.xy * invW.y;
    daxa_f32vec2 ndc2 = pt2.xy * invW.z;  
    float invDet = 1.0f / (determinant(daxa_f32mat2x2(ndc2 - ndc1, ndc0 - ndc1)));
    ret.m_ddx = daxa_f32vec3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * invDet * invW;
    ret.m_ddy = daxa_f32vec3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * invDet * invW;
    float ddxSum = dot(ret.m_ddx, daxa_f32vec3(1,1,1));
    float ddySum = dot(ret.m_ddy, daxa_f32vec3(1,1,1));   
    daxa_f32vec2 deltaVec = pixelNdc - ndc0;
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
daxa_f32vec3 interpolate_with_deriv(BarycentricDeriv deriv, float v0, float v1, float v2)
{
    daxa_f32vec3 mergedV = daxa_f32vec3(v0, v1, v2);
    daxa_f32vec3 ret;
    ret.x = dot(mergedV, deriv.m_lambda);
    ret.y = dot(mergedV, deriv.m_ddx);
    ret.z = dot(mergedV, deriv.m_ddy);
    return ret;
}

float visgeo_interpolate_float(daxa_f32vec3 derivator, float v0, float v1, float v2)
{
    daxa_f32vec3 mergedV = daxa_f32vec3(v0, v1, v2);
    float ret;
    ret = dot(mergedV, derivator);
    return ret;
}

daxa_f32vec2 visgeo_interpolate_vec2(daxa_f32vec3 derivator, daxa_f32vec2 v0, daxa_f32vec2 v1, daxa_f32vec2 v2)
{
    daxa_f32vec3 merged_x = daxa_f32vec3(v0.x, v1.x, v2.x);
    daxa_f32vec3 merged_y = daxa_f32vec3(v0.y, v1.y, v2.y);
    daxa_f32vec2 ret;
    ret.x = dot(merged_x, derivator);
    ret.y = dot(merged_y, derivator);
    return ret;
}

daxa_f32vec3 visgeo_interpolate_vec3(daxa_f32vec3 derivator, daxa_f32vec3 v0, daxa_f32vec3 v1, daxa_f32vec3 v2)
{
    daxa_f32vec3 merged_x = daxa_f32vec3(v0.x, v1.x, v2.x);
    daxa_f32vec3 merged_y = daxa_f32vec3(v0.y, v1.y, v2.y);
    daxa_f32vec3 merged_z = daxa_f32vec3(v0.z, v1.z, v2.z);
    daxa_f32vec3 ret;
    ret.x = dot(merged_x, derivator);
    ret.y = dot(merged_y, derivator);
    ret.z = dot(merged_z, derivator);
    return ret;
}

daxa_f32vec4 visgeo_interpolate_vec4(daxa_f32vec3 derivator, daxa_f32vec4 v0, daxa_f32vec4 v1, daxa_f32vec4 v2)
{
    daxa_f32vec3 merged_x = daxa_f32vec3(v0.x, v1.x, v2.x);
    daxa_f32vec3 merged_y = daxa_f32vec3(v0.y, v1.y, v2.y);
    daxa_f32vec3 merged_z = daxa_f32vec3(v0.z, v1.z, v2.z);
    daxa_f32vec3 merged_w = daxa_f32vec3(v0.w, v1.w, v2.w);
    daxa_f32vec4 ret;
    ret.x = dot(merged_x, derivator);
    ret.y = dot(merged_y, derivator);
    ret.z = dot(merged_z, derivator);
    ret.w = dot(merged_w, derivator);
    return ret;
}

struct VisbufferTriangleData
{
    uint meshlet_instance_index;
    uint meshlet_triangle_index;
    MeshletInstance meshlet_instance;
    BarycentricDeriv bari_deriv;
    daxa_u32vec3 vertex_indices;
    daxa_f32vec3 world_position;
    daxa_f32vec3 world_normal;
    daxa_f32vec3 world_tangent;
    daxa_f32 depth;
    daxa_f32vec2 uv;
    daxa_f32vec2 uv_ddx;
    daxa_f32vec2 uv_ddy;
};

VisbufferTriangleData visgeo_triangle_data(
    uint triangle_id, 
    daxa_f32vec2 xy, 
    daxa_f32vec2 screen_size,
    daxa_f32vec2 inv_screen_size,
    daxa_f32mat4x4 view_proj, 
    daxa_BufferPtr(MeshletInstancesBufferHead) meshlet_instances,
    daxa_BufferPtr(GPUMesh) meshes,
    daxa_BufferPtr(daxa_f32mat4x3) combined_transforms)
{

    VisbufferTriangleData ret;
    ret.meshlet_instance_index = TRIANGLE_ID_GET_MESHLET_INSTANCE_INDEX(triangle_id);
    ret.meshlet_triangle_index = TRIANGLE_ID_GET_MESHLET_TRIANGLE_INDEX(triangle_id);

    #if GPU_ASSERTS
        const uint meshlet_instance_count = meshlet_instances.pass_counts[0] + meshlet_instances.pass_counts[1];
        if (!(ret.meshlet_instance_index < meshlet_instance_count))
        {
            printf(GPU_ASSERT_STRING" Invalid Triangle ID passed to visgeo_triangle_data");
            ret.meshlet_instance_index = {};
            ret.meshlet_triangle_index = {};
            return ret;
        }
    #endif

    ret.meshlet_instance = deref_i(deref(meshlet_instances).meshlets, ret.meshlet_instance_index);

    GPUMesh mesh = deref_i(meshes, ret.meshlet_instance.mesh_index);
    Meshlet meshlet = deref_i(mesh.meshlets, ret.meshlet_instance.meshlet_index);

    const daxa_u32vec3 micro_indices = daxa_u32vec3(
        get_micro_index(mesh.micro_indices, meshlet.micro_indices_offset + ret.meshlet_triangle_index * 3 + 0),
        get_micro_index(mesh.micro_indices, meshlet.micro_indices_offset + ret.meshlet_triangle_index * 3 + 1),
        get_micro_index(mesh.micro_indices, meshlet.micro_indices_offset + ret.meshlet_triangle_index * 3 + 2)
    );

    ret.vertex_indices = daxa_u32vec3(
        deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + micro_indices.x),
        deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + micro_indices.y),
        deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + micro_indices.z)
    );

    daxa_f32mat4x4 model_matrix = mat_4x3_to_4x4(deref_i(combined_transforms, ret.meshlet_instance.entity_index));

    const daxa_f32vec2 ndc_xy = ((xy + 0.5f) * inv_screen_size) * 2.0f - 1.0f;

    const daxa_f32vec3[3] vertex_positions = daxa_f32vec3[3](
        deref_i(mesh.vertex_positions, ret.vertex_indices.x),
        deref_i(mesh.vertex_positions, ret.vertex_indices.y),
        deref_i(mesh.vertex_positions, ret.vertex_indices.z)
    );

    const daxa_f32vec4[3] world_vertex_positions = daxa_f32vec4[3](
        mul(model_matrix, daxa_f32vec4(vertex_positions[0],1)),
        mul(model_matrix, daxa_f32vec4(vertex_positions[1],1)),
        mul(model_matrix, daxa_f32vec4(vertex_positions[2],1))
    );

    const daxa_f32vec4[3] clipspace_vertex_positions = daxa_f32vec4[3](
        mul(view_proj, world_vertex_positions[0]),
        mul(view_proj, world_vertex_positions[1]),
        mul(view_proj, world_vertex_positions[2])
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

    const daxa_f32vec2 interp_zw = visgeo_interpolate_vec2(
        ret.bari_deriv.m_lambda,
        clipspace_vertex_positions[0].zw,
        clipspace_vertex_positions[1].zw,
        clipspace_vertex_positions[2].zw
    );
    ret.depth = interp_zw.r / interp_zw.g;

    const daxa_f32vec3[3] vertex_normals = daxa_f32vec3[3](
        deref_i(mesh.vertex_normals, ret.vertex_indices.x),
        deref_i(mesh.vertex_normals, ret.vertex_indices.y),
        deref_i(mesh.vertex_normals, ret.vertex_indices.z)
    );

    // WARNING: WE ACTUALLY NEED THE TRANSPOSE INVERSE HERE
    const daxa_f32vec4[3] worldspace_vertex_normals = daxa_f32vec4[3](
        mul(model_matrix, daxa_f32vec4(vertex_normals[0], 0)),
        mul(model_matrix, daxa_f32vec4(vertex_normals[1], 0)),
        mul(model_matrix, daxa_f32vec4(vertex_normals[2], 0))
    );

    ret.world_normal = normalize(visgeo_interpolate_vec3(
        ret.bari_deriv.m_lambda, 
        worldspace_vertex_normals[0].xyz,
        worldspace_vertex_normals[1].xyz,
        worldspace_vertex_normals[2].xyz
    ));

    const daxa_f32vec2[3] vertex_uvs = daxa_f32vec2[3](
        deref_i(mesh.vertex_uvs, ret.vertex_indices.x),
        deref_i(mesh.vertex_uvs, ret.vertex_indices.y),
        deref_i(mesh.vertex_uvs, ret.vertex_indices.z)
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

    // Calculae Tangent.
    {
        // Credit: https://stackoverflow.com/questions/5255806/how-to-calculate-tangent-and-binormal
        /// derivations of the fragment position
        daxa_f32vec3 p_dx = visgeo_interpolate_vec3(
            ret.bari_deriv.m_ddx, 
            world_vertex_positions[0].xyz,
            world_vertex_positions[1].xyz,
            world_vertex_positions[2].xyz
        );
        daxa_f32vec3 p_dy = visgeo_interpolate_vec3(
            ret.bari_deriv.m_ddy, 
            world_vertex_positions[0].xyz,
            world_vertex_positions[1].xyz,
            world_vertex_positions[2].xyz
        );
        // derivations of the texture coordinate
        daxa_f32vec2 tc_dx = ret.uv_ddx;
        daxa_f32vec2 tc_dy = ret.uv_ddy;
        // compute initial tangent and bi-tangent
        daxa_f32vec3 t = normalize( tc_dy.y * p_dx - tc_dx.y * p_dy );
        // get new tangent from a given mesh normal
        daxa_f32vec3 x = cross(ret.world_normal, t);
        t = cross(x, ret.world_normal);
        t = normalize(t);
        ret.world_tangent = t;
    }

    return ret;
}