#pragma once

#include "../shader_shared/visbuffer.inl"
#include "../shader_shared/geometry.inl"
#include "../shader_lib/debug.glsl"
#include "../shader_lib/misc.hlsl"
#include "../shader_lib/geometry.hlsl"

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
    ddySum    *= -1.0f; 
    ret.m_ddy *= -1;
    float interpW_ddx = 1.0f / (interpInvW + ddxSum);
    float interpW_ddy = 1.0f / (interpInvW + ddySum);   
    ret.m_ddx = interpW_ddx*(ret.m_lambda*interpInvW + ret.m_ddx) - ret.m_lambda;
    ret.m_ddy = interpW_ddy*(ret.m_lambda*interpInvW + ret.m_ddy) - ret.m_lambda;   
    
    // Modification by Ipotrick. Prevents broken barycentrics for shallow angles caused by floating point imprecision
    ret.m_lambda = ret.m_lambda * rcp(ret.m_lambda.x + ret.m_lambda.y + ret.m_lambda.z + 0.00001f);

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

struct VisbufferTriangleGeometry
{
    TriangleGeometry tri_geo;
    TriangleGeometryPoint tri_geo_point;
    float depth;
    uint meshlet_triangle_index;
    uint meshlet_instance_index;
    uint meshlet_index;
};

VisbufferTriangleGeometry visgeo_triangle_data(
    uint triangle_id, 
    daxa_f32vec2 xy, 
    daxa_f32vec2 screen_size,
    daxa_f32vec2 inv_screen_size,
    daxa_f32mat4x4 view_proj, 
    daxa_BufferPtr(MeshletInstancesBufferHead) meshlet_instances,
    daxa_BufferPtr(GPUMesh) meshes,
    daxa_BufferPtr(daxa_f32mat4x3) combined_transforms)
{

    VisbufferTriangleGeometry ret;
    ret.meshlet_instance_index = TRIANGLE_ID_GET_MESHLET_INSTANCE_INDEX(triangle_id);
    ret.meshlet_triangle_index = TRIANGLE_ID_GET_MESHLET_TRIANGLE_INDEX(triangle_id);

    #if GPU_ASSERT_ENABLE
        const uint meshlet_instance_count = meshlet_instances.pass_counts[0] + meshlet_instances.pass_counts[1];
        if (!(ret.meshlet_instance_index < meshlet_instance_count))
        {
            GPU_ASSERT(ret.meshlet_instance_index < meshlet_instance_count);
            ret = {};
            return ret;
        }
    #endif

    MeshletInstance meshlet_instance = deref_i(deref(meshlet_instances).meshlets, ret.meshlet_instance_index);
    ret.tri_geo.entity_index = meshlet_instance.entity_index;
    ret.meshlet_index = meshlet_instance.meshlet_index;
    ret.tri_geo.mesh_index = meshlet_instance.mesh_index;
    ret.tri_geo.material_index = meshlet_instance.material_index;
    ret.tri_geo.in_mesh_group_index = meshlet_instance.in_mesh_group_index;
    ret.tri_geo.mesh_instance_index = meshlet_instance.mesh_instance_index;


    GPUMesh mesh = deref_i(meshes, ret.tri_geo.mesh_index);
    Meshlet meshlet = deref_i(mesh.meshlets, ret.meshlet_index);

    const daxa_u32vec3 micro_indices = daxa_u32vec3(
        get_micro_index(mesh.micro_indices, meshlet.micro_indices_offset + ret.meshlet_triangle_index * 3 + 0),
        get_micro_index(mesh.micro_indices, meshlet.micro_indices_offset + ret.meshlet_triangle_index * 3 + 1),
        get_micro_index(mesh.micro_indices, meshlet.micro_indices_offset + ret.meshlet_triangle_index * 3 + 2)
    );

    ret.tri_geo.vertex_indices = daxa_u32vec3(
        deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + micro_indices.x),
        deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + micro_indices.y),
        deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + micro_indices.z)
    );

    daxa_f32mat4x4 model_matrix = mat_4x3_to_4x4(deref_i(combined_transforms, ret.tri_geo.entity_index));

    const daxa_f32vec2 ndc_xy = ((xy + 0.5f) * inv_screen_size) * 2.0f - 1.0f;

    const daxa_f32vec3[3] vertex_positions = daxa_f32vec3[3](
        deref_i(mesh.vertex_positions, ret.tri_geo.vertex_indices.x),
        deref_i(mesh.vertex_positions, ret.tri_geo.vertex_indices.y),
        deref_i(mesh.vertex_positions, ret.tri_geo.vertex_indices.z)
    );

    const daxa_f32vec3[3] world_vertex_positions = daxa_f32vec3[3](
        mul(model_matrix, daxa_f32vec4(vertex_positions[0],1)).xyz,
        mul(model_matrix, daxa_f32vec4(vertex_positions[1],1)).xyz,
        mul(model_matrix, daxa_f32vec4(vertex_positions[2],1)).xyz
    );

    const daxa_f32vec4[3] clipspace_vertex_positions = daxa_f32vec4[3](
        mul(view_proj, float4(world_vertex_positions[0].xyz, 1.0f)),
        mul(view_proj, float4(world_vertex_positions[1].xyz, 1.0f)),
        mul(view_proj, float4(world_vertex_positions[2].xyz, 1.0f))
    );
    
    BarycentricDeriv bari_deriv = calc_bary_and_deriv(
        clipspace_vertex_positions[0],
        clipspace_vertex_positions[1],
        clipspace_vertex_positions[2],
        ndc_xy,
        screen_size
    );
    ret.tri_geo.barycentrics = bari_deriv.m_lambda;

    ret.tri_geo_point.world_position = interpolate_vec3(
        bari_deriv.m_lambda, 
        world_vertex_positions[0].xyz,
        world_vertex_positions[1].xyz,
        world_vertex_positions[2].xyz
    );

    const daxa_f32vec2 interp_zw = interpolate_vec2(
        bari_deriv.m_lambda,
        clipspace_vertex_positions[0].zw,
        clipspace_vertex_positions[1].zw,
        clipspace_vertex_positions[2].zw
    );
    ret.depth = interp_zw.r / interp_zw.g;

    const daxa_f32vec3[3] vertex_normals = daxa_f32vec3[3](
        deref_i(mesh.vertex_normals, ret.tri_geo.vertex_indices.x),
        deref_i(mesh.vertex_normals, ret.tri_geo.vertex_indices.y),
        deref_i(mesh.vertex_normals, ret.tri_geo.vertex_indices.z)
    );

    // WARNING: WE ACTUALLY NEED THE TRANSPOSE INVERSE HERE
    const daxa_f32vec4[3] worldspace_vertex_normals = daxa_f32vec4[3](
        mul(model_matrix, daxa_f32vec4(vertex_normals[0], 0)),
        mul(model_matrix, daxa_f32vec4(vertex_normals[1], 0)),
        mul(model_matrix, daxa_f32vec4(vertex_normals[2], 0))
    );

    ret.tri_geo_point.world_normal = normalize(interpolate_vec3(
        bari_deriv.m_lambda, 
        worldspace_vertex_normals[0].xyz,
        worldspace_vertex_normals[1].xyz,
        worldspace_vertex_normals[2].xyz
    ));


    ret.tri_geo_point.uv = {};
    ret.tri_geo_point.uv_ddx = {};
    ret.tri_geo_point.uv_ddy = {};
    daxa_f32vec2[3] vertex_uvs = {};
    if (mesh.vertex_uvs != Ptr<float2>(0))
    {
        vertex_uvs = daxa_f32vec2[3](
            deref_i(mesh.vertex_uvs, ret.tri_geo.vertex_indices.x),
            deref_i(mesh.vertex_uvs, ret.tri_geo.vertex_indices.y),
            deref_i(mesh.vertex_uvs, ret.tri_geo.vertex_indices.z)
        );

        ret.tri_geo_point.uv = interpolate_vec2(
            bari_deriv.m_lambda, 
            vertex_uvs[0],
            vertex_uvs[1],
            vertex_uvs[2]
        );

        ret.tri_geo_point.uv_ddx = interpolate_vec2(
            bari_deriv.m_ddx, 
            vertex_uvs[0],
            vertex_uvs[1],
            vertex_uvs[2]
        );

        ret.tri_geo_point.uv_ddy = interpolate_vec2(
            bari_deriv.m_ddy, 
            vertex_uvs[0],
            vertex_uvs[1],
            vertex_uvs[2]
        );
    }

    // Calculate Face Normal
    ret.tri_geo_point.face_normal = normalize(cross(world_vertex_positions[1].xyz - world_vertex_positions[0].xyz, world_vertex_positions[2].xyz - world_vertex_positions[0].xyz));

    if ((mesh.vertex_uvs != Ptr<float2>(0)) && !all(ret.tri_geo_point.world_normal == float3(0,0,0)))
    {
        geom_compute_uv_tangent(world_vertex_positions, vertex_uvs, ret.tri_geo_point.world_tangent, ret.tri_geo_point.world_bitangent);
    }
    else // When no uvs are available we still want a tangent to construct a tbn even if its not aligned to anything
    {
        ret.tri_geo_point.world_tangent = geom_compute_arb_tangent(world_vertex_positions);
    }

    return ret;
}