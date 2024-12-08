#pragma once

#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>

#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/debug.glsl"
#include "shader_lib/transform.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/sky_util.glsl"
#include "shader_lib/vsm_util.glsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/volumetric.hlsl"
#include "shader_lib/geometry.hlsl"

func rt_get_triangle_geo(
    float3 position, 
    float3 barycentrics,
    uint tlas_instance_index,
    uint tlas_instance_geometry_index,
    uint tlas_instance_geometry_triangle_index,
    GPUMesh* meshes,
    uint* entity_to_meshgroup,
    GPUMeshGroup* mesh_groups,
    daxa_f32mat4x3* combined_transforms) -> TriangleGeometry
{
    TriangleGeometry ret = {};
    ret.barycentrics = barycentrics;
    ret.entity_index = tlas_instance_index;
    uint mesh_group_index =  entity_to_meshgroup[ret.entity_index];
    const GPUMeshGroup* mesh_group = &mesh_groups[mesh_group_index];
    uint lod = 0;
    const uint mesh_index = mesh_group->mesh_lod_group_indices[tlas_instance_geometry_index] * MAX_MESHES_PER_LOD_GROUP + lod;
    ret.mesh_index = mesh_index;

    GPUMesh* mesh = meshes + ret.mesh_index;

    const daxa_u32vec3 vertex_indices = daxa_u32vec3(
        mesh.primitive_indices[tlas_instance_geometry_triangle_index * 3 + 0],
        mesh.primitive_indices[tlas_instance_geometry_triangle_index * 3 + 1],
        mesh.primitive_indices[tlas_instance_geometry_triangle_index * 3 + 2]
    );
    ret.vertex_indices = vertex_indices;

    return ret;
}

func rt_get_triangle_geo_point(
    float3 position, 
    float3 barycentric,
    TriangleGeometry surf_geo,
    GPUMesh* meshes,
    uint* entity_to_meshgroup,
    GPUMeshGroup* mesh_groups,
    daxa_f32mat4x3* combined_transforms) -> TriangleGeometryPoint
{
    TriangleGeometryPoint ret = {};

    daxa_f32mat4x4 model_matrix = mat_4x3_to_4x4(combined_transforms[surf_geo.entity_index]);

    GPUMesh* mesh = meshes + surf_geo.mesh_index;
    
    const daxa_f32vec3[3] vertex_positions = daxa_f32vec3[3](
        deref_i(mesh.vertex_positions, surf_geo.vertex_indices.x),
        deref_i(mesh.vertex_positions, surf_geo.vertex_indices.y),
        deref_i(mesh.vertex_positions, surf_geo.vertex_indices.z)
    );

    const daxa_f32vec4[3] world_vertex_positions = daxa_f32vec4[3](
        mul(model_matrix, daxa_f32vec4(vertex_positions[0],1)),
        mul(model_matrix, daxa_f32vec4(vertex_positions[1],1)),
        mul(model_matrix, daxa_f32vec4(vertex_positions[2],1))
    );    

    ret.world_position = interpolate_vec3(
        barycentric, 
        world_vertex_positions[0].xyz,
        world_vertex_positions[1].xyz,
        world_vertex_positions[2].xyz
    );
    
    const daxa_f32vec3[3] vertex_normals = daxa_f32vec3[3](
        deref_i(mesh.vertex_normals, surf_geo.vertex_indices.x),
        deref_i(mesh.vertex_normals, surf_geo.vertex_indices.y),
        deref_i(mesh.vertex_normals, surf_geo.vertex_indices.z)
    );

    // WARNING: WE ACTUALLY NEED THE TRANSPOSE INVERSE HERE
    const daxa_f32vec4[3] worldspace_vertex_normals = daxa_f32vec4[3](
        mul(model_matrix, daxa_f32vec4(vertex_normals[0], 0)),
        mul(model_matrix, daxa_f32vec4(vertex_normals[1], 0)),
        mul(model_matrix, daxa_f32vec4(vertex_normals[2], 0))
    );    
    
    ret.world_normal = normalize(interpolate_vec3(
        barycentric, 
        worldspace_vertex_normals[0].xyz,
        worldspace_vertex_normals[1].xyz,
        worldspace_vertex_normals[2].xyz
    ));    
    
    const daxa_f32vec2[3] vertex_uvs = daxa_f32vec2[3](
        deref_i(mesh.vertex_uvs, surf_geo.vertex_indices.x),
        deref_i(mesh.vertex_uvs, surf_geo.vertex_indices.y),
        deref_i(mesh.vertex_uvs, surf_geo.vertex_indices.z)
    );

    ret.uv = interpolate_vec2(
        barycentric, 
        vertex_uvs[0],
        vertex_uvs[1],
        vertex_uvs[2]
    );
    ret.uv_ddx = float2(0,0);
    ret.uv_ddy = float2(0,0);

    // Calculate Tangent.
    {
        float3 d_p1 = vertex_positions[1] - vertex_positions[1];
        float3 d_p2 = vertex_positions[2] - vertex_positions[1];
        float2 d_uv1 = vertex_uvs[1] - vertex_uvs[0];
        float2 d_uv2 = vertex_uvs[2] - vertex_uvs[0];
        float r = 1.0f / (d_uv1.x * d_uv2.y - d_uv2.x * d_uv1.y);

        ret.world_tangent = normalize(r * ( d_uv2.y * d_p1 - d_uv1.y * d_p2 ));
    }

    return ret;
}

struct SurfaceShadingData
{
    float3 albedo;
    float alpha;
    // float3 normal;
    // float3 geo_normal;
    // float3 position;
};

// Can be used to generate gbuffer values
func generate_surface_shading_data(RenderGlobalData* globals, TriangleGeometry tri_geo) -> SurfaceShadingData
{
    SurfaceShadingData ret = {};
    ret.alpha = 1.0f;
    ret.albedo = tri_geo.material.base_color;
    if (!tri_geo.material.diffuse_texture_id.is_empty())
    {
        float4 diffuse_fetch = Texture2D<float4>::get(tri_geo.material.diffuse_texture_id).SampleGrad(globals.samplers.linear_repeat_ani.get(), tri_geo.uv, tri_geo.uv_ddx, tri_geo.uv_ddy);
        ret.albedo *= diffuse_fetch.rgb;
        ret.alpha = diffuse_fetch.a;
    }

    return ret;
}

func shade_surface(RenderGlobalData* globals, SurfaceShadingData data, float3 incoming_ray) -> float4
{
    return float4(data.albedo, data.alpha);
}