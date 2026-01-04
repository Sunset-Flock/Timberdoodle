#pragma once

#include <daxa/daxa.inl>

#include "shader_lib/debug.glsl"
#include "shader_lib/transform.glsl"
#include "shader_lib/sky_util.glsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/geometry.hlsl"

#include "shader_shared/geometry.inl"
#include "shader_shared/globals.inl"

static const float RAY_MIN_POSITION_OFFSET = 0.01f;

float3 rt_calc_ray_start(float3 position, float3 geo_normal, float3 view_ray)
{
    float float_error_scaled_offset = RAY_MIN_POSITION_OFFSET * (max(position.x, max(position.y, position.z)) + 1.0f);
    return position + (geo_normal - view_ray * 2) * RAY_MIN_POSITION_OFFSET;
}

float rayquery_shadow_path(RaytracingAccelerationStructure tlas, float3 origin, float3 dir, float t_max, RenderGlobalData* globals, MeshInstance* mesh_instances)
{
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;

    const float t_min = 0.1f;

    RayDesc my_ray = {
        origin,
        t_min,
        dir,
        t_max*1.001f,
    };

    // Set up a trace.  No work is done yet.
    q.TraceRayInline(
        tlas,
        0, // OR'd with flags above
        0xFFFF,
        my_ray);

    bool hit = false;

    while(q.Proceed()) {
        // Examine and act on the result of the traversal.
        // Was a hit committed?
        if(q.CommittedStatus() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            TriangleGeometry tri_geo = rt_get_triangle_geo(
                q.CandidateRayBarycentrics(),
                q.CandidateInstanceID(),
                q.CandidateGeometryIndex(),
                q.CandidatePrimitiveIndex(),
                globals->scene.meshes,
                globals->scene.entity_to_meshgroup,
                globals->scene.mesh_groups,
                mesh_instances
            );

            const uint primitive_index = q.CandidateRayPrimitiveIndex();
    
            const uint mesh_instance_index = q.CandidateInstanceID();
            MeshInstance* mesh_instance = mesh_instances + mesh_instance_index;
            GPUMesh *mesh = globals->scene.meshes + mesh_instance->mesh_index;
            if ((mesh.vertex_uvs == Ptr<float2>(0)) || mesh.material_index == INVALID_MANIFEST_INDEX)
            {
                q.CommitNonOpaqueTriangleHit();
                hit = true;
            }

            GPUMaterial *material = globals->scene.materials + mesh.material_index;
            if (!material.alpha_discard_enabled || (material.opacity_texture_id.is_empty() && material.diffuse_texture_id.is_empty()))
            {
                q.CommitNonOpaqueTriangleHit();
                hit = true;
            }

            const int primitive_indices[3] = {
                mesh.primitive_indices[3 * primitive_index],
                mesh.primitive_indices[3 * primitive_index + 1],
                mesh.primitive_indices[3 * primitive_index + 2],
            };

            const float2 uvs[3] = {
                mesh.vertex_uvs[primitive_indices[0]],
                mesh.vertex_uvs[primitive_indices[1]],
                mesh.vertex_uvs[primitive_indices[2]],
            };
            const float2 interp_uv = uvs[0] + q.CandidateRayBarycentrics().x * (uvs[1] - uvs[0]) + q.CandidateRayBarycentrics().y* (uvs[2] - uvs[0]);


            float opacity = 1.0f;
            if(material.opacity_texture_id.value != 0 && material.alpha_discard_enabled)
            {
                opacity = Texture2D<float>::get(material.opacity_texture_id)
                    .SampleLevel(SamplerState::get(globals->samplers.linear_repeat), interp_uv, 5).r;
            }
            else if(material.diffuse_texture_id.value != 0 && material.alpha_discard_enabled)
            {
                opacity = Texture2D<float4>::get(material.diffuse_texture_id)
                    .SampleLevel(SamplerState::get(globals->samplers.linear_repeat), interp_uv, 5).a;
            }

            if (opacity > 0.5)
            {
                q.CommitNonOpaqueTriangleHit();
                hit = true;
            }
        }
    }



    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        return t_max;
    }
    else
    {
        // return q.CommittedRayT();
        return q.CandidateTriangleRayT();
    }
}

float rayquery_free_path(RaytracingAccelerationStructure tlas, float3 origin, float3 dir, float t_max)
{
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_NON_OPAQUE> q;

    const float t_min = 0.000f;

    RayDesc my_ray = {
        origin,
        t_min,
        dir,
        t_max*1.001f,
    };

    // Set up a trace.  No work is done yet.
    q.TraceRayInline(
        tlas,
        0, // OR'd with flags above
        0xFFFF,
        my_ray);

    q.Proceed();

    bool hit = false;
    // Examine and act on the result of the traversal.
    // Was a hit committed?
    if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        hit = true;
    }


    if (!hit)
    {
        return t_max;
    }
    else
    {
        return q.CandidateTriangleRayT();
    }
}

func rt_get_triangle_geo(
    float2 rt_barycentrics,
    uint tlas_instance_index,
    uint tlas_instance_geometry_index,
    uint tlas_instance_geometry_triangle_index,
    GPUMesh* meshes,
    uint* entity_to_meshgroup,
    GPUMeshGroup* mesh_groups,
    MeshInstance* mesh_instances) -> TriangleGeometry
{
    TriangleGeometry ret = {};

    float3 barycentrics = float3(1.0f - (rt_barycentrics.x + rt_barycentrics.y), rt_barycentrics.x, rt_barycentrics.y );
    ret.barycentrics = barycentrics;

    const uint mesh_instance_index = tlas_instance_index;
    MeshInstance mesh_instance = mesh_instances[mesh_instance_index];
    ret.entity_index = mesh_instance.entity_index;
    ret.mesh_index = mesh_instance.mesh_index;
    ret.in_mesh_group_index = mesh_instance.in_mesh_group_index;
    ret.mesh_instance_index = mesh_instance_index;

    GPUMesh* mesh = meshes + ret.mesh_index;
    ret.material_index = mesh.material_index;

    const daxa_u32vec3 vertex_indices = daxa_u32vec3(
        mesh.primitive_indices[tlas_instance_geometry_triangle_index * 3 + 0],
        mesh.primitive_indices[tlas_instance_geometry_triangle_index * 3 + 1],
        mesh.primitive_indices[tlas_instance_geometry_triangle_index * 3 + 2]
    );
    ret.vertex_indices = vertex_indices;

    return ret;
}

func rt_get_triangle_geo_point(
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

    const daxa_f32vec3[3] world_vertex_positions = daxa_f32vec3[3](
        mul(model_matrix, daxa_f32vec4(vertex_positions[0],1)).xyz,
        mul(model_matrix, daxa_f32vec4(vertex_positions[1],1)).xyz,
        mul(model_matrix, daxa_f32vec4(vertex_positions[2],1)).xyz
    );    

    ret.world_position = interpolate_vec3(
        surf_geo.barycentrics, 
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
        surf_geo.barycentrics, 
        worldspace_vertex_normals[0].xyz,
        worldspace_vertex_normals[1].xyz,
        worldspace_vertex_normals[2].xyz
    ));    
    
    daxa_f32vec2[3] vertex_uvs = {};
    if (mesh.vertex_uvs != Ptr<float2>(0))
    {
        vertex_uvs = daxa_f32vec2[3](
            deref_i(mesh.vertex_uvs, surf_geo.vertex_indices.x),
            deref_i(mesh.vertex_uvs, surf_geo.vertex_indices.y),
            deref_i(mesh.vertex_uvs, surf_geo.vertex_indices.z)
        );

        ret.uv = interpolate_vec2(
            surf_geo.barycentrics, 
            vertex_uvs[0],
            vertex_uvs[1],
            vertex_uvs[2]
        );
    }
    ret.uv_ddx = float2(0.1, 0.0);
    ret.uv_ddy = float2(0.0, 0.1);

    // Calculate Face Normal
    ret.face_normal = normalize(cross(world_vertex_positions[1].xyz - world_vertex_positions[0].xyz, world_vertex_positions[2].xyz - world_vertex_positions[0].xyz));

    // Calculate Tangent.
    if ((mesh.vertex_uvs != Ptr<float2>(0)) && !all(ret.world_normal == float3(0,0,0)))
    {
        geom_compute_uv_tangent(world_vertex_positions, vertex_uvs, ret.world_tangent, ret.world_bitangent);
    }
    else // When no uvs are available we still want a tangent to construct a tbn even if its not aligned to anything
    {
        ret.world_tangent = geom_compute_arb_tangent(world_vertex_positions);
    }

    return ret;
}

struct RTLightVisibilityTester : LightVisibilityTesterI
{
    RaytracingAccelerationStructure tlas;
    RenderGlobalData* globals;
    float3 origin;
    float sun_light(MaterialPointData material_point, float3 incoming_ray)
    {
        let sky = globals->sky_settings;

        float t_max = 10000.0f;
        float3 start = rt_calc_ray_start(material_point.position, material_point.geometry_normal, incoming_ray);
        float3 dir = sky.sun_direction;
        float t = rayquery_free_path(tlas, start, dir, t_max);

        bool path_occluded = t != t_max;

        return path_occluded ? 0.0f : 1.0f;
    }
    float point_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        return 0.0f;
    }
    float spot_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        return 1.0f;
    }
}

func rt_is_alpha_hit(
    RenderGlobalData* globals,
    MeshInstancesBufferHead* mesh_instances,
    GPUMesh* meshes,
    GPUMaterial* materials,
    float2 barycentrics
    ) -> bool
{
    const float3 hit_location = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    const uint primitive_index = PrimitiveIndex();
    
    const uint mesh_instance_index = InstanceID();
    MeshInstance* mesh_instance = mesh_instances.instances + mesh_instance_index;
    GPUMesh *mesh = meshes + mesh_instance->mesh_index;
    if ((mesh.vertex_uvs == Ptr<float2>(0)) || mesh.material_index == INVALID_MANIFEST_INDEX)
    {
        return true;
    }

    GPUMaterial *material = materials + mesh.material_index;
    if (!material.alpha_discard_enabled || (material.opacity_texture_id.is_empty() && material.diffuse_texture_id.is_empty()))
    {
        return true;
    }

    const int primitive_indices[3] = {
        mesh.primitive_indices[3 * primitive_index],
        mesh.primitive_indices[3 * primitive_index + 1],
        mesh.primitive_indices[3 * primitive_index + 2],
    };

    const float2 uvs[3] = {
        mesh.vertex_uvs[primitive_indices[0]],
        mesh.vertex_uvs[primitive_indices[1]],
        mesh.vertex_uvs[primitive_indices[2]],
    };
    const float2 interp_uv = uvs[0] + barycentrics.x * (uvs[1] - uvs[0]) + barycentrics.y* (uvs[2] - uvs[0]);


    float opacity = 1.0f;
    if(material.opacity_texture_id.value != 0 && material.alpha_discard_enabled)
    {
        opacity = Texture2D<float>::get(material.opacity_texture_id)
            .SampleLevel(SamplerState::get(globals->samplers.linear_repeat), interp_uv, 5).r;
    }
    else if(material.diffuse_texture_id.value != 0 && material.alpha_discard_enabled)
    {
        opacity = Texture2D<float4>::get(material.diffuse_texture_id)
            .SampleLevel(SamplerState::get(globals->samplers.linear_repeat), interp_uv, 5).a;
    }
    return opacity > 0.5;
}