#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "ray_tracing.inl"

#include "shader_lib/visbuffer.glsl"

#define PI 3.1415926535897932384626433832795

static uint _rand_state;
void rand_seed(uint seed) {
    _rand_state = seed;
}

float rand() {
    // https://www.pcg-random.org/
    _rand_state = _rand_state * 747796405u + 2891336453u;
    uint result = ((_rand_state >> ((_rand_state >> 28u) + 4u)) ^ _rand_state) * 277803737u;
    result = (result >> 22u) ^ result;
    return result / 4294967295.0;
}

float rand_normal_dist() {
    float theta = 2.0 * PI * rand();
    float rho = sqrt(-2.0 * log(rand()));
    return rho * cos(theta);
}

float3 rand_dir() {
    return normalize(float3(
        rand_normal_dist(),
        rand_normal_dist(),
        rand_normal_dist()));
}

float3 rand_hemi_dir(float3 nrm) {
    float3 result = rand_dir();
    return result * sign(dot(nrm, result));
}

[[vk::push_constant]] RayTraceAmbientOcclusionPush rt_ao_push;
[numthreads(RT_AO_X, RT_AO_Y, 1)]
[shader("compute")]
void entry_rt_ao(
    uint3 svtid : SV_DispatchThreadID
){
    let push = rt_ao_push;
    const int2 index = svtid.xy;
    const float2 screen_uv = float2(svtid.xy) * push.attach.globals->settings.render_target_size_inv;

    uint triangle_id;
    if(all(lessThan(index, push.attach.globals->settings.render_target_size)))
    {
        triangle_id = push.attach.vis_image.get().Load(int3(index, 0), int2(0)).x;
    } else {
        triangle_id = INVALID_TRIANGLE_ID;
    }

    float4 output_value = float4(0);
    float4 debug_value = float4(0);

    bool triangle_id_valid = triangle_id != INVALID_TRIANGLE_ID;

    if(triangle_id_valid)
    {
        float4x4 view_proj;
        float3 camera_position;
        if(push.attach.globals->settings.draw_from_observer == 1)
        {
            view_proj = push.attach.globals->observer_camera.view_proj;
            camera_position = push.attach.globals->observer_camera.position;
        }
        else 
        {
            view_proj = push.attach.globals->camera.view_proj;
            camera_position = push.attach.globals->camera.position;
        }

        MeshletInstancesBufferHead* instantiated_meshlets = push.attach.instantiated_meshlets;
        GPUMesh* meshes = push.attach.meshes;
        daxa_f32mat4x3* combined_transforms = push.attach.combined_transforms;
        VisbufferTriangleData tri_data = visgeo_triangle_data(
            triangle_id,
            float2(index),
            push.attach.globals->settings.render_target_size,
            push.attach.globals->settings.render_target_size_inv,
            view_proj,
            instantiated_meshlets,
            meshes,
            combined_transforms
        );
        float3 normal = tri_data.world_normal;
        
        const uint AO_RAY_COUNT = push.attach.globals.settings.ao_samples;

        const uint thread_seed = (svtid.x * push.attach.globals->settings.render_target_size.y + svtid.y) * push.attach.globals.frame_index;
        rand_seed(AO_RAY_COUNT * thread_seed);
        uint hit_count = 0;
        for (uint ray_i = 0; ray_i < AO_RAY_COUNT; ++ray_i)
        {
            // Instantiate ray query object.
            // Template parameter allows driver to generate a specialized
            // implementation.
            RayQuery<RAY_FLAG_CULL_NON_OPAQUE |
                    RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
                    RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;

            const float t_min = 0.01f;
            const float t_max = 15.0f;

            RayDesc my_ray = {
                tri_data.world_position,
                t_min,
                rand_hemi_dir(normal),
                t_max,
            };

            // Set up a trace.  No work is done yet.
            q.TraceRayInline(
                daxa::acceleration_structures[push.attach.tlas.index()],
                0, // OR'd with flags above
                0xFFFF,
                my_ray);

            // Proceed() below is where behind-the-scenes traversal happens,
            // including the heaviest of any driver inlined code.
            // In this simplest of scenarios, Proceed() only needs
            // to be called once rather than a loop:
            // Based on the template specialization above,
            // traversal completion is guaranteed.
            q.Proceed();

            // Examine and act on the result of the traversal.
            // Was a hit committed?
            if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
            {
                hit_count += 1;
            }
            else // COMMITTED_NOTHING
                // From template specialization,
                // COMMITTED_PROCEDURAL_PRIMITIVE can't happen.
            {
                // Do miss shading
            }
        }

        let ao_value = 1.0f - float(hit_count) * rcp(AO_RAY_COUNT);
        push.attach.ao_image.get()[svtid.xy] = ao_value;
        //push.attach.debug_image.get()[svtid.xy] = float4(ao_value.xxx, 1.0f);
    }
}

struct RayPayload
{
    bool hit;
}

[shader("raygeneration")]
void ray_gen()
{
    let push = rt_ao_push;
    const int2 index = DispatchRaysIndex().xy;
    const float2 screen_uv = index * push.attach.globals->settings.render_target_size_inv;

    uint triangle_id;
    if(all(lessThan(index, push.attach.globals->settings.render_target_size)))
    {
        triangle_id = push.attach.vis_image.get().Load(int3(index, 0), int2(0)).x;
    } else {
        triangle_id = INVALID_TRIANGLE_ID;
    }

    float4 output_value = float4(0);
    float4 debug_value = float4(0);

    bool triangle_id_valid = triangle_id != INVALID_TRIANGLE_ID;

    if(triangle_id_valid)
    {
        float4x4 view_proj;
        float3 camera_position;
        if(push.attach.globals->settings.draw_from_observer == 1)
        {
            view_proj = push.attach.globals->observer_camera.view_proj;
            camera_position = push.attach.globals->observer_camera.position;
        }
        else 
        {
            view_proj = push.attach.globals->camera.view_proj;
            camera_position = push.attach.globals->camera.position;
        }

        MeshletInstancesBufferHead* instantiated_meshlets = push.attach.instantiated_meshlets;
        GPUMesh* meshes = push.attach.meshes;
        daxa_f32mat4x3* combined_transforms = push.attach.combined_transforms;
        VisbufferTriangleData tri_data = visgeo_triangle_data(
            triangle_id,
            float2(index),
            push.attach.globals->settings.render_target_size,
            push.attach.globals->settings.render_target_size_inv,
            view_proj,
            instantiated_meshlets,
            meshes,
            combined_transforms
        );
        float3 normal = tri_data.world_normal;
        
        const uint AO_RAY_COUNT = push.attach.globals.settings.ao_samples;

        const uint thread_seed = (index.x * push.attach.globals->settings.render_target_size.y + index.y) * push.attach.globals.frame_index;
        rand_seed(AO_RAY_COUNT * thread_seed);
        uint hit_count = 0;

        RayDesc ray = {};
        ray.Origin = tri_data.world_position;
        ray.TMax = 15.0f;
        ray.TMin = 0.01f;

        RaytracingAccelerationStructure tlas = daxa::acceleration_structures[push.attach.tlas.index()];
        RayPayload payload = {};
        for (uint ray_i = 0; ray_i < AO_RAY_COUNT; ++ray_i)
        {
            ray.Direction = rand_hemi_dir(normal);
            TraceRay(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, 0, 0, 0, ray, payload);
            hit_count += payload.hit ? 1 : 0;
        }

        let ao_value = 1.0f - float(hit_count) * rcp(AO_RAY_COUNT);
        push.attach.ao_image.get()[index.xy] = ao_value;
        //push.attach.debug_image.get()[index.xy] = float4(ao_value.xxx, 1.0f);
    }
}

[shader("anyhit")]
void any_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    const float3 hit_location = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    const uint geometry_index = GeometryIndex();
    const uint primitive_index = PrimitiveIndex() * 3;
    GPUMesh *mesh = {};
    if(rt_ao_push.attach.globals.settings.enable_merged_scene_blas)
    {
        let geo_indirect_info = rt_ao_push.attach.geo_inst_indirections[geometry_index];
        mesh = &rt_ao_push.attach.meshes[geo_indirect_info.mesh_index];
    }
    else
    {
        const uint custom_instance_index = InstanceID();
        const GPUMeshGroup * mesh_group = &rt_ao_push.attach.mesh_groups[custom_instance_index];
        const uint mesh_index = mesh_group->mesh_indices[geometry_index];
        mesh = &rt_ao_push.attach.meshes[mesh_index];
    }

    const int primitive_indices[3] = {
        mesh.primitive_indices[primitive_index],
        mesh.primitive_indices[primitive_index + 1],
        mesh.primitive_indices[primitive_index + 2],
    };

    const float2 uvs[3] = {
        mesh.vertex_uvs[primitive_indices[0]],
        mesh.vertex_uvs[primitive_indices[1]],
        mesh.vertex_uvs[primitive_indices[2]],
    };
    const float2 interp_uv = uvs[0] + attr.barycentrics.x * (uvs[1] - uvs[0]) + attr.barycentrics.y* (uvs[2] - uvs[0]);
    const GPUMaterial *material = &rt_ao_push.attach.material_manifest[mesh.material_index];
    const bool has_opacity_texture = !material.opacity_texture_id.is_empty();
    const bool alpha_discard_enabled = material.alpha_discard_enabled;
    if(has_opacity_texture && alpha_discard_enabled)
    {
        let oppacity_tex = Texture2D<float>::get(material.opacity_texture_id);
        let oppacity = oppacity_tex.SampleLevel(SamplerState::get(rt_ao_push.attach.globals->samplers.linear_repeat), interp_uv, 2).r;
        if(oppacity < 0.5) {
            IgnoreHit();
        }
    }
}

[shader("closesthit")]
void closest_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.hit = true;
}

[shader("miss")]
void miss(inout RayPayload payload)
{
    payload.hit = false;
}


#define DENOISER_TAP_WIDTH 3
[[vk::push_constant]] RTAODenoiserPush denoiser_push;
[shader("compute")]
[numthreads(RTAO_DENOISER_X, RTAO_DENOISER_Y, 1)]
void entry_rtao_denoiser(uint2 svtid : SV_DispatchThreadID)
{
    let p = denoiser_push;

    if (any(greaterThan(svtid, p.size)))
    {
        return;
    }

    // Reads and box blurrs a 6x6 area.
    // Uses 3x3 taps. Each tap is doing a 2x2 linear interpolation.
    // All samples are averaged for the box blur.
    const int2 start_index = int2(svtid) - int2(2 * (DENOISER_TAP_WIDTH/2), 2 * (DENOISER_TAP_WIDTH/2));
    const float2 start_uv = float2(start_index) * p.inv_size;
    const float2 tap_uv_increment = 2.0f * p.inv_size;
    float average = 0.0f;
    for (uint i = 0; i < DENOISER_TAP_WIDTH * DENOISER_TAP_WIDTH; ++i)
    {
        let x = i % DENOISER_TAP_WIDTH;
        let y = i / DENOISER_TAP_WIDTH;

        let v = p.attach.src.get().SampleLevel(p.attach.globals.samplers.linear_clamp.get(), start_uv + tap_uv_increment * float2(x,y), 0.0f).r;
        average += v * (1.0f / (DENOISER_TAP_WIDTH * DENOISER_TAP_WIDTH));
    }
    p.attach.dst.get()[svtid] = average;
}