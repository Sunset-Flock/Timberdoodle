#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "rtao.inl"

#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/raytracing.hlsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/transform.hlsl"

#define PI 3.1415926535897932384626433832795

[[vk::push_constant]] RayTraceAmbientOcclusionPush rt_ao_push;

float3 rand_dir() {
    return normalize(float3(
        rand() * 2.0f - 1.0f,
        rand() * 2.0f - 1.0f,
        rand() * 2.0f - 1.0f));
}

float3 rand_hemi_dir(float3 nrm) {
    float3 result = rand_dir();
    return result * sign(dot(nrm, result));
}

float2 concentric_sample_disc()
{
    float r = sqrt(rand());
    float theta = rand() * 2 * PI;
    return float2(cos(theta), sin(theta)) * r;
}

float3 cosine_sample_hemi()
{
    float2 d = concentric_sample_disc();
    float z = sqrt(max(0.0f, 1.0f - d.x * d.x - d.y * d.y));
    return float3(d.x, d.y, z);
}

struct RayPayload
{
    bool miss;
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
        triangle_id = push.attach.view_cam_visbuffer.get()[index].x;
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

        MeshletInstancesBufferHead* instantiated_meshlets = push.attach.meshlet_instances;
        GPUMesh* meshes = push.attach.globals.scene.meshes;
        daxa_f32mat4x3* combined_transforms = push.attach.globals.scene.entity_combined_transforms;
        VisbufferTriangleGeometry visbuf_tri = visgeo_triangle_data(
            triangle_id,
            float2(index),
            push.attach.globals->settings.render_target_size,
            push.attach.globals->settings.render_target_size_inv,
            view_proj,
            instantiated_meshlets,
            meshes,
            combined_transforms
        );
        TriangleGeometry tri_geo = visbuf_tri.tri_geo;
        TriangleGeometryPoint tri_point = visbuf_tri.tri_geo_point;
        float depth = visbuf_tri.depth;
        uint meshlet_triangle_index = visbuf_tri.meshlet_triangle_index;
        uint meshlet_instance_index = visbuf_tri.meshlet_instance_index;
        uint meshlet_index = visbuf_tri.meshlet_index;
        const float3 detail_normal = uncompress_normal_octahedral_32(push.attach.view_cam_detail_normals.get()[index].r);
        const float3 primary_ray = normalize(tri_point.world_position - push.attach.globals.camera.position);
        const float3 corrected_face_normal = flip_normal_to_incoming(detail_normal, detail_normal, primary_ray);
        const float3 sample_pos = rt_calc_ray_start(tri_point.world_position, corrected_face_normal, primary_ray);
        const float3x3 tbn = transpose(float3x3(tri_point.world_tangent, cross(tri_point.world_tangent, corrected_face_normal), corrected_face_normal));
            
        const uint AO_RAY_COUNT = push.attach.globals.settings.ao_samples;
        const uint thread_seed = (index.x * push.attach.globals->settings.render_target_size.y + index.y) * push.attach.globals.frame_index;
        rand_seed(AO_RAY_COUNT * thread_seed);

        RayDesc ray = {};
        ray.Origin = sample_pos;
        ray.TMax = 2.0f;
        ray.TMin = 0.0f;

        RaytracingAccelerationStructure tlas = daxa::acceleration_structures[push.attach.tlas.index()];
        RayPayload payload = {};
        float ao_factor = 0.0f;
        for (uint ray_i = 0; ray_i < AO_RAY_COUNT; ++ray_i)
        {
            const float3 hemi_sample = cosine_sample_hemi();
            const float3 sample_dir = mul(tbn, hemi_sample);
            ray.Direction = sample_dir;
            payload.miss = false; // need to set to false as we skip the closest hit shader
            TraceRay(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, 0, 0, 0, ray, payload);
            ao_factor += payload.miss ? 0 : 1;
        }

        let ao_value = 1.0f - ao_factor * rcp(AO_RAY_COUNT);
        push.attach.ao_image.get()[index.xy] = ao_value;
        //push.attach.debug_image.get()[index.xy] = float4(ao_value.xxx, 1.0f);
    }
}

[shader("anyhit")]
void any_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = rt_ao_push;

    if (!rt_is_alpha_hit(
        push.attach.globals,
        push.attach.mesh_instances,
        push.attach.globals.scene.meshes,
        push.attach.globals.scene.materials,
        attr.barycentrics))
    {
        IgnoreHit();
    }
}

[shader("closesthit")]
void closest_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = rt_ao_push;

    payload.miss = false;

    let primitive_index = PrimitiveIndex();
    let instance_id = InstanceID();
    let geometry_index = GeometryIndex();
    
    TriangleGeometry tri_geo = rt_get_triangle_geo(
        attr.barycentrics,
        instance_id,
        geometry_index,
        primitive_index,
        push.attach.globals.scene.meshes,
        push.attach.globals.scene.entity_to_meshgroup,
        push.attach.globals.scene.mesh_groups,
        push.attach.mesh_instances.instances
    );

    if (tri_geo.material_index != INVALID_MANIFEST_INDEX)
    {
        const GPUMaterial material = push.attach.globals.scene.materials[tri_geo.material_index];

        bool emissive = any(material.emissive_color > 0.0f);
        payload.miss = emissive;
    }
}

[shader("miss")]
void miss(inout RayPayload payload)
{
    payload.miss = true;
}


#define DENOISER_TAP_WIDTH 1
[[vk::push_constant]] RTAODenoiserPush denoiser_push;
[shader("compute")]
[numthreads(RTAO_DENOISER_X, RTAO_DENOISER_Y, 1)]
void entry_rtao_denoiser(int2 index : SV_DispatchThreadID)
{
    let push = denoiser_push;

    if (any(greaterThan(index, push.size)))
    {
        return;
    }


    // Blur new samples.
    float blurred_new_ao = {};
    {
        // Reads and box blurrs a 6x6 area.
        // Uses 3x3 taps. Each tap is doing a 2x2 linear interpolation.
        // All samples are averaged for the box blur.
        const int2 start_index = index - int2(2 * (DENOISER_TAP_WIDTH/2), 2 * (DENOISER_TAP_WIDTH/2));
        const float2 start_uv = float2(start_index) * push.inv_size;
        const float2 tap_uv_increment = 2.0f * push.inv_size;
        for (uint i = 0; i < DENOISER_TAP_WIDTH * DENOISER_TAP_WIDTH; ++i)
        {
            let x = i % DENOISER_TAP_WIDTH;
            let y = i / DENOISER_TAP_WIDTH;

            let v = push.attach.src.get().SampleLevel(push.attach.globals.samplers.linear_clamp.get(), start_uv + tap_uv_increment * float2(x,y), 0.0f).r;
            blurred_new_ao += v * (1.0f / (DENOISER_TAP_WIDTH * DENOISER_TAP_WIDTH));
        }
    }

    float4 prev_frame_ndc = {};
    {
        float4x4 view_proj = push.attach.globals->camera.view_proj;
        float4x4 view_proj_prev = push.attach.globals->camera_prev_frame.view_proj;
        float3 camera_position = push.attach.globals->camera.position;
        float non_linear_depth = push.attach.depth.get()[index];
        float3 pixel_world_position = pixel_index_to_world_space(push.attach.globals.camera, index, non_linear_depth);

        prev_frame_ndc = mul(view_proj_prev, float4(pixel_world_position, 1.0f));
        prev_frame_ndc.xyz /= prev_frame_ndc.w;
    }

    float accepted_history_ao = {};
    float acceptance = 0.0f;
    {
        float2 pixel_prev_uv = (prev_frame_ndc.xy * 0.5f + 0.5f);
        float2 pixel_prev_index = pixel_prev_uv * float2(push.attach.globals.camera.screen_size);
        float2 base_texel = max(float2(0,0), floor(pixel_prev_index - 0.5f)); 
        float2 interpolants = clamp((pixel_prev_index - (base_texel + 0.5f)), float2(0,0), float2(1,1));
        float2 gather_uv = (base_texel + 1.0f) / float2(push.attach.globals.camera.screen_size);
        
        float4 history_ao = push.attach.history.get().GatherRed(push.attach.globals.samplers.nearest_clamp.get(), gather_uv).wzxy;
        float4 history_depth = push.attach.history.get().GatherGreen(push.attach.globals.samplers.linear_clamp.get(), gather_uv).wzxy;
        //history_ao = max(history_ao, 0.0f);
        float4 sample_linear_weights = float4(
            (1.0f - interpolants.x) * (1.0f - interpolants.y),
            interpolants.x * (1.0f - interpolants.y),
            (1.0f - interpolants.x) * interpolants.y,
            interpolants.x * interpolants.y
        );

        float linear_prev_depth = linearise_depth(prev_frame_ndc.z, push.attach.globals.camera.near_plane);
        float interpolated_ao = 0.0f;
        float interpolated_ao_weight_sum = 0.0f;
        for (int i = 0; i < 4; ++i)
        {
            float depth = history_depth[i];
            float linear_depth = linearise_depth(depth, push.attach.globals.camera.near_plane);
            float depth_diff = abs(linear_prev_depth - linear_depth);
            float acceptable_diff = 0.05f;

            if (depth_diff < acceptable_diff)
            {
                interpolated_ao_weight_sum += sample_linear_weights[i];
                interpolated_ao += history_ao[i] * sample_linear_weights[i];
            }
        }
        if (interpolated_ao_weight_sum > 0.01f)
        {
            acceptance = 1.0f;
            accepted_history_ao = interpolated_ao * rcp(interpolated_ao_weight_sum);
        }
        if (any(base_texel < 0.0f || (base_texel + 1.0f >= float2(push.attach.globals.camera.screen_size))))
        {
            acceptance = 0.0f;
        }
        //accepted_history_ao = push.attach.history.get().SampleLevel(push.attach.globals.samplers.linear_clamp.get(), pixel_prev_uv, 0.0f).x;
        push.attach.debug_image.get()[index].xy = interpolated_ao;
    }

    float exp_average = lerp(1.0f, 0.02f, acceptance);

    float new_ao = lerp(accepted_history_ao, blurred_new_ao, exp_average);
    float new_depth = push.attach.depth.get()[index];

    push.attach.dst.get()[index] = float2(new_ao, new_depth);
}