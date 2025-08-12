#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "rtgi_trace_diffuse.inl"
#include "rtgi_trace_diffuse_shared.hlsl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/raytracing.hlsl"
#include "shader_lib/transform.hlsl"

#include "rtgi_shared.hlsl"

#define PI 3.1415926535897932384626433832795

[shader("raygeneration")]
void ray_gen()
{
    let clk_start = clockARB();
    let push = rtgi_trace_diffuse_push;
    const int2 dtid = DispatchRaysIndex().xy;

    const float depth = push.attach.view_cam_half_res_depth.get()[dtid];
    if(depth > 0.0f)
    {
        const float2 pixel_index = float2(dtid.xy * 2u) + 0.5f;
        const CameraInfo camera = push.attach.globals.view_camera;
        const float3 world_position = pixel_index_to_world_space(camera, pixel_index, depth);
        const float3 face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[dtid].r);
        const float3 primary_ray = normalize(world_position - push.attach.globals.view_camera.position);
            
        if (push.debug_primary_trace)
        {
            RayPayload payload = {};
            
            RayDesc ray = {};
            ray.Origin = camera.position;
            ray.TMax = 1000000000.0f;
            ray.TMin = 0.0f;
            ray.Direction = primary_ray;

            payload.color = float3(0,0,0);
            TraceRay(push.attach.tlas.get(), 0, ~0, 0, 0, 0, ray, payload);

            float4 value = float4(payload.color, 1.0f);
            push.attach.rtgi_diffuse_raw.get()[dtid.xy] = float4(payload.color,payload.t);
            return;
        }

        float4 acc = float4( 0, 0, 0, 0 );
        float2 acc2 = float2( 0, 0 );

        static const uint SAMPLES = 1;
        for (uint i = 0; i < SAMPLES; ++i)
        {
            const float3 sample_pos = rt_calc_ray_start(world_position, face_normal, primary_ray);
            const float3 world_tangent = normalize(cross(face_normal, float3(0,0,1) + 0.0001));
            const float3x3 tbn = transpose(float3x3(world_tangent, cross(world_tangent, face_normal), face_normal));
                
            const uint thread_seed = (dtid.x * push.attach.globals->settings.render_target_size.y + dtid.y) * push.attach.globals.frame_index + i;
            rand_seed(thread_seed);
            const float3 importance_rand_hemi_sample = rand_cosine_sample_hemi();

            RayPayload payload = {};
            payload.dtid = dtid;

            RayDesc ray = {};
            ray.Origin = sample_pos;
            ray.TMax = TMAX;
            ray.TMin = 0.0f;

            const float3 sample_dir = mul(tbn, importance_rand_hemi_sample);
            ray.Direction = sample_dir;
            TraceRay(push.attach.tlas.get(), 0, ~0, 0, 0, 0, ray, payload);

            #if RTGI_USE_SH
                float4 sh_y_new;
                float2 cocg_new;
                radiance_to_y_co_cg_sh(payload.color, sample_dir, sh_y_new, cocg_new);
                acc += sh_y_new * rcp(SAMPLES);
                acc2 += cocg_new * rcp(SAMPLES);
            #else
                acc += float4(payload.color, 0.0f) * rcp(SAMPLES);
            #endif
        }

        push.attach.rtgi_diffuse_raw.get()[dtid.xy] = acc;
        push.attach.rtgi_diffuse2_raw.get()[dtid.xy] = acc2;
    }

    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTGI_TRACE_DIFFUSE_CLOCKS)
    {
        let clk_end = clockARB();
        push.attach.clocks_image.get()[dtid] = uint(clk_end - clk_start);
    }
}

[shader("anyhit")]
void any_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = rtgi_trace_diffuse_push;

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