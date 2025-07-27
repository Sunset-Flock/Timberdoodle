#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "rtgi_trace_diffuse.inl"

#include "shader_lib/misc.hlsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/raytracing.hlsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/shading.hlsl"

#define PI 3.1415926535897932384626433832795

[[vk::push_constant]] RtgiTraceDiffusePush rtgi_trace_diffuse_push;

struct RayPayload
{
    float3 color;    
    float t;
};

static const float3 sky_color = float3(0.5f, 0.7f, 1.0f);
static const float TMAX = 100000000000.0f; // Arbitrary large value

[shader("raygeneration")]
void ray_gen()
{
    let clk_start = clockARB();
    let push = rtgi_trace_diffuse_push;
    const int2 dtid = DispatchRaysIndex().xy;

    const float depth = push.attach.view_cam_half_res_depth.get()[dtid];
    float2 pixel_index = float2(dtid.xy * 2u) + 0.5f;

    if(depth > 0.0f)
    {
        CameraInfo camera = push.attach.globals.view_camera;
        const float3 world_position = pixel_index_to_world_space(camera, pixel_index, depth);
        const float3 face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[dtid].r);
        const float3 primary_ray = normalize(world_position - push.attach.globals.view_camera.position);
        
        const float3 sample_pos = rt_calc_ray_start(world_position, face_normal, primary_ray);
        const float3 world_tangent = normalize(cross(face_normal, float3(0,0,1) + 0.0001));
        const float3x3 tbn = transpose(float3x3(world_tangent, cross(world_tangent, face_normal), face_normal));
            
        const uint thread_seed = (dtid.x * push.attach.globals->settings.render_target_size.y + dtid.y) * push.attach.globals.frame_index;
        rand_seed(thread_seed);
        const float3 importance_rand_hemi_sample = rand_cosine_sample_hemi();

        RayPayload payload = {};

        RayDesc ray = {};
        ray.Origin = sample_pos;
        ray.TMax = TMAX;
        ray.TMin = 0.0f;

        float ao_factor = 0.0f;
        const float3 sample_dir = mul(tbn, importance_rand_hemi_sample);
        ray.Direction = sample_dir;
        payload = (RayPayload)(0);
        TraceRay(push.attach.tlas.get(), 0, ~0, 0, 0, 0, ray, payload);

        push.attach.rtgi_diffuse_raw.get()[dtid.xy] = float4(payload.color,payload.t);
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

func trace_shadow_ray(RaytracingAccelerationStructure tlas, float3 position, float3 light_position, float3 flat_normal, float3 incoming_ray) -> bool
{
    float3 start = rt_calc_ray_start(position, flat_normal, incoming_ray);

    RayDesc ray = {};
    ray.Direction = normalize(light_position - position);
    ray.Origin = start;
    ray.TMax = length(light_position - position) * 1.01f;
    ray.TMin = 0.0f;

    RayPayload payload = {};
    TraceRay(tlas, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, ~0, 0, 0, 0, ray, payload);

    return payload.t == TMAX;
}

[shader("closesthit")]
void closest_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = rtgi_trace_diffuse_push;

    const float3 position = WorldRayDirection() + WorldRayDirection() * RayTCurrent();
    const float3 sun_position = push.attach.globals.sky_settings.sun_direction * TMAX * 2 + position;
    const float3 surface_normal = float3(0,0,0);

    let miss = trace_shadow_ray(
        push.attach.tlas.get(),
        position,
        sun_position,
        surface_normal,
        WorldRayDirection());

    if (miss)
    {
        payload.color = sky_color;
        payload.t = TMAX;
    }
    else
    {
        payload.color = float3(0.0f, 0.0f, 0.0f);
        payload.t = RayTCurrent();
    }
}

[shader("miss")]
void miss(inout RayPayload payload)
{
    payload.color = sky_color;
    payload.t = RayTCurrent();
}
