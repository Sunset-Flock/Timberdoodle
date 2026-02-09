#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "rtgi_trace_diffuse.inl"
#include "rtgi_trace_diffuse_shared.hlsl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/raytracing.hlsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/pgi.hlsl"

#include "rtgi_shared.hlsl"

#define GOLDEN_RATIO 1.6181
#define PI 3.1415926535897932384626433832795

#define STBN_SIZE uint3(128,128,32)
#define STBN_GRID_SIZE uint3(128,128,32)

float2 rand_stbn2d(Texture2DArray<float4> stbn2d_image, uint2 pixel, int frame)
{
    const uint z_wrap = 0;//frame / STBN_SIZE.z;
    const uint2 xy_wrap = pixel / STBN_SIZE.xy;
    const uint z = frame % STBN_SIZE.z;// + xy_wrap.x + xy_wrap.y * 17;
    pixel = (pixel + uint2(GOLDEN_RATIO * float2(STBN_SIZE.xy * z_wrap))) % STBN_SIZE.xy;
    return stbn2d_image[uint3(pixel,z)].xy;
}

float3 rand_stbnCosDir(Texture2DArray<float4> stbn2d_image, uint2 pixel, int frame)
{
    pixel = pixel % STBN_GRID_SIZE.xy;
    const uint z_wrap = frame / STBN_SIZE.z;
    const uint z = frame % STBN_SIZE.z;
    pixel = (pixel + uint2(GOLDEN_RATIO * float2(STBN_SIZE.xy * z_wrap))) % STBN_SIZE.xy;
    return stbn2d_image[uint3(pixel,z)].xyz * 2.0f - 1.0f;
}

float2 rand_concentric_sample_disc_stbn(uint2 pixel_frame)
{
    let push = rtgi_trace_diffuse_push;
    float2 rr = rand_stbn2d(Texture2DArray<float4>::get(push.attach.globals.stbn2d), pixel_frame.xy, push.attach.globals.frame_index);
    rr = abs(rr);
    float r = rr.x;
    float theta = rr.y * 2 * PI;
    return float2(cos(theta), sin(theta)) * r;
}

float3 rand_cosine_sample_hemi_stbn(uint2 pixel_frame)
{
    float2 d = rand_concentric_sample_disc_stbn(pixel_frame);
    float z = sqrt(max(0.0f, 1.0f - d.x * d.x - d.y * d.y));
    return float3(d.x, d.y, z);
}

__generic<uint N>
func linear_to_perceptual(vector<float, N> v) -> vector<float, N> 
{
    return log(max(v, 0.01f) + 1.0f);
}

__generic<uint N>
func perceptual_to_linear(vector<float, N> v) -> vector<float, N> 
{
    return exp(v) - 1.0f;
}

[shader("raygeneration")]
void ray_gen()
{
    let clk_start = clockARB();
    let push = rtgi_trace_diffuse_push;
    const int2 dtid = DispatchRaysIndex().xy;

    const float depth = push.attach.view_cam_half_res_depth.get()[dtid];
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
        TraceRay(RaytracingAccelerationStructure::get(push.attach.tlas), 0, ~0, 0, 0, 0, ray, payload);

        float4 value = float4(payload.color, 1.0f);
        push.attach.diffuse_raw.get()[dtid.xy] = float4(payload.color,payload.t);
        return;
    }

    float4 acc = float4( 0, 0, 0, 0 );
    float2 acc2 = float2( 0, 0 );

    const uint thread_seed = dtid.x * push.attach.globals->settings.render_target_size.y + dtid.y + push.attach.globals.frame_index * push.attach.globals->settings.render_target_size.y * push.attach.globals->settings.render_target_size.x;
    rand_seed(thread_seed);
    float2 rr_stbn = rand_stbn2d(Texture2DArray<float4>::get(push.attach.globals.stbn2d), dtid.xy, push.attach.globals.frame_index);
    float2 rr = float2(rand(), rand());

    if(depth > 0.0f)
    {
        static const uint SAMPLES = 1;
        for (uint i = 0; i < SAMPLES; ++i)
        {
            const float3 sample_pos = rt_calc_ray_start(world_position, face_normal, primary_ray);
            const float3 world_tangent = normalize(cross(face_normal, float3(0,0,1) + 0.0001));
            const float3x3 tbn = transpose(float3x3(world_tangent, cross(world_tangent, face_normal), face_normal));
                
            const uint thread_seed = (dtid.x * push.attach.globals->settings.render_target_size.y + dtid.y) * push.attach.globals.frame_index + i;
            rand_seed(thread_seed);
            // const float3 importance_rand_hemi_sample = rand_cosine_sample_hemi();
            // const float3 importance_rand_hemi_sample = rand_cosine_sample_hemi_stbn( pixel_index );
            
            float3 importance_rand_hemi_sample;
            if (dtid.x > (push.attach.globals.settings.render_target_size.x/4) && false)
            {
                // importance_rand_hemi_sample = rand_cosine_sample_hemi_stbn( pixel_index );
                importance_rand_hemi_sample = rand_stbnCosDir(Texture2DArray<float4>::get(push.attach.globals.stbnCosDir), pixel_index, push.attach.globals.frame_index);
            }
            else
            {
                importance_rand_hemi_sample = rand_cosine_sample_hemi();
            }

            RayPayload payload = {};
            payload.dtid = dtid;

            #if RTGI_USE_PGI_RADIANCE_ON_MISS
            float pgi_cascade = pgi_select_cascade_smooth_spherical(push.attach.globals.pgi_settings, sample_pos - push.attach.globals.view_camera.position);
            float t_max = float(1u << uint(ceil(pgi_cascade))) * push.attach.globals.pgi_settings.cascades[0].probe_spacing.x * RTGI_USE_PGI_RADIANCE_ON_MISS_TMAX_SCALE;
            #else
            float t_max = 100000000000.0f;
            #endif

            RayDesc ray = {};
            ray.Origin = sample_pos;
            ray.TMax = t_max;
            ray.TMin = 0.0f;

            const float3 sample_dir = mul(tbn, importance_rand_hemi_sample);
            ray.Direction = sample_dir;
            const uint flags = {};//RAY_FLAG_FORCE_OPAQUE; 
            TraceRay(RaytracingAccelerationStructure::get(push.attach.tlas), flags, ~0, 0, 0, 0, ray, payload);

            float4 sh_y_new;
            float2 cocg_new;
            radiance_to_y_co_cg_sh((payload.color * VALUE_MULTIPLIER), sample_dir, sh_y_new, cocg_new);
            acc += sh_y_new * rcp(SAMPLES);
            acc2 += cocg_new * rcp(SAMPLES);
        }

        push.attach.diffuse_raw.get()[dtid.xy] = acc;
        push.attach.diffuse2_raw.get()[dtid.xy] = acc2;
    }
    else
    {
        push.attach.diffuse_raw.get()[dtid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        push.attach.diffuse2_raw.get()[dtid.xy] = float2(0.0f, 0.0f);
    }

    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTGI_TRACE_CLOCKS)
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