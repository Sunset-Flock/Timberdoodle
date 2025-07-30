#pragma once

#include "rtgi_adaptive_blur.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"

#define POWER_PRE_BLUR 0
#define POWER_SAMPLE_STRENGTH 1.5f

#if POWER_PRE_BLUR
#define POWER_SAMPLE(X) pow(X, (1.0f/POWER_SAMPLE_STRENGTH))
#define DE_POWER_SAMPLE(X) pow(X, POWER_SAMPLE_STRENGTH)
#else
#define POWER_SAMPLE(X) X
#define DE_POWER_SAMPLE(X) X
#endif

[[vk::push_constant]] RtgiAdaptiveBlurPush rtgi_adaptive_blur_push;
[[vk::push_constant]] RtgiPreBlurPush rtgi_pre_blur_push;

func get_geometry_weight(float2 inv_render_target_size, float near_plane, float depth, float3 vs_position, float3 vs_normal, float3 other_vs_position) -> float
{
    const float plane_distance = abs(dot(other_vs_position - vs_position, vs_normal));
    // The further away the pixel is, the larger difference we allow.
    // The scale is proportional to the size the pixel takes up in world space.
    const float pixel_size_on_near_plane = inv_render_target_size.y;
    const float near_plane_ws_size = near_plane * 2;
    const float pixel_ws_size = pixel_size_on_near_plane * near_plane_ws_size * rcp(depth + 0.0000001f);
    const float threshold_scale = 3.0f; // a larger factor leads to more bleeding across edges but also less noise on small details
    const float threshold = pixel_ws_size * threshold_scale; 

    const float validity = step( plane_distance, threshold );
    return validity;
}

func get_normal_diffuse_weight(float3 normal, float3 other_normal) -> float
{
    const float validity = max(0.0f, dot(normal, other_normal));
    const float tight_validity = pow(validity, 8.0f);
    return tight_validity;
}

static const float3 g_Poisson8[8] =
{
    float3( -0.4706069, -0.4427112, +0.6461146 ),
    float3( -0.9057375, +0.3003471, +0.9542373 ),
    float3( -0.3487388, +0.4037880, +0.5335386 ),
    float3( +0.1023042, +0.6439373, +0.6520134 ),
    float3( +0.5699277, +0.3513750, +0.6695386 ),
    float3( +0.2939128, -0.1131226, +0.3149309 ),
    float3( +0.7836658, -0.4208784, +0.8895339 ),
    float3( +0.1564120, -0.8198990, +0.8346850 )
};

[shader("compute")]
[numthreads(RTGI_ADAPTIVE_BLUR_DIFFUSE_X,RTGI_ADAPTIVE_BLUR_DIFFUSE_Y,1)]
func entry_blur_diffuse(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_adaptive_blur_push;
    if (any(dtid.xy >= push.size))
    {
        return;
    }

    // Load and precalculate constants
    const CameraInfo camera = push.attach.globals->view_camera;
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);
    const uint2 halfres_pixel_index = dtid;

    // Load half res depth, normal and sample count
    const float pixel_depth = push.attach.view_cam_half_res_depth.get()[halfres_pixel_index];
    const float pixel_vs_depth = linearise_depth(pixel_depth, camera.near_plane);
    const float pixel_samplecnt = push.attach.rtgi_samplecnt.get()[halfres_pixel_index];
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[halfres_pixel_index]);
    
    // reconstruct pixel positions based on depth
    const float2 uv = (float2(dtid.xy) + 0.5f) * inv_half_res_render_target_size;
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 world_position_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 world_position = world_position_pre_div.xyz / world_position_pre_div.w;
    const float3 vs_position = mul(camera.view, float4(world_position, 1.0f)).xyz;
    const float3 vs_normal = mul(camera.view, float4(pixel_face_normal, 0.0f)).xyz;

    if (pixel_depth == 0.0f)
    {
        push.attach.rtgi_diffuse_blurred.get()[halfres_pixel_index] = float4(0,0,0,0);
        return;
    }

    // Construct tangent bases matrix and setup rand for sample generation
    const float3 world_tangent = normalize(cross(pixel_face_normal, float3(0,0,1) + 0.0001));
    const float3 world_bitangent = cross(world_tangent, pixel_face_normal);
    const uint thread_seed = (dtid.x * push.attach.globals->settings.render_target_size.y + dtid.y) * push.attach.globals.frame_index;
    rand_seed(thread_seed);

    // Sample disc around normal
    const uint SAMPLE_COUNT = 8;
    const float BLUR_PIXEL_RADIUS = 32; // 32 pixels wide
    const float pixel_ws_size = inv_half_res_render_target_size.y * camera.near_plane * rcp(pixel_depth + 0.000000001f);
    const float blur_radius_scale = (push.attach.globals.rtgi_settings.history_frames - pixel_samplecnt) / push.attach.globals.rtgi_settings.history_frames;
    const float blur_radius = max(3.5f, BLUR_PIXEL_RADIUS * blur_radius_scale);
    float weight_accum = 0.0f;
    float3 blurred_diffuse_accum = float3(0.0f,0.0f,0.0f);
    for (uint s = 0; s < SAMPLE_COUNT; ++s)
    {
        // Calculate sample position
        const float2 sample_2d = rand_concentric_sample_disc() * blur_radius * pixel_ws_size;// rand_concentric_sample_disc() * blur_radius * pixel_ws_size;
        const float3 sample_ws = world_position + world_tangent * sample_2d.x + world_bitangent * sample_2d.y;
        const float4 sample_ndc_prev_div = mul(camera.view_proj, float4(sample_ws, 1.0f));
        const float3 sample_ndc = sample_ndc_prev_div.xyz / sample_ndc_prev_div.w;
        const float2 sample_uv = sample_ndc.xy * 0.5f + 0.5f;
        const uint2 sample_index = uint2(sample_uv * half_res_render_target_size);

        // Load sample data
        const float3 sample_value_diffuse = push.attach.rtgi_diffuse_accumulated.get()[sample_index].rgb;
        const float3 sample_value_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[sample_index]);
        const float sample_value_samplecnt = push.attach.rtgi_samplecnt.get()[sample_index];
        const float sample_value_depth = push.attach.view_cam_half_res_depth.get()[sample_index];
        const float3 sample_value_ndc = float3(sample_ndc.xy, sample_value_depth);
        const float4 sample_value_ws_pre_div = mul(camera.inv_proj, float4(sample_value_ndc, 1.0f));
        const float3 sample_value_ws = sample_value_ws_pre_div.xyz / sample_value_ws_pre_div.w;

        // Calculate validity weights
        const float depth_valid_weight = sample_value_depth != 0.0f ? 1.0f : 0.0f;
        const float geometric_weight = get_geometry_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, vs_position, vs_normal, sample_value_ws);
        const float normal_weight = get_normal_diffuse_weight(pixel_face_normal, sample_value_normal);
        const float sample_count_weight = sample_value_samplecnt / float(push.attach.globals.rtgi_settings.history_frames);
        const float weight = depth_valid_weight * geometric_weight * normal_weight * sample_count_weight;

        // Accumulate blurred diffuse
        weight_accum += weight;
        blurred_diffuse_accum += weight * sample_value_diffuse;
    }

    // Calculate blurred diffuse and fallback blending
    // Some pixels find nearly no suitable spacial samples,
    // if less than 1/4th of the samples matter, we start to fallback to the original diffuse
    const float3 blurry_diffuse = blurred_diffuse_accum * rcp(weight_accum + 0.00000001f);
    const float low_weight_fallback_blend = max(0.0f, 1.0f - weight_accum / (SAMPLE_COUNT/4)); 
    const float3 original_diffuse = push.attach.rtgi_diffuse_accumulated.get()[halfres_pixel_index].rgb;
    const float3 fallback_diffuse = original_diffuse;
    const float3 blurred_diffuse = lerp(blurry_diffuse, fallback_diffuse, low_weight_fallback_blend);

    push.attach.rtgi_diffuse_blurred.get()[halfres_pixel_index] = float4(blurred_diffuse, 1.0f);
}

[shader("compute")]
[numthreads(RTGI_PRE_BLUR_DIFFUSE_X,RTGI_PRE_BLUR_DIFFUSE_Y,1)]
func entry_pre_blur_diffuse(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_pre_blur_push;
    if (any(dtid.xy >= push.size))
    {
        return;
    }

    // Load and precalculate constants
    const CameraInfo camera = push.attach.globals->view_camera;
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);
    const uint2 halfres_pixel_index = dtid;

    // Load half res depth and normal
    const float pixel_depth = push.attach.view_cam_half_res_depth.get()[halfres_pixel_index];
    const float pixel_vs_depth = linearise_depth(pixel_depth, camera.near_plane);
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[halfres_pixel_index]);
    
    // reconstruct pixel positions based on depth
    const float2 uv = (float2(dtid.xy) + 0.5f) * inv_half_res_render_target_size;
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 world_position_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 world_position = world_position_pre_div.xyz / world_position_pre_div.w;
    const float3 vs_position = mul(camera.view, float4(world_position, 1.0f)).xyz;
    const float3 vs_normal = mul(camera.view, float4(pixel_face_normal, 0.0f)).xyz;

    if (pixel_depth == 0.0f)
    {
        push.attach.rtgi_diffuse_raw_pre_blurred.get()[halfres_pixel_index] = float4(0,0,0,0);
        return;
    }

    // Construct tangent bases matrix and setup rand for sample generation
    const float3 world_tangent = normalize(cross(pixel_face_normal, float3(0,0,1) + 0.0001));
    const float3 world_bitangent = cross(world_tangent, pixel_face_normal);
    const uint thread_seed = (dtid.x * push.attach.globals->settings.render_target_size.y + dtid.y) * push.attach.globals.frame_index;
    rand_seed(thread_seed);

    // Sample disc around normal
    const uint SAMPLE_COUNT = 8;
    const float BLUR_PIXEL_RADIUS = 8; // 8 pixels wide
    const float pixel_ws_size = inv_half_res_render_target_size.y * camera.near_plane * rcp(pixel_depth + 0.000000001f);
    const float blur_radius = max(1.5f, BLUR_PIXEL_RADIUS);
    float weight_accum = 0.0f;
    float3 blurred_diffuse_accum = float3(0.0f,0.0f,0.0f);
    for (uint s = 0; s < SAMPLE_COUNT; ++s)
    {
        // Calculate sample position
        const float2 sample_2d = rand_concentric_sample_disc() * blur_radius * pixel_ws_size;
        const float3 sample_ws = world_position + world_tangent * sample_2d.x + world_bitangent * sample_2d.y;
        const float4 sample_ndc_prev_div = mul(camera.view_proj, float4(sample_ws, 1.0f));
        const float3 sample_ndc = sample_ndc_prev_div.xyz / sample_ndc_prev_div.w;
        const float2 sample_uv = sample_ndc.xy * 0.5f + 0.5f;
        const uint2 sample_index = uint2(sample_uv * half_res_render_target_size);

        // Load sample data
        const float3 sample_value_diffuse = push.attach.rtgi_diffuse_raw.get()[sample_index].rgb;
        const float3 sample_value_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[sample_index]);
        const float sample_value_depth = push.attach.view_cam_half_res_depth.get()[sample_index];
        const float3 sample_value_ndc = float3(sample_ndc.xy, sample_value_depth);
        const float4 sample_value_ws_pre_div = mul(camera.inv_proj, float4(sample_value_ndc, 1.0f));
        const float3 sample_value_ws = sample_value_ws_pre_div.xyz / sample_value_ws_pre_div.w;

        // Calculate validity weights
        const float depth_valid_weight = sample_value_depth != 0.0f ? 1.0f : 0.0f;
        const float geometric_weight = get_geometry_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, vs_position, vs_normal, sample_value_ws);
        const float normal_weight = get_normal_diffuse_weight(pixel_face_normal, sample_value_normal);
        const float weight = depth_valid_weight * geometric_weight * normal_weight;

        // Accumulate blurred diffuse
        weight_accum += weight;
        blurred_diffuse_accum += weight * POWER_SAMPLE(sample_value_diffuse);
    }

    // Calculate blurred diffuse and fallback blending
    // Some pixels find nearly no suitable spacial samples,
    // if less than 1/4th of the samples matter, we start to fallback to the original diffuse
    const float3 blurry_diffuse = DE_POWER_SAMPLE(blurred_diffuse_accum * rcp(weight_accum + 0.00000001f));
    const float low_weight_fallback_blend = max(0.0f, 1.0f - weight_accum / (SAMPLE_COUNT/4)); 
    const float3 original_diffuse = push.attach.rtgi_diffuse_raw.get()[halfres_pixel_index].rgb;
    const float3 fallback_diffuse = original_diffuse;
    const float3 blurred_diffuse = lerp(blurry_diffuse, fallback_diffuse, low_weight_fallback_blend);

    push.attach.rtgi_diffuse_raw_pre_blurred.get()[halfres_pixel_index] = float4(blurred_diffuse, 1.0f);
}