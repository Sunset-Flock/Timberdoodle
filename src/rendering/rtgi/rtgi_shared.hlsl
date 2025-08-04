#pragma once

#include "shader_shared/rtgi.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"

func get_geometry_weight_threshold(float2 inv_render_target_size, float near_plane, float depth) -> float
{
    // The further away the pixel is, the larger difference we allow.
    // The scale is proportional to the size the pixel takes up in world space.
    const float pixel_size_on_near_plane = inv_render_target_size.y;
    const float near_plane_ws_size = near_plane * 2;
    const float pixel_ws_size = pixel_size_on_near_plane * near_plane_ws_size * rcp(depth + 0.0000001f);
    return pixel_ws_size;
}

func get_geometry_weight(float2 inv_render_target_size, float near_plane, float depth, float3 vs_position, float3 vs_normal, float3 other_vs_position,
    float threshold_scale = 3.0f, // a larger factor leads to more bleeding across edges but also less noise on small details
) -> float
{
    // We assume 0 positional difference in view space xy. Good enough approximation.
    const float plane_distance = abs(dot(other_vs_position.z - vs_position.z, vs_normal.z));
    const float threshold = get_geometry_weight_threshold(inv_render_target_size, near_plane, depth) * threshold_scale;
    const float validity = step( plane_distance, threshold );
    return validity;
}

func get_geometry_weight4(
    float2 inv_render_target_size, 
    float near_plane, 
    float depth, 
    float3 vs_position, 
    float3 vs_normal, 
    float4 other_quad_depths, 
    float threshold_scale = 3.0f // a larger factor leads to more bleeding across edges but also less noise on small details
) -> float4
{
    // We assume 0 positional difference in view space xy. Good enough approximation.
    const float4 plane_distances = {
        abs((linearise_depth(other_quad_depths.x, near_plane) - vs_position.z) * vs_normal.z),
        abs((linearise_depth(other_quad_depths.y, near_plane) - vs_position.z) * vs_normal.z),
        abs((linearise_depth(other_quad_depths.z, near_plane) - vs_position.z) * vs_normal.z),
        abs((linearise_depth(other_quad_depths.w, near_plane) - vs_position.z) * vs_normal.z),
    };
    const float threshold = get_geometry_weight_threshold(inv_render_target_size, near_plane, depth) * threshold_scale;
    const float4 validity = step( plane_distances, threshold );
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