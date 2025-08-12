#pragma once

#include "shader_shared/rtgi.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"


#define RTGI_USE_POISSON_DISC 0

#define RTGI_SPATIAL_FILTER_SAMPLES 16
#define RTGI_SPATIAL_FILTER_RADIUS_MIN 2
#define RTGI_SPATIAL_FILTER_RADIUS_MAX 32

#define RTGI_SPATIAL_FILTER_SAMPLES_PRE_BLUR 8
#define RTGI_SPATIAL_FILTER_RADIUS_PRE_BLUR_MIN 0
#define RTGI_SPATIAL_FILTER_RADIUS_PRE_BLUR_MAX 5

#define RTGI_POST_BLUR_LUMA_DIFF_RADIUS_SCALE 1


func luma_of(float3 color) -> float
{
    return (color.r + 2.0f * color.g + color.b) * 0.25f;
}

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
    float threshold_scale = 2.0f, // a larger factor leads to more bleeding across edges but also less noise on small details
) -> float
{
    const float plane_distance = abs(dot(other_vs_position - vs_position, vs_normal));
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
    float threshold_scale = 2.0f // a larger factor leads to more bleeding across edges but also less noise on small details
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

float get_gaussian_weight( float r )
{
    return exp( -0.66 * r * r ); // assuming r is normalized to 1
}

#define RTGI_USE_SH 1

// Sh functions from:
// https://github.com/NVIDIA-RTX/NRD/blob/03d5e0b2015c6eaf122d8e8f95b0527f6f03633e/Shaders/Include/NRD.hlsli#L361

float3 linear_to_y_co_cg( float3 color )
{
    float y = dot( color, float3( 0.25, 0.5, 0.25 ) );
    float Co = dot( color, float3( 0.5, 0.0, -0.5 ) );
    float Cg = dot( color, float3( -0.25, 0.5, -0.25 ) );

    return float3( y, Co, Cg );
}

float3 y_co_cg_to_linear( float3 color )
{
    float t = color.x - color.z;

    float3 r;
    r.y = color.x + color.z;
    r.x = t + color.y;
    r.z = t - color.y;

    return max( r, 0.0 );
}

float3 y_co_cg_to_linear_corrected( float y, float sh_y_0, float2 co_cg )
{
    y = max( y, 0.0 );
    co_cg *= ( y + 1e-6 ) / ( sh_y_0 + 1e-6 );

    return y_co_cg_to_linear( float3( y, co_cg ) );
}


float3 sh_resolve_diffuse( float4 sh_y, float2 co_cg, float3 normal )
{
    float y = dot( normal, sh_y.xyz ) + 0.5 * sh_y.w;

    return y_co_cg_to_linear_corrected( y, sh_y.w, co_cg );
}

float4 y_to_sh(float y, float3 direction)
{
    float sh0 = y;
    float3 sh1 = direction * y;
    return float4(sh1, sh0);
}

void radiance_to_y_co_cg_sh(float3 radiance, float3 direction, out float4 sh_y, out float2 co_cg)
{
    float3 y_co_cg = linear_to_y_co_cg(radiance);
    co_cg = y_co_cg.gb;

    sh_y = y_to_sh(y_co_cg.x, direction);
}