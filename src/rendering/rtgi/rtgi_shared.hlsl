#pragma once

#include "shader_shared/rtgi.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"


#define RTGI_USE_POISSON_DISC 0

#define RTGI_SPATIAL_FILTER_SAMPLES 8

// This casues dark areas to be too dark often.
// Consider to find a way to get rid of this and have constant filter size
#define RTGI_SPATIAL_FILTER_RADIUS_MAX 22
#define RTGI_SPATIAL_FILTER_DISOCCLUSION_EXPANSION 1

#define RTGI_DISOCCLUSION_SCALING 1
#define RTGI_DISOCCLUSION_FLOOD_FILL 1

// With a value of 32 AND a value of 32 accumulated frames,
// Effectively, each pixel can AT MOST double its brightness every frame PRE blurring.
#define RTGI_FIREFLY_FILTER 1
#define RTGI_FIREFLY_FILTER_TIGHT_AGRESSIVE 1
#define RTGI_TEMPORAL_FIREFLY_FILTER_THRESHOLD 32.0f
#define RTGI_SPATIAL_FIREFLY_FILTER_THRESHOLD_FIRST_FRAME 8.0f

#define VALUE_MULTIPLIER (1e4f)

func ws_pixel_size(float2 inv_render_target_size, float near_plane, float depth) -> float
{
    // The further away the pixel is, the larger difference we allow.
    // The scale is proportional to the size the pixel takes up in world space.
    const float pixel_size_on_near_plane = inv_render_target_size.y;
    const float near_plane_ws_size = near_plane * 2;
    const float pixel_ws_size = pixel_size_on_near_plane * near_plane_ws_size * rcp(depth + 0.0000001f);
    return pixel_ws_size;
}

func planar_surface_distance(float2 inv_render_target_size, float near_plane, float depth, float3 vs_position, float3 vs_normal, float3 other_vs_position) -> float
{
    const float plane_distance = dot(other_vs_position - vs_position, vs_normal);
    return plane_distance * rcp(ws_pixel_size(inv_render_target_size, near_plane, depth));
}

// geometry weight is used as a hard cutoff for edge stopping when spatial blurring
func planar_surface_weight(float2 inv_render_target_size, float near_plane, float depth, float3 vs_position, float3 vs_normal, float3 other_vs_position,
    float threshold_scale = 2.0f,                   // a larger factor leads to more bleeding across edges but also less noise on small details
    bool only_consider_negative_distance = false    // when calculating the valid sample footprint we want to ignore "enclosed" pixels as we handle those separately.
) -> float
{
    const float plane_distance = only_consider_negative_distance ? abs(min(0.0f, dot(other_vs_position - vs_position, vs_normal))) : abs(dot(other_vs_position - vs_position, vs_normal));
    const float threshold = ws_pixel_size(inv_render_target_size, near_plane, depth) * threshold_scale;
    const float validity = step( plane_distance, threshold );
    return validity;
}

// geometry weight used as a hard cutoff when reprojecting temporal data
// returns 1 when the given points are likely to be the same geometry
func surface_weight(float2 inv_render_target_size, float near_plane, float depth, float3 vs_position, float3 vs_normal, float3 other_vs_position, float3 other_vs_normal,
    float threshold_scale = 2.0f, // a larger factor leads to more bleeding across edges but also less noise on small details
) -> float
{
    const float pixel_size = ws_pixel_size(inv_render_target_size, near_plane, depth);
    const float plane_distanceA = abs(dot(other_vs_position - vs_position, vs_normal));
    const float plane_distanceB = abs(dot(vs_position - other_vs_position, other_vs_normal));
    const float distance = abs(distance(vs_position, other_vs_position));
    const float plane_distance_threshold = pixel_size * threshold_scale;
    // In rare cases, two pixels will accidentally lie on the same plane but are actually far away from each other.
    // The distance test rejects those cases.
    const float dist_threshold = pixel_size * 32.0f * threshold_scale; 
    const float validity = 
        step(distance, dist_threshold) *
        step(plane_distanceA, plane_distance_threshold) * 
        step(plane_distanceB, plane_distance_threshold);
    return validity;
}
func depth_distances4(
    float2 inv_render_target_size, 
    float near_plane, 
    float depth, 
    float3 vs_position, 
    float3 vs_normal, 
    float4 other_quad_depths
) -> float4
{
    // We assume 0 positional difference in view space xy. Good enough approximation.
    const float4 plane_distances = {
        abs((linearise_depth(other_quad_depths.x, near_plane) - vs_position.z)),
        abs((linearise_depth(other_quad_depths.y, near_plane) - vs_position.z)),
        abs((linearise_depth(other_quad_depths.z, near_plane) - vs_position.z)),
        abs((linearise_depth(other_quad_depths.w, near_plane) - vs_position.z)),
    };
    return plane_distances * rcp(ws_pixel_size(inv_render_target_size, near_plane, depth));
}

func planar_surface_weight4(
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
    const float threshold = ws_pixel_size(inv_render_target_size, near_plane, depth) * threshold_scale;
    const float4 validity = step( plane_distances, threshold );
    return validity;
}

// Due to the low resolution the tracing and de-noising runs at we have to use normals only as a strong suggestion, not for cutoff.
func normal_similarity_weight(float3 normal, float3 other_normal) -> float
{
    const float validity = (max(0.1f, dot(normal, other_normal) + 0.85f)) * (1.0f / 1.85f);
    const float tight_validity = (square(validity));
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

float y_co_cg_to_brightness(float3 yCoCg)
{
    const float3 linear = y_co_cg_to_linear(yCoCg);
    return linear.r + linear.g * 2.0f + linear.b * 0.5f;
}

float rgb_brightness(float3 linear)
{
    return linear.r + linear.g * 2.0f + linear.b * 0.5f;
}

float3 y_co_cg_to_linear_corrected( float y, float sh_y_0, float2 co_cg )
{
    y = max( y, 0.0 );
    co_cg *= ( y + 1e-6 ) / ( sh_y_0 + 1e-6 );

    return y_co_cg_to_linear( float3( y, co_cg ) );
}


float3 sh_resolve_diffuse( float4 sh_y, float2 co_cg, float3 normal )
{
    float y = max(dot( normal, sh_y.xyz ) + 0.5f * sh_y.w, sh_y.w * 0.1f);
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