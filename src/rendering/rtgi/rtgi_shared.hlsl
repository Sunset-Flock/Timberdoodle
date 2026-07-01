#pragma once

#include "shader_shared/rtgi.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"

#define RTGI_USE_QUAD 1
#define RTGI_QUAD_FILTER_EXTENT 1  // valid: 1 (3x3), 2 (5x5)
#define RTGI_QUAD_FILTER_STRIDE 1  // cell spacing of the outer gather; >1 widens reach at same tap count (e.g. extent 1 + stride 2)

#define VALUE_MULTIPLIER (1e4f)

struct PixelData
{
    float2 uv;
    float3 ndc;
    float3 position_ws;
    float3 position_vs;
    float3 normal_ws;
    float3 normal_vs;
};

func calc_pixel_data(
    uint2 dtid,
    float2 inv_half_res_render_target_size,
    const CameraInfo camera,
    Texture2D<float> depth_tex,
    Texture2D<uint> normals_tex) -> PixelData
{
    PixelData pd;
    pd.uv          = (float2(dtid) + 0.5f) * inv_half_res_render_target_size;
    pd.ndc         = float3(pd.uv * 2.0f - 1.0f, depth_tex[dtid]);
    const float4 pos_pre_div = mul(camera.inv_view_proj, float4(pd.ndc, 1.0f));
    pd.position_ws = pos_pre_div.xyz / pos_pre_div.w;
    pd.position_vs = mul(camera.view, float4(pd.position_ws, 1.0f)).xyz;
    pd.normal_ws   = uncompress_normal_octahedral_32(normals_tex[dtid]);
    pd.normal_vs   = mul(camera.view, float4(pd.normal_ws, 0.0f)).xyz;
    return pd;
}

func calc_pixel_width_ws(float2 inv_render_target_size, float near_plane, float depth) -> float
{
    // The further away the pixel is, the larger difference we allow.
    // The scale is proportional to the size the pixel takes up in world space.
    const float pixel_size_on_near_plane = inv_render_target_size.y;
    const float near_plane_ws_size = near_plane * 2;
    const float pixel_width_ws = pixel_size_on_near_plane * near_plane_ws_size * rcp(depth + 0.0000001f);
    return pixel_width_ws;
}

func calc_plane_distance(float3 a_pos, float3 a_norm, float3 b_pos) -> float
{
    return dot(a_pos - b_pos, a_norm);
}

func calc_similar_surface_weight(const float rcp_pixel_width_ws, const float3 a_pos, const float3 a_norm, const float3 b_pos, const float3 b_norm, const float px_threshold = 1.2f) -> float
{
    const float within_acceptable_plane_dist_a = 1.0f - saturate(calc_plane_distance(a_pos, a_norm, b_pos) * rcp(px_threshold) * rcp_pixel_width_ws);
    const float within_acceptable_plane_dist_b = 1.0f - saturate(calc_plane_distance(b_pos, b_norm, a_pos) * rcp(px_threshold) * rcp_pixel_width_ws);
    return within_acceptable_plane_dist_a * within_acceptable_plane_dist_b;
}

// Same as surface_similarity but also rejects pairs of points that share a plane yet are
// far apart in world space — guards against accidental coplanar matches across large distances.
func calc_similar_surface_weight_dist_limited(const float rcp_pixel_width_ws, const float3 a_pos, const float3 a_norm, const float3 b_pos, const float3 b_norm, const float px_threshold = 1.2f) -> float
{
    const float surface_weight              = calc_similar_surface_weight(rcp_pixel_width_ws, a_pos, a_norm, b_pos, b_norm, px_threshold);
    const float dist_in_pixel_widths        = abs(distance(a_pos, b_pos)) * rcp_pixel_width_ws;
    const float pixel_widths_threshold      = 32.0f * px_threshold;
    return step(dist_in_pixel_widths, pixel_widths_threshold) * surface_weight;
}

// Due to the low resolution the tracing and de-noising runs at we have to use normals only as a strong suggestion, not for cutoff.
func calc_similar_normal_weight(float3 normal, float3 other_normal) -> float
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

float calc_gaussian_weight( float r )
{
    return exp( -0.66f * square(r * 2.71828182846f * 0.5f) ); // assuming r is normalized to 1
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