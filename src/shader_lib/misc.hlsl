#pragma once

#include "daxa/daxa.inl"

#include "shader_shared/shared.inl"
#include "shader_shared/geometry.inl"

func firstbitlow_uint4(uint4 v) -> uint
{
    uint vec_mask = 
        (v[0] > 0 ? 1u << 0u : 0u) |
        (v[1] > 0 ? 1u << 1u : 0u) |
        (v[2] > 0 ? 1u << 2u : 0u) |
        (v[3] > 0 ? 1u << 3u : 0u);
    uint first_scalar_with_bits = firstbitlow(vec_mask);
    return firstbitlow(v[first_scalar_with_bits]);
}

func firstbithigh_uint4(uint4 v) -> uint
{
    uint vec_mask = 
        (v[0] > 0 ? 1u << 0u : 0u) |
        (v[1] > 0 ? 1u << 1u : 0u) |
        (v[2] > 0 ? 1u << 2u : 0u) |
        (v[3] > 0 ? 1u << 3u : 0u);
    uint first_scalar_with_bits = firstbithigh(vec_mask);
    return firstbithigh(v[first_scalar_with_bits]);
}

float3 TurboColormap(float x)
{
    const float4 kRedVec4 = float4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const float4 kGreenVec4 = float4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const float4 kBlueVec4 = float4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const float2 kRedVec2 = float2(-152.94239396, 59.28637943);
    const float2 kGreenVec2 = float2(4.27729857, 2.82956604);
    const float2 kBlueVec2 = float2(-89.90310912, 27.34824973);

    x = clamp(x, 0, 1);
    float4 v4 = float4( 1.0, x, x * x, x * x * x);
    float2 v2 = v4.zw * v4.z;
    return float3(
      dot(v4, kRedVec4)   + dot(v2, kRedVec2),
      dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
      dot(v4, kBlueVec4)  + dot(v2, kBlueVec2)
    );
}

// Perceptual heatmap ramp that also increases in brightness monotonically.
// x in [0,1]: black -> blue -> blurple(purple) -> red -> orange -> yellow.
float3 Heatmap(float x)
{
    x = clamp(x, 0.0, 1.0);

    // Control points along the ramp, getting continuously brighter.
    // Non-uniform stop positions give more room to the mid-tones (purple->red->orange)
    // where the eye is most sensitive.
    const float3 c0 = float3(0.00, 0.00, 0.00); // black
    const float3 c1 = float3(0.05, 0.02, 0.35); // deep blue
    const float3 c2 = float3(0.35, 0.05, 0.72); // blurple / violet
    const float3 c3 = float3(0.78, 0.08, 0.42); // magenta-red
    const float3 c4 = float3(0.97, 0.20, 0.10); // red
    const float3 c5 = float3(1.00, 0.60, 0.05); // orange
    const float3 c6 = float3(1.00, 0.95, 0.35); // yellow
    const float3 c7 = float3(1.00, 1.00, 1.00); // white

    // Positions of each control point along x (monotonic, non-uniform).
    const float p0 = 0.00, p1 = 0.13, p2 = 0.28, p3 = 0.44;
    const float p4 = 0.60, p5 = 0.76, p6 = 0.90, p7 = 1.00;

    float3 col;
    if (x < p1)      col = lerp(c0, c1, (x - p0) / (p1 - p0));
    else if (x < p2) col = lerp(c1, c2, (x - p1) / (p2 - p1));
    else if (x < p3) col = lerp(c2, c3, (x - p2) / (p3 - p2));
    else if (x < p4) col = lerp(c3, c4, (x - p3) / (p4 - p3));
    else if (x < p5) col = lerp(c4, c5, (x - p4) / (p5 - p4));
    else if (x < p6) col = lerp(c5, c6, (x - p5) / (p6 - p5));
    else             col = lerp(c6, c7, (x - p6) / (p7 - p6));
    return col;
}

float3 hsv2rgb(float3 c) {
    float4 k = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * lerp(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}

float IdFloatScramble(uint id)
{
    let SCRAMBLE = 0.172426234237f;
    return frac(float(id) * SCRAMBLE);
}

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

float2 rand_concentric_sample_disc()
{
    float r = sqrt(rand());
    float theta = rand() * 2 * PI;
    return float2(cos(theta), sin(theta)) * r;
}

float3 rand_cosine_sample_hemi()
{
    float2 d = rand_concentric_sample_disc();
    float z = sqrt(max(0.0f, 1.0f - d.x * d.x - d.y * d.y));
    return float3(d.x, d.y, z);
}

[ForceInline]
func AtomicMaxU64(__ref uint64_t dest, uint64_t value) -> uint64_t
{
    uint64_t original_value;
    spirv_asm
    {
        OpExtension "SPV_EXT_shader_image_int64";
        OpCapability Int64Atomics;
        OpCapability Int64ImageEXT;
        %origin:$$uint64_t = OpAtomicUMax &dest Device None $value;
        OpStore &original_value %origin
    };
    return original_value;
}

[ForceInline]
func AtomicAddU64(__ref uint64_t dest, uint64_t value) -> uint64_t
{
    uint64_t original_value;
    spirv_asm
    {
        OpExtension "SPV_EXT_shader_image_int64";
        OpCapability Int64Atomics;
        OpCapability Int64ImageEXT;
        %origin:$$uint64_t = OpAtomicIAdd &dest Device None $value;
        OpStore &original_value %origin
    };
    return original_value;
}

/// ===== From Shadertoy: https://www.shadertoy.com/view/llfcRl =====

uint packSnorm2x12(float2 v) 
{ 
    uint2 d = uint2(round(2047.5 + v*2047.5));
    return d.x|(d.y<<12u);
}

float2 unpackSnorm2x12(uint d) 
{
    return float2(uint2(d,d>>12)&4095u)/2047.5 - 1.0;
}

float msign( float v )
{
    return (v.x>=0.0) ? 1.0 : -1.0;
}

float2 msign2( float2 v )
{
    return float2( (v.x>=0.0) ? 1.0 : -1.0, 
                 (v.y>=0.0) ? 1.0 : -1.0 );
}

uint compress_normal_octahedral_24( in float3 nor )
{
    nor /= ( abs( nor.x ) + abs( nor.y ) + abs( nor.z ) );
    nor.xy = (nor.z >= 0.0) ? nor.xy : (1.0-abs(nor.yx))*msign2(nor.xy);
    return packSnorm2x12(nor.xy);
}

float3 uncompress_normal_octahedral_24( uint data )
{
    float2 v = unpackSnorm2x12(data);
    float3 nor = float3(v, 1.0 - abs(v.x) - abs(v.y));
    float t = max(-nor.z,0.0);
    nor.x += (nor.x>0.0)?-t:t;
    nor.y += (nor.y>0.0)?-t:t;
    return normalize( nor );
}

uint compress_normal_octahedral_32( in float3 nor )
{
    nor.xy /= ( abs( nor.x ) + abs( nor.y ) + abs( nor.z ) );
    nor.xy  = (nor.z >= 0.0) ? nor.xy : (1.0-abs(nor.yx))*msign2(nor.xy);
    uint2 d = uint2(round(32767.5 + nor.xy*32767.5));  
    return d.x|(d.y<<16u);
}

float3 uncompress_normal_octahedral_32( uint data )
{
    uint2 iv = uint2( data, data>>16u ) & 65535u;
    float2 v = float2(iv)/32767.5 - 1.0;
    float3 nor = float3(v, 1.0 - abs(v.x) - abs(v.y));
    float t = max(-nor.z,0.0);
    nor.x += (nor.x>0.0)?-t:t;
    nor.y += (nor.y>0.0)?-t:t;
    return normalize( nor );
}


float2 map_octahedral(float3 nor) {
    const float fac = 1.0f / (abs(nor.x) + abs(nor.y) + abs(nor.z));
    nor.x *= fac;
    nor.y *= fac;
    if (nor.z < 0.0f) {
        const float2 temp = nor.xy;
        nor.x = (1.0f - abs(temp.y)) * msign(temp.x);
        nor.y = (1.0f - abs(temp.x)) * msign(temp.y);
    }
    return float2(nor.x, nor.y) * 0.5f + 0.5f;
}

float3 unmap_octahedral(float2 v) {
    v = v * 2.0f - 1.0f;
    float3 nor = float3(v, 1.0f - abs(v.x) - abs(v.y)); // Rune Stubbe's version,
    float t = max(-nor.z, 0.0f);                        // much faster than original
    nor.x += (nor.x > 0.0f) ? -t : t;                   // implementation of this
    nor.y += (nor.y > 0.0f) ? -t : t;                   // technique
    return normalize(nor);
}

float interpolate_float(daxa_f32vec3 derivator, float v0, float v1, float v2)
{
    daxa_f32vec3 mergedV = daxa_f32vec3(v0, v1, v2);
    float ret;
    ret = dot(mergedV, derivator);
    return ret;
}

daxa_f32vec2 interpolate_vec2(daxa_f32vec3 derivator, daxa_f32vec2 v0, daxa_f32vec2 v1, daxa_f32vec2 v2)
{
    daxa_f32vec3 merged_x = daxa_f32vec3(v0.x, v1.x, v2.x);
    daxa_f32vec3 merged_y = daxa_f32vec3(v0.y, v1.y, v2.y);
    daxa_f32vec2 ret;
    ret.x = dot(merged_x, derivator);
    ret.y = dot(merged_y, derivator);
    return ret;
}

daxa_f32vec3 interpolate_vec3(daxa_f32vec3 derivator, daxa_f32vec3 v0, daxa_f32vec3 v1, daxa_f32vec3 v2)
{
    daxa_f32vec3 merged_x = daxa_f32vec3(v0.x, v1.x, v2.x);
    daxa_f32vec3 merged_y = daxa_f32vec3(v0.y, v1.y, v2.y);
    daxa_f32vec3 merged_z = daxa_f32vec3(v0.z, v1.z, v2.z);
    daxa_f32vec3 ret;
    ret.x = dot(merged_x, derivator);
    ret.y = dot(merged_y, derivator);
    ret.z = dot(merged_z, derivator);
    return ret;
}

daxa_f32vec4 interpolate_vec4(daxa_f32vec3 derivator, daxa_f32vec4 v0, daxa_f32vec4 v1, daxa_f32vec4 v2)
{
    daxa_f32vec3 merged_x = daxa_f32vec3(v0.x, v1.x, v2.x);
    daxa_f32vec3 merged_y = daxa_f32vec3(v0.y, v1.y, v2.y);
    daxa_f32vec3 merged_z = daxa_f32vec3(v0.z, v1.z, v2.z);
    daxa_f32vec3 merged_w = daxa_f32vec3(v0.w, v1.w, v2.w);
    daxa_f32vec4 ret;
    ret.x = dot(merged_x, derivator);
    ret.y = dot(merged_y, derivator);
    ret.z = dot(merged_z, derivator);
    ret.w = dot(merged_w, derivator);
    return ret;
}

__generic<uint N>
func square(vector<float, N> x) -> vector<float, N>
{
    return x * x;
}

func square(float x) -> float
{
    return x * x;
}

struct Bilinear
{
    float2 origin;
    float2 weights;
};

Bilinear get_bilinear_filter( float2 uv, float2 texSize )
{
    Bilinear ret;
    ret.origin = floor( uv * texSize - 0.5f );
    ret.weights = frac( uv * texSize - 0.5f );
    return ret;
}

float4 get_bilinear_custom_weights( Bilinear f, float4 custumWeights )
{
    float4 weights;
    weights.x = ( 1.0f - f.weights.x ) * ( 1.0f - f.weights.y );
    weights.y = f.weights.x * ( 1.0f - f.weights.y );
    weights.z = ( 1.0f - f.weights.x ) * f.weights.y;
    weights.w = f.weights.x * f.weights.y;
    return weights * custumWeights;
}

float4 apply_bilinear_custom_weights( float4 s00, float4 s10, float4 s01, float4 s11, float4 w, bool normalize = true )
{
    float4 wsum = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
    return wsum * ( normalize ? rcp( dot( w, 1.0f ) ) : 1.0f );
}

float3 apply_bilinear_custom_weights( float3 s00, float3 s10, float3 s01, float3 s11, float4 w, bool normalize = true )
{
    float3 wsum = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
    return wsum * ( normalize ? rcp( dot( w, 1.0f ) ) : 1.0f );
}

float2 apply_bilinear_custom_weights( float2 s00, float2 s10, float2 s01, float2 s11, float4 w, bool normalize = true )
{
    float2 wsum = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
    return wsum * ( normalize ? rcp( dot( w, 1.0f ) ) : 1.0f );
}

float apply_bilinear_custom_weights( float s00, float s10, float s01, float s11, float4 w, bool normalize = true )
{
    float wsum = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
    return wsum * ( normalize ? rcp( dot( w, 1.0f ) ) : 1.0f );
}

int2 flip_oob_index(int2 index, int2 max_index)
{
    index.x = index.x < 0 ? abs(index.x) : index.x;
    index.y = index.y < 0 ? abs(index.y) : index.y;
    index.x = index.x > max_index.x ? (max_index.x - (index.x - max_index.x)) : index.x;
    index.y = index.y > max_index.y ? (max_index.y - (index.y - max_index.y)) : index.y;
    return index;
}

struct RayAABBResult
{
    float near;
    float far;
};

// Returns the near and far intersection points.
// Ray missing the aabb is implied by (near >= far).
RayAABBResult intersect_ray_with_aabb( const float3 ray_origin, const float3 ray_dir, AABB aabb)
{
    const float3 aabb_min = aabb.center - (aabb.size * 0.5f);
    const float3 aabb_max = aabb.center + (aabb.size * 0.5f);

    const float3 t_min = (aabb_min - ray_origin) / ray_dir;
    const float3 t_max = (aabb_max - ray_origin) / ray_dir;
    const float3 t1 = min(t_min, t_max);
    const float3 t2 = max(t_min, t_max);
    const float t_near = max(max(t1.x, t1.y), t1.z);
    const float t_far = min(min(t2.x, t2.y), t2.z);

    // t_near and t_far can be negative in the inverse of the view direction.
    // To remove the side effect of reporting a false hit I max the t_near with 0.0f.
    // This also has the added benefit of having near be as 0 when inside of the AABB.
    return RayAABBResult(max(t_near, 0.0), t_far);
}

// Picks the representative pixel of a 2x2 downsample block by depth.
// The representative is the depth most central to the quad — not the closest, not the furthest —
// found as the L1 geometric median of the four depths, which rejects both near outliers (a stray
// foreground sliver) and far outliers (a background pixel bleeding in).
// A small foreground term nudges the pick toward the nearer pixels. It is scaled by each pixel's
// distance to the quad's nearest depth, so it only matters when the quad straddles a large depth gap
// and vanishes when all four pixels sit close together.
// Ties (flat surfaces, near-identical depths) keep the lower index; the layout is idx = x | (y<<1),
// so that is top-then-left, resolving flat edges consistently instead of wobbling side to side.
// Sky pixels (depth == 0) are excluded and only win when all four are sky.
int quad_downsample_representative(float depths[4])
{
    // Reversed-Z: the largest depth is the nearest.
    float max_depth = max(max(depths[0], depths[1]), max(depths[2], depths[3]));

    // Small weight on distance-to-nearest. Kept low so it only breaks otherwise-close ties, and
    // because it multiplies the depth gap it stays negligible on tight quads.
    const float foreground_bias = 0.25f;

    // L1 distance of each depth to all others (lower = more central = better representative), plus
    // the foreground bias. Sky pixels stay "infinitely bad" so they never beat a real surface.
    float depth_dist[4] = { 1e30f, 1e30f, 1e30f, 1e30f };
    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        float dist = 
            depths[0] != 0.0f ? abs(depths[i] - depths[0]) : 0.0f +
            depths[1] != 0.0f ? abs(depths[i] - depths[1]) : 0.0f +
            depths[2] != 0.0f ? abs(depths[i] - depths[2]) : 0.0f +
            depths[3] != 0.0f ? abs(depths[i] - depths[3]) : 0.0f +
            foreground_bias * (max_depth - depths[i]);
        depth_dist[i] = depths[i] != 0.0f ? dist : 1e30f;
    }

    // Distances within this (depth-scaled) band count as equal, so a flat surface's near-identical
    // depths all tie and fall through to the spatial (top-then-left) tiebreak below.
    const float repr_tol = 0.02f * max_depth;

    int rep = 0; // default when every pixel is sky
    [unroll]
    for (int i = 1; i < 4; ++i)
    {
        if (depths[i] == 0.0f) continue;
        if (depths[rep] == 0.0f) { rep = i; continue; }

        // Only replace when strictly better than the tolerance band; on a tie we keep the lower
        // index (top-then-left), since i increases through the loop.
        if (depth_dist[i] < depth_dist[rep] - repr_tol)
            rep = i;
    }
    return rep;
}

/// ===== =====