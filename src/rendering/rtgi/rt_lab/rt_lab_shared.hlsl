#pragma once

#include "shader_shared/globals.inl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "../rtgi_shared.hlsl" // RTGI_RADIANCE_SCALE
#include "rt_lab_rays.inl" // RT_LAB_GRADIENT_CELL
#include "shader_lib/debug.glsl" // Heatmap
#include "shader_lib/raytracing.hlsl" // rt_calc_ray_start

// Lab resolution geometry: every lab pass works on the lab_depth (linear view depth, 0 = sky) and lab_face_normal
// images the trace writes, so full and half res run the same code.

func rt_lab_ndc(uint2 pixel, float2 inv_size) -> float2
{
    return (float2(pixel) + 0.5f) * inv_size * 2.0f - 1.0f;
}

// Hard ray budget of one frame (ray list entries the trace may use, rt_lab_rays.inl): ray_budget rays per lab pixel,
// 1 ray per pixel without redistribution, never more than the ray list capacity.
func rt_lab_ray_budget(uint2 size, RtLabSettings settings, uint ray_capacity) -> uint
{
    const uint pixel_count = size.x * size.y;
    const uint budget = settings.redistribution_enabled != 0 ? uint(float(pixel_count) * max(settings.ray_budget, 0.0f)) : pixel_count;
    return min(budget, ray_capacity);
}

// Lab geometry of one pixel from the g-buffer: half res reuses the g-buffer's half res linear depth, full res converts NDC
// depth (infinite reversed Z). Returns the linear depth (0 = sky).
func rt_lab_load_geometry(Texture2D<float> view_cam_depth, Texture2D<uint> view_cam_face_normals, Texture2D<float> half_res_depth,
    Texture2D<uint> half_res_face_normals, CameraInfo camera, bool half_res, uint2 pixel, out uint packed_normal) -> float
{
    if (half_res)
    {
        packed_normal = half_res_face_normals[pixel];
        return half_res_depth[pixel];
    }
    packed_normal = view_cam_face_normals[pixel];
    const float ndc_depth = view_cam_depth[pixel];
    return ndc_depth > 0.0f ? camera.near_plane / ndc_depth : 0.0f;
}

// Rec. 709 luminance.
func rt_lab_luminance(float3 v) -> float
{
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

// Radiance (RTGI_RADIANCE_SCALE scaled) -> exposed irradiance units (display ~0..1, the INDIRECT_DIFFUSE view's units).
func rt_lab_exposed(float scaled_radiance, float exposure) -> float
{
    return scaled_radiance * PI * exposure / RTGI_RADIANCE_SCALE;
}

// Lab pixel whose geometry represents a temporal gradient cell in the edge aware filtering (rt_lab_rays.inl).
func rt_lab_gradient_cell_pixel(uint2 cell, uint2 lab_size) -> uint2
{
    return min(cell * RT_LAB_GRADIENT_CELL + RT_LAB_GRADIENT_CELL / 2, lab_size - 1u);
}

// COARSE edge stopping between a point and a gradient cell's representative (the gradient only needs to stay on roughly
// the same surface): plane distance along the point's normal with a smooth falloff exp(-(d / tolerance)^2) (d in units
// of 1 / rcp_width_ws), times a smooth normal test ((1 + dot) / 2)^RT_LAB_GRADIENT_NORMAL_POWER (0 at 180 degrees).
func rt_lab_gradient_surface_weight(float3 position, float3 normal, float3 cell_position, float3 cell_normal, float rcp_width_ws, float tolerance) -> float
{
    const float plane_distance = abs(dot(cell_position - position, normal)) * rcp_width_ws;
    return exp(-square(plane_distance / max(tolerance, 1e-3f))) *
           pow(saturate(0.5f + 0.5f * dot(normal, cell_normal)), RT_LAB_GRADIENT_NORMAL_POWER);
}

// Representative geometry of a gradient-res cell (rt_lab_gradient_cell_pixel) straight from the g-buffer (usable before
// the demand writes this frame's lab geometry). Returns the linear depth (0 = sky).
func rt_lab_cell_geometry(Texture2D<float> view_cam_depth, Texture2D<uint> view_cam_face_normals, Texture2D<float> half_res_depth,
    Texture2D<uint> half_res_face_normals, CameraInfo camera, uint2 lab_size, bool half_res, uint2 cell, out float3 position, out uint packed_normal) -> float
{
    const uint2 pixel = rt_lab_gradient_cell_pixel(cell, lab_size);
    const float depth = rt_lab_load_geometry(view_cam_depth, view_cam_face_normals, half_res_depth, half_res_face_normals, camera, half_res, pixel, packed_normal);
    position = linear_depth_to_world_space(camera, rt_lab_ndc(pixel, rcp(float2(lab_size))), depth);
    return depth;
}

// 1 = alpha (the cut) is the MAX over the matching 2x2 cells instead of interpolated (a cut cell, e.g. at a moving shadow
// edge, is not diluted by uncut neighbors); the mean stays interpolated. Measured 2026-10 bistro sun sweeps: lag p50 about
// -10 frames (street) / -7 (wall), static scene unchanged. 0 = interpolated alpha.
#define RT_LAB_ANTI_LAG_MAX_ALPHA_UPSCALE 1

// TAA style anti-lag history clip (rt_lab_anti_lag.hlsl): the slow coarse mean is clipped into the fast mean's noise band
// fast x (1 +- RT_LAB_ANTI_LAG_CLIP_SIGMAS x sigma), sigma = relative std of the fast mean (floored), BEFORE the MSE cut
// is decided; the temporal pass multiplies the pixel histories by the same ratio clipped / slow (edge aware upscale).
// The color jumps to the noise band edge, no samples are dropped, the cut only handles what stays inside the band.
// The ratio rides in the coarse history's frames channel (.z): f16 frames (low bits) | f16 ratio (high bits).
// Shader only (hot reload safe); 0 = old behaviour.
#define RT_LAB_ANTI_LAG_CLIP 1
#define RT_LAB_ANTI_LAG_CLIP_SIGMAS 2.0f
#define RT_LAB_ANTI_LAG_CLIP_MIN_RELATIVE_STD 0.02f

func rt_lab_anti_lag_pack_frames(float frames, float ratio) -> float
{
#if RT_LAB_ANTI_LAG_CLIP
    // Positive f16 ratio in the high bits: the f32 exponent stays far below inf / NaN, the value stays > 0 when frames > 0.
    return asfloat((f32tof16(clamp(ratio, 0.01f, 100.0f)) << 16u) | f32tof16(max(frames, 0.0f)));
#else
    return frames;
#endif
}

func rt_lab_anti_lag_unpack_frames(float packed) -> float
{
#if RT_LAB_ANTI_LAG_CLIP
    return f16tof32(asuint(packed) & 0xFFFFu);
#else
    return packed;
#endif
}

func rt_lab_anti_lag_unpack_ratio(float packed) -> float
{
#if RT_LAB_ANTI_LAG_CLIP
    return f16tof32(asuint(packed) >> 16u);
#else
    return 1.0f;
#endif
}

// Clip ratio of the coarse anti-lag at a lab pixel (RT_LAB_ANTI_LAG_CLIP): same edge aware 2x2 upscale as
// rt_lab_coarse_anti_lag_stats, cells without statistics skipped, 1 = nothing to rescale.
func rt_lab_coarse_anti_lag_ratio(Texture2D<float4> coarse, CameraInfo camera, uint2 lab_size, uint2 pixel, float3 position, float3 normal,
    float rcp_pixel_width_ws) -> float
{
#if RT_LAB_ANTI_LAG_CLIP
    uint2 cells;
    coarse.GetDimensions(cells.x, cells.y);
    const float2 inv_lab_size = rcp(float2(lab_size));
    const float2 cell_position = (float2(pixel) + 0.5f) / float(RT_LAB_GRADIENT_CELL) - 0.5f;
    const int2 origin = int2(floor(cell_position));
    const float2 f = cell_position - float2(origin);
    float sum = 0.0f;
    float weight_sum = 0.0f;
    [unroll] for (uint i = 0u; i < 4u; ++i)
    {
        const int2 offset = int2(i & 1u, i >> 1u);
        const uint2 cell = uint2(clamp(origin + offset, int2(0, 0), int2(cells) - 1));
        const float4 data = coarse[cell];
        if (!(data.z > 0.0f) || !(data.w > 0.0f) || !all(isfinite(data)))
        {
            continue;
        }
        const float bilinear_weight = (offset.x == 1 ? f.x : 1.0f - f.x) * (offset.y == 1 ? f.y : 1.0f - f.y);
        const float3 cell_world = linear_depth_to_world_space(camera, rt_lab_ndc(rt_lab_gradient_cell_pixel(cell, lab_size), inv_lab_size), data.w);
        const float weight = bilinear_weight * rt_lab_gradient_surface_weight(position, normal, cell_world, normal, rcp_pixel_width_ws, RT_LAB_GRADIENT_UPSCALE_PLANE_TOLERANCE);
        sum += rt_lab_anti_lag_unpack_ratio(data.z) * weight;
        weight_sum += weight;
    }
    return weight_sum > 1e-3f ? sum / weight_sum : 1.0f;
#else
    return 1.0f;
#endif
}

// Coarse temporal anti-lag statistics (rt_lab_anti_lag.inl) at a lab pixel: (coarse temporal mean luminance, clamp alpha) of
// the coarse history, edge aware 2x2 bilinear upscale like rt_lab_temporal_gradient (alpha interpolates linearly, a
// length cap would not). Returns false when no cell holds statistics.
func rt_lab_coarse_anti_lag_stats(Texture2D<float4> coarse, CameraInfo camera, uint2 lab_size, uint2 pixel, float3 position, float3 normal,
    float rcp_pixel_width_ws, out float2 moments) -> bool
{
    uint2 cells;
    coarse.GetDimensions(cells.x, cells.y);
    const float2 inv_lab_size = rcp(float2(lab_size));
    const float2 cell_position = (float2(pixel) + 0.5f) / float(RT_LAB_GRADIENT_CELL) - 0.5f;
    const int2 origin = int2(floor(cell_position));
    const float2 f = cell_position - float2(origin);
    float2 plain = float2(0.0f, 0.0f);
    float plain_weight = 0.0f;
    float2 sum = float2(0.0f, 0.0f);
    float weight_sum = 0.0f;
    float max_alpha = 0.0f;
    [unroll] for (uint i = 0u; i < 4u; ++i)
    {
        const int2 offset = int2(i & 1u, i >> 1u);
        const uint2 cell = uint2(clamp(origin + offset, int2(0, 0), int2(cells) - 1));
        const float4 data = coarse[cell];
        if (!(data.z > 0.0f) || !(data.w > 0.0f) || !all(isfinite(data)))
        {
            continue;
        }
        const float bilinear_weight = (offset.x == 1 ? f.x : 1.0f - f.x) * (offset.y == 1 ? f.y : 1.0f - f.y);
        plain += data.xy * bilinear_weight;
        plain_weight += bilinear_weight;
        const float3 cell_world = linear_depth_to_world_space(camera, rt_lab_ndc(rt_lab_gradient_cell_pixel(cell, lab_size), inv_lab_size), data.w);
        // Only the plane distance along the pixel's normal (the cell's normal is not stored).
        const float weight = bilinear_weight * rt_lab_gradient_surface_weight(position, normal, cell_world, normal, rcp_pixel_width_ws, RT_LAB_GRADIENT_UPSCALE_PLANE_TOLERANCE);
        sum += data.xy * weight;
        weight_sum += weight;
        if (weight > 1e-3f)
        {
            max_alpha = max(max_alpha, data.y);
        }
    }
    if (weight_sum > 1e-3f)
    {
        moments = sum / weight_sum;
#if RT_LAB_ANTI_LAG_MAX_ALPHA_UPSCALE
        moments.y = max_alpha;
#endif
        return true;
    }
    moments = plain_weight > 1e-3f ? plain / plain_weight : float2(0.0f, 0.0f);
    return plain_weight > 1e-3f;
}

// History length cap of a clamp alpha: alpha = 1 / (length + 1) -> length <= 1 / alpha - 1. The color is not touched,
// only the history length (sample count) shrinks.
func rt_lab_coarse_anti_lag_cap(float history_length, float alpha) -> float
{
    if (!(alpha > 1e-6f))
    {
        return history_length;
    }
    return min(history_length, max(1.0f / alpha - 1.0f, 0.0f));
}

// temporal_gradient_deadzone: lambda' = saturate((lambda - d) / (1 - d)).
func rt_lab_gradient_deadzone(RtLabSettings settings, float lambda) -> float
{
    const float d = clamp(settings.temporal_gradient_deadzone, 0.0f, 0.99f);
    return saturate((lambda - d) / (1.0f - d));
}

// The temporal gradient's .x / .w are f16 pairs (pixel relative modes; the Kalman mode always uses the plain layout).
func rt_lab_gradient_packed(RtLabSettings settings) -> bool
{
    return settings.temporal_gradient_pixel_relative != RT_LAB_GRADIENT_RELATIVE_SAMPLES && (settings.temporal_gradient_cap_mode & RT_LAB_GRADIENT_CAP_KALMAN) == 0;
}

// f16 pair packing of the temporal gradient's .x / .w (temporal_gradient_pixel_relative != 0).
func rt_lab_gradient_pack_pair(float a, float b) -> uint
{
    return (f32tof16(clamp(a, -65000.0f, 65000.0f)) & 0xFFFFu) | (f32tof16(clamp(b, -65000.0f, 65000.0f)) << 16u);
}

func rt_lab_gradient_unpack_pair(uint packed) -> float2
{
    return float2(f16tof32(packed & 0xFFFFu), f16tof32(packed >> 16u));
}

// The temporal gradient cell's (lambda, |mean delta|, signed lambda, signed mean delta); unpacked layout (mode 0):
// deltas 0.
func rt_lab_gradient_cell_values(uint4 data, bool packed) -> float4
{
    if (packed)
    {
        const float2 x = rt_lab_gradient_unpack_pair(data.x);
        const float2 w = rt_lab_gradient_unpack_pair(data.w);
        return float4(x, w);
    }
    return float4(asfloat(data.x), 0.0f, asfloat(data.w), 0.0f);
}

// Temporal gradient at a lab pixel: (lambda (dilated, unscaled), |mean delta| (dilated), signed undilated lambda, signed
// undilated mean delta), deltas only with packed (temporal_gradient_pixel_relative != 0). Edge aware 2x2 bilinear
// upscale of the cells: bilinear weight x the pixel vs cell distance + loose normal weight (rt_lab_gradient_distance_weight,
// no plane test), renormalized; no matching cell -> plain bilinear.
func rt_lab_temporal_gradient_full(Texture2D<uint4> temporal_gradient, CameraInfo camera, uint2 lab_size, uint2 pixel, float3 position,
    float3 normal, float rcp_pixel_width_ws, bool packed) -> float4
{
    uint2 cells;
    temporal_gradient.GetDimensions(cells.x, cells.y);
    const float2 inv_lab_size = rcp(float2(lab_size));
    const float2 cell_position = (float2(pixel) + 0.5f) / float(RT_LAB_GRADIENT_CELL) - 0.5f;
    const int2 origin = int2(floor(cell_position));
    const float2 f = cell_position - float2(origin);
    float4 plain = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 lambda_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    [unroll] for (uint i = 0u; i < 4u; ++i)
    {
        const int2 offset = int2(i & 1u, i >> 1u);
        const uint2 cell = uint2(clamp(origin + offset, int2(0, 0), int2(cells) - 1));
        const float bilinear_weight = (offset.x == 1 ? f.x : 1.0f - f.x) * (offset.y == 1 ? f.y : 1.0f - f.y);
        const uint4 data = temporal_gradient[cell];
        const float4 lambda = rt_lab_gradient_cell_values(data, packed);
        const float cell_depth = asfloat(data.y);
        plain += lambda * bilinear_weight;
        if (cell_depth > 0.0f)
        {
            const uint2 cell_pixel = rt_lab_gradient_cell_pixel(cell, lab_size);
            const float3 cell_world = linear_depth_to_world_space(camera, rt_lab_ndc(cell_pixel, inv_lab_size), cell_depth);
            const float weight = bilinear_weight * rt_lab_gradient_distance_weight(camera.position, position, normal, cell_world,
                uncompress_normal_octahedral_32(data.z), rcp_pixel_width_ws, length(float2(cell_pixel) - float2(pixel)));
            lambda_sum += lambda * weight;
            weight_sum += weight;
        }
    }
    return weight_sum > 1e-3f ? lambda_sum / weight_sum : plain;
}

// Temporal gradient lambda scaled by temporal_gradient_scale (scale mode: reprojected history lengths x (1 - this)).
func rt_lab_temporal_gradient(Texture2D<uint4> temporal_gradient, CameraInfo camera, uint2 lab_size, uint2 pixel, float3 position,
    float3 normal, float rcp_pixel_width_ws, float scale) -> float
{
    return saturate(rt_lab_temporal_gradient_full(temporal_gradient, camera, lab_size, pixel, position, normal, rcp_pixel_width_ws, false).x * scale);
}

// Per sample perceptual variance luma var / (mean + floor)^2 (std of log(L + floor)) from last frame's variance guide
// estimate and history mean at the nearest previous pixel; floor = variance_laziness_perceptual_floor x display white.
// < 0 = unknown (off screen last frame / broken).
func rt_lab_previous_perceptual_variance(Texture2D<float4> history, Texture2D<float4> variance, CameraInfo* prev_camera,
    RtLabSettings settings, uint2 size, float3 position, float exposure) -> float
{
    const float4 prev_clip = mul(prev_camera.view_proj, float4(position, 1.0f));
    const float2 prev_uv = (prev_clip.xy / prev_clip.w) * 0.5f + 0.5f;
    if (!(prev_clip.w > 0.0f) || any(prev_uv <= 0.0f) || any(prev_uv >= 1.0f))
    {
        return -1.0f;
    }
    const uint2 pixel = min(uint2(prev_uv * float2(size)), size - 1u);
    const float3 sample_variance = variance[pixel].rgb;
    const float mean = rt_lab_luminance(history[pixel].rgb);
    const float perceptual_floor = max(settings.variance_laziness_perceptual_floor, 1e-4f) * RTGI_RADIANCE_SCALE / (PI * exposure);
    const float perceptual_variance = dot(max(sample_variance, 0.0f), square(float3(0.2126f, 0.7152f, 0.0722f))) / square(max(mean, 0.0f) + perceptual_floor);
    return isfinite(perceptual_variance) ? perceptual_variance : -1.0f;
}

// Temporal gradient history cap (temporal_gradient_cap_mode, see rt_lab_settings.inl): the history length limit for an
// unscaled lambda; 1e30 = no cap. perceptual_variance < 0 (unknown) skips the MSE cap.
func rt_lab_gradient_history_cap(RtLabSettings settings, float lambda, float perceptual_variance, float history_length) -> float
{
    if ((settings.temporal_gradient_cap_mode & RT_LAB_GRADIENT_CAP_KALMAN) != 0)
    {
        // lambda = the information loss L: N' = N / (1 + N k L / v) (continuous, no threshold). Unknown v: 1.
        const float v = perceptual_variance >= 0.0f ? max(perceptual_variance, 1e-4f) : 1.0f;
        const float kept = history_length / (1.0f + history_length * max(settings.temporal_gradient_kalman_k, 0.0f) * max(lambda, 0.0f) / v);
        return max(kept, min(history_length, max(settings.temporal_gradient_cap_min, 0.0f)));
    }
    float cap = 1e30f;
    if ((settings.temporal_gradient_cap_mode & RT_LAB_GRADIENT_CAP_MSE) != 0 && perceptual_variance >= 0.0f)
    {
        const float excess = square(lambda) - square(RT_LAB_GRADIENT_CAP_LAMBDA_FLOOR);
        if (excess > 0.0f)
        {
            cap = min(cap, max(settings.temporal_gradient_cap_k, 0.0f) * perceptual_variance / excess);
        }
    }
    const float tolerance = max(settings.temporal_gradient_settle_tolerance, 1e-4f);
    if ((settings.temporal_gradient_cap_mode & RT_LAB_GRADIENT_CAP_SETTLE) != 0 && lambda > tolerance)
    {
        const float growth = pow(lambda / tolerance, 1.0f / max(settings.temporal_gradient_settle_frames, 1.0f));
        cap = min(cap, 1.0f / max(growth - 1.0f, 1e-6f));
    }
    if (cap < 1e30f)
    {
        cap = max(cap, max(settings.temporal_gradient_cap_min, 0.0f));
    }
    return cap;
}

// History rescale factor (temporal_gradient_rescale) from the signed undilated lambda: sum new / sum old of the re-traced
// rays assuming one change direction in the footprint.
func rt_lab_gradient_rescale(RtLabSettings settings, float signed_lambda) -> float
{
    if ((settings.temporal_gradient_cap_mode & RT_LAB_GRADIENT_CAP_KALMAN) != 0)
    {
        // signed_lambda = the log correction c (already Wiener shrunk + clamped).
        return exp(signed_lambda * clamp(settings.temporal_gradient_rescale_strength, 0.0f, 1.0f));
    }
    const float s = clamp(signed_lambda, -RT_LAB_GRADIENT_RESCALE_MAX_LAMBDA, RT_LAB_GRADIENT_RESCALE_MAX_LAMBDA);
    const float ratio = s >= 0.0f ? 1.0f / (1.0f - s) : 1.0f + s;
    return pow(ratio, clamp(settings.temporal_gradient_rescale_strength, 0.0f, 1.0f));
}

// World space width of one pixel of an image of height size_y at linear depth.
func rt_lab_pixel_width_ws(CameraInfo camera, float size_y, float linear_depth) -> float
{
    return 2.0f * linear_depth / (abs(camera.proj[1][1]) * size_y);
}

// Half res lab: lab_depth carries the index (0..3, x + 2y) of the full res sub-pixel the gen_gbuffer representative
// depth came from in its 2 lowest mantissa bits (relative depth change ~2e-7, invisible to every other consumer).
// Written by the demand pass, read by rt_lab_ray_launch. Sky (0) stays untagged.
func rt_lab_tag_subpixel(float depth, uint subpixel) -> float
{
    return depth > 0.0f ? asfloat((asuint(depth) & ~3u) | (subpixel & 3u)) : depth;
}

func rt_lab_depth_subpixel(float depth) -> uint
{
    return asuint(depth) & 3u;
}

// Launch data of a lab pixel's diffuse rays: returns the origin, writes the pixel's surface point and the ray t min.
// Shared by the trace and the temporal gradient selection (gather), which records it for the next frame's re-trace.
// RTGI ray start (rtgi_load_trace_pixel + rtgi_specular_ray_desc, rtgi_trace.hlsl): ws_px_size =
// rtgi_half_res_pixel_width_ws (half res pixel width, at full AND half res lab), origin = rt_calc_ray_start - primary_ray
// * ws_px_size + face_normal * ws_px_size, t min = ws_px_size * 0.5.
// Half res: the position is the representative's depth at its OWN full res sub-pixel (rt_lab_depth_subpixel), NOT at the
// half res pixel center. A quad straddling a floor / wall contact put the floor depth on the center ray = behind the
// wall -> rays started behind it and returned outside light = dashed bright pixels along concave edges (user 2026-10).
func rt_lab_ray_launch(CameraInfo camera, uint2 pixel, uint2 size, float depth, float3 face_normal, out float3 world_pos, out float t_min) -> float3
{
    float2 ndc = rt_lab_ndc(pixel, rcp(float2(size)));
    const float full_width = rcp(camera.inv_screen_size.x);
    if (float(size.x) * 1.5f < full_width && depth > 0.0f)
    {
        const uint subpixel = rt_lab_depth_subpixel(depth);
        const uint2 full_res_pixel = pixel * 2u + uint2(subpixel & 1u, subpixel >> 1u);
        ndc = (float2(full_res_pixel) + 0.5f) * camera.inv_screen_size * 2.0f - 1.0f;
    }
    world_pos = rtgi_half_res_depth_to_world_space(camera, ndc, depth);
    const float2 half_res_inv_render_target_size = camera.inv_screen_size * 2.0f;
    const float ws_px_size = depth > 0.0f ? rtgi_half_res_pixel_width_ws(half_res_inv_render_target_size, camera.near_plane, depth) : 0.0f;
    const float3 primary_ray = normalize(world_pos - camera.position);
    t_min = ws_px_size * 0.5f;
    return rt_calc_ray_start(world_pos, face_normal, primary_ray) - primary_ray * ws_px_size + face_normal * ws_px_size;
}

// Seed of a lab ray's direction sample (trace): per frame (animate_noise), pixel and ray of the pixel. Ray 0 = the old
// 1 ray per pixel seed.
func rt_lab_ray_seed(RtLabSettings settings, float trunk_flt_frame_index, uint2 pixel, uint pixel_ray) -> uint
{
    const uint frame_seed = settings.animate_noise ? uint(trunk_flt_frame_index) * 257u : 0u;
    return frame_seed + pixel.x * 9629u + pixel.y * 10069u + pixel_ray * 1000003u;
}

// Seed of a lab ray's hit shading (rand() inside the shading, e.g. VSM PCF rotations), derived from its ray seed (own
// stream), 24 bits: stored in the temporal gradient record so the re-trace shades with the SAME random numbers.
func rt_lab_hit_seed(uint ray_seed) -> uint
{
    return ((ray_seed ^ 0x68E31DA4u) * 2654435761u + 0x7F4A7C15u) & RT_LAB_HIT_SEED_MASK;
}

// Relative difference of two luminances: |a - b| / min(a, b). Scale free, and loud exactly where convergence is worst:
// when one of the two caught a rare bright event and the other did not, min stays small and the ratio explodes.
// Luminances below RT_LAB_RELATIVE_DIFFERENCE_FLOOR (exposed irradiance) count as the floor.
#define RT_LAB_RELATIVE_DIFFERENCE_FLOOR 0.01f
func rt_lab_relative_difference(float a, float b) -> float
{
    return abs(a - b) / max(min(a, b), RT_LAB_RELATIVE_DIFFERENCE_FLOOR);
}

// Temporal gradient at a pixel as (lambda, signed lambda) from rt_lab_temporal_gradient_full's output
// (temporal_gradient_pixel_relative, RT_LAB_GRADIENT_RELATIVE_*). SAMPLES: the samples relative lambda unchanged.
// PIXEL: the absolute exposed (|delta|, signed delta) divided by the pixel's own previous history luminance L (nearest
// previous pixel, like rt_lab_previous_perceptual_variance): max(L + max(delta, 0), floor) = max(old, new) of the pixel,
// so the rescale ratio (rt_lab_gradient_rescale) is exactly (L + delta) / L. MIN: the smaller of both (signed: smaller
// magnitude). No history there: (0, 0) (nothing to keep or cut).
func rt_lab_gradient_pixel_relative(RtLabSettings settings, float4 gradient, Texture2D<float4> history, CameraInfo* prev_camera,
    uint2 size, float3 position, float exposure) -> float2
{
    const int mode = settings.temporal_gradient_pixel_relative;
    if (!rt_lab_gradient_packed(settings))
    {
        return gradient.xz;
    }
    const float4 prev_clip = mul(prev_camera.view_proj, float4(position, 1.0f));
    const float2 prev_uv = (prev_clip.xy / prev_clip.w) * 0.5f + 0.5f;
    if (!(prev_clip.w > 0.0f) || any(prev_uv <= 0.0f) || any(prev_uv >= 1.0f))
    {
        return float2(0.0f, 0.0f);
    }
    const float4 previous = history[min(uint2(prev_uv * float2(size)), size - 1u)];
    if (!(previous.a > 0.0f) || any(!isfinite(previous.rgb)))
    {
        return float2(0.0f, 0.0f);
    }
    const float luminance = rt_lab_exposed(max(rt_lab_luminance(previous.rgb), 0.0f), exposure);
    const float denominator = max(luminance + max(gradient.w, 0.0f), RT_LAB_RELATIVE_DIFFERENCE_FLOOR);
    const float2 pixel_relative = float2(saturate(gradient.y / denominator), clamp(gradient.w / denominator, -1.0f, 1.0f));
    if (mode == RT_LAB_GRADIENT_RELATIVE_MIN)
    {
        return float2(min(gradient.x, pixel_relative.x), abs(gradient.z) < abs(pixel_relative.y) ? gradient.z : pixel_relative.y);
    }
    return pixel_relative;
}

// Variance guide (fine + coarse spatial filters): likelihood that center and tap estimate the same mean. Both are
// N(mean, variance / n) (mean_variance = per sample variance / effective sample count n), their difference is
// N(0, var_c / n_c + var_t / n_t). z^2 averaged over rgb, weight exp(-z^2 / (2 k^2)), k = tolerance in sigmas.
// tail_nu > 0: Student-t weight (1 + z^2 / (nu k^2))^(-(nu + 1) / 2) instead (RtLabSettings::variance_guide_heavy_tail).
func rt_lab_variance_guide_likelihood(float3 mean_c, float3 mean_variance_c, float3 mean_t, float3 mean_variance_t,
    float3 variance_floor, float rcp_two_k_sq, float tail_nu) -> float
{
    const float3 difference = mean_c - mean_t;
    const float3 z_sq = difference * difference / (mean_variance_c + mean_variance_t + variance_floor);
    const float z_sq_mean = dot(z_sq, 1.0f / 3.0f);
    if (tail_nu > 0.0f)
    {
        // z^2 / (nu k^2) = z^2 x 2 rcp_two_k_sq / nu
        return pow(1.0f + z_sq_mean * 2.0f * rcp_two_k_sq / tail_nu, -0.5f * (tail_nu + 1.0f));
    }
    return exp(-z_sq_mean * rcp_two_k_sq);
}

// Student-t nu of the variance guide, 0 = gaussian (RtLabSettings::variance_guide_heavy_tail).
func rt_lab_variance_guide_tail_nu(RtLabSettings* settings) -> float
{
    return settings->variance_guide_heavy_tail != 0 ? max(settings->variance_guide_tail_nu, 0.1f) : 0.0f;
}

// Effective sample count used in the variance guide's standard error (RtLabSettings::variance_guide_count_cap_enabled).
func rt_lab_variance_guide_count(RtLabSettings* settings, float count) -> float
{
    const float n = max(count, 1e-3f);
    return settings->variance_guide_count_cap_enabled != 0 ? min(n, max(settings->variance_guide_count_cap, 1.0f)) : n;
}

// Variance floor of the guide: 0.1 % of display white per channel, so black / unlit channels (variance 0) do not
// reject everything.
func rt_lab_variance_guide_floor(float exposure) -> float3
{
    const float luma_floor = 0.1f * RT_LAB_RELATIVE_DIFFERENCE_FLOOR * RTGI_RADIANCE_SCALE / (PI * exposure);
    return float3(luma_floor * luma_floor, luma_floor * luma_floor, luma_floor * luma_floor);
}

// Log heatmap of a relative difference: 0.1% -> 0 (cold), 1% -> 1/3, 10% -> 2/3, >= 100% -> 1 (hot).
func rt_lab_relative_difference_heatmap(float relative) -> float3
{
    return Heatmap(saturate((log10(max(relative, 1e-6f)) + 3.0f) / 3.0f));
}

// Spatial images (blur / a-trous / coarse passes) store the effective sample count in alpha DIVIDED by this: the counts
// grow huge (coarse ~7x per pass) and f16 overflows at 65504. f16 relative precision is the same at any magnitude, so
// /128 costs no precision and moves the top to ~8.4M. The temporal history alpha stays the plain history length.
#define RT_LAB_SPATIAL_COUNT_SCALE 128.0f

// Effective sample count of a spatial image texel: first_pass = the input is the temporal history (plain history
// length, clamped to sample_count_input_clamp), else a spatial image (alpha x RT_LAB_SPATIAL_COUNT_SCALE).
func rt_lab_spatial_read_count(float alpha, bool first_pass, float clamp_value) -> float
{
    return first_pass ? ((clamp_value > 0.0f) ? min(alpha, clamp_value) : alpha) : alpha * RT_LAB_SPATIAL_COUNT_SCALE;
}

// Coarse spatial fallback top up: the blend t of lerp(fine, coarse, t) that tops the effective sample count up to
// target. Fine / coarse are treated as estimates with variance 1 / N, so the blend has
//   N_eff(t) = 1 / ((1 - t)^2 / N_fine + t^2 / N_coarse),
// and t is the smaller root of N_eff(t) = target. A linear lerp of the counts would not work: the a-trous count compounds
// (~7x per pass, overcounts reused samples), N_coarse ~1M would give t ~0; here a huge N_coarse just means
// t ~ 1 - sqrt(N_fine / target). Unreachable target: the minimum variance mix t = N_coarse / (N_fine + N_coarse). Fine
// already at target: 0. Capped by max_blend. Ignores that fine and coarse share samples (N_eff a bit optimistic).
func rt_lab_coarse_top_up(float fine_count, float coarse_count, float target, float max_blend, out float effective_count) -> float
{
    const float n_fine = max(fine_count, 1e-3f);
    const float n_coarse = max(coarse_count, 1e-3f);
    effective_count = n_fine;
    if (!(target > n_fine) || !(max_blend > 0.0f))
    {
        return 0.0f;
    }
    // (1/n_f + 1/n_c) t^2 - 2/n_f t + (1/n_f - 1/target) = 0
    const float a = 1.0f / n_fine + 1.0f / n_coarse;
    const float b = -2.0f / n_fine;
    const float c = 1.0f / n_fine - 1.0f / target;
    const float discriminant = b * b - 4.0f * a * c;
    const float t_minimum_variance = n_coarse / (n_fine + n_coarse);
    float t = discriminant >= 0.0f ? (-b - sqrt(discriminant)) / (2.0f * a) : t_minimum_variance;
    t = min(clamp(t, 0.0f, t_minimum_variance), saturate(max_blend));
    effective_count = 1.0f / (square(1.0f - t) / n_fine + square(t) / n_coarse);
    return t;
}

// Ray length guide (RtLabSettings::ray_length_guide_*): the RTGI max visibility distance of a lab pixel at linear depth,
// R = ray_length_guide_radius_px x the RTGI half res pixel width (rtgi_half_res_pixel_width_ws, = what RTGI's
// calc_ray_shortness gets with max_visibility_pixel_range; same at full and half res lab).
func rt_lab_ray_length_max_distance(RenderGlobalData* globals, float linear_depth) -> float
{
    return globals.rt_lab_settings.ray_length_guide_radius_px *
           rtgi_half_res_pixel_width_ws(globals.settings.render_target_size_inv * 2.0f, globals.view_camera.near_plane, linear_depth);
}

// Firefly energy weighting (RtLabSettings::firefly_weight_enabled): accumulated raw (unclamped) luminance / accumulated
// filtered luminance of a pixel, >= 1, capped (a history with ~0 filtered energy but some raw energy would explode).
#define RT_LAB_FIREFLY_WEIGHT_MAX 1024.0f
func rt_lab_firefly_factor(float raw_luminance, float filtered_luminance) -> float
{
    if (!(raw_luminance > 0.0f) || !(filtered_luminance > 0.0f))
    {
        return 1.0f;
    }
    return clamp(raw_luminance / filtered_luminance, 1.0f, RT_LAB_FIREFLY_WEIGHT_MAX);
}

// Most the expected tap distance may grow over the camera facing one on slanted surfaces (= 1 / sin(view angle), 16 = ~3.6
// degrees grazing): beyond, the tangent plane says little and a far depth jump would pass.
#define RT_LAB_EDGE_MAX_FORESHORTENING 16.0f

// Coarse fallback edge stopping (RtLabSettings::coarse_*): no plane test. World distance relative to the distance the tap
// would have on the center's TANGENT PLANE (where the tap's camera ray meets it; ~1 on the same surface at any view angle,
// depth jumps are larger), smooth falloff exp(-(max(d_rel - 1, 0) / tolerance)^2), times a smooth normal test
// ((1 + dot) / 2)^power (0 at 180 degrees). The expected distance never goes below the camera facing one
// (offset_px pixel widths).
// BUG FIXED 2026-10: the expected distance used to be the camera facing one in every direction. On a slanted surface a
// screen step along the slope covers 1 / sin(view angle) more world distance -> those taps read as far, the ones across the
// slope did not -> the footprint squashed along the slope (blob -> stripe at <= ~45 degrees), compounding over a-trous passes.
func rt_lab_coarse_edge_weight(float3 camera_position, float3 position, float3 normal, float3 tap_position, float3 tap_normal,
    float rcp_pixel_width_ws, float offset_px, float distance_tolerance, float normal_power) -> float
{
    const float facing_distance = offset_px / rcp_pixel_width_ws;
    const float3 tap_ray = tap_position - camera_position;
    const float ray_dot = dot(normal, tap_ray);
    const float plane_dot = dot(normal, position - camera_position);
    float expected_distance = facing_distance * RT_LAB_EDGE_MAX_FORESHORTENING;
    if (abs(ray_dot) > 1e-12f)
    {
        const float s = plane_dot / ray_dot; // tap ray parameter at the tangent plane (1 = the tap itself)
        if (s > 0.0f)
        {
            expected_distance = distance(camera_position + tap_ray * s, position);
        }
    }
    expected_distance = clamp(expected_distance, facing_distance, facing_distance * RT_LAB_EDGE_MAX_FORESHORTENING);
    const float relative_distance = distance(tap_position, position) / expected_distance;
    return exp(-square(max(relative_distance - 1.0f, 0.0f) / max(distance_tolerance, 1e-3f))) *
           pow(saturate(0.5f + 0.5f * dot(normal, tap_normal)), normal_power);
}

// Temporal gradient blur + upscale edge stopping: the coarse fallback's relative world distance (no plane test; offset_px =
// screen distance of the two points in lab pixels) x a somewhat loose normal weight.
func rt_lab_gradient_distance_weight(float3 camera_position, float3 position, float3 normal, float3 tap_position, float3 tap_normal, float rcp_pixel_width_ws, float offset_px) -> float
{
    return rt_lab_coarse_edge_weight(camera_position, position, normal, tap_position, tap_normal, rcp_pixel_width_ws, max(offset_px, 1.0f),
        RT_LAB_GRADIENT_DISTANCE_TOLERANCE, RT_LAB_GRADIENT_DISTANCE_NORMAL_POWER);
}

// RTGI SH-Y + CoCg (rtgi_shared.hlsl) of one cosine distributed ray: sh_y = (direction * Y, Y).
// sh_resolve_diffuse evaluates dot(n, sh.xyz) + 0.5 * sh.w. For cosine distributed rays around n under uniform radiance
// E[direction] = 2/3 n, so it returns 7/6 Y: RT_LAB_SH_RESOLVE_NORMALIZATION scales that back to Y, so the coarse result
// matches the plain rgb mean the fine path carries where both see the same surface.
#define RT_LAB_SH_RESOLVE_NORMALIZATION (6.0f / 7.0f)
func rt_lab_sh_resolve(float4 sh_y, float2 cocg, float3 normal) -> float3
{
    return sh_resolve_diffuse(sh_y, cocg, normal) * RT_LAB_SH_RESOLVE_NORMALIZATION;
}

// 1 on the same plane, falls linearly to 0 at tolerance_px pixel widths of distance to EITHER surface's plane.
func rt_lab_plane_weight(float3 a_pos, float3 a_normal, float3 b_pos, float3 b_normal, float rcp_pixel_width_ws, float tolerance_px) -> float
{
    const float3 d = b_pos - a_pos;
    const float dist = max(abs(dot(d, a_normal)), abs(dot(d, b_normal))) * rcp_pixel_width_ws;
    return saturate(1.0f - dist / max(tolerance_px, 1e-4f));
}
