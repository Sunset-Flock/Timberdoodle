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
    }
    if (weight_sum > 1e-3f)
    {
        moments = sum / weight_sum;
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

// Temporal gradient lambda (rt_lab_rays.inl) at a lab pixel, scaled by temporal_gradient_scale. Edge aware 2x2 bilinear
// upscale of the cells: bilinear weight x the pixel vs cell coarse surface test (rt_lab_gradient_surface_weight, plane
// falloff RT_LAB_GRADIENT_UPSCALE_PLANE_TOLERANCE pixel widths), renormalized; no matching cell -> plain bilinear. Reprojected
// history lengths are multiplied by 1 - this.
func rt_lab_temporal_gradient(Texture2D<uint4> temporal_gradient, CameraInfo camera, uint2 lab_size, uint2 pixel, float3 position,
    float3 normal, float rcp_pixel_width_ws, float scale) -> float
{
    uint2 cells;
    temporal_gradient.GetDimensions(cells.x, cells.y);
    const float2 inv_lab_size = rcp(float2(lab_size));
    const float2 cell_position = (float2(pixel) + 0.5f) / float(RT_LAB_GRADIENT_CELL) - 0.5f;
    const int2 origin = int2(floor(cell_position));
    const float2 f = cell_position - float2(origin);
    float plain = 0.0f;
    float lambda_sum = 0.0f;
    float weight_sum = 0.0f;
    [unroll] for (uint i = 0u; i < 4u; ++i)
    {
        const int2 offset = int2(i & 1u, i >> 1u);
        const uint2 cell = uint2(clamp(origin + offset, int2(0, 0), int2(cells) - 1));
        const float bilinear_weight = (offset.x == 1 ? f.x : 1.0f - f.x) * (offset.y == 1 ? f.y : 1.0f - f.y);
        const uint4 data = temporal_gradient[cell];
        const float lambda = asfloat(data.x);
        const float cell_depth = asfloat(data.y);
        plain += lambda * bilinear_weight;
        if (cell_depth > 0.0f)
        {
            const float3 cell_world = linear_depth_to_world_space(camera, rt_lab_ndc(rt_lab_gradient_cell_pixel(cell, lab_size), inv_lab_size), cell_depth);
            const float weight = bilinear_weight * rt_lab_gradient_surface_weight(position, normal, cell_world,
                uncompress_normal_octahedral_32(data.z), rcp_pixel_width_ws, RT_LAB_GRADIENT_UPSCALE_PLANE_TOLERANCE);
            lambda_sum += lambda * weight;
            weight_sum += weight;
        }
    }
    const float lambda = weight_sum > 1e-3f ? lambda_sum / weight_sum : plain;
    return saturate(lambda * scale);
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

// Variance guide (fine + coarse spatial filters): likelihood that center and tap estimate the same mean. Both are
// N(mean, variance / n) (mean_variance = per sample variance / effective sample count n), their difference is
// N(0, var_c / n_c + var_t / n_t). z^2 averaged over rgb, weight exp(-z^2 / (2 k^2)), k = tolerance in sigmas.
func rt_lab_variance_guide_likelihood(float3 mean_c, float3 mean_variance_c, float3 mean_t, float3 mean_variance_t,
    float3 variance_floor, float rcp_two_k_sq) -> float
{
    const float3 difference = mean_c - mean_t;
    const float3 z_sq = difference * difference / (mean_variance_c + mean_variance_t + variance_floor);
    return exp(-dot(z_sq, 1.0f / 3.0f) * rcp_two_k_sq);
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

// Coarse fallback edge stopping (RtLabSettings::coarse_*): no plane test. World distance relative to what a camera facing
// surface at this screen offset would give (~1; slanted surfaces and depth jumps are larger), smooth falloff
// exp(-(max(d_rel - 1, 0) / tolerance)^2), times a smooth normal test ((1 + dot) / 2)^power (0 at 180 degrees).
func rt_lab_coarse_edge_weight(float3 position, float3 normal, float3 tap_position, float3 tap_normal, float rcp_pixel_width_ws,
    float offset_px, float distance_tolerance, float normal_power) -> float
{
    const float relative_distance = distance(tap_position, position) * rcp_pixel_width_ws / offset_px;
    return exp(-square(max(relative_distance - 1.0f, 0.0f) / max(distance_tolerance, 1e-3f))) *
           pow(saturate(0.5f + 0.5f * dot(normal, tap_normal)), normal_power);
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
