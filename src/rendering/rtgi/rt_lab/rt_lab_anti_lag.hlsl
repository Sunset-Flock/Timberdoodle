#pragma once

#include "rt_lab_anti_lag.inl"

#include "rt_lab_shared.hlsl"
#include "shader_lib/debug.glsl"

[[vk::push_constant]] RtLabAntiLagDownsamplePush rt_lab_anti_lag_downsample_push;
[[vk::push_constant]] RtLabAntiLagBlurPush rt_lab_anti_lag_blur_push;

// One thread per cell: luminance moments over all rays of the cell's lab pixels this frame.
[shader("compute")]
[numthreads(RT_LAB_ANTI_LAG_X, RT_LAB_ANTI_LAG_Y, 1)]
func entry_rt_lab_anti_lag_downsample(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_anti_lag_downsample_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    float3 sums = float3(0.0f, 0.0f, 0.0f);
    for (uint i = 0u; i < RT_LAB_GRADIENT_CELL * RT_LAB_GRADIENT_CELL; ++i)
    {
        const uint2 pixel = dtid * RT_LAB_GRADIENT_CELL + uint2(i % RT_LAB_GRADIENT_CELL, i / RT_LAB_GRADIENT_CELL);
        if (any(pixel >= push.lab_size))
        {
            continue;
        }
        const uint2 alloc = push.attach.ray_alloc.get()[pixel];
        for (uint k = 0u; k < alloc.y; ++k)
        {
            const float luminance = rt_lab_luminance(push.attach.rays[alloc.x + k].radiance);
            if (isfinite(luminance))
            {
                sums += float3(luminance, luminance * luminance, 1.0f);
            }
        }
    }
    push.attach.cell_sums.get()[dtid] = float4(sums, 0.0f);
}

#define RT_LAB_ANTI_LAG_FAST_MIN_RELATIVE_STD 0.1f // floor of the fast relative std in the fast firefly clamp (escape from a stuck fast history)

// Last frame's coarse histories at this cell's surface point: bilinear over the previous cells, each tap's point rebuilt
// from the slow history's stored depth with the previous camera, coarse plane test along this cell's normal. The fast
// history (out) is gathered with the same weights. frames = 0: none.
func rt_lab_anti_lag_reproject(RtLabAntiLagBlurPush push, float3 position, float3 normal, float rcp_cell_width_ws, out float4 fast) -> float4
{
    fast = float4(0.0f, 0.0f, 0.0f, 0.0f);
    CameraInfo* prev_camera = &push.attach.globals.view_camera_prev_frame;
    const float4 prev_clip = mul(prev_camera.view_proj, float4(position, 1.0f));
    if (!(prev_clip.w > 0.0f))
    {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float2 prev_uv = (prev_clip.xy / prev_clip.w) * 0.5f + 0.5f;
    // Cell c is represented by lab pixel c x CELL + CELL / 2, whose center sits at c x CELL + CELL / 2 + 0.5.
    const float2 cell_coord = (prev_uv * float2(push.lab_size) - float(RT_LAB_GRADIENT_CELL / 2) - 0.5f) / float(RT_LAB_GRADIENT_CELL);
    const int2 origin = int2(floor(cell_coord));
    const float2 f = cell_coord - float2(origin);
    const float2 inv_lab_size = rcp(float2(push.lab_size));
    float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 fast_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    [unroll] for (uint i = 0u; i < 4u; ++i)
    {
        const int2 offset = int2(i & 1u, i >> 1u);
        const int2 cell = origin + offset;
        if (any(cell < 0) || any(cell >= int2(push.size)))
        {
            continue;
        }
        const float4 history = push.attach.history_previous.get()[cell];
        const float4 tap_fast = push.attach.fast_history_previous.get()[cell];
        if (!(history.w > 0.0f) || !(history.z > 0.0f) || !all(isfinite(history)))
        {
            continue;
        }
        const float3 tap_position = linear_depth_to_world_space(*prev_camera, rt_lab_ndc(rt_lab_gradient_cell_pixel(uint2(cell), push.lab_size), inv_lab_size), history.w);
        // Same normal for both: only the plane distance along this cell's normal is tested.
        const float weight = (offset.x == 1 ? f.x : 1.0f - f.x) * (offset.y == 1 ? f.y : 1.0f - f.y) *
            rt_lab_gradient_surface_weight(position, normal, tap_position, normal, rcp_cell_width_ws, RT_LAB_ANTI_LAG_BLUR_PLANE_TOLERANCE);
        sum += history * weight;
        fast_sum += (all(isfinite(tap_fast)) ? tap_fast : float4(0.0f, 0.0f, 0.0f, 0.0f)) * weight;
        weight_sum += weight;
    }
    if (weight_sum > 1e-3f)
    {
        fast = fast_sum / weight_sum;
        return sum / weight_sum;
    }
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
}

func rt_lab_anti_lag_debug_mode_active(int mode) -> bool
{
    return mode == DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG_FRAME_MEAN || mode == DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG_RELATIVE_STD ||
           mode == DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG_FRAME_RELATIVE_STD;
}

// Relative std sigma / mean of luminance moments (mean, second moment), log heatmap 1 % .. 1000 % (per sample spreads at
// ~1 ray per pixel sit around 100 %). Black: no data.
func rt_lab_anti_lag_relative_std_color(float2 moments) -> float3
{
    if (!(moments.x > 0.0f))
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    const float relative_std = sqrt(max(moments.y - moments.x * moments.x, 0.0f)) / moments.x;
    return Heatmap(saturate((log10(max(relative_std, 1e-6f)) + 2.0f) / 3.0f));
}

// Draws one cell (RT_LAB_GRADIENT_CELL lab pixels, the lab may be half res) of the cell res debug views.
func rt_lab_anti_lag_write_debug(RtLabAntiLagBlurPush push, uint2 cell, float3 color)
{
    const uint lab_scale = max(push.attach.globals.settings.render_target_size.x / push.lab_size.x, 1u);
    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, cell,
        float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend), RT_LAB_GRADIENT_CELL * lab_scale);
}

[shader("compute")]
[numthreads(RT_LAB_ANTI_LAG_X, RT_LAB_ANTI_LAG_Y, 1)]
func entry_rt_lab_anti_lag_blur(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_anti_lag_blur_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    let settings = push.attach.globals.rt_lab_settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    const bool vertical = push.vertical != 0u;
    const bool half_res = push.lab_size.x != push.attach.globals.settings.render_target_size.x;

    float3 center_position;
    uint center_packed_normal;
    const float center_depth = rt_lab_cell_geometry(push.attach.view_cam_depth.get(), push.attach.view_cam_face_normals.get(),
        push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get(), camera, push.lab_size, half_res,
        dtid, center_position, center_packed_normal);
    if (center_depth == 0.0f)
    {
        if (vertical)
        {
            push.attach.history.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            push.attach.fast_history.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (rt_lab_anti_lag_debug_mode_active(push.attach.globals.settings.debug_draw_mode))
            {
                rt_lab_anti_lag_write_debug(push, dtid, float3(0.0f, 0.0f, 0.0f));
            }
        }
        else
        {
            push.attach.blur_dst.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        return;
    }
    const float3 center_normal = uncompress_normal_octahedral_32(center_packed_normal);
    const float rcp_cell_width_ws = rcp(rt_lab_pixel_width_ws(camera, float(push.lab_size.y), center_depth) * float(RT_LAB_GRADIENT_CELL));

    // Edge aware gaussian over the moment sums (sum of w x sums: the result stays a sample weighted sum).
    const int radius = clamp(settings.coarse_anti_lag_blur_radius, 0, RT_LAB_GRADIENT_MAX_BLUR_RADIUS);
    const float sigma = max(float(radius) * 0.5f, 0.5f);
    const float rcp_two_sigma_sq = rcp(2.0f * sigma * sigma);
    const int2 direction = vertical ? int2(0, 1) : int2(1, 0);
    float3 sums = float3(0.0f, 0.0f, 0.0f);
    for (int d = -radius; d <= radius; ++d)
    {
        const int2 cell = int2(dtid) + direction * d;
        if (any(cell < 0) || any(cell >= int2(push.size)))
        {
            continue;
        }
        float weight = exp(-float(d * d) * rcp_two_sigma_sq);
        if (d != 0)
        {
            float3 tap_position;
            uint tap_packed_normal;
            if (rt_lab_cell_geometry(push.attach.view_cam_depth.get(), push.attach.view_cam_face_normals.get(),
                    push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get(), camera, push.lab_size, half_res,
                    uint2(cell), tap_position, tap_packed_normal) == 0.0f)
            {
                continue;
            }
            weight *= rt_lab_gradient_surface_weight(center_position, center_normal, tap_position,
                uncompress_normal_octahedral_32(tap_packed_normal), rcp_cell_width_ws, RT_LAB_ANTI_LAG_BLUR_PLANE_TOLERANCE);
            if (weight <= 0.0f)
            {
                continue;
            }
        }
        sums += push.attach.blur_src.get()[cell].xyz * weight;
    }
    if (!vertical)
    {
        push.attach.blur_dst.get()[dtid] = float4(sums, 0.0f);
        return;
    }

    // Temporal (RTGI fast history style, see rtgi_temporal.hlsl): the blurred frame mean x feeds a coarse FAST history
    // (fast mean, fast relative variance over coarse_anti_lag_history_frames frames, x firefly clamped against the fast stats
    // once the window is full) and the SLOW coarse mean (capped at the denoiser's history length). RTGI anti-lag on the slow history length N (once N > window):
    // cap = min(N, N x (1 + sqrt(adaptation variance) x coarse_anti_lag_variance_blend)^2 x (1 / max(1, slow / fast))^2).
    // The slow history is capped by it before x is blended in; alpha = 1 / (cap + 1) caps the pixel histories in the
    // temporal pass. Slow: .x mean, .y alpha (0 = no cut), .z frames, .w depth. Fast: .x mean, .y rel variance, .z frames.
    // A frame without rays in the footprint keeps both histories and cuts nothing.
    float4 fast = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 history = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (settings.reset_history == 0)
    {
        history = rt_lab_anti_lag_reproject(push, center_position, center_normal, rcp_cell_width_ws, fast);
    }
    const float2 moments = sums.z > 1e-3f ? sums.xy / sums.z : float2(0.0f, 0.0f); // this frame's blurred (mean, second moment)
    const float frame_variance = max(moments.y - moments.x * moments.x, 0.0f);
    float anti_lag_alpha = 0.0f;
    if (sums.z > 1e-3f)
    {
        const float luminance_floor = RT_LAB_RELATIVE_DIFFERENCE_FLOOR * RTGI_RADIANCE_SCALE / (PI * push.attach.globals.exposure);
        const float window = float(max(settings.coarse_anti_lag_history_frames, 1));
        float x = moments.x;
        if (fast.z < 0.5f)
        {
            fast = float4(x, 0.0f, 1.0f, 0.0f);
        }
        else
        {
            const float fast_frames = min(fast.z + 1.0f, window + 1.0f);
            const float fast_blend = 1.0f / (1.0f + min(fast_frames, window));
            const float fast_mean_prev = fast.x;
            const float fast_var_prev = fast.y;
            if (fast_frames > window)
            {
                // Floors on both terms: a fast mean of ~0 (dark before the change) or a fast variance of ~0 (smooth static
                // cell) would otherwise clamp every new frame onto the old mean forever (variance from the clamped x stays 0).
                const float firefly_ceiling = max(fast_mean_prev, luminance_floor) *
                    (1.0f + max(sqrt(fast_var_prev), RT_LAB_ANTI_LAG_FAST_MIN_RELATIVE_STD) * RT_LAB_ANTI_LAG_FAST_FIREFLY_STD);
                x = min(x, firefly_ceiling);
            }
            const float fast_mean = lerp(fast_mean_prev, x, fast_blend);
            const float variance_point = min(square((x - fast_mean_prev) / max(fast_mean_prev, luminance_floor)), 4.0f);
            fast = float4(fast_mean, lerp(fast_var_prev, variance_point, fast_blend), fast_frames, 0.0f);
            if (history.z > window)
            {
                float cap = history.z;
                if ((settings.coarse_anti_lag_enabled & RT_LAB_ANTI_LAG_MSE_OPTIMAL) != 0)
                {
                    // MSE optimal length: an EMA of length N carries noise var / N, the old samples a lag bias ~ the slow-fast
                    // difference d -> N* = k x var / d^2. var = fast relative variance (frame to frame noise of the blurred
                    // coarse mean); d^2 debiased by the fast mean's own noise (EMA: var x blend / (2 - blend)). Continuous in d:
                    // no threshold, rises again on its own as the slow mean catches up. Never below the fast window.
                    const float fast_var = fast.y;
                    const float difference = abs(history.x - fast_mean) / max(min(history.x, fast_mean), luminance_floor);
                    const float fast_mean_noise = fast_var * fast_blend / (2.0f - fast_blend);
                    const float excess = max(square(difference) - fast_mean_noise, 0.0f);
                    if (excess > 0.0f)
                    {
                        const float optimal = settings.coarse_anti_lag_variance_blend * fast_var / excess;
                        cap = min(history.z, max(optimal, window));
                    }
                }
                else
                {
                    const float adaptation_point = min(square(abs(fast_mean - x) / max(fast_mean, luminance_floor)), 4.0f);
                    const float adaptation_variance = lerp(fast_var_prev, adaptation_point, fast_blend);
                    const float variance_scaling = square(1.0f + sqrt(adaptation_variance) * settings.coarse_anti_lag_variance_blend);
                    const float slow_to_fast = max(history.x, fast_mean) / (min(history.x, fast_mean) + luminance_floor);
                    const float mean_diff_scaling = square(1.0f / max(1.0f, slow_to_fast));
                    cap = min(history.z, history.z * variance_scaling * mean_diff_scaling);
                }
                anti_lag_alpha = cap < history.z ? 1.0f / (cap + 1.0f) : 0.0f;
            }
        }
        if (history.z < 0.5f)
        {
            history.xz = float2(moments.x, 1.0f);
        }
        else
        {
            // The slow side: as long as the denoiser's own history (the pixel histories this cap acts on).
            const float max_frames = float(max(settings.temporal_progressive != 0 ? settings.progressive_max_samples : settings.max_history_frames, 1));
            const float frames = rt_lab_coarse_anti_lag_cap(history.z, anti_lag_alpha);
            const float alpha = max(1.0f / (frames + 1.0f), 1.0f / max_frames);
            history.x = lerp(history.x, moments.x, alpha);
            // Blend only: the cut length is used for this blend, the stored length keeps growing.
            history.z = min(((settings.coarse_anti_lag_enabled & RT_LAB_ANTI_LAG_MODE_MASK) == RT_LAB_ANTI_LAG_BLEND_ONLY ? history.z : frames) + 1.0f, max_frames);
        }
    }
    history.y = anti_lag_alpha;
    push.attach.fast_history.get()[dtid] = fast;
    history.w = center_depth;
    push.attach.history.get()[dtid] = history;

    const int debug_mode = push.attach.globals.settings.debug_draw_mode;
    if (rt_lab_anti_lag_debug_mode_active(debug_mode))
    {
        float3 color = float3(0.0f, 0.0f, 0.0f);
        if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG_FRAME_MEAN)
        {
            // This frame's blurred mean, exposed, x / (1 + x) gray (no clipping). Magenta: no ray in the footprint this frame.
            const float exposed = rt_lab_exposed(moments.x, push.attach.globals.exposure);
            color = sums.z > 1e-3f ? float3(1.0f, 1.0f, 1.0f) * (exposed / (1.0f + exposed)) : float3(1.0f, 0.0f, 1.0f) * 0.25f;
        }
        else if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG_RELATIVE_STD)
        {
            // Fast history relative std sqrt(fast relative variance) (the anti-lag's noise term), same log heatmap.
            color = fast.z > 0.0f ? Heatmap(saturate((log10(max(sqrt(max(fast.y, 0.0f)), 1e-6f)) + 2.0f) / 3.0f)) : float3(0.0f, 0.0f, 0.0f);
        }
        else
        {
            color = rt_lab_anti_lag_relative_std_color(moments);
        }
        rt_lab_anti_lag_write_debug(push, dtid, color);
    }
}
