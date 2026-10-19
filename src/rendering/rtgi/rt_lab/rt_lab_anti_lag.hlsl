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

// == Kalman anti-lag (bit RT_LAB_ANTI_LAG_KALMAN in coarse_anti_lag_enabled, rt_lab_settings.inl, with a mode) ==
// Replaces fast history / MSE / RTGI heuristic / TAA clip. Per cell Kalman filter of the LOG coarse mean m (fast history
// .x m, .y P, .z last outlier sign, .w initialized): predict with the temporal gradient's correction (m += ln ratio,
// P += its loss L) + RT_LAB_ANTI_LAG_KALMAN_Q; measure z = ln(blurred frame mean), R = ray variance x sum w^2 n /
// (sum w n)^2 / mean^2; robust gate (heavy tails at 1 rpp): the first frame outside RT_LAB_ANTI_LAG_KALMAN_GATE sigmas
// only moves by the gate (firefly?), a second one in the same direction resets (real change). The pixel histories' mean
// (slow history) vs m: the bias is corrected (James-Stein shrink at 2 sigma, ratio channel like the clip), only the
// uncorrectable rest (ln 20 clamp) + the correction's own uncertainty cut the length N / (1 + N k L / v).
#define RT_LAB_ANTI_LAG_KALMAN_Q 1e-4f
#define RT_LAB_ANTI_LAG_KALMAN_GATE 3.0f
#define RT_LAB_ANTI_LAG_KALMAN_K 20.0f
#define RT_LAB_ANTI_LAG_KALMAN_SHRINK_SIGMAS 2.0f

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
        float4 history = push.attach.history_previous.get()[cell];
        history.z = rt_lab_anti_lag_unpack_frames(history.z); // frames (+ clip ratio, RT_LAB_ANTI_LAG_CLIP)
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
    float sum_weight_sq_rays = 0.0f; // sum w^2 x rays (separable: the vertical pass weights the horizontal pass's)
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
        const float4 tap_sums = push.attach.blur_src.get()[cell];
        sums += tap_sums.xyz * weight;
        sum_weight_sq_rays += weight * weight * (vertical ? tap_sums.w : tap_sums.z);
    }
    if (!vertical)
    {
        push.attach.blur_dst.get()[dtid] = float4(sums, sum_weight_sq_rays);
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
        // The temporal gradient's history rescale (the pixel histories get the same factor in the temporal pass).
        if (push.gradient_bound != 0u && settings.temporal_gradient_rescale != 0 && !rt_lab_gradient_packed(settings) && all(dtid < push.size))
        {
            uint2 gradient_cells;
            push.attach.temporal_gradient.get().GetDimensions(gradient_cells.x, gradient_cells.y);
            const uint4 gradient = push.attach.temporal_gradient.get()[min(dtid, gradient_cells - 1u)];
            if (asfloat(gradient.y) > 0.0f)
            {
                const float ratio = rt_lab_gradient_rescale(settings, asfloat(gradient.w));
                history.x *= ratio;
                if ((uint(settings.coarse_anti_lag_enabled) & RT_LAB_ANTI_LAG_KALMAN) != 0)
                {
                    // Kalman predict: the measured change + its uncertainty (the Kalman gradient's loss L).
                    if (fast.w > 0.5f)
                    {
                        fast.x += log(ratio);
                        fast.y += (settings.temporal_gradient_cap_mode & RT_LAB_GRADIENT_CAP_KALMAN) != 0 ? max(asfloat(gradient.x), 0.0f) : 0.0f;
                    }
                }
                else
                {
                    fast.x *= ratio;
                }
            }
        }
    }
    const float2 moments = sums.z > 1e-3f ? sums.xy / sums.z : float2(0.0f, 0.0f); // this frame's blurred (mean, second moment)
    const float frame_variance = max(moments.y - moments.x * moments.x, 0.0f);
    float anti_lag_alpha = 0.0f;
    float clip_ratio = 1.0f; // RT_LAB_ANTI_LAG_CLIP: clipped / slow, applied to the pixel histories by the temporal pass
    const bool kalman = (uint(settings.coarse_anti_lag_enabled) & RT_LAB_ANTI_LAG_KALMAN) != 0;
    if (kalman && sums.z > 1e-3f)
    {
        const float luminance_floor = RT_LAB_RELATIVE_DIFFERENCE_FLOOR * RTGI_RADIANCE_SCALE / (PI * push.attach.globals.exposure);
        const float mean = moments.x;
        const float z = log(mean + luminance_floor);
        const float per_ray_variance = frame_variance / square(mean + luminance_floor); // relative (log) per ray
        const float r = max(per_ray_variance * sum_weight_sq_rays / square(sums.z), 1e-6f);
        float m = z;
        float p = r;
        float outlier = 0.0f;
        if (fast.w > 0.5f && settings.reset_history == 0)
        {
            const float p_predicted = fast.y + RT_LAB_ANTI_LAG_KALMAN_Q;
            const float innovation = z - fast.x;
            const float innovation_variance = p_predicted + r;
            const float limit = RT_LAB_ANTI_LAG_KALMAN_GATE * sqrt(innovation_variance);
            const float gain = p_predicted / innovation_variance;
            if (abs(innovation) > limit && fast.z * sign(innovation) > 0.5f)
            {
                m = z; // second frame outside the gate in the same direction: a real change, reset
                p = r;
            }
            else
            {
                // Inside the gate: the usual update. First frame outside: moves only by the gate (could be a firefly).
                outlier = abs(innovation) > limit ? sign(innovation) : 0.0f;
                m = fast.x + gain * clamp(innovation, -limit, limit);
                p = (1.0f - gain) * p_predicted;
            }
        }
        fast = float4(m, p, outlier, 1.0f);
        if (history.z < 0.5f)
        {
            history.xz = float2(mean, 1.0f);
        }
        else
        {
            const float max_frames = float(max(settings.temporal_progressive != 0 ? settings.progressive_max_samples : settings.max_history_frames, 1));
            const float frames_now = history.z;
            // The pixel histories hold the slow mean: its bias vs m, noise of the difference = P + the slow EMA's own noise.
            const float bias = log(history.x + luminance_floor) - m;
            const float bias_variance = p + r / max(2.0f * frames_now - 1.0f, 1.0f);
            const float shrink_sigmas_sq = square(RT_LAB_ANTI_LAG_KALMAN_SHRINK_SIGMAS);
            const float shrink = max(0.0f, 1.0f - shrink_sigmas_sq * bias_variance / max(square(bias), 1e-12f)); // James-Stein
            const float correction = clamp(-shrink * bias, -RT_LAB_GRADIENT_KALMAN_MAX_LOG, RT_LAB_GRADIENT_KALMAN_MAX_LOG);
            clip_ratio = exp(correction);
            history.x *= clip_ratio;
            // Information loss: the uncorrectable rest beyond its noise + the correction's own uncertainty.
            const float residual = bias + correction;
            const float loss = max(square(residual) - shrink_sigmas_sq * bias_variance, 0.0f) + square(shrink) * p;
            const float v = max(per_ray_variance, 1e-4f);
            const float cap = frames_now / (1.0f + frames_now * RT_LAB_ANTI_LAG_KALMAN_K * loss / v);
            anti_lag_alpha = cap < frames_now - 0.5f ? 1.0f / (max(cap, 1.0f) + 1.0f) : 0.0f;
            const float frames = rt_lab_coarse_anti_lag_cap(frames_now, anti_lag_alpha);
            const float alpha = max(1.0f / (frames + 1.0f), 1.0f / max_frames);
            history.x = lerp(history.x, mean, alpha);
            history.z = min(((settings.coarse_anti_lag_enabled & RT_LAB_ANTI_LAG_MODE_MASK) == RT_LAB_ANTI_LAG_BLEND_ONLY ? frames_now : frames) + 1.0f, max_frames);
        }
    }
    else if (!kalman && sums.z > 1e-3f)
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
            const uint noise_mode = uint(settings.coarse_anti_lag_enabled) & (RT_LAB_ANTI_LAG_NOISE_MAD | RT_LAB_ANTI_LAG_NOISE_SPATIAL);
            // Relative variance of the fast history's noise estimate (.y holds a relative MAD in NOISE_MAD mode).
            const float fast_var_prev = noise_mode == RT_LAB_ANTI_LAG_NOISE_MAD ? square(fast.y) * (0.5f * PI) : fast.y;
            if (fast_frames > window)
            {
                // Floors on both terms: a fast mean of ~0 (dark before the change) or a fast variance of ~0 (smooth static
                // cell) would otherwise clamp every new frame onto the old mean forever (variance from the clamped x stays 0).
                const float firefly_ceiling = max(fast_mean_prev, luminance_floor) *
                    (1.0f + max(sqrt(fast_var_prev), RT_LAB_ANTI_LAG_FAST_MIN_RELATIVE_STD) * RT_LAB_ANTI_LAG_FAST_FIREFLY_STD);
                x = min(x, firefly_ceiling);
            }
            const float fast_mean = lerp(fast_mean_prev, x, fast_blend);
            const float relative_deviation = abs(x - fast_mean_prev) / max(fast_mean_prev, luminance_floor);
            float noise_point = min(square(relative_deviation), 4.0f);
            if (noise_mode == RT_LAB_ANTI_LAG_NOISE_MAD)
            {
                noise_point = min(relative_deviation, 2.0f);
            }
            else if (noise_mode == RT_LAB_ANTI_LAG_NOISE_SPATIAL)
            {
                // Predicted frame to frame relative variance of the blurred mean: ray variance x sum w^2 n / (sum w n)^2 / mean^2.
                noise_point = min(frame_variance * sum_weight_sq_rays / square(sums.z) / square(max(moments.x, luminance_floor)), 4.0f);
            }
            fast = float4(fast_mean, lerp(fast.y, noise_point, fast_blend), fast_frames, 0.0f);
            const float fast_var_now = noise_mode == RT_LAB_ANTI_LAG_NOISE_MAD ? square(fast.y) * (0.5f * PI) : fast.y;
#if RT_LAB_ANTI_LAG_CLIP
            if (history.z > window && history.x > 0.0f)
            {
                // TAA style: the slow mean is clipped into the fast mean's noise band; the pixel histories follow (ratio).
                const float fast_mean_std = max(sqrt(max(fast_var_now, 0.0f) * fast_blend / (2.0f - fast_blend)), RT_LAB_ANTI_LAG_CLIP_MIN_RELATIVE_STD);
                const float band = RT_LAB_ANTI_LAG_CLIP_SIGMAS * fast_mean_std;
                const float clipped = clamp(history.x, fast_mean * max(1.0f - band, 0.0f), fast_mean * (1.0f + band));
                clip_ratio = clamp(clipped / history.x, 0.01f, 100.0f);
                history.x = clipped;
            }
#endif
            if (history.z > window)
            {
                float cap = history.z;
                if ((settings.coarse_anti_lag_enabled & RT_LAB_ANTI_LAG_MSE_OPTIMAL) != 0)
                {
                    // MSE optimal length: an EMA of length N carries noise var / N, the old samples a lag bias ~ the slow-fast
                    // difference d -> N* = k x var / d^2. var = fast relative variance (frame to frame noise of the blurred
                    // coarse mean); d^2 debiased by the fast mean's own noise (EMA: var x blend / (2 - blend)). Continuous in d:
                    // no threshold, rises again on its own as the slow mean catches up. Never below the fast window.
                    const float fast_var = fast_var_now;
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
    history.z = rt_lab_anti_lag_pack_frames(history.z, clip_ratio);
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
            const float fast_relative_std = (settings.coarse_anti_lag_enabled & RT_LAB_ANTI_LAG_NOISE_MAD) != 0 ? fast.y * sqrt(0.5f * PI) : sqrt(max(fast.y, 0.0f));
            color = fast.z > 0.0f ? Heatmap(saturate((log10(max(fast_relative_std, 1e-6f)) + 2.0f) / 3.0f)) : float3(0.0f, 0.0f, 0.0f);
        }
        else
        {
            color = rt_lab_anti_lag_relative_std_color(moments);
        }
        rt_lab_anti_lag_write_debug(push, dtid, color);
    }
}
