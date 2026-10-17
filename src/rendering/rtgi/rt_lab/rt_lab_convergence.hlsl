#pragma once

#include "rt_lab_convergence.inl"

#include "rt_lab_shared.hlsl" // RTGI_RADIANCE_SCALE
#include "shader_lib/debug.glsl"

[[vk::push_constant]] RtLabConvergenceCapturePush rt_lab_convergence_capture_push;
[[vk::push_constant]] RtLabConvergenceAnalyzePush rt_lab_convergence_analyze_push;

func rt_lab_convergence_luminance(float3 v) -> float
{
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

// Stored split predictions (unexposed luminance) in exposed luminance; negative entries (no prediction) stay negative.
func rt_lab_convergence_load_prediction(float2 stored, float exposure_scale) -> float2
{
    return float2(stored.x >= 0.0f ? stored.x * exposure_scale : -1.0f, stored.y >= 0.0f ? stored.y * exposure_scale : -1.0f);
}

// Predicted error of custom - progressive: both runs' errors add in variance. Without a reference prediction (its
// split history off) only the custom one. -1 = no prediction.
func rt_lab_convergence_combined_prediction(float2 stored) -> float
{
    if (!(stored.x >= 0.0f) || !isfinite(stored.x))
    {
        return -1.0f;
    }
    if (!(stored.y >= 0.0f) || !isfinite(stored.y))
    {
        return stored.x;
    }
    return sqrt(stored.x * stored.x + stored.y * stored.y);
}

[shader("compute")]
[numthreads(RT_LAB_CONVERGENCE_X, RT_LAB_CONVERGENCE_Y, 1)]
func entry_rt_lab_convergence_capture(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_convergence_capture_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const float4 value = push.attach.resolved.get()[dtid];
    // Post-spatial split history error prediction of this run: |resolved split difference|, unexposed luminance (as the
    // captured diffuse: the analysis exposes both with its own frame's exposure).
    float predicted = -1.0f;
    if (push.split_enabled != 0u)
    {
        const float2 split_difference = push.attach.split_difference.get()[dtid];
        if (split_difference.y >= 0.999f)
        {
            predicted = abs(split_difference.x) / RTGI_RADIANCE_SCALE;
        }
    }
    float2 stored_prediction = push.attach.predicted_error.get()[dtid];
    if (push.target == RT_LAB_CAPTURE_PROGRESSIVE)
    {
        push.attach.progressive.get()[dtid] = value;
        stored_prediction.y = predicted;
    }
    else
    {
        push.attach.custom.get()[dtid] = value;
        stored_prediction.x = predicted;
    }
    push.attach.predicted_error.get()[dtid] = stored_prediction;
}

groupshared uint gs_histogram[RT_LAB_CONVERGENCE_HISTOGRAM_BINS];

// Prediction ratio view: per pixel, predicted and actual error are each ONE random draw of |N(0, sigma)|, so their ratio
// is noise even for a perfect prediction. Pooled over a (2r+1)^2 window: sqrt(sum p^2 / sum a^2).
#define RT_LAB_CONVERGENCE_RATIO_POOL_RADIUS 4

func rt_lab_convergence_pooled_prediction_stops(int2 center) -> float
{
    let push = rt_lab_convergence_analyze_push;
    const float scale = PI * push.attach.globals.exposure;
    float sum_predicted_squared = 0.0f;
    float sum_actual_squared = 0.0f;
    for (int y = -RT_LAB_CONVERGENCE_RATIO_POOL_RADIUS; y <= RT_LAB_CONVERGENCE_RATIO_POOL_RADIUS; ++y)
    {
        for (int x = -RT_LAB_CONVERGENCE_RATIO_POOL_RADIUS; x <= RT_LAB_CONVERGENCE_RATIO_POOL_RADIUS; ++x)
        {
            const int2 tap = center + int2(x, y);
            if (any(tap < 0) || any(tap >= int2(push.size)))
            {
                continue;
            }
            const float predicted = rt_lab_convergence_combined_prediction(rt_lab_convergence_load_prediction(push.attach.predicted_error.get()[tap], scale));
            if (push.attach.view_cam_depth.get()[tap] == 0.0f || !(predicted >= 0.0f))
            {
                continue;
            }
            const float3 difference = (push.attach.custom.get()[tap].rgb - push.attach.progressive.get()[tap].rgb) * scale;
            const float actual = min(abs(rt_lab_convergence_luminance(difference)), RT_LAB_CONVERGENCE_PREDICTION_CLAMP);
            sum_predicted_squared += square(min(predicted, RT_LAB_CONVERGENCE_PREDICTION_CLAMP));
            sum_actual_squared += square(actual);
        }
    }
    return 0.5f * log2(max(sum_predicted_squared, 1e-12f) / max(sum_actual_squared, 1e-12f));
}

// Units: exposed irradiance, the same as DEBUG_DRAW_MODE_INDIRECT_DIFFUSE (resolved radiance * pi * exposure).
// Difference = custom - progressive. Squared error = mean over rgb of the squared difference. The histogram bins the
// signed luminance difference over [-range, range]; outliers are counted separately (below / above range), NOT in the
// end bins. Sky pixels are skipped.
[shader("compute")]
[numthreads(RT_LAB_CONVERGENCE_X, RT_LAB_CONVERGENCE_Y, 1)]
func entry_rt_lab_convergence_analyze(uint2 dtid : SV_DispatchThreadID, uint group_index : SV_GroupIndex)
{
    let push = rt_lab_convergence_analyze_push;
    if (group_index < RT_LAB_CONVERGENCE_HISTOGRAM_BINS)
    {
        gs_histogram[group_index] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    // No early outs: every thread has to reach the barrier and the wave reductions.
    const bool in_bounds = all(dtid < push.size);
    const bool valid = in_bounds && push.attach.view_cam_depth.get()[dtid] != 0.0f;
    float squared_error = 0.0f;
    float signed_error = 0.0f;
    float3 progressive = float3(0.0f, 0.0f, 0.0f);
    float3 custom = float3(0.0f, 0.0f, 0.0f);
    bool below_range = false;
    bool above_range = false;
    // Split history error prediction check (-1 = no prediction for this pixel).
    float predicted = -1.0f;
    float actual = 0.0f;
    if (valid)
    {
        const float scale = PI * push.attach.globals.exposure;
        progressive = push.attach.progressive.get()[dtid].rgb * scale;
        custom = push.attach.custom.get()[dtid].rgb * scale;
        const float3 difference = custom - progressive;
        squared_error = min(dot(difference * difference, float3(1.0f, 1.0f, 1.0f) / 3.0f), 1000.0f);
        signed_error = clamp(dot(difference, float3(0.2126f, 0.7152f, 0.0722f)), -1000.0f, 1000.0f);
        if (!isfinite(squared_error) || !isfinite(signed_error))
        {
            squared_error = 0.0f;
            signed_error = 0.0f;
        }
        const float range = max(push.histogram_range, 1e-6f);
        const float normalized = (signed_error / range) * 0.5f + 0.5f;
        below_range = normalized < 0.0f;
        above_range = normalized >= 1.0f;
        if (!below_range && !above_range)
        {
            const uint bin = min(uint(normalized * float(RT_LAB_CONVERGENCE_HISTOGRAM_BINS)), RT_LAB_CONVERGENCE_HISTOGRAM_BINS - 1u);
            InterlockedAdd(gs_histogram[bin], 1u);
        }
        const float stored_prediction = rt_lab_convergence_combined_prediction(rt_lab_convergence_load_prediction(push.attach.predicted_error.get()[dtid], scale));
        if (stored_prediction >= 0.0f)
        {
            predicted = min(stored_prediction, RT_LAB_CONVERGENCE_PREDICTION_CLAMP);
            actual = min(abs(signed_error), RT_LAB_CONVERGENCE_PREDICTION_CLAMP);
        }
    }

    // Debug views.
    const int debug_mode = push.attach.globals.settings.debug_draw_mode;
    if (in_bounds && debug_mode >= DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_PROGRESSIVE && debug_mode <= DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_PREDICTION_RATIO)
    {
        const float debug_scale = push.attach.globals.settings.debug_visualization_scale;
        float3 color = float3(0.0f, 0.0f, 0.0f);
        if (valid)
        {
            switch (debug_mode)
            {
                case DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_PROGRESSIVE:
                {
                    color = progressive;
                    break;
                }
                case DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_CUSTOM:
                {
                    color = custom;
                    break;
                }
                case DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_DIFFERENCE:
                {
                    // Red: custom brighter than progressive, blue: darker.
                    const float v = signed_error * debug_scale;
                    color = v > 0.0f ? float3(v, 0.0f, 0.0f) : float3(0.0f, 0.0f, -v);
                    break;
                }
                case DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_DIFFERENCE_SQUARED:
                {
                    color = Heatmap(squared_error * debug_scale);
                    break;
                }
                case DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_RELATIVE_DIFFERENCE:
                {
                    // |C - P| / min(C, P), same log heatmap as RT_LAB_SPLIT_RELATIVE_ERROR.
                    color = rt_lab_relative_difference_heatmap(rt_lab_relative_difference(
                        rt_lab_convergence_luminance(custom), rt_lab_convergence_luminance(progressive)));
                    break;
                }
                case DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_PREDICTED_ERROR:
                {
                    // Custom run only. Magenta: no prediction (split history off or a half had no samples).
                    const float custom_predicted = rt_lab_convergence_load_prediction(push.attach.predicted_error.get()[dtid], PI * push.attach.globals.exposure).x;
                    color = !(custom_predicted >= 0.0f) ? float3(1.0f, 0.0f, 1.0f) : Heatmap(custom_predicted * debug_scale);
                    break;
                }
                default:
                {
                    // Pooled log2(RMS predicted / RMS actual): red = overestimate, blue = underestimate, saturated at
                    // +-3 stops, black = exact.
                    if (predicted < 0.0f)
                    {
                        color = float3(1.0f, 0.0f, 1.0f);
                    }
                    else
                    {
                        const float stops = clamp(rt_lab_convergence_pooled_prediction_stops(int2(dtid)), -3.0f, 3.0f) / 3.0f;
                        color = stops > 0.0f ? float3(stops, 0.0f, 0.0f) : float3(0.0f, 0.0f, -stops);
                    }
                    break;
                }
            }
        }
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid, float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend));
    }

    // Error sums: one 64 bit atomic per wave and sum.
    const float wave_squared_error = WaveActiveSum(squared_error);
    const float wave_positive_error = WaveActiveSum(max(signed_error, 0.0f));
    const float wave_negative_error = WaveActiveSum(max(-signed_error, 0.0f));
    const uint wave_pixels = WaveActiveCountBits(valid);
    const uint wave_below_range = WaveActiveCountBits(below_range);
    const uint wave_above_range = WaveActiveCountBits(above_range);
    const bool has_prediction = predicted >= 0.0f;
    const float p = has_prediction ? predicted : 0.0f;
    const float a = has_prediction ? actual : 0.0f;
    const float wave_predicted = WaveActiveSum(p);
    const float wave_predicted_squared = WaveActiveSum(p * p);
    const float wave_actual = WaveActiveSum(a);
    const float wave_actual_squared = WaveActiveSum(a * a);
    const float wave_predicted_actual = WaveActiveSum(p * a);
    const uint wave_predicted_pixels = WaveActiveCountBits(has_prediction);
    if (all(dtid == uint2(0, 0)))
    {
        push.attach.globals.readback.rt_lab_convergence_histogram_range = push.histogram_range;
        push.attach.globals.readback.rt_lab_convergence_step = push.step;
    }
    if (WaveIsFirstLane() && wave_pixels > 0u)
    {
        ReadbackValues* readback = push.attach.globals.readback;
        InterlockedAdd(readback.rt_lab_convergence_below_range, wave_below_range);
        InterlockedAdd(readback.rt_lab_convergence_above_range, wave_above_range);
        AtomicAddU64(readback.rt_lab_convergence_squared_error_sum, uint64_t(wave_squared_error * RT_LAB_CONVERGENCE_FIXED_SCALE));
        AtomicAddU64(readback.rt_lab_convergence_positive_error_sum, uint64_t(wave_positive_error * RT_LAB_CONVERGENCE_FIXED_SCALE));
        AtomicAddU64(readback.rt_lab_convergence_negative_error_sum, uint64_t(wave_negative_error * RT_LAB_CONVERGENCE_FIXED_SCALE));
        InterlockedAdd(readback.rt_lab_convergence_pixels, wave_pixels);
        if (wave_predicted_pixels > 0u)
        {
            AtomicAddU64(readback.rt_lab_convergence_predicted_sum, uint64_t(wave_predicted * RT_LAB_CONVERGENCE_FIXED_SCALE));
            AtomicAddU64(readback.rt_lab_convergence_predicted_squared_sum, uint64_t(wave_predicted_squared * RT_LAB_CONVERGENCE_FIXED_SCALE));
            AtomicAddU64(readback.rt_lab_convergence_actual_sum, uint64_t(wave_actual * RT_LAB_CONVERGENCE_FIXED_SCALE));
            AtomicAddU64(readback.rt_lab_convergence_actual_squared_sum, uint64_t(wave_actual_squared * RT_LAB_CONVERGENCE_FIXED_SCALE));
            AtomicAddU64(readback.rt_lab_convergence_predicted_actual_sum, uint64_t(wave_predicted_actual * RT_LAB_CONVERGENCE_FIXED_SCALE));
            InterlockedAdd(readback.rt_lab_convergence_predicted_pixels, wave_predicted_pixels);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (group_index < RT_LAB_CONVERGENCE_HISTOGRAM_BINS && gs_histogram[group_index] > 0u)
    {
        InterlockedAdd(push.attach.globals.readback.rt_lab_convergence_histogram[group_index], gs_histogram[group_index]);
    }
}
