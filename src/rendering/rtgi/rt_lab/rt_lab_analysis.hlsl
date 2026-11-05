#pragma once

#include "rt_lab_analysis.inl"

[[vk::push_constant]] RtLabStatsPush rt_lab_stats_push;
[[vk::push_constant]] RtLabPickerPush rt_lab_picker_push;

#define RT_LAB_STATS_THREADS (RT_LAB_STATS_GROUP * RT_LAB_STATS_GROUP)

groupshared uint gs_stats_histograms[2 * RT_LAB_STATS_HIST_WORDS];
groupshared float gs_stats_reduce[RT_LAB_STATS_PARTIAL_WORDS][RT_LAB_STATS_THREADS];

// Contribution of one value to the moment words of its metric (count, sum, sum of squares, min, max, positive count, sum of log10, sum of squared log10).
func rt_lab_stats_set(uint thread, uint metric, bool valid, float value)
{
    const uint base = metric * RT_LAB_STATS_METRIC_WORDS;
    const bool positive = valid && value > 1e-12f;
    const float log_value = positive ? log10(value) : 0.0f;
    gs_stats_reduce[base + 0u][thread] = valid ? 1.0f : 0.0f;
    gs_stats_reduce[base + 1u][thread] = valid ? value : 0.0f;
    gs_stats_reduce[base + 2u][thread] = valid ? value * value : 0.0f;
    gs_stats_reduce[base + 3u][thread] = valid ? value : 3.4e38f;
    gs_stats_reduce[base + 4u][thread] = valid ? value : 0.0f;
    gs_stats_reduce[base + 5u][thread] = positive ? 1.0f : 0.0f;
    gs_stats_reduce[base + 6u][thread] = log_value;
    gs_stats_reduce[base + 7u][thread] = log_value * log_value;
}

func rt_lab_analysis_luma(float3 v) -> float
{
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

// Bin (0..BINS-1), BINS = under, BINS + 1 = over of a value in a log10 range.
func rt_lab_stats_bin(float value, float log_min, float log_max) -> uint
{
    if (!(value > 0.0f))
    {
        return RT_LAB_STATS_BINS;
    }
    if (!isfinite(value))
    {
        return RT_LAB_STATS_BINS + 1u;
    }
    const float t = (log10(value) - log_min) / max(log_max - log_min, 1e-4f);
    if (t < 0.0f)
    {
        return RT_LAB_STATS_BINS;
    }
    if (t >= 1.0f)
    {
        return RT_LAB_STATS_BINS + 1u;
    }
    return min(uint(t * float(RT_LAB_STATS_BINS)), RT_LAB_STATS_BINS - 1u);
}

[shader("compute")]
[numthreads(RT_LAB_STATS_GROUP, RT_LAB_STATS_GROUP, 1)]
func entry_rt_lab_stats(uint2 dtid : SV_DispatchThreadID, uint gi : SV_GroupIndex, uint2 gid : SV_GroupID)
{
    let push = rt_lab_stats_push;
    for (uint i = gi; i < 2u * RT_LAB_STATS_HIST_WORDS; i += RT_LAB_STATS_THREADS)
    {
        gs_stats_histograms[i] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    const int2 size = int2(push.attach.globals.settings.render_target_size);
    const int2 pixel = int2(dtid);
    bool luma_valid = false;
    bool variance_valid = false;
    float luma_value = 0.0f;
    float variance_value = 0.0f;
    if (all(pixel < size) && push.attach.view_cam_depth.get()[pixel] != 0.0f)
    {
        const float center = rt_lab_analysis_luma(push.attach.resolved.get()[pixel].rgb);
        InterlockedAdd(gs_stats_histograms[rt_lab_stats_bin(center, push.luma_log_min, push.luma_log_max)], 1u);
        luma_valid = isfinite(center);
        luma_value = center;

        // 5x5 spatial variance of the luminance over the geometry pixels of the window (unbiased).
        float sum = 0.0f;
        float sum_squares = 0.0f;
        float count = 0.0f;
        for (int y = -RT_LAB_STATS_VARIANCE_RADIUS; y <= RT_LAB_STATS_VARIANCE_RADIUS; ++y)
        {
            for (int x = -RT_LAB_STATS_VARIANCE_RADIUS; x <= RT_LAB_STATS_VARIANCE_RADIUS; ++x)
            {
                const int2 tap = pixel + int2(x, y);
                if (any(tap < int2(0, 0)) || any(tap >= size) || push.attach.view_cam_depth.get()[tap] == 0.0f)
                {
                    continue;
                }
                const float luma = rt_lab_analysis_luma(push.attach.resolved.get()[tap].rgb);
                if (!isfinite(luma))
                {
                    continue;
                }
                sum += luma;
                sum_squares += luma * luma;
                count += 1.0f;
            }
        }
        if (count >= 2.0f)
        {
            const float mean = sum / count;
            const float variance = max(sum_squares - count * mean * mean, 0.0f) / (count - 1.0f);
            InterlockedAdd(gs_stats_histograms[RT_LAB_STATS_HIST_WORDS + rt_lab_stats_bin(variance, push.variance_log_min, push.variance_log_max)], 1u);
            variance_valid = isfinite(variance);
            variance_value = variance;
        }
    }
    rt_lab_stats_set(gi, 0u, luma_valid, luma_value);
    rt_lab_stats_set(gi, 1u, variance_valid, variance_value);
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction of the moment words (sums add, min / max combine).
    for (uint stride = RT_LAB_STATS_THREADS / 2u; stride > 0u; stride /= 2u)
    {
        if (gi < stride)
        {
            for (uint word = 0u; word < RT_LAB_STATS_PARTIAL_WORDS; ++word)
            {
                const uint kind = word % RT_LAB_STATS_METRIC_WORDS;
                const float other = gs_stats_reduce[word][gi + stride];
                float value = gs_stats_reduce[word][gi];
                if (kind == 3u)
                {
                    value = min(value, other);
                }
                else if (kind == 4u)
                {
                    value = max(value, other);
                }
                else
                {
                    value += other;
                }
                gs_stats_reduce[word][gi] = value;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const uint group_index = gid.y * push.groups_x + gid.x;
    if (gi < RT_LAB_STATS_PARTIAL_WORDS && group_index < RT_LAB_STATS_MAX_GROUPS)
    {
        push.attach.stats[push.slot * RT_LAB_STATS_SLOT_WORDS + RT_LAB_STATS_PARTIALS_OFFSET + group_index * RT_LAB_STATS_PARTIAL_WORDS + gi] = asuint(gs_stats_reduce[gi][0]);
    }

    for (uint i = gi; i < 2u * RT_LAB_STATS_HIST_WORDS; i += RT_LAB_STATS_THREADS)
    {
        const uint value = gs_stats_histograms[i];
        if (value != 0u)
        {
            InterlockedAdd(push.attach.stats[push.slot * RT_LAB_STATS_SLOT_WORDS + RT_LAB_STATS_HEADER_WORDS + i], value);
        }
    }
    if (all(gid == uint2(0, 0)) && gi == 0u)
    {
        push.attach.stats[push.slot * RT_LAB_STATS_SLOT_WORDS] = push.stamp;
    }
}

func rt_lab_picker_store(uint index, float value)
{
    rt_lab_picker_push.attach.picker[index] = asuint(value);
}

[shader("compute")]
[numthreads(RT_LAB_PICKER_COUNT, 1, 1)]
func entry_rt_lab_picker(uint picker_index : SV_GroupIndex)
{
    let push = rt_lab_picker_push;
    const uint slot_base = push.slot * RT_LAB_PICKER_SLOT_WORDS;
    if (picker_index == 0u)
    {
        push.attach.picker[slot_base + 0u] = push.stamp;
        push.attach.picker[slot_base + 1u] = push.active_mask;
    }
    if (((push.active_mask >> picker_index) & 1u) == 0u)
    {
        return;
    }
    const uint base = slot_base + RT_LAB_PICKER_HEADER_WORDS + picker_index * RT_LAB_PICKER_WORDS;
    const int2 size = int2(push.attach.globals.settings.render_target_size);
    const int2 pixel = int2(push.screen_pixel[picker_index]);
    const bool geometry = push.attach.view_cam_depth.get()[pixel] != 0.0f;
    const float3 color = push.attach.resolved.get()[pixel].rgb;

    float sum = 0.0f;
    float sum_squares = 0.0f;
    float count = 0.0f;
    float luma_min = 1e30f;
    float luma_max = 0.0f;
    for (int y = -RT_LAB_STATS_VARIANCE_RADIUS; y <= RT_LAB_STATS_VARIANCE_RADIUS; ++y)
    {
        for (int x = -RT_LAB_STATS_VARIANCE_RADIUS; x <= RT_LAB_STATS_VARIANCE_RADIUS; ++x)
        {
            const int2 tap = pixel + int2(x, y);
            if (any(tap < int2(0, 0)) || any(tap >= size) || push.attach.view_cam_depth.get()[tap] == 0.0f)
            {
                continue;
            }
            const float luma = rt_lab_analysis_luma(push.attach.resolved.get()[tap].rgb);
            if (!isfinite(luma))
            {
                continue;
            }
            sum += luma;
            sum_squares += luma * luma;
            count += 1.0f;
            luma_min = min(luma_min, luma);
            luma_max = max(luma_max, luma);
        }
    }
    const float mean = count > 0.0f ? sum / count : 0.0f;
    const float variance = count >= 2.0f ? max(sum_squares - count * mean * mean, 0.0f) / (count - 1.0f) : 0.0f;

    push.attach.picker[base + 0u] = geometry ? 1u : 0u;
    push.attach.picker[base + 1u] = uint(pixel.x);
    push.attach.picker[base + 2u] = uint(pixel.y);
    rt_lab_picker_store(base + 3u, color.x);
    rt_lab_picker_store(base + 4u, color.y);
    rt_lab_picker_store(base + 5u, color.z);
    rt_lab_picker_store(base + 6u, mean);
    rt_lab_picker_store(base + 7u, variance);
    rt_lab_picker_store(base + 8u, count);
    rt_lab_picker_store(base + 9u, push.attach.globals.exposure);
    rt_lab_picker_store(base + 10u, count > 0.0f ? luma_min : 0.0f);
    rt_lab_picker_store(base + 11u, luma_max);
}
