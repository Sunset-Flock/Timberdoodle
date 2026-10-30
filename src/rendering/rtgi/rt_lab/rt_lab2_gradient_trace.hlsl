#pragma once

#include "rt_lab2_gradient.inl"

#include "rt_lab_trace.hlsl" // rt_lab_trace_radiance (hit shading of the lab rays) + RTGI_SHADE_CONTEXT

[[vk::push_constant]] RtLab2GradientTracePush rt_lab2_gradient_trace_push;

func rt_lab2_gradient_trace_capacity(RtLab2Settings s) -> uint
{
    const float log2_capacity = s.cell_capacity_log2 > 0.0f ? s.cell_capacity_log2 : float(RT_LAB2_CELL_CAPACITY_LOG2_DEFAULT);
    return 1u << uint(clamp(log2_capacity, 12.0f, 24.0f));
}

[shader("compute")]
[numthreads(RT_LAB2_CELL_PASS_X, 1, 1)]
func entry_rt_lab2_gradient_trace(uint dtid : SV_DispatchThreadID)
{
    let push = rt_lab2_gradient_trace_push;
    const RtLab2Settings s = *push.settings;
    if (dtid >= rt_lab2_gradient_trace_capacity(s))
    {
        return;
    }
    RtLab2Cell* cell = &push.attach.cells[dtid];
    if (cell.key == 0u || cell.rec_valid == 0u)
    {
        return;
    }
    const RtLabSettings settings = push.attach.globals.rt_lab_settings;
    let ctx = RTGI_SHADE_CONTEXT(push.attach);
    const float floor_luminance = 0.01f * RTGI_RADIANCE_SCALE / (PI * max(push.attach.globals.exposure, 1e-6f));
    float new_luminances[RT_LAB2_GRADIENT_RECORDS];
    float old_luminances[RT_LAB2_GRADIENT_RECORDS];
    uint count = 0u;
    float sum_new = 0.0f;
    float sum_old = 0.0f;
    const uint valid = min(cell.rec_valid, uint(RT_LAB2_GRADIENT_RECORDS));
    for (uint k = 0u; k < valid; ++k)
    {
        const RtLab2GradientRecord record = push.attach.records[dtid * uint(RT_LAB2_GRADIENT_RECORDS) + k];
        if ((record.data & RT_LAB_GRADIENT_RECORD_MAGIC_MASK) != RT_LAB_GRADIENT_RECORD_MAGIC || any(!isfinite(record.origin)) || !isfinite(record.old_luminance))
        {
            continue;
        }
        RayDesc ray = {};
        ray.Origin = record.origin;
        ray.Direction = uncompress_normal_octahedral_32(record.direction);
        ray.TMin = record.t_min;
        ray.TMax = RT_LAB_RAY_T_MAX;
        float hit_t;
        const float new_luminance = rt_lab_luminance(rt_lab_trace_radiance(ctx, settings, ray, record.data & RT_LAB_HIT_SEED_MASK, hit_t) * RTGI_RADIANCE_SCALE);
        if (!isfinite(new_luminance))
        {
            continue;
        }
        new_luminances[count] = new_luminance;
        old_luminances[count] = record.old_luminance;
        sum_new += new_luminance;
        sum_old += record.old_luminance;
        count += 1u;
    }
    if (count == 0u)
    {
        return;
    }
    // Ratio of the sums (dark rays regularized by the display floor), variance by the delta method (needs 2 records, else only the floor).
    const float regularized_new = sum_new + float(count) * floor_luminance;
    const float regularized_old = sum_old + float(count) * floor_luminance;
    const float ratio = regularized_new / regularized_old;
    float variance = 0.0f;
    if (count >= 2u)
    {
        for (uint k = 0u; k < count; ++k)
        {
            const float difference = (new_luminances[k] + floor_luminance) - ratio * (old_luminances[k] + floor_luminance);
            variance += difference * difference;
        }
        variance /= regularized_new * regularized_new;
    }
    cell.meas_z = clamp(log(ratio), -RT_LAB2_GRADIENT_MAX_LOG, RT_LAB2_GRADIENT_MAX_LOG);
    cell.meas_r = max(variance, max(s.gradient_min_variance, 1e-6f));
}
