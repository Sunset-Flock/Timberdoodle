#pragma once

#include "rt_lab3.hlsl" // reservoir helpers, the temporal reuse, the ReSTIR push constants (same head and push as the ReSTIR temporal pass)

// == ReTIR temporal pass ================================================================================================
// The reservoir is processed like the ReSTIR temporal pass: reprojected history merge, then the rays of the pixel as RIS candidates. The final reservoir
// (history of the next frame) is the same, RIS does not depend on the order. The difference: the rays are added ONE AT A TIME and the estimate of the
// running reservoir after each ray is kept. out_samples = mean of these estimates (.rgb) and their count (.a): the denoiser gets all of them.

[shader("compute")]
[numthreads(RT_LAB3_PASS_X, RT_LAB3_PASS_Y, 1)]
func entry_rt_lab_retir_temporal(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab3_temporal_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const RtLab3Settings s = *push.settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    const float depth = push.attach.lab_depth.get()[dtid];
    if (depth == 0.0f)
    {
        rt_lab3_store(push.attach.out_a.get(), push.attach.out_b.get(), push.attach.out_c.get(), dtid, rt_lab3_empty_reservoir());
        push.attach.out_samples.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    const float2 size = float2(push.size);
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, rcp(size)), depth);
    const float3 normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
    const float pixel_width_ws = rt_lab_pixel_width_ws(camera, size.y, depth);
    const float rcp_pixel_width_ws = rcp(pixel_width_ws);
    rand_seed(rt_lab3_seed(push.attach.globals, dtid, 1u));

    const uint2 alloc = push.attach.ray_alloc.get()[dtid];
    RtLab3Merge merge;
    merge.selected = rt_lab3_empty_reservoir();
    merge.selected_p_hat = 0.0f;
    merge.weight_sum = 0.0f;
    merge.count_sum = 0.0f;
    merge.ray_history = 0.0f;
    const float ray_history_cap = max(push.attach.globals.rt_lab_settings.fast_convergence_samples, 1.0f);
    rt_lab3_temporal_reuse(push, s, position, normal, pixel_width_ws, rcp_pixel_width_ws, ray_history_cap, merge);

    RtLab3Reservoir selected = merge.selected;
    float selected_p_hat = merge.selected_p_hat;
    float weight_sum = merge.weight_sum;
    float count_sum = merge.count_sum;

    float3 estimate_sum = float3(0.0f, 0.0f, 0.0f);
    for (uint k = 0u; k < alloc.y; ++k)
    {
        const RtLab3Ray ray = push.attach.rays[alloc.x + k];
        count_sum += 1.0f;
        const float luminance = rt_lab_luminance(ray.radiance);
        const float candidate_weight = PI * luminance; // p_hat / p_source = luminance cos / (cos / pi)
        if (all(isfinite(ray.radiance)) && all(isfinite(ray.position)) && candidate_weight > 0.0f)
        {
            weight_sum += candidate_weight;
            if (rand() * weight_sum < candidate_weight)
            {
                selected.position = ray.position;
                selected.normal = ray.hit_normal;
                selected.age = 0u;
                selected.radiance = ray.radiance;
                selected_p_hat = luminance * max(dot(normal, uncompress_normal_octahedral_32(ray.direction)), 0.0f);
            }
        }
        // The estimate of the reservoir after this ray (same formula as the shade pass).
        if (selected_p_hat > 0.0f)
        {
            const float contribution_weight = weight_sum / (count_sum * selected_p_hat);
            const float3 to_sample = selected.position - position;
            const float3 estimate = selected.radiance * (max(dot(normal, to_sample / max(length(to_sample), 1e-4f)), 0.0f) * contribution_weight / PI);
            if (all(isfinite(estimate)))
            {
                estimate_sum += estimate;
            }
        }
    }
    push.attach.out_samples.get()[dtid] = alloc.y > 0u ? float4(estimate_sum / float(alloc.y), float(alloc.y)) : float4(0.0f, 0.0f, 0.0f, 0.0f);

    selected.count = count_sum;
    selected.ray_history = min(merge.ray_history + float(alloc.y), ray_history_cap);
    selected.weight = (selected_p_hat > 0.0f && count_sum > 0.0f) ? weight_sum / (count_sum * selected_p_hat) : 0.0f;
    rt_lab3_store(push.attach.out_a.get(), push.attach.out_b.get(), push.attach.out_c.get(), dtid, selected);
}
