#pragma once

#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "pgi_update.inl"
#include "../../shader_lib/pgi.hlsl"
#include "../../shader_lib/misc.hlsl"
#include "../../shader_lib/debug.glsl"
#include "../../shader_lib/raytracing.hlsl"
#include "../../shader_lib/SH.hlsl"

func octahedtral_texel_wrap(int2 index, int resolution) -> int2
{
    // Octahedral texel clamping is very strange..
    if (index.y >= resolution)
    {
        // Flip y on the edge of the texture
        index.y = (resolution-1) - (index.y-resolution);

        // Flip x on the middle of the texture.
        index.x = resolution - 1 - index.x;
    }
    
    if (index.y < 0)
    {
        // Flip y on the edge of the texture
        index.y = -index.y - 1;

        // Flip x on the middle of the texture.
        index.x = resolution - 1 - index.x;
    }

    if (index.x >= resolution)
    {
        // Flip x on the edge of the texture
        index.x = (resolution-1) - (index.x-resolution);

        // Flip y on the middle of the texture.
        index.y = resolution - 1 - index.y;
    }
    
    if (index.x < 0)
    {
        // Flip x on the edge of the texture
        index.x = -index.x - 1;

        // Flip y on the middle of the texture.
        index.y = resolution - 1 - index.y;
    }

    return index;
}

__generic<let N : uint>
func write_probe_texel_with_border(RWTexture2DArray<vector<float,N>> tex, int2 probe_res, int3 base_index, int2 probe_texel, vector<float, N> value, uint layer_offset = 0)
{
    base_index.z += layer_offset;
    bool2 on_edge = probe_texel == 0 || probe_texel == (probe_res - 1);
    int2 mirror_index = probe_texel - probe_res/2;
    mirror_index += int2(mirror_index >= int2(0,0)) * int2(1,1);
    bool border_corner = all(on_edge);
    tex[int3(probe_texel, 0) + base_index] = value;
    
    if (border_corner)
    {
        // Diagonal Border Texel
        int2 diag_index = -mirror_index;
        diag_index += int2(mirror_index >= int2(0,0)) * int2(-1,-1);
        diag_index += probe_res/2;
        tex[int3(diag_index,0) + base_index] = value;
    }

    if (on_edge.x)
    {
        int2 edge_index = int2(mirror_index.x + sign(mirror_index.x), -mirror_index.y);
        edge_index += int2(edge_index >= int2(0,0)) * int2(-1,-1);
        edge_index += probe_res/2;
        tex[int3(edge_index,0) + base_index] = value;
    }

    if (on_edge.y)
    {
        int2 edge_index = int2(-mirror_index.x, mirror_index.y + sign(mirror_index.y));
        edge_index += int2(edge_index >= int2(0,0)) * int2(-1,-1);
        edge_index += probe_res/2;
        tex[int3(edge_index,0) + base_index] = value;
    }
}

[[vk::push_constant]] PGIUpdateProbeTexelsPush update_probe_texels_push;

// Every probe update adds this to the probe validity (see the probe reposition pass), a fresh probe has validity 0.
#define PGI_PROBE_VALIDITY_PER_UPDATE 0.05f

func abs_radiance(float3 radiance) -> float
{
    return radiance.r + radiance.g * 2.0f + radiance.b * 0.5f;
}

#define PGI_PERCEPTUAL_EXPONENT 4.0f
func perceptual_lerp3(float3 a, float3 b, float v) -> float3
{
    return pow(lerp(pow(a + 0.0000001f, (1.0f/PGI_PERCEPTUAL_EXPONENT)), pow(b + 0.0000001f, (1.0f/PGI_PERCEPTUAL_EXPONENT)), v), PGI_PERCEPTUAL_EXPONENT);
}

func fast_uint_div(uint a, uint b) -> uint
{
    return uint(floor(float(a) / float(b) + rcp(float(1u << 12u))));
}

// Maps an update thread to the probe (index into the indirect update list) and the texel it updates.
// Returns false for threads that are past the last updated probe.
// The dispatch is 1D (workgroups wrapped into rows of PGI_UPDATE_LINEAR_WG_X): consecutive threads walk the texels of one probe row major, then the next probe.
func pgi_update_thread_to_probe_texel(
    uint flat_thread,
    int texel_res,
    uint probe_update_count,
    out uint indirect_index,
    out int2 probe_texel) -> bool
{
    let texels_per_probe = uint(texel_res * texel_res);
    // flat_thread stays below 2^24 (max updates per frame * texels per probe), so the float based division is exact enough.
    indirect_index = fast_uint_div(flat_thread, texels_per_probe);
    let texel_index = flat_thread - indirect_index * texels_per_probe;
    probe_texel.y = int(fast_uint_div(texel_index, uint(texel_res)));
    probe_texel.x = int(texel_index) - probe_texel.y * texel_res;
    return indirect_index < probe_update_count;
}

func entry_update_probe_color(
    uint flat_thread,
)
{
    let push = update_probe_texels_push;
    PGISettings* settings = &push.attach.globals.pgi_settings;
    PGISettings reg_settings = *settings;
    const int probe_texel_res = settings.probe_color_resolution;

    uint indirect_index = {};
    int4 probe_index = {};
    int2 probe_texel = {};
    {
        let is_valid = pgi_update_thread_to_probe_texel(flat_thread, reg_settings.probe_color_resolution, push.attach.probe_indirections.probe_update_count, indirect_index, probe_texel);
        if (!is_valid)
        {
            return;
        }
        uint indirect_package = ((uint*)(push.attach.probe_indirections + 1))[indirect_index];
        probe_index = pgi_unpack_indirect_probe(indirect_package);
    }
    PGICascade reg_cascade = settings->cascades[probe_index.w];
    
    const uint frame_index = push.attach.globals.frame_index;
    if (any(greaterThanEqual(probe_index.xyz, reg_settings.probe_count)))
    {
        return;
    }

    GPU_ASSERT_COMPARE_INT(probe_texel.x, >=, 0);
    GPU_ASSERT_COMPARE_INT(probe_texel.x, <, reg_settings.probe_color_resolution);
    GPU_ASSERT_COMPARE_INT(probe_texel.y, >=, 0);
    GPU_ASSERT_COMPARE_INT(probe_texel.y, <, reg_settings.probe_color_resolution);
    
    PGIProbeInfo probe_info = PGIProbeInfo::load(reg_settings, reg_cascade, push.attach.probe_info.get(), probe_index);
    float3 probe_position = pgi_probe_index_to_worldspace(reg_settings, reg_cascade, probe_info, probe_index);
    float2 probe_texel_uv = (float2(probe_texel) + 0.5f) * rcp(probe_texel_res);
    float3 probe_texel_normal = pgi_probe_uv_to_probe_normal(probe_texel_uv);
    float2 probe_texel_min_uv = (float2(probe_texel)) * rcp(probe_texel_res);
    float2 probe_texel_max_uv = (float2(probe_texel) + 1.0f) * rcp(probe_texel_res);

    int3 probe_texture_base_index = pgi_probe_texture_base_offset<HAS_BORDER>(reg_settings, reg_cascade, probe_texel_res, probe_index);
    int3 probe_texture_index = probe_texture_base_index + int3(probe_texel, 0);

    int s = reg_settings.probe_trace_resolution;

    float4 cosine_convoluted_trace_result = float4(0.0f,0.0f,0.0f, 0.0f);
    uint3 trace_result_texture_base_index = uint3(pgi_indirect_index_to_trace_tex_offset(reg_settings, indirect_index), 0);
    float2 trace_texel_noise = pgi_probe_trace_noise(probe_index, frame_index); // used to reconstruct directions used for traces.
    float acc_weight = 0.0f;
    float rcp_s = rcp(s);
    Texture2DArray<float4> trace_result_tex = push.attach.trace_result.get();

    float4 prev_frame_texel = push.attach.probe_color.get()[probe_texture_index];
    float3 prev_frame_irradiance = prev_frame_texel.rgb;
    // The hysteresis of a texel is the cosine convolved hysteresis of the last frame (the same cosine lobe the irradiance is convolved with),
    // not the texel's own value.
    // KNOWN RACE (kept on purpose): the neighbors are read from the probe color image that this pass also writes, so a neighbor texel may already hold its NEW value of
    // this frame. Which ones depends on the thread order of the GPU: the result is nondeterministic and not reproducible between frames.
    // The hysteresis RISES by at most 0.1 per update but can DROP by much more (a large relative difference takes it from 0.95 to 0 in one update), so the jitter is not
    // always small: it changes the blend weight of the texel (0.95 vs 0.9 doubles its update speed).
    // Clean fix if it ever matters: compute this blur in the trace integrate pass (it runs before the color update and writes nothing to the probe color image)
    // into a small transient image with the layout of the trace summary, and read that here.
    RWTexture2DArray<float4> prev_hysteresis_tex = push.attach.probe_color.get();
    float hysteresis_blur_sum = 0.0f;
    float hysteresis_blur_weight = 0.0f;

    float3 radiance = float3(0.0f,0.0f,0.0f);
    float radiance_weight = 0.0f;
    // A weighted (front facing, cos > 0) texel exists: the new irradiance stands for data, if none of them got a ray the old value is kept.
    bool irradiance_has_weighted_texel = false;
    Texture2DArray<uint> trace_count_tex = push.attach.trace_count.get();
    for (int y = 0; y < s; ++y)
    for (int x = 0; x < s; ++x)
    {
        float2 trace_tex_uv = (float2(x,y) + trace_texel_noise) * rcp_s;
        float3 trace_direction = pgi_probe_uv_to_probe_normal(trace_tex_uv); // Trace direction is identical to the one used in tracer.
        float cos_weight = max(0.0f, dot(trace_direction, probe_texel_normal));
        {
            // Hysteresis of the probe texel the trace direction falls into, same cosine weight as the irradiance.
            const int2 hysteresis_texel = clamp(int2(trace_tex_uv * float(probe_texel_res)), int2(0, 0), int2(probe_texel_res - 1, probe_texel_res - 1));
            hysteresis_blur_sum += cos_weight * prev_hysteresis_tex[probe_texture_base_index + int3(hysteresis_texel, 0)].a;
            hysteresis_blur_weight += cos_weight;
        }
        int3 sample_texture_index = trace_result_texture_base_index + int3(x,y,0);
        // Number of valid rays integrated into this trace texel, 0 = no data.
        const uint packed_count = trace_count_tex[sample_texture_index];
        const float ray_count = float(packed_count & 0xFFFFu);
        const bool has_rays = ray_count > 0.0f;
        // Horvitz-Thompson: a covered texel stands for 1 / coverage probability texels (the ones that got no ray are not seen).
        const float inverse_coverage = has_rays ? 65535.0f * rcp(float(max(packed_count >> 16u, 1u))) : 0.0f;
        // If statement on cos weight would REDUCE PERFORMANCE. Reads in branches lead to poor latency hiding due to long scoreboard stalls.
        {
            float4 sample = trace_result_tex[sample_texture_index].rgba;

            // Firefly filter
            if (true) 
            {
                const float firely_clamp_threshold_ratio = 32.0f;
                const float relative_radiance_difference = abs_radiance(sample.rgb) / ( 0.0001f + abs_radiance(prev_frame_irradiance) );
                const float supression_factor = min(1.0f, firely_clamp_threshold_ratio / relative_radiance_difference );
                sample.rgb *= supression_factor;
            }

            const bool backface_hit = sample.a < 0.0f;
            if (!backface_hit)
            {
                // Texels without rays carry no data, they are left out of the estimate.
                const float sample_weight = cos_weight * inverse_coverage;
                cosine_convoluted_trace_result += sample * sample_weight;
                acc_weight += sample_weight;
                irradiance_has_weighted_texel = irradiance_has_weighted_texel || cos_weight > 0.0f;
            }

            const bool trace_within_probe_texel = all(trace_tex_uv >= probe_texel_min_uv && trace_tex_uv <= probe_texel_max_uv);
            if (trace_within_probe_texel)
            {
                radiance += has_rays ? sample.rgb : float3(0.0f, 0.0f, 0.0f);
                radiance_weight += has_rays ? 1.0f : 0.0f;
            }
        }

    }

    // Without any valid ray inside the texel the radiance is not updated.
    if (radiance_weight > 0.0f)
    {
        radiance *= rcp(radiance_weight);
    }

    // If we have 0 weight we have no samples to blend so we dont blend at all.
    if (acc_weight > 0.0f)
    {
        // We div by the sum of cosine weights to reduce variance.
        // To compensate we have to multiply the result by 2.
        cosine_convoluted_trace_result *= rcp(2.0f * acc_weight);
    }

    float3 new_irradiance = cosine_convoluted_trace_result.rgb;

    // Automatic Hysteresis
    float hysteresis = prev_frame_texel.a;
    if (hysteresis_blur_weight > 0.0f)
    {
        hysteresis = hysteresis_blur_sum * rcp(hysteresis_blur_weight);
    }
    {
        const float HYSTERESIS_UPDATE_RATE = 0.1f;
        const float HYSTERESIS_MIN_RELATIVE_CHANGE = 1.5f;
        const float MAX_CONSIDERED_RELATIVE_DIFFERENCE = 32.0f;
        const float HYSTERESIS_SMALL_LIGHT_BIAS = 0.000001f;

        const float3 power_scaled_min = min(prev_frame_irradiance, new_irradiance) + HYSTERESIS_SMALL_LIGHT_BIAS;
        const float3 relative_difference3 = abs(prev_frame_irradiance - new_irradiance) / power_scaled_min;
        const float relative_difference = min(max3(relative_difference3.x, relative_difference3.y, relative_difference3.z), MAX_CONSIDERED_RELATIVE_DIFFERENCE);
        const float BASE_CONFIDENCE_GAIN = HYSTERESIS_UPDATE_RATE;
        const float RELATIVE_DIFFERENCE_SCALING = BASE_CONFIDENCE_GAIN / HYSTERESIS_MIN_RELATIVE_CHANGE;
        hysteresis += -relative_difference * RELATIVE_DIFFERENCE_SCALING + BASE_CONFIDENCE_GAIN;
        hysteresis = clamp(hysteresis, 0.0f, 0.95f);
    }
    float blend = hysteresis;
    // default initialize probe lighting with higher cascade value if possible
    if (probe_info.validity < 0.5f)
    {
        if (probe_index.w + 1 < reg_settings.cascade_count)
        {
            int4 higher_cascade_probe = ((probe_index.xyz - int3(reg_settings.probe_count/2)/2 + reg_settings.probe_count/2), probe_index.w + 1);
            
            PGIProbeInfo higher_probe_info = PGIProbeInfo::load(reg_settings, reg_cascade, push.attach.probe_info.get(), higher_cascade_probe);
            
            if (higher_probe_info.validity > 1.0f)
            {
                int3 higher_cascade_probe_texture_base_index = pgi_probe_texture_base_offset<HAS_BORDER>(reg_settings, reg_cascade, probe_texel_res, higher_cascade_probe);
                int3 higher_cascade_probe_texture_index = higher_cascade_probe_texture_base_index + int3(probe_texel, 0);
                float3 higher_probe_irrad = push.attach.probe_color.get()[higher_cascade_probe_texture_index].rgb;
                new_irradiance = higher_probe_irrad;
            }
        }
        blend = 0.0f;
        hysteresis = 0.5f;
    }
    
    // Perceptual lerp helps with blending to dark values
    // Fresh probes take the new (or higher cascade) value completely. Otherwise the old value is kept when weighted texels exist but none of them got a ray.
    const bool fresh_probe = probe_info.validity < 0.5f;
    const bool irradiance_no_data = irradiance_has_weighted_texel && acc_weight <= 0.0f;
    new_irradiance = perceptual_lerp3(new_irradiance, prev_frame_irradiance, (fresh_probe || !irradiance_no_data) ? blend : 1.0f);
    new_irradiance = max(new_irradiance, float3(0,0,0)); // remove nans, what can i say...

    // Perceptual lerp helps with blending to dark values
    const float radiance_blend = fresh_probe ? 0.0f : (radiance_weight > 0.0f ? 0.75f * hysteresis : 1.0f);
    const float3 prev_exact_radiance = push.attach.probe_color.get()[probe_texture_index + int3(0, 0, reg_settings.cascade_count * reg_settings.probe_count.z)].rgb;
    float3 new_radiance = perceptual_lerp3(radiance, prev_exact_radiance, radiance_blend);
    new_radiance = max(new_radiance, float3(0,0,0)); // remove nans, what can i say...

    write_probe_texel_with_border(push.attach.probe_color.get(), reg_settings.probe_color_resolution, probe_texture_base_index, probe_texel, float4(new_irradiance, hysteresis));
    write_probe_texel_with_border(push.attach.probe_color.get(), reg_settings.probe_color_resolution, probe_texture_base_index, probe_texel, float4(new_radiance, 1.0f), reg_settings.cascade_count * settings->probe_count.z);
}

func entry_update_probe_visibility(
    uint flat_thread,
)
{
    let push = update_probe_texels_push;
    PGISettings* settings = &push.attach.globals.pgi_settings;
    PGISettings reg_settings = *settings;
    let probe_texel_res = reg_settings.probe_visibility_resolution;

    uint indirect_index = {};
    int4 probe_index = {};
    int2 probe_texel = {};
    {
        let is_valid = pgi_update_thread_to_probe_texel(flat_thread, reg_settings.probe_visibility_resolution, push.attach.probe_indirections.probe_update_count, indirect_index, probe_texel);
        if (!is_valid)
        {
            return;
        }
        uint indirect_package = ((uint*)(push.attach.probe_indirections + 1))[indirect_index];
        probe_index = pgi_unpack_indirect_probe(indirect_package);
    }
    PGICascade reg_cascade = settings->cascades[probe_index.w];

    uint frame_index = push.attach.globals.frame_index;

    if (any(greaterThanEqual(probe_index.xyz, reg_settings.probe_count)))
    {
        return;
    }
    
    PGIProbeInfo probe_info = PGIProbeInfo::load(reg_settings, reg_cascade, push.attach.probe_info.get(), probe_index);
    float3 probe_position = pgi_probe_index_to_worldspace(reg_settings, reg_cascade, probe_info, probe_index);
    float2 probe_texel_min_uv = (float2(probe_texel)) * rcp(probe_texel_res);
    float2 probe_texel_max_uv = (float2(probe_texel) + 1.0f) * rcp(probe_texel_res);
    float2 probe_texel_uv = (float2(probe_texel) + 0.5f) * rcp(probe_texel_res);
    float3 probe_texel_normal = pgi_probe_uv_to_probe_normal((probe_texel_max_uv + probe_texel_min_uv) * 0.5f);

    int3 probe_texture_base_index = pgi_probe_texture_base_offset<HAS_BORDER>(reg_settings, reg_cascade, probe_texel_res, probe_index);
    
    int3 probe_texture_index = probe_texture_base_index + int3(probe_texel, 0);

    float2 relevant_trace_blend = float2(0.0f, 0.0f);
    int valid_trace_count = 0;
    uint3 trace_result_texture_base_index = uint3(pgi_indirect_index_to_trace_tex_offset(reg_settings, indirect_index), 0);
    float2 trace_texel_noise = pgi_probe_trace_noise(probe_index, frame_index); // used to reconstruct directions used for traces.
    const float max_depth = reg_cascade.max_visibility_distance;
    float2 prev_frame_visibility = push.attach.probe_visibility.get()[probe_texture_index];
    float acc_cos_weights = {};
    Texture2DArray<uint> trace_count_tex = push.attach.trace_count.get();
    int s = reg_settings.probe_trace_resolution;
    float rcp_s = rcp(s) * 0.999999f; // The multiplication with 0.999999f ensures that the calculated uv never reaches 1.0f.

    static const float COS_POWER = 25.0f;
    static const float MIN_ACCEPTED_POWER_COS = 0.001f;
    // Based on the COS_POWER and the MIN_ACCEPTED_COS_POWER, we can calculate the largest accepted cos value.
    static const float MIN_ACCEPTED_COS = acos(pow(MIN_ACCEPTED_POWER_COS, 1.0f / COS_POWER));

    Texture2DArray<float4> trace_result_tex = push.attach.trace_result.get();

    const float rcp_texel_size = rcp(float(reg_settings.probe_visibility_resolution));

    // At 0.15 RADIANS, the power cos weight becomes 0.01.
    // This calculation assumes the least distortion and the highest local texel resolution of an octahedral mapped sphere.
    // At 0.15 * 2 = 0.3 as a RADIANS range to sample trace texels
    // Roughly this means we sample a 0.3 x 0.3 uv area of the trace texels
    static const float RELEVANT_RANGE = 0.3; // from 0 - 1
    const int relevant_texel_range = int(ceil(float(reg_settings.probe_trace_resolution) * RELEVANT_RANGE));

    const float2 probe_trace_uv_min = probe_texel_uv - float2(RELEVANT_RANGE,RELEVANT_RANGE) * 0.5f;
    const int2 probe_trace_index_min = int2(floor(probe_trace_uv_min * reg_settings.probe_trace_resolution));

#if PGI_SHARP_DEPTH && false
    {
        int3 sample_texture_index = trace_result_texture_base_index + int3(probe_texel,0);
        float trace_depth = trace_result_tex[sample_texture_index].a;
        bool is_backface = trace_depth < 0.0f;

        trace_depth = abs(trace_depth);
        trace_depth = min(trace_depth, max_depth);
        if (is_backface) // Backface probe killer.
        { 
            relevant_trace_blend.x += -trace_depth * PGI_BACKFACE_DIST_SCALE;
            relevant_trace_blend.y += 0.0f;
            acc_cos_weights += PGI_BACKFACE_DIST_SCALE;
        }
        else
        {
            relevant_trace_blend.x += trace_depth;
            // Smooth out hard contacts. Its always better to have a minimum difference to the average.
            const float DIFF_TO_AVERAGE_BIAS = 0.01f;
            float difference_to_average = abs(max(0.0f, prev_frame_visibility.x) - trace_depth) + DIFF_TO_AVERAGE_BIAS * max_depth;
            relevant_trace_blend.y += square(difference_to_average);
            acc_cos_weights += 1.0f;
        }
        valid_trace_count = 1;
    }
#else
    for (int y = 0; y < relevant_texel_range; ++y)
    for (int x = 0; x < relevant_texel_range; ++x)
    {
        int2 probe_trace_tex_index = (int2(x,y) + probe_trace_index_min);
        probe_trace_tex_index = octahedtral_texel_wrap(probe_trace_tex_index, reg_settings.probe_trace_resolution);
        float2 trace_tex_uv = (probe_trace_tex_index + trace_texel_noise) * rcp_s;
        float3 trace_direction = pgi_probe_uv_to_probe_normal(trace_tex_uv); // Trace direction is identical to the one used in tracer.
        float cos_weight = (dot(trace_direction, probe_texel_normal));        
        int3 sample_texture_index = trace_result_texture_base_index + int3(probe_trace_tex_index,0);
        float power_cos_weight = pow(cos_weight, COS_POWER);
        // If statement on cos weight would REDUCE PERFORMANCE. Reads in branches lead to poor latency hiding due to long scoreboard stalls.
        {
            #if PGI_SHARP_DEPTH
            const bool trace_within_probe_texel = all(trace_tex_uv >= probe_texel_min_uv && trace_tex_uv <= probe_texel_max_uv);
            if (!trace_within_probe_texel)
            {
                continue;
            }
            #endif

            // Trace texels without a valid ray have no data, they are skipped.
            if ((trace_count_tex[sample_texture_index] & 0xFFFFu) == 0u)
            {
                continue;
            }

            float trace_depth = trace_result_tex[sample_texture_index].a;
            bool is_backface = trace_depth < 0.0f;

            trace_depth = abs(trace_depth);
            trace_depth = min(trace_depth, max_depth);
            {
                if (is_backface) // Backface probe killer.
                {
                    relevant_trace_blend.x += -trace_depth * cos_weight * PGI_BACKFACE_DIST_SCALE;
                    relevant_trace_blend.y += 0.0f;
                    acc_cos_weights += cos_weight * PGI_BACKFACE_DIST_SCALE;
                }
                else
                {
                    relevant_trace_blend.x += trace_depth * power_cos_weight;
                    // Smooth out hard contacts. Its always better to have a minimum difference to the average.
                    const float DIFF_TO_AVERAGE_BIAS = 0.01f;
                    float difference_to_average = abs(max(0.0f, prev_frame_visibility.x) - trace_depth) + DIFF_TO_AVERAGE_BIAS * max_depth;
                    relevant_trace_blend.y += square(difference_to_average) * power_cos_weight;
                    acc_cos_weights += power_cos_weight;
                }
                valid_trace_count += 1;
            }
        }
    }
#endif
    
    if (valid_trace_count > 0)
    {
        relevant_trace_blend *= rcp(acc_cos_weights);

        float update_factor = 0.01f;

        float ray_to_texel_ratio = float(reg_settings.probe_trace_resolution) / float(reg_settings.probe_visibility_resolution);
        update_factor *= ray_to_texel_ratio;

        if (probe_info.validity < 0.5f)
        {
            update_factor = 1.0f;
        }

        float2 value = lerp(prev_frame_visibility, relevant_trace_blend, update_factor);

        write_probe_texel_with_border(push.attach.probe_visibility.get(), reg_settings.probe_visibility_resolution, probe_texture_base_index, probe_texel, value);
    }
}

[shader("compute")]
[numthreads(PGI_UPDATE_WG_SIZE,1,1)]
func entry_update_probe_texels(
    int3 gid : SV_GroupID,
    uint group_index : SV_GroupIndex,
)
{
    let push = update_probe_texels_push;
    // Workgroups are wrapped into rows of PGI_UPDATE_LINEAR_WG_X.
    let flat_thread = (uint(gid.y) * PGI_UPDATE_LINEAR_WG_X + uint(gid.x)) * PGI_UPDATE_WG_SIZE + group_index;
    if (push.update_radiance)
    {
        entry_update_probe_color(flat_thread);
    }
    else
    {
        entry_update_probe_visibility(flat_thread);
    }
}

[[vk::push_constant]] PGIUpdateProbesPush update_probes_push;

#define PGI_DESIRED_RELATIVE_DISTANCE 0.3f 
#define PGI_RELATIVE_REPOSITIONING_STEP 0.2f
#define PGI_MAX_RELATIVE_REPOSITIONING 0.66f
#define PGI_ACCEPTABLE_SURFACE_DISTANCE (PGI_DESIRED_RELATIVE_DISTANCE * 0.3)
#define PGI_BACKFACE_ESCAPE_RANGE (PGI_DESIRED_RELATIVE_DISTANCE * 3)
#define PGI_PROBE_VIEW_DISTANCE 1.0

// The probe ray cdf (see PGI_RAY_CDF_ONE in pgi.hlsl) is built here, in the update of the probe.

func pgi_probe_radiance_luma(float3 radiance) -> float
{
    return max(0.0f, dot(radiance, float3(0.2126f, 0.7152f, 0.0722f)));
}

// One wave per probe. Each lane owns a contiguous section of the texels:
// - lane local: luma section sum, later the prefix sum inside the section
// - wave: ONE exclusive prefix sum over the section sums gives every section its start
// - the base distribution is patched into the finished prefix sum
func pgi_update_probe_ray_cdf(PGISettings reg_settings, PGICascade reg_cascade, int4 probe_index, int lane)
{
    let push = update_probes_push;
    let res = reg_settings.probe_color_resolution;
    let texel_count = res * res;

    // Section of this lane: contiguous texels in row major order. Lanes past the last texel get an empty section.
    let section_size = round_up_div(texel_count, WARP_SIZE);
    let section_begin = min(lane * section_size, texel_count);
    let section_end = min(section_begin + section_size, texel_count);

    // Source (radiance texels) and destination (cdf buffer) addressing.
    Texture2DArray<float4> radiance_tex = push.attach.probe_color.get();
    // Radiance lives in the upper half of the color layers.
    int3 radiance_base = pgi_probe_texture_base_offset<HAS_BORDER>(reg_settings, reg_cascade, res, probe_index);
    radiance_base.z += reg_settings.cascade_count * reg_settings.probe_count.z;
    // The cdf is a flat buffer: texel_count consecutive values per probe, in stable probe order.
    uint* cdf_ptr = push.attach.probe_ray_cdf + pgi_probe_stable_linear_index(reg_settings, reg_cascade, probe_index) * uint(texel_count);

    // Step 1, lane local: sum of the luma over this lane's section.
    float section_sum = 0.0f;
    for (int i = section_begin; i < section_end; ++i)
    {
        int y = int(fast_uint_div(uint(i), uint(res)));
        int2 texel = int2(i - y * res, y);
        section_sum += pgi_probe_radiance_luma(radiance_tex[radiance_base + int3(texel, 0)].rgb);
    }

    // Step 2, wave: the only wave prefix sum over the section sums, exclusive: the start of this lane's section.
    let section_start = WavePrefixSum(section_sum);
    let section_end_sum = section_start + section_sum; // inclusive prefix sum
    // Lanes past the last texel have empty sections, so the inclusive sum of the last lane is the total.
    let total = WaveReadLaneAt(section_end_sum, WARP_SIZE - 1);

    // Step 3, base distribution: patch it into the finished prefix sum.
    // Adding the constant base_weight to every texel weight adds (i + 1) * base_weight to the inclusive prefix sum of texel i.
    // The base weight is a settings fraction of the average texel luma (the share of uniformly spread rays is fraction / (1 + fraction)).
    // A probe without any radiance (fresh or invalid) is sampled uniformly (every weight 1).
    let has_radiance = total > 0.0f;
    let base_weight = has_radiance ? (total * rcp(float(texel_count)) * max(0.0f, reg_settings.probe_ray_base_fraction)) : 1.0f;
    let patched_total = (has_radiance ? total : 0.0f) + float(texel_count) * base_weight;
    let scale = float(PGI_RAY_CDF_ONE) * rcp(patched_total);

    // Step 4, lane local: prefix sum inside the section, offset by the section start,
    // patched with the base distribution, normalized to PGI_RAY_CDF_ONE and written out.
    float cumulative = has_radiance ? section_start : 0.0f;
    for (int i = section_begin; i < section_end; ++i)
    {
        int y = int(fast_uint_div(uint(i), uint(res)));
        int2 texel = int2(i - y * res, y);
        cumulative += pgi_probe_radiance_luma(radiance_tex[radiance_base + int3(texel, 0)].rgb);
        let patched = (has_radiance ? cumulative : 0.0f) + float(i + 1) * base_weight;
        uint cdf = min(PGI_RAY_CDF_ONE, uint(patched * scale + 0.5f));
        // Rounding must not leave the last value off the exact total.
        if (i == texel_count - 1)
        {
            cdf = PGI_RAY_CDF_ONE;
        }
        cdf_ptr[i] = cdf;
    }
}

[shader("compute")]
[numthreads(WARP_SIZE,1,1)]
func entry_update_probe(
    int group_id : SV_GroupID,
    int group_index : SV_GroupIndex)
{
    let push = update_probes_push;
    PGISettings* settings = &push.attach.globals.pgi_settings;

    uint indirect_index = {};
    int4 probe_index = {};
    {
        indirect_index = group_id;
        let overhang = indirect_index >= push.attach.probe_indirections.probe_update_count;
        if (overhang)
        {
            return;
        }

        uint indirect_package = ((uint*)(push.attach.probe_indirections + 1))[indirect_index];
        probe_index = pgi_unpack_indirect_probe(indirect_package);
    }

    PGISettings reg_settings = *settings;
    PGICascade reg_cascade = settings->cascades[probe_index.w];

    int3 stable_index = pgi_probe_to_stable_index(reg_settings, reg_cascade, probe_index);

    if (any(greaterThanEqual(probe_index.xyz, reg_settings.probe_count)))
    {
        return;
    }

    uint3 trace_base_texel = uint3(pgi_indirect_index_to_trace_tex_offset(reg_settings, indirect_index), 0);

    PGIProbeInfo probe_info = {}; // The dummy here is intentionally used to query default world space position.
    float3 original_probe_position = pgi_probe_index_to_worldspace(reg_settings, reg_cascade, probe_info, probe_index);
    probe_info = PGIProbeInfo::load(reg_settings, reg_cascade, push.attach.probe_info.get(), probe_index);
    float3 probe_position = pgi_probe_index_to_worldspace(reg_settings, reg_cascade, probe_info, probe_index);

    rand_seed(push.attach.globals.frame_index);

    // As the offset is in "probe space" (-1 to 1 in xyz between probes), we convert all directions and values into probe space as well.
    const float SOME_LARGE_VALUE = 10000.0f;
    float2 texel_trace_noise = pgi_probe_trace_noise(probe_index, push.attach.globals.frame_index);
    const float max_probe_distance = reg_cascade.max_visibility_distance;
    int s = reg_settings.probe_trace_resolution;
    float trace_res_rcp = rcp(float(reg_settings.probe_trace_resolution));
    Texture2DArray<float4> trace_result_tex = push.attach.trace_result.get();
    let wave_iterations = round_up_div(s*s, WARP_SIZE);

    // Calculated per lane, then averaged over wavefront.
    float3 lane_average_hit_offset = {};
    float3 lane_average_backface_hit_offset = {};
    float lane_closest_backface_dist = SOME_LARGE_VALUE;
    float lane_closest_frontface_dist = SOME_LARGE_VALUE;
    float3 lane_closest_backface_dir = {};
    float lane_backface_count = {};
    // Trace texels with at least one valid ray. Texels without data are skipped (their distance would read as a hit at distance 0).
    float lane_valid_ray_texel_count = {};
    Texture2DArray<uint> trace_count_tex = push.attach.trace_count.get();

    for (int wave_i = 0; wave_i < wave_iterations; ++wave_i)
    {
        int i = wave_i * WARP_SIZE + group_index;
        int y = fast_uint_div(i, s); // Can not be negative, no need to clamp.
        int x = i - y * s;
        if (i >= (s*s))
        {
            continue;
        }
        int2 probe_local_texel = int2(x,y);
        int3 texel = trace_base_texel + int3(x,y,0);
        float2 uv = (float2(probe_local_texel) + texel_trace_noise) * trace_res_rcp;
        float3 ray_dir = pgi_probe_uv_to_probe_normal(uv);
        
        if (trace_count_tex[texel] == 0u)
        {
            continue;
        }
        lane_valid_ray_texel_count += 1.0f;

        float trace_result_a = trace_result_tex[texel].a;
        bool is_backface_hit = trace_result_a < 0.0f;
        float trace_distance = (is_backface_hit ? -trace_result_a : trace_result_a);
        float3 probe_space_hit_position_no_offset = trace_distance * ray_dir * reg_cascade.probe_spacing_rcp;
        float3 probe_space_hit_position = probe_space_hit_position_no_offset + probe_info.offset;
        float probe_space_dist = length(probe_space_hit_position_no_offset);
        float3 probe_space_ray_dir = normalize(ray_dir * reg_cascade.probe_spacing_rcp);

        probe_space_dist = min(PGI_PROBE_VIEW_DISTANCE, probe_space_dist);

        lane_average_hit_offset += probe_space_ray_dir * max(0.5f, probe_space_dist);

        if (is_backface_hit)
        {
            // Only move out probes that are close to a surface.
            // Probes need a lot of room to move when they cross the surface,
            // if they are very deep they have no space to reposition after they cross the surface.
            if (probe_space_dist < PGI_BACKFACE_ESCAPE_RANGE)
            {
                if (probe_space_dist < lane_closest_backface_dist)
                {
                    lane_closest_backface_dist = probe_space_dist;
                    lane_closest_backface_dir = probe_space_ray_dir;
                }
                lane_average_backface_hit_offset += probe_space_ray_dir * probe_space_dist;
                lane_backface_count += 1;
            }
        }
        else
        {
            lane_closest_frontface_dist = min(lane_closest_frontface_dist, probe_space_dist);
        }
    }

    let backface_count = WaveActiveSum(lane_backface_count);
    let wave_lane_closest_backface_dist = WaveActiveMin(lane_closest_backface_dist);
    let lane_has_min = wave_lane_closest_backface_dist == lane_closest_backface_dist;
    let elect = firstbitlow(WaveActiveBallot(lane_has_min).x);
    var closest_backface_dist = WaveShuffle(wave_lane_closest_backface_dist, elect);
    var closest_backface_dir = WaveShuffle(lane_closest_backface_dir, elect);
    let closest_frontface_dist = WaveActiveMin(lane_closest_frontface_dist);
    var average_hit_offset = WaveActiveSum(lane_average_hit_offset);
    var average_backface_hit_offset = WaveActiveSum(lane_average_backface_hit_offset);

    pgi_update_probe_ray_cdf(reg_settings, reg_cascade, probe_index, group_index);

    // Average over the trace texels that have data. Without any the offset stays 0.
    let valid_ray_texel_count = WaveActiveSum(lane_valid_ray_texel_count);
    average_hit_offset *= rcp(max(valid_ray_texel_count, 1.0f));
    if (backface_count > 0)
    {
        average_backface_hit_offset *= rcp(float(backface_count));
    }
    
    float3 spring_force = {};
    if (group_index < 6)
    {
        int x = (group_index == 0 ? 1 : 0) + (group_index == 1 ? -1 : 0);
        int y = (group_index == 2 ? 1 : 0) + (group_index == 3 ? -1 : 0);
        int z = (group_index == 4 ? 1 : 0) + (group_index == 5 ? -1 : 0);
        int3 other_probe_index_offset = int3(x,y,z);
        int4 other_probe_index = probe_index + int4(other_probe_index_offset, 0);
        other_probe_index.xyz = clamp(other_probe_index.xyz, int3(0,0,0), reg_settings.probe_count - 1);

        if (all(other_probe_index != probe_index))
        {
            PGIProbeInfo other_info = PGIProbeInfo::load(reg_settings, reg_cascade, push.attach.probe_info_copy.get(), other_probe_index);

            float3 equilibrium_diff = lerp(other_info.offset - probe_info.offset, - probe_info.offset, 0.1f);
            float diff_magnitude = min(1.0f, length(equilibrium_diff));
            if (length(equilibrium_diff) > 0.0f)
            {
                spring_force += equilibrium_diff * diff_magnitude * diff_magnitude * diff_magnitude;
            }
        }
    }
    spring_force = WaveActiveSum(spring_force);

    if (!WaveIsFirstLane())
    {
        return;
    }

    probe_info.validity += PGI_PROBE_VALIDITY_PER_UPDATE;

    // Calculate backface attraction
    bool too_few_backface_hits = backface_count < ceil(float(s) * 0.1f);
    if (too_few_backface_hits)
    {
        closest_backface_dist = SOME_LARGE_VALUE;
        closest_backface_dir = {};
    }
    float backface_escape_distance = too_few_backface_hits ? 0.0f : (closest_backface_dist + PGI_ACCEPTABLE_SURFACE_DISTANCE * 0.5f);
    float backface_escape_power = backface_escape_distance * rcp(PGI_BACKFACE_ESCAPE_RANGE);


    // Calculate frontface attraction and repulsion
    float3 estimated_freedom_direction = normalize(average_hit_offset); // Average of hit points generally gives a high quality outward vector
    float frontface_repulse_distance = 0.0f;
    float frontface_repulse_power = 0.0f;
    if (backface_escape_distance == 0.0f)
    {
        frontface_repulse_distance = clamp(PGI_DESIRED_RELATIVE_DISTANCE - closest_frontface_dist, 0.0f, PGI_DESIRED_RELATIVE_DISTANCE);
        frontface_repulse_power = frontface_repulse_distance * rcp(PGI_DESIRED_RELATIVE_DISTANCE);
    }


    // Calculate probe grid spring attraction and repulsion 
    float spring_force_distance = 0.0f;
    float3 spring_force_dir = {};
    if ((backface_escape_distance == 0.0f) && (reg_settings.probe_repositioning_spring_force != 0))
    {
        spring_force_distance = length(spring_force);
        spring_force_distance *= 1.0f - frontface_repulse_power;

        if (spring_force_distance > 0.001f)
        {
            spring_force_dir = normalize(spring_force);
        }

        // We dont want the spring force to move the probe closer to geometry.
        // To prevent the spring force pushing probes into geometry,
        // we calculate the part of the spring force pointing towards geometry,
        // and stir the spring force away.
        if (spring_force_distance > 0.001f)
        {
            float3 towards_geometry_direction = -estimated_freedom_direction;
            float geometry_dir_projected_spring_force = max(0.0f, dot(towards_geometry_direction, spring_force_dir));
            // deflect direction pointing towards geometry
            spring_force_dir = spring_force_dir - geometry_dir_projected_spring_force * towards_geometry_direction;
            spring_force_dir = normalize(spring_force_dir);
            // reduce distance pointing intowards geometry
            spring_force_distance *= (1.0f - geometry_dir_projected_spring_force);
        }
    }

    if (probe_info.validity >= 1.0f)
    {
        backface_escape_distance = 0.0f;
        frontface_repulse_distance = 0.0f;
        spring_force_distance = 0.0f;
    }

    float3 adjustment_vector = 
        closest_backface_dir * min(backface_escape_distance, PGI_RELATIVE_REPOSITIONING_STEP) +
        estimated_freedom_direction * min(frontface_repulse_distance * PGI_RELATIVE_REPOSITIONING_STEP, PGI_RELATIVE_REPOSITIONING_STEP) +
        spring_force_dir * min(spring_force_distance * PGI_RELATIVE_REPOSITIONING_STEP, PGI_RELATIVE_REPOSITIONING_STEP);


    if (reg_settings.debug_draw_repositioning_forces)
    {
        ShaderDebugLineDraw backface_force = {};
        backface_force.color = float3(0.5,0,0);
        backface_force.start = probe_position;
        backface_force.end = probe_position + closest_backface_dir * backface_escape_distance * reg_cascade.probe_spacing;
        debug_draw_line(push.attach.globals.debug, backface_force);

        ShaderDebugLineDraw frontface_force = {};
        frontface_force.color = float3(0,0.5,0);
        frontface_force.start = probe_position;
        frontface_force.end = probe_position + estimated_freedom_direction * frontface_repulse_distance * reg_cascade.probe_spacing;
        debug_draw_line(push.attach.globals.debug, frontface_force);

        ShaderDebugLineDraw spring_force = {};
        spring_force.color = float3(0,0,0.5);
        spring_force.start = probe_position;
        spring_force.end = probe_position + spring_force_dir * spring_force_distance * reg_cascade.probe_spacing;
        debug_draw_line(push.attach.globals.debug, spring_force);
    }

    // Invalidate probes that are either too close to a surface or see backfaces
    if (closest_backface_dist != SOME_LARGE_VALUE || closest_frontface_dist < PGI_ACCEPTABLE_SURFACE_DISTANCE)
    {
        probe_info.validity = 0.0f;
    }
    
    // When the window moves, the probe data on the new border must be invalidated.
    int4 prev_frame_probe_index = pgi_probe_index_to_prev_frame(reg_settings, reg_cascade, probe_index);
    bool prev_frame_invalid = any(prev_frame_probe_index.xyz < int3(0,0,0) || prev_frame_probe_index.xyz >= (reg_settings.probe_count));
    if (prev_frame_invalid)
    {
        probe_info.validity = 0.0f;
        probe_info.offset = {};
        adjustment_vector = {};
    }

    if (!reg_settings.probe_repositioning)
    {
        push.attach.probe_info.get()[stable_index] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    float3 new_offset = probe_info.offset;
    float3 curr_offset = probe_info.offset;
    new_offset = curr_offset + adjustment_vector;
    new_offset = clamp(new_offset, -(float3)PGI_MAX_RELATIVE_REPOSITIONING, (float3)PGI_MAX_RELATIVE_REPOSITIONING);      

    // Border brobes must not converge in position as they need the surrounding probes to find their proper location.
    let border_flux = 1;
    let is_border_probe = any(probe_index.xyz < border_flux.xxx || probe_index.xyz >= (reg_settings.probe_count - border_flux.xxx));
    if (is_border_probe)
    {
        probe_info.validity = 0.0f;
    }

    push.attach.probe_info.get()[stable_index] = float4(new_offset, probe_info.validity);
}

[[vk::push_constant]] PGIPreUpdateProbesPush pre_update_probes_push;

[shader("compute")]
[numthreads(PGI_PRE_UPDATE_XYZ,PGI_PRE_UPDATE_XYZ,PGI_PRE_UPDATE_XYZ)]
func entry_pre_update_probes(int3 dtid : SV_DispatchThreadID, int group_index : SV_GroupIndex)
{
    let push = pre_update_probes_push;
    PGISettings* settings = &push.attach.globals.pgi_settings;
    let cascade = lowp_i32_as_f32_div(dtid.z, settings.probe_count.z);
    int4 probe_index = int4(dtid.xy, dtid.z - settings.probe_count.z * cascade, cascade);

    PGISettings reg_settings = *settings;
    PGICascade reg_cascade = settings->cascades[probe_index.w];

    int3 stable_index = pgi_probe_to_stable_index(reg_settings, reg_cascade, probe_index);

    if (any(probe_index.xyz >= reg_settings.probe_count))
    {
        return;
    }

    if (cascade > reg_settings.cascade_count)
    {
        return;
    }

    int4 prev_frame_probe_index = pgi_probe_index_to_prev_frame(reg_settings, reg_cascade, probe_index);

    // The last probe in each dimension is never a base probe.
    // We clear these probes request counter to 0.
    uint probe_base_request = 0;
    let is_base_probe = pgi_is_cell_base_probe(reg_settings, probe_index);
    if (is_base_probe)
    {
        // New probes are those that came into existence due to the window moving to a new location.
        // Any probes that occupy space that was not taken in the prior frame are new.
        let is_prev_base_probe = pgi_is_cell_base_probe(reg_settings, prev_frame_probe_index);

        // Each base probe is responsible for maintaining the cells request counter.
        if (is_prev_base_probe)
        {
            uint request_package = push.attach.requests.get()[stable_index];
            uint direct_request_timer = request_package & 0xFF;
            uint indirect_request_timer = (request_package >> 8) & 0xFF;
            direct_request_timer = max(direct_request_timer, 1) - 1;
            indirect_request_timer = max(indirect_request_timer, 1) - 1;
            request_package = direct_request_timer | (indirect_request_timer << 8);

            probe_base_request = request_package;
        }
    }

    // New probes are those that came into existence due to the window moving to a new location.
    // Any probes that occupy space that was not taken in the prior frame are new.
    let is_new_probe = any(prev_frame_probe_index.xyz < int3(0,0,0)) || any(prev_frame_probe_index.xyz >= (reg_settings.probe_count));

    let fetch = push.attach.probe_info.get()[stable_index];
    PGIProbeInfo probe_info = PGIProbeInfo(fetch.xyz, fetch.a);

    // Each probe is a vertex in up to 8 cells.
    // Check each cells base probes request counter to determine if probe is requested.
    bool requested = false;
    bool update = false;
    uint request_index = ~0u;
    uint update_index = ~0u;
    bool probe_directly_requested = false;
    bool probe_indirectly_requested = false;
    if (!is_new_probe)
    {
        // Each Probe is part of up to 8 cells (8 Probes form the vertices of a cell)
        // A probe request is always only marked in the "base" probe of a cell.
        // The base probe is the one that has the lowest index.
        for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
        for (int z = 0; z < 2; ++z)
        {
            let other_probe = int4(max(int3(0,0,0), probe_index.xyz - int3(x,y,z)), probe_index.w);
            int4 other_probe_prev_frame = pgi_probe_index_to_prev_frame(reg_settings, reg_cascade, other_probe);
            let other_probe_is_new_base_probe = !pgi_is_cell_base_probe(reg_settings, other_probe_prev_frame);
            if (other_probe_is_new_base_probe)
            {
                continue;
            }

            let other_probe_stable_index = pgi_probe_to_stable_index(reg_settings, reg_cascade, other_probe);

            uint request_package = push.attach.requests.get()[other_probe_stable_index];
            uint direct_request_timer = request_package & 0xFF;
            uint indirect_request_timer = (request_package >> 8) & 0xFF;
            let cell_directly_requested = direct_request_timer != 0;
            let cell_indirectly_requested = indirect_request_timer != 0;

            probe_directly_requested = probe_directly_requested || cell_directly_requested;
            probe_indirectly_requested = probe_indirectly_requested || cell_indirectly_requested;
            requested = requested || cell_directly_requested || cell_indirectly_requested;
        }

        if (requested)
        {
            InterlockedAdd(push.attach.probe_indirections.detailed_probe_count, 1, request_index);
        }

        requested = requested && (request_index < PGI_MAX_REQUESTED_PROBES);

        // Increase update rate for new probes. Without this, probes take too long to initialize after beeing unveiled.
        uint update_rate = probe_info.validity < 0.5f ? min(PGI_UPDATE_RATE_1_OF_2, reg_settings.update_rate) : reg_settings.update_rate;

        switch(update_rate)
        {
        case PGI_UPDATE_RATE_FULL:
            update = requested;
            break;
        case PGI_UPDATE_RATE_1_OF_2:
            int every_2nd = (probe_index.x & 0x1) ^ (probe_index.y & 0x1) ^ (probe_index.z & 0x1) ^ (push.attach.globals.frame_index & 0x1);
            update = requested && (every_2nd != 0);
            break;
        case PGI_UPDATE_RATE_1_OF_8:
            int every_8th = ((probe_index.x/2 & 0x1) + (probe_index.y/2 & 0x1) * 2 + (probe_index.z/2 & 0x1) * 4) == (push.attach.globals.frame_index % 8);
            update = requested && (every_8th != 0);
            break;
        case PGI_UPDATE_RATE_1_OF_16:
            int every_16th = ((probe_index.x/2 & 0x1) + (probe_index.y/2 & 0x1) * 2 + (probe_index.z/4 & 0x3) * 4) == (push.attach.globals.frame_index % 16);
            update = requested && (every_16th != 0);
            break;
        case PGI_UPDATE_RATE_1_OF_32:
            int every_32th = ((probe_index.x/2 & 0x1) + (probe_index.y/4 & 0x3) * 2 + (probe_index.z/4 & 0x3) * 8) == (push.attach.globals.frame_index % 32);
            update = requested && (every_32th != 0);
            break;
        case PGI_UPDATE_RATE_1_OF_64:
            int every_64th = ((probe_index.x/4 & 0x3) + (probe_index.y/4 & 0x3) * 4 + (probe_index.z/4 & 0x3) * 16) == (push.attach.globals.frame_index % 64);
            update = requested && (every_64th != 0);
            break;
        }

        if (update)
        {
            InterlockedAdd(push.attach.probe_indirections.probe_update_count, 1, update_index);
        }

        update = update && (update_index < PGI_MAX_UPDATES_PER_FRAME);
    }

    push.attach.requests.get()[stable_index] = probe_base_request | (uint(probe_directly_requested) << 16) | (uint(probe_indirectly_requested) << 17);

    if (update)
    {
        // We need a list of all updated probes 
        ((uint*)(push.attach.probe_indirections + 1))[update_index] = pgi_pack_indirect_probe(probe_index);
    }

    if (requested)
    {
        // We need a list of all active probes 
        ((uint*)(push.attach.probe_indirections + 1))[request_index + PGI_MAX_UPDATES_PER_FRAME] = pgi_pack_indirect_probe(probe_index);
    }

    if (!requested)
    {
        probe_info = {}; 
    }

    // Used for Debug Draws
    float3 probe_position = pgi_probe_index_to_worldspace(reg_settings, reg_cascade, probe_info, probe_index);

    // Debug Draw Probe Grid
    if (reg_settings.debug_draw_grid && (probe_info.validity != 0.0f))
    {
        for (int other_i = 0; other_i < 3; ++other_i)
        {
            int4 other_probe_index = probe_index;
            other_probe_index[other_i] += 1;
            
            let other_index_in_range = all(other_probe_index.xyz >= int3(0,0,0) && other_probe_index.xyz < reg_settings.probe_count);
            if (!other_index_in_range)
                continue;

            PGIProbeInfo other_info = PGIProbeInfo::load(reg_settings, reg_cascade, push.attach.probe_info.get(), other_probe_index);

            let other_valid = other_info.validity != 0.0f;
            if (other_valid)
            {
                float3 other_pos = pgi_probe_index_to_worldspace(reg_settings, reg_cascade, other_info, other_probe_index);
                ShaderDebugLineDraw line = {};
                line.start = probe_position;
                line.end = other_pos;
                line.color = Heatmap(float(cascade) * rcp(8));
                debug_draw_line(push.attach.globals.debug, line);
            }
        }
    }

    if (reg_settings.debug_draw_repositioning && (probe_info.validity != 0.0f))
    {
        ShaderDebugLineDraw line = {};
        line.start = probe_position;
        line.end = pgi_probe_index_to_worldspace(reg_settings, reg_cascade, PGIProbeInfo::null(), probe_index);
        line.color = float3(1,1,0);
        debug_draw_line(push.attach.globals.debug, line);
    }

    push.attach.probe_info.get()[stable_index] = float4(probe_info.offset, probe_info.validity);
    push.attach.probe_info_copy.get()[stable_index] = float4(probe_info.offset, probe_info.validity);

    if (group_index == 0)
    {
        uint finished_workgroups = 0;
        InterlockedAdd(*push.workgroups_finished, 1, finished_workgroups);
        finished_workgroups += 1;

        let last_to_finish = finished_workgroups == push.total_workgroups;
        if (last_to_finish)
        {
            let probe_update_count = min(PGI_MAX_UPDATES_PER_FRAME, push.attach.probe_indirections.probe_update_count);
            let detailed_probe_count = push.attach.probe_indirections.detailed_probe_count;
            let probe_update_workgroups_x = probe_update_count; //round_up_div(probe_update_count, 64);

            push.attach.globals.readback.requested_probes = probe_update_count;

            // One thread per texel, probes one after the other. Groups wrapped into rows to stay below the per dimension dispatch limit.
            const uint threads_per_group = uint(PGI_UPDATE_WG_SIZE);
            const uint groups_x_max = uint(PGI_UPDATE_LINEAR_WG_X);
            let color_groups = round_up_div(probe_update_count * uint(reg_settings.probe_color_resolution * reg_settings.probe_color_resolution), threads_per_group);
            let visibility_groups = round_up_div(probe_update_count * uint(reg_settings.probe_visibility_resolution * reg_settings.probe_visibility_resolution), threads_per_group);

            let shade_rays_threads = 0;//settings.probe_trace_resolution * settings.probe_trace_resolution * probe_update_count;
            let shade_rays_wgs = 0;//round_up_div(shade_rays_threads, PGI_SHADE_RAYS_X * PGI_SHADE_RAYS_Y * PGI_SHADE_RAYS_Z);

            push.attach.probe_indirections.probe_update_dispatch = DispatchIndirectStruct(probe_update_workgroups_x,1,1);
            push.attach.probe_indirections.probe_trace_dispatch = DispatchIndirectStruct(probe_update_count * reg_settings.probe_ray_count, 1, 1);
            push.attach.probe_indirections.probe_color_update_dispatch = DispatchIndirectStruct(min(color_groups, groups_x_max), round_up_div(color_groups, groups_x_max), 1);
            push.attach.probe_indirections.probe_visibility_update_dispatch = DispatchIndirectStruct(min(visibility_groups, groups_x_max), round_up_div(visibility_groups, groups_x_max), 1);
            // The trace integration runs one thread per trace texel (trace resolution), same linear layout as the texel updates.
            let trace_integrate_groups = round_up_div(probe_update_count * uint(reg_settings.probe_trace_resolution * reg_settings.probe_trace_resolution), threads_per_group);
            push.attach.probe_indirections.probe_trace_integrate_dispatch = DispatchIndirectStruct(min(trace_integrate_groups, groups_x_max), round_up_div(trace_integrate_groups, groups_x_max), 1);
            push.attach.probe_indirections.probe_shade_rays_dispatch = DispatchIndirectStruct(shade_rays_wgs, 1, 1);
            push.attach.probe_indirections.probe_debug_draw_dispatch = DrawIndexedIndirectStruct(
                PGI_DEBUG_PROBE_MESH_INDICES * 3,
                (reg_settings.debug_probe_draw_mode != 0 || reg_settings.probe_inspect_pick) ? detailed_probe_count : 0,
                0,0,0
            );
        }
    }
}

[[vk::push_constant]] PGITraceIntegratePush trace_integrate_push;

// One thread per trace texel, probes one after the other (same linear layout as the texel updates).
// Integrates the rays of the texel into the summary: mean radiance (rgb) and mean signed hit distance (a), exactly what the
// probe update passes read as the "ray" of the texel.
// Without guiding the rays of a texel is the one ray at its own atlas position, with guiding the rays [start, end) the probe ray cdf
// assigned to the texel (the cdf the trace of this frame used, the probe update has not rewritten it yet).
// Ray r is stored at the atlas position (r % trace_res, r / trace_res).
[shader("compute")]
[numthreads(PGI_UPDATE_WG_SIZE,1,1)]
func entry_pgi_trace_integrate(
    int3 gid : SV_GroupID,
    uint group_index : SV_GroupIndex)
{
    let push = trace_integrate_push;
    PGISettings* settings = &push.attach.globals.pgi_settings;
    let trace_res = settings->probe_trace_resolution;
    // Workgroups are wrapped into rows of PGI_UPDATE_LINEAR_WG_X.
    let flat_thread = (uint(gid.y) * PGI_UPDATE_LINEAR_WG_X + uint(gid.x)) * PGI_UPDATE_WG_SIZE + group_index;

    uint indirect_index = {};
    int2 trace_texel = {};
    let probe_update_count = min(PGI_MAX_UPDATES_PER_FRAME, push.attach.probe_indirections.probe_update_count);
    if (!pgi_update_thread_to_probe_texel(flat_thread, trace_res, probe_update_count, indirect_index, trace_texel))
    {
        return;
    }

    // The summary has one entry per trace texel and updated probe, the ray atlas one slot per ray (taller tiles with more than one ray per texel).
    let atlas_base = int3(int2(pgi_indirect_index_to_trace_tex_offset(*settings, indirect_index)), 0);
    let ray_atlas_base = int3(int2(pgi_indirect_index_to_ray_tex_offset(*settings, indirect_index)), 0);
    Texture2DArray<float4> rays = push.attach.trace_result.get();

    // The rays of this texel.
    uint ray_begin = uint(trace_texel.y * trace_res + trace_texel.x);
    uint ray_end = ray_begin + 1u;
    // Probability that the texel got a ray this frame (1 without guiding: every texel has its own ray).
    float coverage_probability = 1.0f;
    PGISettings reg_settings = *settings;
    if (pgi_ray_guiding_active(reg_settings))
    {
        uint indirect_package = ((uint*)(push.attach.probe_indirections + 1))[indirect_index];
        int4 probe_index = pgi_unpack_indirect_probe(indirect_package);
        PGICascade reg_cascade = settings->cascades[probe_index.w];
        let texel_count = trace_res * trace_res;
        uint* cdf = (uint*)(push.attach.probe_ray_cdf) + pgi_probe_stable_linear_index(reg_settings, reg_cascade, probe_index) * uint(texel_count);
        // A probe without a cdf (never written) was traced with the uniform cdf.
        {
            let cdf_valid = cdf[texel_count - 1] == PGI_RAY_CDF_ONE;
            let ray_count = reg_settings.probe_ray_count;
            let offset = pgi_ray_cdf_offset(pgi_probe_trace_noise(probe_index, int(push.attach.globals.frame_index)));
            let texel_index = int(ray_begin);
            ray_begin = pgi_ray_cdf_ray_start(texel_index > 0 ? uint64_t(pgi_ray_cdf_at(cdf, cdf_valid, texel_count, texel_index - 1)) * uint64_t(ray_count) : uint64_t(0), offset);
            ray_end = pgi_ray_cdf_ray_start(uint64_t(pgi_ray_cdf_at(cdf, cdf_valid, texel_count, texel_index)) * uint64_t(ray_count), offset);
            // Systematic sampling: a texel with ray_count * (its cdf share) < 1 gets a ray with exactly that probability.
            let cdf_high = pgi_ray_cdf_at(cdf, cdf_valid, texel_count, texel_index);
            let cdf_low = texel_index > 0 ? pgi_ray_cdf_at(cdf, cdf_valid, texel_count, texel_index - 1) : 0u;
            coverage_probability = clamp(float(cdf_high - cdf_low) * (1.0f / float(PGI_RAY_CDF_ONE)) * float(ray_count), 0.02f, 1.0f);
        }
    }

    float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    uint valid_ray_count = 0u;
    for (uint ray_index = ray_begin; ray_index < ray_end; ++ray_index)
    {
        let ray_y = int(ray_index) / trace_res;
        let ray = rays[ray_atlas_base + int3(int2(int(ray_index) - ray_y * trace_res, ray_y), 0)];
        // Only valid rays count (a non finite result would poison the mean).
        if (all(isfinite(ray)))
        {
            sum += ray;
            valid_ray_count += 1u;
        }
    }
    // A texel without a valid ray has no data: count 0, the probe update passes skip it.
    let mean = valid_ray_count > 0u ? sum * rcp(float(valid_ray_count)) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    push.attach.trace_summary.get()[atlas_base + int3(trace_texel, 0)] = mean;
    // Low 16 bits: valid ray count. High 16 bits: the probability the texel was covered (unorm), the irradiance update weights covered texels by 1 / it.
    // Without it texels that got no ray are dropped, which biases the estimate towards the texels the cdf favors (the bright ones).
    push.attach.trace_summary_count.get()[atlas_base + int3(trace_texel, 0)] = valid_ray_count > 0u ? (min(valid_ray_count, 0xFFFFu) | (uint(coverage_probability * 65535.0f + 0.5f) << 16u)) : 0u;
}

[[vk::push_constant]] PGIInspectPush pgi_inspect_push;

groupshared uint gs_inspect_update_index;
groupshared uint gs_inspect_new_probe;
groupshared uint gs_inspect_ray_count;
groupshared float3 gs_inspect_ray_origin;

// Probe inspection, one workgroup. Copies the selected probe (texels, ray cdf) into the host visible inspection buffer
// and keeps the last traced rays of the probe. The probe is only traced on the frames it is updated, the rays of
// the last trace stay in the buffer and are drawn every frame.
[shader("compute")]
[numthreads(PGI_INSPECT_WG_SIZE,1,1)]
func entry_pgi_inspect(int group_index : SV_GroupIndex)
{
    let push = pgi_inspect_push;
    PGISettings reg_settings = push.attach.globals.pgi_settings;
    PGIInspectData* data = push.attach.inspect;
    let grid_coord = int4(push.grid_coord[0], push.grid_coord[1], push.grid_coord[2], push.grid_coord[3]);
    let frame_index = push.attach.globals.frame_index;

    if (grid_coord.w < 0 || grid_coord.w >= reg_settings.cascade_count)
    {
        return;
    }
    PGICascade reg_cascade = reg_settings.cascades[grid_coord.w];
    // The selection is the absolute probe grid coordinate, so it stays on the same probe while the window scrolls:
    // grid coordinate = window index + window offset - count / 2 + 1.
    let window_index = grid_coord.xyz - reg_cascade.window_to_stable_index_offset + reg_settings.probe_count / 2 - 1;
    let probe_index = int4(window_index, grid_coord.w);
    let in_window = all(window_index >= int3(0,0,0)) && all(window_index < reg_settings.probe_count);
    if (!in_window)
    {
        if (group_index == 0)
        {
            data->grid_coord = grid_coord;
            data->in_window = 0u;
            data->valid = 1u;
            data->ray_frame = 0xFFFFFFFFu;
            data->ray_count = 0u;
        }
        return;
    }

    let res = reg_settings.probe_color_resolution;
    let texel_count = min(res * res, PGI_INSPECT_MAX_TEXELS);
    let trace_res = reg_settings.probe_trace_resolution;
    let ray_count = min(int(reg_settings.probe_ray_count), PGI_INSPECT_MAX_TEXELS);

    PGIProbeInfo probe_info = PGIProbeInfo::load(reg_settings, reg_cascade, push.attach.probe_info.get(), probe_index);
    let probe_position = pgi_probe_index_to_worldspace(reg_settings, reg_cascade, probe_info, probe_index);

    // Find out if the probe was traced this frame (it is in the update list) and if it is a different probe than the last frame's.
    if (group_index == 0)
    {
        gs_inspect_update_index = 0xFFFFFFFFu;
        let stored = data->grid_coord;
        gs_inspect_new_probe = (data->valid == 0u || data->in_window == 0u || any(stored != grid_coord)) ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    let update_count = min(PGI_MAX_UPDATES_PER_FRAME, push.attach.probe_indirections.probe_update_count);
    let packed_probe = pgi_pack_indirect_probe(probe_index);
    for (uint i = uint(group_index); i < update_count; i += PGI_INSPECT_WG_SIZE)
    {
        if (((uint*)(push.attach.probe_indirections + 1))[i] == packed_probe)
        {
            InterlockedMin(gs_inspect_update_index, i);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    let found = gs_inspect_update_index != 0xFFFFFFFFu;
    let new_probe = gs_inspect_new_probe != 0u;
    if (group_index == 0)
    {
        gs_inspect_ray_count = found ? uint(ray_count) : (new_probe ? 0u : data->ray_count);
        gs_inspect_ray_origin = found ? probe_position : data->ray_origin;
    }
    GroupMemoryBarrierWithGroupSync();
    let draw_ray_count = int(gs_inspect_ray_count);
    let ray_origin = gs_inspect_ray_origin;

    // Texels: irradiance (with hysteresis in .a), radiance, ray cdf.
    {
        int3 radiance_base = pgi_probe_texture_base_offset<HAS_BORDER>(reg_settings, reg_cascade, res, probe_index);
        int3 irradiance_base = radiance_base;
        radiance_base.z += reg_settings.cascade_count * reg_settings.probe_count.z;
        uint* cdf = (uint*)(push.attach.probe_ray_cdf) + pgi_probe_stable_linear_index(reg_settings, reg_cascade, probe_index) * uint(res * res);
        Texture2DArray<float4> color_tex = push.attach.probe_color.get();
        for (int i = group_index; i < texel_count; i += PGI_INSPECT_WG_SIZE)
        {
            int y = i / res;
            int2 texel = int2(i - y * res, y);
            data->irradiance[i] = color_tex[irradiance_base + int3(texel, 0)];
            data->radiance[i] = color_tex[radiance_base + int3(texel, 0)];
            data->cdf[i] = cdf[i];
        }
    }

    // Request counters of the cells this probe is a vertex of, and why the probe would be deleted (read after this frame's pre update and sampling).
    if (group_index == 0)
    {
        Texture2DArray<uint> requests_tex = push.attach.probe_requests.get();
        bool any_requested = false;
        for (int c = 0; c < 8; ++c)
        {
            let other_probe = int4(max(int3(0, 0, 0), probe_index.xyz - int3(c & 1, (c >> 1) & 1, (c >> 2) & 1)), probe_index.w);
            uint direct_timer = 0xFFFFFFFFu;
            uint indirect_timer = 0xFFFFFFFFu;
            if (pgi_is_cell_base_probe(reg_settings, other_probe))
            {
                let package = requests_tex[pgi_probe_to_stable_index(reg_settings, reg_cascade, other_probe)];
                direct_timer = package & 0xFFu;
                indirect_timer = (package >> 8u) & 0xFFu;
                any_requested = any_requested || direct_timer != 0u || indirect_timer != 0u;
            }
            data->request_cell_direct[c] = direct_timer;
            data->request_cell_indirect[c] = indirect_timer;
        }
        data->request_package = requests_tex[pgi_probe_to_stable_index(reg_settings, reg_cascade, probe_index)];
        data->requested = any_requested ? 1u : 0u;
        let prev_frame_probe_index = pgi_probe_index_to_prev_frame(reg_settings, reg_cascade, probe_index);
        data->is_new_probe = (any(prev_frame_probe_index.xyz < int3(0, 0, 0)) || any(prev_frame_probe_index.xyz >= reg_settings.probe_count)) ? 1u : 0u;
    }

    // Visibility texels: mean ray distance (depth) and its deviation, at the visibility resolution.
    {
        let vis_res = reg_settings.probe_visibility_resolution;
        let vis_texel_count = min(vis_res * vis_res, PGI_INSPECT_MAX_TEXELS);
        int3 visibility_base = pgi_probe_texture_base_offset<HAS_BORDER>(reg_settings, reg_cascade, vis_res, probe_index);
        Texture2DArray<float2> visibility_tex = push.attach.probe_visibility.get();
        for (int i = group_index; i < vis_texel_count; i += PGI_INSPECT_WG_SIZE)
        {
            int y = i / vis_res;
            int2 texel = int2(i - y * vis_res, y);
            data->visibility[i] = visibility_tex[visibility_base + int3(texel, 0)];
        }
    }

    // Rays: fresh from the trace result when the probe was traced this frame, else the stored ones. Every ray is drawn by the thread that writes it.
    {
        Texture2DArray<float4> trace_result_tex = push.attach.trace_result.get();
        uint3 trace_base = uint3(0,0,0);
        if (found)
        {
            trace_base = uint3(pgi_indirect_index_to_ray_tex_offset(reg_settings, gs_inspect_update_index), 0);
        }
        let exposure = push.attach.globals.exposure;
        for (int r = group_index; r < draw_ray_count; r += PGI_INSPECT_WG_SIZE)
        {
            PGIInspectRay ray = data->rays[r];
            if (found)
            {
                int y = r / trace_res;
                int x = r - y * trace_res;
                // The inspection runs before the probe update rewrites the cdf, it is the cdf this frame's trace used.
                let uv = pgi_probe_ray_uv(reg_settings, reg_cascade, (uint*)(push.attach.probe_ray_cdf), probe_index, frame_index, uint(r));
                float4 trace = trace_result_tex[trace_base + uint3(x, y, 0)];
                ray.direction = pgi_probe_uv_to_probe_normal(uv);
                ray.distance = trace.a;
                ray.radiance = trace.rgb;
                data->rays[r] = ray;
            }
            if (push.show_rays != 0u)
            {
                let is_backface = ray.distance < 0.0f;
                let length = min(abs(ray.distance), reg_cascade.max_visibility_distance * 1.5f);
                ShaderDebugLineDraw line = {};
                line.start = ray_origin;
                line.end = ray_origin + ray.direction * length;
                line.color = is_backface ? float3(0.6f, 0.0f, 0.0f) : clamp(ray.radiance * exposure, float3(0.03f), float3(1.0f));
                debug_draw_line(push.attach.globals.debug, line);
            }
        }
    }

    if (group_index == 0)
    {
        // Marker on the selected probe, always drawn (independent of show rays).
        ShaderDebugCircleDraw marker = {};
        marker.position = probe_position;
        marker.color = float3(1.0f, 1.0f, 1.0f);
        marker.radius = 0.04f * reg_cascade.max_visibility_distance;
        debug_draw_circle(push.attach.globals.debug, marker);
    }

    if (group_index == 0)
    {
        data->grid_coord = grid_coord;
        data->in_window = 1u;
        data->valid = 1u;
        data->write_frame = frame_index;
        data->probe_position = probe_position;
        data->validity = probe_info.validity;
        data->max_visibility_distance = reg_cascade.max_visibility_distance;
        data->color_resolution = uint(res);
        data->trace_resolution = uint(trace_res);
        data->visibility_resolution = uint(reg_settings.probe_visibility_resolution);
        if (found)
        {
            data->ray_frame = frame_index;
            data->ray_count = uint(ray_count);
            data->ray_origin = probe_position;
        }
        else if (new_probe)
        {
            data->ray_frame = 0xFFFFFFFFu;
            data->ray_count = 0u;
            data->ray_origin = probe_position;
        }
    }
}