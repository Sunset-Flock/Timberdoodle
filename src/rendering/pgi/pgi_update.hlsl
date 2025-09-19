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

func abs_radiance(float3 radiance) -> float
{
    return radiance.r + radiance.g * 2.0f + radiance.b * 0.5f;
}

#define PGI_PERCEPTUAL_EXPONENT 4.0f
func perceptual_lerp3(float3 a, float3 b, float v) -> float3
{
    return pow(lerp(pow(a + 0.0000001f, (1.0f/PGI_PERCEPTUAL_EXPONENT)), pow(b + 0.0000001f, (1.0f/PGI_PERCEPTUAL_EXPONENT)), v), PGI_PERCEPTUAL_EXPONENT);
}

func entry_update_probe_color(
    int3 dtid,
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
        uint probes_in_column = uint(float(48) * rcp(float(reg_settings.probe_color_resolution)));
        uint indirect_index_y = uint(float(dtid.y) * rcp(float(reg_settings.probe_color_resolution)));
        uint indirect_index_x = uint(float(dtid.x) * rcp(float(reg_settings.probe_color_resolution)));
        indirect_index = indirect_index_x * probes_in_column + indirect_index_y;
        let is_overhang = indirect_index >= push.attach.probe_indirections.probe_update_count;
        if (is_overhang)
        {
            return;
        }
        uint indirect_package = ((uint*)(push.attach.probe_indirections + 1))[indirect_index];
        probe_index = pgi_unpack_indirect_probe(indirect_package);
        probe_texel.x = dtid.x - int(reg_settings.probe_color_resolution * indirect_index_x);
        probe_texel.y = dtid.y - int(reg_settings.probe_color_resolution * indirect_index_y);
    }
    PGICascade reg_cascade = settings->cascades[probe_index.w];
    
    const uint frame_index = push.attach.globals.frame_index;
    if (any(greaterThanEqual(probe_index.xyz, reg_settings.probe_count)))
    {
        return;
    }
    
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

    float3 radiance = float3(0.0f,0.0f,0.0f);
    float radiance_weight = 0.0f;
    for (int y = 0; y < s; ++y)
    for (int x = 0; x < s; ++x)
    {
        float2 trace_tex_uv = (float2(x,y) + trace_texel_noise) * rcp_s;
        float3 trace_direction = pgi_probe_uv_to_probe_normal(trace_tex_uv); // Trace direction is identical to the one used in tracer.
        float cos_weight = max(0.0f, dot(trace_direction, probe_texel_normal));
        int3 sample_texture_index = trace_result_texture_base_index + int3(x,y,0);
        // If statement on cos weight would REDUCE PERFORMANCE. Reads in branches lead to poor latency hiding due to long scoreboard stalls.
        {
            float4 sample = trace_result_tex[sample_texture_index].rgba;

            // Firely filter
            if (true) 
            {
                const float firely_clamp_threshold_ratio = 64.0f;
                const float relative_radiance_difference = abs_radiance(sample.rgb) / ( 0.000001f + abs_radiance(prev_frame_irradiance) );
                const float supression_factor = min(1.0f, firely_clamp_threshold_ratio / relative_radiance_difference );
                sample.rgb *= supression_factor;
            }
            
            const bool backface_hit = sample.a < 0.0f;
            if (!backface_hit)
            {
                cosine_convoluted_trace_result += sample * cos_weight;
                acc_weight += cos_weight;
            }

            const bool trace_within_probe_texel = all(trace_tex_uv >= probe_texel_min_uv && trace_tex_uv <= probe_texel_max_uv);
            if (trace_within_probe_texel)
            {
                radiance += sample.rgb;
                radiance_weight += 1.0f;
            }
        }

    }

    radiance *= rcp(radiance_weight);
    
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
    {
        const float HYSTERESIS_UPDATE_RATE = 0.05f;
        const float HYSTERESIS_MIN_RELATIVE_CHANGE = 2.0f;
        const float MAX_MAX_RELATIVE_DIFFERENCE = 3.0f;

        const float3 power_scaled_min = min(prev_frame_irradiance, new_irradiance);
        const float3 relative_difference = abs(prev_frame_irradiance - new_irradiance) / power_scaled_min;
        const float max_relative_difference = min(max3(relative_difference.x, relative_difference.y, relative_difference.z), MAX_MAX_RELATIVE_DIFFERENCE);
        const float BASE_CONFIDENCE_GAIN = HYSTERESIS_UPDATE_RATE;
        const float RELATIVE_DIFFERENCE_SCALING = BASE_CONFIDENCE_GAIN / HYSTERESIS_MIN_RELATIVE_CHANGE;
        hysteresis += -max_relative_difference * RELATIVE_DIFFERENCE_SCALING + BASE_CONFIDENCE_GAIN;
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
    new_irradiance = perceptual_lerp3(new_irradiance, prev_frame_irradiance, blend);
    new_irradiance = max(new_irradiance, float3(0,0,0)); // remove nans, what can i say...

    // Perceptual lerp helps with blending to dark values
    const float radiance_blend = probe_info.validity < 0.5f ? 0.0f : 0.5f;
    const float3 prev_exact_radiance = push.attach.probe_color.get()[probe_texture_index + int3(0, 0, reg_settings.cascade_count * reg_settings.probe_count.z)].rgb;
    float3 new_radiance = perceptual_lerp3(radiance, prev_exact_radiance, radiance_blend);
    new_radiance = max(new_radiance, float3(0,0,0)); // remove nans, what can i say...

    write_probe_texel_with_border(push.attach.probe_color.get(), reg_settings.probe_color_resolution, probe_texture_base_index, probe_texel, float4(new_irradiance, hysteresis));
    write_probe_texel_with_border(push.attach.probe_color.get(), reg_settings.probe_color_resolution, probe_texture_base_index, probe_texel, float4(new_radiance, 1.0f), reg_settings.cascade_count * settings->probe_count.z);
}

func entry_update_probe_visibility(
    int3 dtid,
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
        uint probes_in_column = uint(float(48) * rcp(float(reg_settings.probe_visibility_resolution)));
        uint indirect_index_y = uint(float(dtid.y) * rcp(float(reg_settings.probe_visibility_resolution)));
        uint indirect_index_x = uint(float(dtid.x) * rcp(float(reg_settings.probe_visibility_resolution)));
        indirect_index = indirect_index_x * probes_in_column + indirect_index_y;
        let is_overhang = indirect_index >= push.attach.probe_indirections.probe_update_count;
        if (is_overhang)
        {
            return;
        }
        uint indirect_package = ((uint*)(push.attach.probe_indirections + 1))[indirect_index];
        probe_index = pgi_unpack_indirect_probe(indirect_package);
        probe_texel.x = dtid.x - reg_settings.probe_visibility_resolution * indirect_index_x;
        probe_texel.y = dtid.y - reg_settings.probe_visibility_resolution * indirect_index_y;
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

    #if defined(DEBUG_PROBE_TEXEL_UPDATE)
    bool debug_mode = any(reg_settings.debug_probe_index != 0);
    if (debug_mode && ((push.attach.globals.frame_index % 256) == 0))
    {
        push.attach.probe_visibility.get()[probe_texture_base_index + int3(probe_texel, 0)] = float2(0,0);
        return;
    }
    bool debug_texel = (all(reg_settings.debug_probe_index.xy == probe_texel));
    if (debug_mode && !debug_texel)
    {
        return;
    }
    #endif

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
    #if defined(DEBUG_PROBE_TEXEL_UPDATE)
        if (debug_mode && debug_texel)
        {
            push.attach.probe_visibility.get()[probe_texture_base_index + int3(probe_trace_tex_index,0)] = float2(0.05,power_cos_weight);
        }
    #endif
        // If statement on cos weight would REDUCE PERFORMANCE. Reads in branches lead to poor latency hiding due to long scoreboard stalls.
        {
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
[numthreads(PGI_UPDATE_WG_XY,PGI_UPDATE_WG_XY,PGI_UPDATE_WG_Z)]
func entry_update_probe_texels(
    int3 dtid : SV_DispatchThreadID,
) 
{
    let push = update_probe_texels_push;
    if (push.update_radiance)
    {
        entry_update_probe_color(dtid);
    }
    else
    {
        entry_update_probe_visibility(dtid);
    }
}

[[vk::push_constant]] PGIUpdateProbesPush update_probes_push;

#define PGI_DESIRED_RELATIVE_DISTANCE 0.3f 
#define PGI_RELATIVE_REPOSITIONING_STEP 0.2f
#define PGI_MAX_RELATIVE_REPOSITIONING 0.4f
#define PGI_ACCEPTABLE_SURFACE_DISTANCE (PGI_DESIRED_RELATIVE_DISTANCE * 0.3)
#define PGI_BACKFACE_ESCAPE_RANGE (PGI_DESIRED_RELATIVE_DISTANCE * 3)
#define PGI_PROBE_VIEW_DISTANCE 1.0

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
    
    for (int wave_i = 0; wave_i < wave_iterations; ++wave_i)
    {
        int i = wave_i * WARP_SIZE + group_index;
        int y = int(float(i) * rcp(float(s))); // Can not be negative, no need to clamp.
        int x = i - y * s;
        if (i >= (s*s))
        {
            continue;
        }
        int2 probe_local_texel = int2(x,y);
        int3 texel = trace_base_texel + int3(x,y,0);
        float2 uv = (float2(probe_local_texel) + texel_trace_noise) * trace_res_rcp;
        float3 ray_dir = pgi_probe_uv_to_probe_normal(uv);
        
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

    average_hit_offset *= rcp(float(s*s));
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

    probe_info.validity += 0.05f;

    // Calculate backface attraction
    bool too_few_backface_hits = backface_count < ceil(float(s*s) * 0.1f);
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
            probe_indirectly_requested = probe_indirectly_requested || cell_directly_requested;
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

    push.attach.requests.get()[stable_index] = probe_base_request | (uint(probe_directly_requested) << 16) | (uint(probe_directly_requested) << 17);

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
                line.color = TurboColormap(float(cascade) * rcp(8));
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

            // Allows for full thread utilization for relevant texel resolutions:
            // 1x1, 2x2, 4x4, 6x6, 8x8, 12x12, 16x16, 24x24, 48x48
            let texel_update_threads_y = 48;
            let texel_update_workgroups_y = 6; // 48/WG_Y = 48/8 = 6

            let radiance_texel_update_probes_y = texel_update_threads_y / reg_settings.probe_color_resolution;
            let radiance_texel_update_probes_x = round_up_div(probe_update_count, radiance_texel_update_probes_y);
            let radiance_texel_update_threads_x = radiance_texel_update_probes_x * reg_settings.probe_color_resolution;
            let radiance_texel_update_workgroups_x = round_up_div(radiance_texel_update_threads_x, 8);

            let visibility_texel_update_probes_y = texel_update_threads_y / reg_settings.probe_visibility_resolution;
            let visibility_texel_update_probes_x = round_up_div(probe_update_count, visibility_texel_update_probes_y);
            let visibility_texel_update_threads_x = visibility_texel_update_probes_x * reg_settings.probe_visibility_resolution;
            let visibility_texel_update_workgroups_x = round_up_div(visibility_texel_update_threads_x, 8);

            let shade_rays_threads = 0;//settings.probe_trace_resolution * settings.probe_trace_resolution * probe_update_count;
            let shade_rays_wgs = 0;//round_up_div(shade_rays_threads, PGI_SHADE_RAYS_X * PGI_SHADE_RAYS_Y * PGI_SHADE_RAYS_Z);

            push.attach.probe_indirections.probe_update_dispatch = DispatchIndirectStruct(probe_update_workgroups_x,1,1);
            push.attach.probe_indirections.probe_trace_dispatch = DispatchIndirectStruct(probe_update_count * reg_settings.probe_trace_resolution * reg_settings.probe_trace_resolution, 1, 1);
            push.attach.probe_indirections.probe_color_update_dispatch = DispatchIndirectStruct(radiance_texel_update_workgroups_x, texel_update_workgroups_y, 1);
            push.attach.probe_indirections.probe_visibility_update_dispatch = DispatchIndirectStruct(visibility_texel_update_workgroups_x, texel_update_workgroups_y, 1);
            push.attach.probe_indirections.probe_shade_rays_dispatch = DispatchIndirectStruct(shade_rays_wgs, 1, 1);
            push.attach.probe_indirections.probe_debug_draw_dispatch = DrawIndexedIndirectStruct(
                PGI_DEBUG_PROBE_MESH_INDICES * 3,
                (reg_settings.debug_probe_draw_mode != 0) ? detailed_probe_count : 0,
                0,0,0
            );
        }
    }
}