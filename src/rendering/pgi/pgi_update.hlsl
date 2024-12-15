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
#include "../../shader_lib/shading.hlsl"
#include "../../shader_lib/raytracing.hlsl"
#include "../../shader_lib/SH.hlsl"

[[vk::push_constant]] PGIUpdateProbesPush update_probes_push;

[shader("compute")]
[numthreads(PGI_UPDATE_WG_XY,PGI_UPDATE_WG_XY,PGI_UPDATE_WG_Z)]
func entry_update_probes(
    int3 dtid : SV_DispatchThreadID,
) 
{
    let push = update_probes_push;
    PGISettings settings = push.attach.globals.pgi_settings;

    let probe_texel_res = settings.probe_surface_resolution;
    
    // Runtime Int Divs are terrible horror but they stay until i am done prototyping.
    let probe_texel = (dtid.xy % probe_texel_res);
    let probe_index = int3(dtid.xy / probe_texel_res, dtid.z);
    uint frame_index = push.attach.globals.frame_index;

    if (any(greaterThanEqual(probe_index, settings.probe_count)))
    {
        return;
    }
    
    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : push.attach.globals.camera.position;
    float3 probe_position = pgi_probe_index_to_worldspace(push.attach.globals.pgi_settings, probe_anchor, probe_index);
    float2 probe_texel_uv = (float2(probe_texel) + 0.5f) * rcp(probe_texel_res);
    float3 probe_texel_normal = pgi_probe_uv_to_probe_normal(probe_texel_uv);

    int3 probe_texture_base_index = pgi_probe_texture_base_offset(settings, probe_texel_res, probe_index);
    int3 probe_texture_index = probe_texture_base_index + int3(probe_texel, 0);

    int s = settings.probe_trace_resolution;

    float4 cosine_convoluted_trace_result = float4(0.0f,0.0f,0.0f, 0.0f);
    int3 trace_result_texture_base_index = pgi_probe_texture_base_offset(settings, settings.probe_trace_resolution, probe_index);
    float2 trace_texel_noise = pgi_probe_trace_noise(probe_index, frame_index); // used to reconstruct directions used for traces.
    float acc_weight = 0.0f;
    float cos_wrap_around = settings.cos_wrap_around;
    float cos_wrap_around_rcp = rcp(cos_wrap_around + 1.0f);
    for (int y = 0; y < s; ++y)
    {
        for (int x = 0; x < s; ++x)
        {
            float2 trace_tex_uv = clamp((float2(x,y) + trace_texel_noise) * rcp(s), 0.0f, 0.999999f);
            float3 trace_direction = pgi_probe_uv_to_probe_normal(trace_tex_uv); // Trace direction is identical to the one used in tracer.
            float cos_weight = max(0.0f, (cos_wrap_around + dot(trace_direction, probe_texel_normal)) * cos_wrap_around_rcp);
            int3 sample_texture_index = trace_result_texture_base_index + int3(x,y,0);
            if (cos_weight > 0.0f)
            {
                float4 sample = push.attach.trace_result.get()[sample_texture_index].rgba;
                cosine_convoluted_trace_result += sample * cos_weight;
                acc_weight += cos_weight;

                #if 0 // draw probe influence
                //if (all(probe_index == settings.debug_probe_index))
                if (probe_texel.y == 0 && probe_texel.x == 0 && update_step == 1){
                    ShaderDebugLineDraw line = {};
                    line.start = probe_position;
                    line.end = probe_position + trace_direction * cos_weight;
                    line.color = sample.rgb;
                    debug_draw_line(push.attach.globals.debug, line);
                }
                #endif
            }
        }
    }
    
    // If we have 0 weight we have no samples to blend so we dont blend at all.
    float update_factor = 0.0f;
    if (acc_weight > 0.0f)
    {
        cosine_convoluted_trace_result *= rcp(acc_weight);
        update_factor = 0.01f;
    }

    float3 new_radiance = cosine_convoluted_trace_result.rgb;
    float3 prev_frame_radiance = push.attach.probe_radiance.get()[probe_texture_index].rgb;
    new_radiance = lerp(prev_frame_radiance, new_radiance, update_factor);
    push.attach.probe_radiance.get()[probe_texture_index].rgb = new_radiance;
}

[shader("compute")]
[numthreads(PGI_UPDATE_WG_XY,PGI_UPDATE_WG_XY,PGI_UPDATE_WG_Z)]
func entry_update_probe_visibility(
    int3 dtid : SV_DispatchThreadID,
) 
{
    let push = update_probes_push;
    PGISettings settings = push.attach.globals.pgi_settings;

    let probe_texel_res = settings.probe_visibility_resolution;
    
    // Runtime Int Divs are terrible horror but they stay until i am done prototyping.
    let probe_texel = (dtid.xy % probe_texel_res);
    let probe_index = int3(dtid.xy / probe_texel_res, dtid.z);
    uint frame_index = push.attach.globals.frame_index;

    if (any(greaterThanEqual(probe_index, settings.probe_count)))
    {
        return;
    }
    
    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : push.attach.globals.camera.position;
    float3 probe_position = pgi_probe_index_to_worldspace(push.attach.globals.pgi_settings, probe_anchor, probe_index);
    float2 probe_texel_min_uv = (float2(probe_texel)) * rcp(probe_texel_res);
    float2 probe_texel_max_uv = (float2(probe_texel) + 1.0f) * rcp(probe_texel_res);
    float3 probe_texel_normal = pgi_probe_uv_to_probe_normal((probe_texel_max_uv + probe_texel_min_uv) * 0.5f);

    int3 probe_texture_base_index = pgi_probe_texture_base_offset(settings, probe_texel_res, probe_index);
    
    int3 probe_texture_index = probe_texture_base_index + int3(probe_texel, 0);

    float2 relevant_trace_blend = float2(0.0f, 0.0f);
    int valid_trace_count = 0;
    int3 trace_result_texture_base_index = pgi_probe_texture_base_offset(settings, settings.probe_trace_resolution, probe_index);
    float2 trace_texel_noise = pgi_probe_trace_noise(probe_index, frame_index); // used to reconstruct directions used for traces.
    const float max_depth = length(settings.probe_range * rcp(settings.probe_count))* 1.01f;
    float2 prev_frame_visibility = push.attach.probe_visibility.get()[probe_texture_index];
    float acc_cos_weights = 0.0f;
    int s = settings.probe_trace_resolution;
    bool deactivate = false;
    for (int y = 0; y < s; ++y)
    for (int x = 0; x < s; ++x)
    {
        float2 trace_tex_uv = clamp((float2(x,y) + trace_texel_noise) * rcp(s), 0.0f, 0.999999f);
        float3 trace_direction = pgi_probe_uv_to_probe_normal(trace_tex_uv); // Trace direction is identical to the one used in tracer.
        float cos_weight = (dot(trace_direction, probe_texel_normal));
        int3 sample_texture_index = trace_result_texture_base_index + int3(x,y,0);
        if (cos_weight > 0.0f)
        {
            float power_cos_weight = pow(cos_weight, 200.0f);
            float trace_depth = push.attach.trace_result.get()[sample_texture_index].a;
            bool is_backface = trace_depth < 0.0f;
            bool is_backface_relevant = abs(trace_depth) < max_depth;
            trace_depth = abs(trace_depth);

            deactivate = deactivate || (trace_depth < 0.05f);
            if (trace_depth < max_depth)
            {
                if (is_backface && is_backface_relevant) // Backface probe killer.
                { 
                    relevant_trace_blend.x += -10000.0f * cos_weight;
                    relevant_trace_blend.y += 0.00001f * cos_weight;
                    acc_cos_weights += 100000 * cos_weight;
                }
                else
                {
                    relevant_trace_blend.x += trace_depth * power_cos_weight;
                    relevant_trace_blend.y += square(abs(max(0.0f, prev_frame_visibility.x) - trace_depth)) * power_cos_weight;
                    acc_cos_weights += power_cos_weight;
                }
                valid_trace_count += 1;
            }
        }
    }
    
    if (valid_trace_count > 0 && !deactivate)
    {
        relevant_trace_blend *= rcp(acc_cos_weights);

        float blending = 0.001f;
        float2 new_blended_val = lerp(prev_frame_visibility, relevant_trace_blend, blending);
        push.attach.probe_visibility.get()[probe_texture_index] = new_blended_val;

        if (all(probe_texel == int2(0,0)) && false)
        {
            float3 trace_direction = pgi_probe_uv_to_probe_normal((probe_texel + 0.5f) * rcp(settings.probe_visibility_resolution));
            ShaderDebugLineDraw line = {};
            line.start = probe_position;
            line.end = probe_position + new_blended_val.x * trace_direction;
            line.color = float3(1,1,1);
            debug_draw_line(push.attach.globals.debug, line);
            ShaderDebugCircleDraw hit = {};
            hit.position = probe_position + trace_direction * new_blended_val.x;
            hit.color = float3(0,1,1);
            hit.radius = 0.1f;
            debug_draw_circle(push.attach.globals.debug, hit);
            float std_dev = sqrt(abs(new_blended_val.x*new_blended_val.x - new_blended_val.y));
            ShaderDebugCircleDraw std_dev_point = {};
            std_dev_point.position = probe_position + trace_direction * (new_blended_val.x - std_dev);
            std_dev_point.color = float3(1,0,1);
            std_dev_point.radius = 0.1f;
            debug_draw_circle(push.attach.globals.debug, std_dev_point);
        }
    }
}