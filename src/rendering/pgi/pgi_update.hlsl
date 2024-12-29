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

[[vk::push_constant]] PGIUpdateProbeTexelsPush update_probe_texels_push;

[shader("compute")]
[numthreads(PGI_UPDATE_WG_XY,PGI_UPDATE_WG_XY,PGI_UPDATE_WG_Z)]
func entry_update_probe_irradiance(
    int3 dtid : SV_DispatchThreadID,
) 
{
    let push = update_probe_texels_push;
    PGISettings settings = push.attach.globals.pgi_settings;

    let probe_texel_res = settings.probe_radiance_resolution;
    
    // Runtime Int Divs are terrible horror but they stay until i am done prototyping.
    let probe_index = int3(float2(dtid.xy) * rcp(probe_texel_res), dtid.z);
    let probe_texel = dtid.xy - probe_index.xy * probe_texel_res;
    uint frame_index = push.attach.globals.frame_index;

    if (any(greaterThanEqual(probe_index, settings.probe_count)))
    {
        return;
    }
    
    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : push.attach.globals.camera.position;
    PGIProbeInfo probe_info = PGIProbeInfo::load(push.attach.probe_info.get(), probe_index);
    float3 probe_position = pgi_probe_index_to_worldspace(push.attach.globals.pgi_settings, probe_info, probe_anchor, probe_index);
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
    for (int x = 0; x < s; ++x)
    {
        float2 trace_tex_uv = (float2(x,y) + trace_texel_noise) * rcp(s);
        float3 trace_direction = pgi_probe_uv_to_probe_normal(trace_tex_uv); // Trace direction is identical to the one used in tracer.
        float cos_weight = max(0.0f, (cos_wrap_around + dot(trace_direction, probe_texel_normal)) * cos_wrap_around_rcp);
        int3 sample_texture_index = trace_result_texture_base_index + int3(x,y,0);
        if (cos_weight > 0.0f)
        {
            float4 sample = push.attach.trace_result.get()[sample_texture_index].rgba;
            cosine_convoluted_trace_result += sample * cos_weight;
            acc_weight += cos_weight;
        }
    }
    
    // If we have 0 weight we have no samples to blend so we dont blend at all.
    float update_factor = 0.0f;
    if (acc_weight > 0.0f)
    {
        cosine_convoluted_trace_result *= rcp(acc_weight);
        update_factor = 0.01f;

        float ray_to_texel_ratio = float(settings.probe_trace_resolution) / float(settings.probe_radiance_resolution);
        update_factor *= ray_to_texel_ratio;
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
    let push = update_probe_texels_push;
    PGISettings settings = push.attach.globals.pgi_settings;

    let probe_texel_res = settings.probe_visibility_resolution;
    
    // Runtime Int Divs are terrible horror but they stay until i am done prototyping.
    let probe_index = int3(float2(dtid.xy) * rcp(probe_texel_res), dtid.z);
    let probe_texel = dtid.xy - probe_index.xy * probe_texel_res;
    uint frame_index = push.attach.globals.frame_index;

    if (any(greaterThanEqual(probe_index, settings.probe_count)))
    {
        return;
    }
    
    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : push.attach.globals.camera.position;
    PGIProbeInfo probe_info = PGIProbeInfo::load(push.attach.probe_info.get(), probe_index);
    float3 probe_position = pgi_probe_index_to_worldspace(push.attach.globals.pgi_settings, probe_info, probe_anchor, probe_index);
    float2 probe_texel_min_uv = (float2(probe_texel)) * rcp(probe_texel_res);
    float2 probe_texel_max_uv = (float2(probe_texel) + 1.0f) * rcp(probe_texel_res);
    float3 probe_texel_normal = pgi_probe_uv_to_probe_normal((probe_texel_max_uv + probe_texel_min_uv) * 0.5f);

    int3 probe_texture_base_index = pgi_probe_texture_base_offset(settings, probe_texel_res, probe_index);
    
    int3 probe_texture_index = probe_texture_base_index + int3(probe_texel, 0);

    float2 relevant_trace_blend = float2(0.0f, 0.0f);
    int valid_trace_count = 0;
    int3 trace_result_texture_base_index = pgi_probe_texture_base_offset(settings, settings.probe_trace_resolution, probe_index);
    float2 trace_texel_noise = pgi_probe_trace_noise(probe_index, frame_index); // used to reconstruct directions used for traces.
    const float max_depth = settings.max_visibility_distance;
    float2 prev_frame_visibility = push.attach.probe_visibility.get()[probe_texture_index];
    float acc_cos_weights = 0.0f;
    int s = settings.probe_trace_resolution;
    for (int y = 0; y < s; ++y)
    for (int x = 0; x < s; ++x)
    {
        float2 trace_tex_uv = clamp((float2(x,y) + trace_texel_noise) * rcp(s), 0.0f, 0.999999f);
        float3 trace_direction = pgi_probe_uv_to_probe_normal(trace_tex_uv); // Trace direction is identical to the one used in tracer.
        float cos_weight = (dot(trace_direction, probe_texel_normal));
        int3 sample_texture_index = trace_result_texture_base_index + int3(x,y,0);
        if (cos_weight > 0.0f)
        {
            float power_cos_weight = pow(cos_weight, 10.0f);
            float trace_depth = push.attach.trace_result.get()[sample_texture_index].a;
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
                    relevant_trace_blend.y += difference_to_average * power_cos_weight;
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

        float ray_to_texel_ratio = float(settings.probe_trace_resolution) / float(settings.probe_visibility_resolution);
        update_factor *= ray_to_texel_ratio;

        float2 new_blended_val = lerp(prev_frame_visibility, relevant_trace_blend, update_factor);
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

[[vk::push_constant]] PGIUpdateProbesPush update_probes_push;

struct FreeSphere
{
    float3 center;
    float radius;
};

func update_free_sphere(inout FreeSphere sphere, float3 point)
{
    // Check if point is in sphere
    // Only update sphere for points that are within it.
    float point_sphere_dist = length(point - sphere.center) - sphere.radius;
    if (point_sphere_dist >= 0.0f)
    {
        return;
    }

    float3 point_to_center = normalize(sphere.center - point);
    float3 new_sphere_end = point_to_center * sphere.radius + sphere.center;
    float3 new_sphere_start = point;
    sphere.center = (new_sphere_start + new_sphere_end) * 0.5f;
    sphere.radius = length(new_sphere_start - new_sphere_end) * 0.5f;
}

[shader("compute")]
[numthreads(PGI_UPDATE_WG_XY,PGI_UPDATE_WG_XY,PGI_UPDATE_WG_Z)]
func entry_update_probe(
    int3 dtid : SV_DispatchThreadID)
{
    int3 probe_index = dtid;
    let push = update_probes_push;
    PGISettings settings = push.attach.globals.pgi_settings;

    if (any(greaterThanEqual(probe_index, settings.probe_count)))
    {
        return;
    }

    int3 trace_base_texel = pgi_probe_texture_base_offset(settings, settings.probe_trace_resolution, probe_index);

    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : push.attach.globals.camera.position;
    PGIProbeInfo probe_info = {}; // The dummy here is intentionally used to query default world space position.
    float3 original_probe_position = pgi_probe_index_to_worldspace(settings, probe_info, probe_anchor, probe_index);
    probe_info = PGIProbeInfo::load(push.attach.probe_info.get(), probe_index);
    float3 probe_position = pgi_probe_index_to_worldspace(settings, probe_info, probe_anchor, probe_index);

    
    if (settings.debug_draw_repositioning)
    {
        ShaderDebugLineDraw line = {};
        line.start = probe_position;
        line.end = original_probe_position;
        line.color = float3(1,1,0);
        debug_draw_line(push.attach.globals.debug, line);
    }

    rand_seed(push.attach.globals.frame_index);

    // As the offset is in "probe space" (-1 to 1 in xyz between probes), we convert all directions and values into probe space as well.
    const float SOME_LARGE_VALUE = 10000.0f;
    float2 texel_trace_noise = pgi_probe_trace_noise(probe_index, push.attach.globals.frame_index);
    const float max_probe_distance = settings.max_visibility_distance;
    int s = settings.probe_trace_resolution;
    float trace_res_rcp = rcp(float(settings.probe_trace_resolution));
    
    FreeSphere free_sphere = FreeSphere({}, 2.0f);

    float closest_backface_dist = SOME_LARGE_VALUE;
    float closest_frontface_dist = SOME_LARGE_VALUE;
    float furthest_frontface_dist = {};
    float3 closest_backface_dir = {};
    float3 closest_frontface_dir = {};
    float3 furthest_frontface_dir = {};
    float backface_count = {};
    float close_frontface_count = {};
    for (int y = 0; y < s; ++y)
    for (int x = 0; x < s; ++x)
    {
        int2 probe_local_texel = int2(x,y);
        int3 texel = trace_base_texel + int3(x,y,0);
        float2 uv = (float2(probe_local_texel) + texel_trace_noise) * trace_res_rcp;
        float3 ray_dir = pgi_probe_uv_to_probe_normal(uv);
        
        float trace_result_a = push.attach.trace_result.get()[texel].a;
        bool is_backface_hit = trace_result_a < 0.0f;
        float trace_distance = (is_backface_hit ? -trace_result_a : trace_result_a);
        float3 probe_space_hit_position_no_offset = trace_distance * ray_dir * settings.probe_spacing_rcp;
        float3 probe_space_hit_position = probe_space_hit_position_no_offset + probe_info.offset;
        float probe_space_dist = length(probe_space_hit_position_no_offset);
        float3 probe_space_ray_dir = normalize(ray_dir * settings.probe_spacing_rcp);

        // todo: determine an optimal value for the hardcoded 2.0f here!
        if (probe_space_dist > 2.0f)
        {
            probe_space_dist = 2.0f + rand();
        }

        if (is_backface_hit)
        {
            // Only move out probes that are close to a surface.
            // Probes need a lot of room to move when they cross the surface,
            // if they are very deep they have no space to reposition after they cross the surface.
            if (probe_space_dist > 0.3f)
            {
                continue;
            }
            if (probe_space_dist < closest_backface_dist)
            {
                closest_backface_dist = probe_space_dist;
                closest_backface_dir = probe_space_ray_dir;
            }
            backface_count += 1;
        }
        else
        {
            if (probe_space_dist > furthest_frontface_dist)
            {
                furthest_frontface_dist = probe_space_dist;
                furthest_frontface_dir = probe_space_ray_dir;
            }
            if (probe_space_dist < closest_frontface_dist)
            {
                closest_frontface_dist = probe_space_dist;
                closest_frontface_dir = probe_space_ray_dir;
                close_frontface_count += 1;
                //update_free_sphere(free_sphere, probe_space_hit_position_no_offset);
            }
        }
    }

    float3 spring_force = {};
    for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
    for (int z = -1; z <= 1; ++z)
    {
        int3 other_probe_index_offset = int3(x,y,z);
        int3 other_probe_index = probe_index + other_probe_index_offset;
        other_probe_index = clamp(other_probe_index, int3(0,0,0), settings.probe_count - 1);

        bool3 sett = other_probe_index_offset != int3(0,0,0);


        if (all(other_probe_index == probe_index) || ((sett.x + sett.y + sett.z) != 1))
            continue;

        PGIProbeInfo other_info = PGIProbeInfo::load(push.attach.probe_info_prev.get(), other_probe_index);

        bool diag = 
            all(other_probe_index_offset == int3(1,0,0)) ||
            all(other_probe_index_offset == int3(0,1,0)) ||
            all(other_probe_index_offset == int3(0,0,1));
        if (diag)
        {
            float3 other_pos = pgi_probe_index_to_worldspace(settings, other_info, probe_anchor, other_probe_index);
            ShaderDebugLineDraw line = {};
            line.start = probe_position;
            line.end = other_pos;
            line.color = float3(0.2,2.0,0.1);
            //debug_draw_line(push.attach.globals.debug, line);
        }

        float3 equilibrium_diff = (other_info.offset - probe_info.offset);
        float diff_magnitude = min(1.0f, length(equilibrium_diff));
        if (length(equilibrium_diff) > 0.0f)
        {
            spring_force += equilibrium_diff * diff_magnitude * diff_magnitude;
        }
    }


    float validity = probe_info.validity + 0.01f;
    

    // Calculate backface attraction
    bool too_few_backface_hits = backface_count < ceil(float(s*s) * 0.1f);
    if (too_few_backface_hits)
    {
        closest_backface_dist = SOME_LARGE_VALUE;
        closest_backface_dir = {};
    }
    float backface_attract_power = clamp(1.0f - closest_backface_dist, 0.0f, 1.0f);

    //if (free_sphere.radius < 1.0f)
    if (false)
    {
        ShaderDebugCircleDraw circle = {};
        circle.color = float3(1,0,0);
        circle.position = probe_position + free_sphere.center * settings.probe_spacing;
        circle.radius = free_sphere.radius * settings.probe_spacing.x;
        debug_draw_circle(push.attach.globals.debug, circle);
        ShaderDebugLineDraw line = {};
        line.color = float3(1,0,0);
        line.start = probe_position;
        line.end = probe_position + free_sphere.center * settings.probe_spacing;
        debug_draw_line(push.attach.globals.debug, line);
    }

    //closest_frontface_dir = normalize(free_sphere.center);
    //closest_frontface_dist = length(free_sphere.center);

    // Calculate frontface attraction and repulsion
    let closest_frontface_dir_orig = closest_frontface_dir;
    bool too_few_close_front_faces = close_frontface_count < ceil(float(s*s) * 0.01f);
    bool front_faces_too_far = false;//closest_frontface_dist > PGI_DESIRED_RELATIVE_DISTANCE;
    if (too_few_close_front_faces || front_faces_too_far)
    {
        closest_frontface_dist = SOME_LARGE_VALUE;
        closest_frontface_dir = {};
        furthest_frontface_dist = {};
        furthest_frontface_dir = {};
    }
    float frontface_repulse_power = clamp(PGI_DESIRED_RELATIVE_DISTANCE - closest_frontface_dist, 0.0f, PGI_DESIRED_RELATIVE_DISTANCE) * rcp(PGI_DESIRED_RELATIVE_DISTANCE) * 20;

    // Calculate probe grid spring attraction and repulsion 
    float spring_power = min(length(spring_force), 1.0f) * (1.0f - min(backface_attract_power + frontface_repulse_power, 1.0f));
    if (spring_power > 0.001f)
    {
        spring_force = spring_force - max(0.0f, dot(closest_frontface_dir_orig, spring_force)) * closest_frontface_dir_orig;
        spring_force = normalize(spring_force + 0.00001f);
    }
    if (settings.probe_repositioning_spring_force == 0)
    {
        spring_power = {};
    }

    float3 adjustment_vector = 
        closest_backface_dir * backface_attract_power +
        lerp(-closest_frontface_dir, furthest_frontface_dir, 0.5) * frontface_repulse_power +
        spring_force * spring_power;

    // Invlidate probes that are either too close to a surface or see backfaces
    if (closest_backface_dist != SOME_LARGE_VALUE || closest_frontface_dist < PGI_DESIRED_RELATIVE_DISTANCE * 0.5)
    {
        validity = 0.0f;
    }

    if (!settings.probe_repositioning)
    {
        push.attach.probe_info.get()[probe_index] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    float3 new_offset = probe_info.offset;
    if (validity < 1.01f)
    {
        float3 curr_offset = probe_info.offset;
        new_offset = curr_offset + adjustment_vector;
        new_offset = clamp(new_offset, float3(-1,-1,-1) * PGI_RELATIVE_REPOSITIONING_STEP, float3(1,1,1) * PGI_RELATIVE_REPOSITIONING_STEP);
        float3 offset_weights = 1.0f - abs(new_offset);
        float offset_weight = offset_weights.x * offset_weights.y * offset_weights.z; // the higher the current offset, the less we allow movement. Helps convergence to a point.
        adjustment_vector *= PGI_RELATIVE_REPOSITIONING_STEP;

        new_offset = curr_offset + adjustment_vector;
        new_offset = clamp(new_offset, -(float3)PGI_MAX_RELATIVE_REPOSITIONING, (float3)PGI_MAX_RELATIVE_REPOSITIONING);        
        //new_offset = float3(0,0,0);
    }
    push.attach.probe_info.get()[probe_index] = float4(new_offset, validity);
}
