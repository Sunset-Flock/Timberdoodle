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

__generic<let N : uint>
func write_probe_texel_with_border(RWTexture2DArray<vector<float,N>> tex, int2 probe_res, int3 base_index, int2 probe_texel, vector<float, N> value)
{
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

[shader("compute")]
[numthreads(PGI_UPDATE_WG_XY,PGI_UPDATE_WG_XY,PGI_UPDATE_WG_Z)]
func entry_update_probe_irradiance(
    int3 dtid : SV_DispatchThreadID,
) 
{
    let push = update_probe_texels_push;
    PGISettings settings = push.attach.globals.pgi_settings;
    let probe_texel_res = settings.probe_radiance_resolution;

    int3 probe_index = {};
    int2 probe_texel = {};
    if (settings.enable_indirect_sparse)
    {
        uint probes_in_column = uint(float(48) * rcp(float(settings.probe_radiance_resolution)));
        uint indirect_index_y = uint(float(dtid.y) * rcp(float(settings.probe_radiance_resolution)));
        uint indirect_index_x = uint(float(dtid.x) * rcp(float(settings.probe_radiance_resolution)));
        uint indirect_index = indirect_index_x * probes_in_column + indirect_index_y;
        let is_overhang = indirect_index >= push.attach.probe_indirections.probe_update_count;
        if (is_overhang)
        {
            return;
        }
        uint indirect_package = ((uint*)(push.attach.probe_indirections + 1))[indirect_index];
        probe_index = int3(
            (indirect_package >> 0) & ((1u << 10u) - 1),
            (indirect_package >> 10) & ((1u << 10u) - 1),
            (indirect_package >> 20) & ((1u << 10u) - 1),
        );
        probe_texel.x = dtid.x - int(settings.probe_radiance_resolution * indirect_index_x);
        probe_texel.y = dtid.y - int(settings.probe_radiance_resolution * indirect_index_y);

        //printf("indirect index %i probe %i,%i,%i texel %i,%i\n", indirect_index, probe_index.x, probe_index.y, probe_index.z, probe_texel.x, probe_texel.y);
    }
    else
    {
        probe_index = int3(float2(dtid.xy) * rcp(probe_texel_res), dtid.z);
        probe_texel = dtid.xy - probe_index.xy * probe_texel_res;
    }
    
    uint frame_index = push.attach.globals.frame_index;

    if (any(greaterThanEqual(probe_index, settings.probe_count)))
    {
        return;
    }
    
    PGIProbeInfo probe_info = PGIProbeInfo::load(settings, push.attach.probe_info.get(), probe_index);
    float3 probe_position = pgi_probe_index_to_worldspace(push.attach.globals.pgi_settings, probe_info, probe_index);
    float2 probe_texel_uv = (float2(probe_texel) + 0.5f) * rcp(probe_texel_res);
    float3 probe_texel_normal = pgi_probe_uv_to_probe_normal(probe_texel_uv);

    int3 probe_texture_base_index = pgi_probe_texture_base_offset<HAS_BORDER>(settings, probe_texel_res, probe_index);
    int3 probe_texture_index = probe_texture_base_index + int3(probe_texel, 0);

    int s = settings.probe_trace_resolution;

    float4 cosine_convoluted_trace_result = float4(0.0f,0.0f,0.0f, 0.0f);
    int3 trace_result_texture_base_index = pgi_probe_texture_base_offset<NO_BORDER>(settings, settings.probe_trace_resolution, probe_index);
    float2 trace_texel_noise = pgi_probe_trace_noise(probe_index, frame_index); // used to reconstruct directions used for traces.
    float acc_weight = 0.0f;
    float cos_wrap_around = settings.cos_wrap_around;
    float cos_wrap_around_rcp = rcp(cos_wrap_around + 1.0f);
    float rcp_s = rcp(s);
    for (int y = 0; y < s; ++y)
    for (int x = 0; x < s; ++x)
    {
        float2 trace_tex_uv = (float2(x,y) + trace_texel_noise) * rcp_s;
        float3 trace_direction = pgi_probe_uv_to_probe_normal(trace_tex_uv); // Trace direction is identical to the one used in tracer.
        float cos_weight = max(0.0f, (cos_wrap_around + dot(trace_direction, probe_texel_normal)) * cos_wrap_around_rcp);
        int3 sample_texture_index = trace_result_texture_base_index + int3(x,y,0);
        if (cos_weight > 0.01f)
        {
            float4 sample = push.attach.trace_result.get()[sample_texture_index].rgba;
            cosine_convoluted_trace_result += sample * cos_weight;
            acc_weight += cos_weight;
        }
    }
    
    // If we have 0 weight we have no samples to blend so we dont blend at all.
    if (acc_weight > 0.0f)
    {
        cosine_convoluted_trace_result *= rcp(acc_weight);
    }

    float3 new_radiance = cosine_convoluted_trace_result.rgb;

    float4 prev_frame_texel = push.attach.probe_radiance.get()[probe_texture_index];
    float3 prev_frame_radiance = prev_frame_texel.rgb;

    // Automatic Hysteresis
    float hysteresis = prev_frame_texel.a;
    {
        float3 lighting_change3 = abs(prev_frame_radiance - new_radiance);
        float lighting_change = max3(lighting_change3.x, lighting_change3.y, lighting_change3.z);
        float prev_frame_max = max3(prev_frame_radiance.x, prev_frame_radiance.y, prev_frame_radiance.z) + 0.01f;
        float factor = (1.0f - smoothstep(0.0f, prev_frame_max * rcp((hysteresis-0.75)*5), lighting_change)) - 0.5f;
        if (factor > 0.0f)
        {
            factor *= 0.1f;
        }
        hysteresis += 0.1 * factor;
        hysteresis = clamp(hysteresis, 0.8f, 0.99f);
    }
    if (probe_info.validity == 0.0f)
    {
        hysteresis = 0.0f;
    }

    new_radiance = lerp(new_radiance, prev_frame_radiance, hysteresis);

    float4 value = float4(new_radiance, hysteresis);
    write_probe_texel_with_border(push.attach.probe_radiance.get(), settings.probe_radiance_resolution, probe_texture_base_index, probe_texel, value);
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

    int3 probe_index = {};
    int2 probe_texel = {};
    if (settings.enable_indirect_sparse)
    {
        uint probes_in_column = uint(float(48) * rcp(float(settings.probe_visibility_resolution)));
        uint indirect_index_y = uint(float(dtid.y) * rcp(float(settings.probe_visibility_resolution)));
        uint indirect_index_x = uint(float(dtid.x) * rcp(float(settings.probe_visibility_resolution)));
        uint indirect_index = indirect_index_x * probes_in_column + indirect_index_y;
        let is_overhang = indirect_index >= push.attach.probe_indirections.probe_update_count;
        if (is_overhang)
        {
            return;
        }
        uint indirect_package = ((uint*)(push.attach.probe_indirections + 1))[indirect_index];
        probe_index = int3(
            (indirect_package >> 0) & ((1u << 10u) - 1),
            (indirect_package >> 10) & ((1u << 10u) - 1),
            (indirect_package >> 20) & ((1u << 10u) - 1),
        );
        probe_texel.x = dtid.x - settings.probe_visibility_resolution * indirect_index_x;
        probe_texel.y = dtid.y - settings.probe_visibility_resolution * indirect_index_y;
    }
    else
    {
        probe_index = int3(float2(dtid.xy) * rcp(probe_texel_res), dtid.z);
        probe_texel = dtid.xy - probe_index.xy * probe_texel_res;
    }

    uint frame_index = push.attach.globals.frame_index;

    if (any(greaterThanEqual(probe_index, settings.probe_count)))
    {
        return;
    }
    
    PGIProbeInfo probe_info = PGIProbeInfo::load(settings, push.attach.probe_info.get(), probe_index);
    float3 probe_position = pgi_probe_index_to_worldspace(push.attach.globals.pgi_settings, probe_info, probe_index);
    float2 probe_texel_min_uv = (float2(probe_texel)) * rcp(probe_texel_res);
    float2 probe_texel_max_uv = (float2(probe_texel) + 1.0f) * rcp(probe_texel_res);
    float3 probe_texel_normal = pgi_probe_uv_to_probe_normal((probe_texel_max_uv + probe_texel_min_uv) * 0.5f);

    int3 probe_texture_base_index = pgi_probe_texture_base_offset<HAS_BORDER>(settings, probe_texel_res, probe_index);
    
    int3 probe_texture_index = probe_texture_base_index + int3(probe_texel, 0);

    float2 relevant_trace_blend = float2(0.0f, 0.0f);
    int valid_trace_count = 0;
    int3 trace_result_texture_base_index = pgi_probe_texture_base_offset<NO_BORDER>(settings, settings.probe_trace_resolution, probe_index);
    float2 trace_texel_noise = pgi_probe_trace_noise(probe_index, frame_index); // used to reconstruct directions used for traces.
    const float max_depth = settings.max_visibility_distance;
    float2 prev_frame_visibility = push.attach.probe_visibility.get()[probe_texture_index];
    float acc_cos_weights = {};
    int s = settings.probe_trace_resolution;
    float rcp_s = rcp(s) * 0.999999f; // The multiplication with 0.999999f ensures that the calculated uv never reaches 1.0f.

    static const float COS_POWER = 50.0f;
    static const float MIN_ACCEPTED_POWER_COS = 0.001f;
    // Based on the COS_POWER and the MIN_ACCEPTED_COS_POWER, we can calculate the largest accepted cos value.
    static const float MAX_ACCEPTED_COS = acos(pow(MIN_ACCEPTED_POWER_COS, 1.0f / COS_POWER));

    Texture2DArray<float4> trace_result_tex = push.attach.trace_result.get();
    for (int y = 0; y < s; ++y)
    for (int x = 0; x < s; ++x)
    {
        float2 trace_tex_uv = (float2(x,y) + trace_texel_noise) * rcp_s;
        float3 trace_direction = pgi_probe_uv_to_probe_normal(trace_tex_uv); // Trace direction is identical to the one used in tracer.
        float cos_weight = (dot(trace_direction, probe_texel_normal));
        int3 sample_texture_index = trace_result_texture_base_index + int3(x,y,0);
        if (cos_weight > MAX_ACCEPTED_COS)
        {
            float power_cos_weight = pow(cos_weight, 50.0f);
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

        float ray_to_texel_ratio = float(settings.probe_trace_resolution) / float(settings.probe_visibility_resolution);
        update_factor *= ray_to_texel_ratio;

        if (probe_info.validity == 0.0f)
        {
            update_factor = 1.0f;
        }

        float2 value = lerp(prev_frame_visibility, relevant_trace_blend, update_factor);
        write_probe_texel_with_border(push.attach.probe_visibility.get(), settings.probe_visibility_resolution, probe_texture_base_index, probe_texel, value);
    }
}

// Y W B B P K#YWBBPK

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

#define PGI_DESIRED_RELATIVE_DISTANCE 0.3f 
#define PGI_RELATIVE_REPOSITIONING_STEP 0.2f
#define PGI_MAX_RELATIVE_REPOSITIONING 1.0f
#define PGI_ACCEPTABLE_SURFACE_DISTANCE (PGI_DESIRED_RELATIVE_DISTANCE * 0.3)
#define PGI_BACKFACE_ESCAPE_RANGE (PGI_DESIRED_RELATIVE_DISTANCE * 3)
#define PGI_PROBE_VIEW_DISTANCE 1.0

[shader("compute")]
[numthreads(PGI_UPDATE_WG_XY,PGI_UPDATE_WG_XY,PGI_UPDATE_WG_Z)]
func entry_update_probe(
    int3 dtid : SV_DispatchThreadID,
    int group_id : SV_GroupID,
    int group_index : SV_GroupIndex)
{
    let push = update_probes_push;
    PGISettings settings = push.attach.globals.pgi_settings;

    int3 probe_index = dtid;
    if (settings.enable_indirect_sparse)
    {
        uint indirect_index = group_id * 64 + group_index;
        let overhang = indirect_index >= push.attach.probe_indirections.probe_update_count;
        if (overhang)
        {
            return;
        }

        uint indirect_package = ((uint*)(push.attach.probe_indirections + 1))[indirect_index];
        probe_index = int3(
            (indirect_package >> 0) & ((1u << 10u) - 1),
            (indirect_package >> 10) & ((1u << 10u) - 1),
            (indirect_package >> 20) & ((1u << 10u) - 1),
        );
    }

    int3 stable_index = pgi_probe_to_stable_index(settings, probe_index);

    if (any(greaterThanEqual(probe_index, settings.probe_count)))
    {
        return;
    }

    int3 trace_base_texel = pgi_probe_texture_base_offset<NO_BORDER>(settings, settings.probe_trace_resolution, probe_index);

    PGIProbeInfo probe_info = {}; // The dummy here is intentionally used to query default world space position.
    float3 original_probe_position = pgi_probe_index_to_worldspace(settings, probe_info, probe_index);
    probe_info = PGIProbeInfo::load(settings, push.attach.probe_info.get(), probe_index);
    float3 probe_position = pgi_probe_index_to_worldspace(settings, probe_info, probe_index);

    rand_seed(push.attach.globals.frame_index);

    // As the offset is in "probe space" (-1 to 1 in xyz between probes), we convert all directions and values into probe space as well.
    const float SOME_LARGE_VALUE = 10000.0f;
    float2 texel_trace_noise = pgi_probe_trace_noise(probe_index, push.attach.globals.frame_index);
    const float max_probe_distance = settings.max_visibility_distance;
    int s = settings.probe_trace_resolution;
    float trace_res_rcp = rcp(float(settings.probe_trace_resolution));

    float3 average_hit_offset = {};
    float3 average_backface_hit_offset = {};

    float closest_backface_dist = SOME_LARGE_VALUE;
    float closest_frontface_dist = SOME_LARGE_VALUE;
    float3 closest_backface_dir = {};
    float backface_count = {};
    Texture2DArray<float4> trace_result_tex = push.attach.trace_result.get();
    for (int y = 0; y < s; ++y)
    for (int x = 0; x < s; ++x)
    {
        int2 probe_local_texel = int2(x,y);
        int3 texel = trace_base_texel + int3(x,y,0);
        float2 uv = (float2(probe_local_texel) + texel_trace_noise) * trace_res_rcp;
        float3 ray_dir = pgi_probe_uv_to_probe_normal(uv);
        
        float trace_result_a = trace_result_tex[texel].a;
        bool is_backface_hit = trace_result_a < 0.0f;
        float trace_distance = (is_backface_hit ? -trace_result_a : trace_result_a);
        float3 probe_space_hit_position_no_offset = trace_distance * ray_dir * settings.probe_spacing_rcp;
        float3 probe_space_hit_position = probe_space_hit_position_no_offset + probe_info.offset;
        float probe_space_dist = length(probe_space_hit_position_no_offset);
        float3 probe_space_ray_dir = normalize(ray_dir * settings.probe_spacing_rcp);

        probe_space_dist = min(PGI_PROBE_VIEW_DISTANCE, probe_space_dist);

        average_hit_offset += probe_space_ray_dir * max(0.5f, probe_space_dist);

        if (is_backface_hit)
        {
            // Only move out probes that are close to a surface.
            // Probes need a lot of room to move when they cross the surface,
            // if they are very deep they have no space to reposition after they cross the surface.
            if (probe_space_dist < PGI_BACKFACE_ESCAPE_RANGE)
            {
                if (probe_space_dist < closest_backface_dist)
                {
                    closest_backface_dist = probe_space_dist;
                    closest_backface_dir = probe_space_ray_dir;
                }
                average_backface_hit_offset += probe_space_ray_dir * probe_space_dist;
                backface_count += 1;
            }
        }
        else
        {
            closest_frontface_dist = min(closest_frontface_dist, probe_space_dist);
        }
    }

    average_hit_offset *= rcp(float(s*s));
    if (backface_count > 0)
    {
        average_backface_hit_offset *= rcp(float(backface_count));
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

        PGIProbeInfo other_info = PGIProbeInfo::load(settings, push.attach.probe_info_prev.get(), other_probe_index);

        float3 equilibrium_diff = lerp(other_info.offset - probe_info.offset, - probe_info.offset, 0.1f);
        float diff_magnitude = min(1.0f, length(equilibrium_diff));
        if (length(equilibrium_diff) > 0.0f)
        {
            spring_force += equilibrium_diff * diff_magnitude * diff_magnitude * diff_magnitude;
        }
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
    if ((backface_escape_distance == 0.0f) && (settings.probe_repositioning_spring_force != 0))
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


    if (settings.debug_draw_repositioning_forces)
    {
        ShaderDebugLineDraw backface_force = {};
        backface_force.color = float3(0.5,0,0);
        backface_force.start = probe_position;
        backface_force.end = probe_position + closest_backface_dir * backface_escape_distance * settings.probe_spacing;
        debug_draw_line(push.attach.globals.debug, backface_force);

        ShaderDebugLineDraw frontface_force = {};
        frontface_force.color = float3(0,0.5,0);
        frontface_force.start = probe_position;
        frontface_force.end = probe_position + estimated_freedom_direction * frontface_repulse_distance * settings.probe_spacing;
        debug_draw_line(push.attach.globals.debug, frontface_force);

        ShaderDebugLineDraw spring_force = {};
        spring_force.color = float3(0,0,0.5);
        spring_force.start = probe_position;
        spring_force.end = probe_position + spring_force_dir * spring_force_distance * settings.probe_spacing;
        debug_draw_line(push.attach.globals.debug, spring_force);
    }

    // Invalidate probes that are either too close to a surface or see backfaces
    if (closest_backface_dist != SOME_LARGE_VALUE || closest_frontface_dist < PGI_ACCEPTABLE_SURFACE_DISTANCE)
    {
        probe_info.validity = 0.0f;
    }
    
    // When the window moves, the probe data on the new border must be invalidated.
    int3 prev_frame_probe_index = pgi_probe_index_to_prev_frame(settings, probe_index);
    bool prev_frame_invalid = any(prev_frame_probe_index < int3(1,1,1) || prev_frame_probe_index >= (settings.probe_count - int3(1,1,1)));
    if (prev_frame_invalid)
    {
        probe_info.validity = 0.0f;
        probe_info.offset = {};
        adjustment_vector = {};
    }

    if (!settings.probe_repositioning)
    {
        push.attach.probe_info.get()[stable_index] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    float3 new_offset = probe_info.offset;
    float3 curr_offset = probe_info.offset;
    new_offset = curr_offset + adjustment_vector;
    new_offset = clamp(new_offset, -(float3)PGI_MAX_RELATIVE_REPOSITIONING, (float3)PGI_MAX_RELATIVE_REPOSITIONING);      

    // Border brobes must not converge in position as they need the surrounding probes to find their proper location.
    let is_border_probe = any(probe_index <= int3(1,1,1) || probe_index >= (settings.probe_count-2));
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
    let settings = push.attach.globals.pgi_settings;
    int3 probe_index = dtid;
    int3 stable_index = pgi_probe_to_stable_index(settings, probe_index);

    if (any(probe_index >= settings.probe_count))
    {
        return;
    }

    int3 prev_frame_probe_index = pgi_probe_index_to_prev_frame(settings, probe_index);

    // The last probe in each dimension is never a base probe.
    // We clear these probes request counter to 0.
    let is_base_probe = pgi_is_cell_base_probe(settings, probe_index);
    if (is_base_probe)
    {
        // New probes are those that came into existence due to the window moving to a new location.
        // Any probes that occupy space that was not taken in the prior frame are new.
        let is_prev_base_probe = pgi_is_cell_base_probe(settings, prev_frame_probe_index);

        // Each base probe is responsible for maintaining the cells request counter.
        uint probe_base_request = 0;
        if (is_prev_base_probe)
        {
            probe_base_request = push.attach.requests.get()[stable_index];
            probe_base_request = max(1,probe_base_request) - 1;
        }
        #if 0
        else
        {
            ShaderDebugAABBDraw aabb = {};
            aabb.position = pgi_probe_index_to_worldspace(settings, PGIProbeInfo(), probe_index) + settings.probe_spacing * 0.5f;
            aabb.size = settings.probe_spacing;
            aabb.color = float3(0,1,0);
            debug_draw_aabb(push.attach.globals.debug, aabb);

            ShaderDebugAABBDraw aabb_prev = {};
            aabb_prev.position = pgi_probe_index_to_worldspace(settings, PGIProbeInfo(), prev_frame_probe_index) + settings.probe_spacing * 0.5f;
            aabb_prev.size = settings.probe_spacing;
            aabb_prev.color = float3(1,0,0);
            debug_draw_aabb(push.attach.globals.debug, aabb_prev);
        }
        #endif
        push.attach.requests.get()[stable_index] = probe_base_request;
    }
    else
    {
        push.attach.requests.get()[stable_index] = 0;
    }

    // New probes are those that came into existence due to the window moving to a new location.
    // Any probes that occupy space that was not taken in the prior frame are new.
    let is_new_probe = any(prev_frame_probe_index < int3(0,0,0)) || any(prev_frame_probe_index >= (settings.probe_count));

    // Each probe is a vertex in up to 8 cells.
    // Check each cells base probes request counter to determine if probe is requested.
    bool requested = false;
    bool update = false;
    uint request_index = ~0u;
    uint update_index = ~0u;
    if (!is_new_probe)
    {
        // Each Probe is part of up to 8 cells (8 Probes form the vertices of a cell)
        // A probe request is always only marked in the "base" probe of a cell.
        // The base probe is the one that has the lowest index.
        for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
        for (int z = 0; z < 2; ++z)
        {
            let other_probe = max(int3(0,0,0), probe_index - int3(x,y,z));
            int3 other_probe_prev_frame = pgi_probe_index_to_prev_frame(settings, other_probe);
            let other_probe_is_new_base_probe = !pgi_is_cell_base_probe(settings, other_probe_prev_frame);
            if (other_probe_is_new_base_probe)
            {
                continue;
            }

            let other_probe_stable_index = pgi_probe_to_stable_index(settings, other_probe);

            uint request = push.attach.requests.get()[other_probe_stable_index];
            requested = requested || (request != 0);
        }

        if (requested)
        {
            InterlockedAdd(push.attach.probe_indirections.detailed_probe_count, 1, request_index);
        }

        switch(settings.update_rate)
        {
        case PGI_UPDATE_RATE_FULL:
            update = requested;
            break;
        case PGI_UPDATE_RATE_1_OF_2:
            int checker_board = (probe_index.x & 0x1) ^ (probe_index.y & 0x1) ^ (probe_index.z & 0x1) ^ (push.attach.globals.frame_index & 0x1);
            update = requested && (checker_board != 0);
            break;
        case PGI_UPDATE_RATE_1_OF_8:
            int every_eightth = ((probe_index.x & 0x1) + (probe_index.y & 0x1) * 2 + (probe_index.z & 0x1) * 4) == (push.attach.globals.frame_index & 0x7);
            update = requested && (every_eightth != 0);
            break;
        case PGI_UPDATE_RATE_1_OF_64:
            int every_64th = ((probe_index.x/4 & 0x3) + (probe_index.y/4 & 0x3) * 16 + (probe_index.z/4 & 0x3) * 4) == ((push.attach.globals.frame_index) & 0x3F);
            update = requested && (every_64th != 0);
            break;
        }

        if (update)
        {
            InterlockedAdd(push.attach.probe_indirections.probe_update_count, 1, update_index);
        }
    }

    if (update)
    {
        // We need a list of all updated probes 
        ((uint*)(push.attach.probe_indirections + 1))[update_index] = ((probe_index.x << 0) | (probe_index.y << 10) | (probe_index.z << 20));
    }

    if (requested)
    {
        // We need a list of all active probes 
        uint active_probes_offset = settings.probe_count.x * settings.probe_count.y * settings.probe_count.z;
        ((uint*)(push.attach.probe_indirections + 1))[request_index + active_probes_offset] = ((probe_index.x << 0) | (probe_index.y << 10) | (probe_index.z << 20));
    }

    PGIProbeInfo probe_info = {};
    if (requested)
    {
        let fetch = push.attach.probe_info.get()[stable_index];
        probe_info = PGIProbeInfo(fetch.xyz, fetch.a);
    }
    push.attach.probe_info.get()[stable_index] = float4(probe_info.offset, probe_info.validity);

    // Used for Debug Draws
    float3 probe_position = pgi_probe_index_to_worldspace(settings, probe_info, probe_index);

    // Debug Draw Probe Grid
    if (settings.debug_draw_grid && (probe_info.validity != 0.0f))
    {
        for (int other_i = 0; other_i < 3; ++other_i)
        {
            int3 other_probe_index = probe_index;
            other_probe_index[other_i] += 1;
            
            let other_index_in_range = all(other_probe_index >= int3(0,0,0) && other_probe_index < settings.probe_count);
            if (!other_index_in_range)
                continue;

            PGIProbeInfo other_info = PGIProbeInfo::load(settings, push.attach.probe_info.get(), other_probe_index);

            let other_valid = other_info.validity != 0.0f;
            if (other_valid)
            {
                float3 other_pos = pgi_probe_index_to_worldspace(settings, other_info, other_probe_index);
                ShaderDebugLineDraw line = {};
                line.start = probe_position;
                line.end = other_pos;
                line.color = float3(0.2,0.0,0.1);
                debug_draw_line(push.attach.globals.debug, line);
            }
        }
    }

    if (settings.debug_draw_repositioning && (probe_info.validity != 0.0f))
    {
        ShaderDebugLineDraw line = {};
        line.start = probe_position;
        line.end = pgi_probe_index_to_worldspace(settings, PGIProbeInfo(), probe_index);
        line.color = float3(1,1,0);
        debug_draw_line(push.attach.globals.debug, line);
    }


    if (group_index == 0)
    {
        uint finished_workgroups = 0;
        InterlockedAdd(*push.workgroups_finished, 1, finished_workgroups);
        finished_workgroups += 1;

        let last_to_finish = finished_workgroups == push.total_workgroups;
        if (last_to_finish)
        {
            let probe_update_count = push.attach.probe_indirections.probe_update_count;
            let detailed_probe_count = push.attach.probe_indirections.detailed_probe_count;
            let probe_update_workgroups_x = round_up_div(probe_update_count, 64);

            push.attach.globals.readback.requested_probes = probe_update_count;

            // Allows for full thread utilization for relevant texel resolutions:
            // 4x4, 6x6, 8x8, 12x12, 16x16, 24x24
            let texel_update_threads_y = 48;
            let texel_update_workgroups_y = 6; // 48/WG_Y = 48/8 = 6

            let radiance_texel_update_probes_y = texel_update_threads_y / settings.probe_radiance_resolution;
            let radiance_texel_update_probes_x = round_up_div(probe_update_count, radiance_texel_update_probes_y);
            let radiance_texel_update_threads_x = radiance_texel_update_probes_x * settings.probe_radiance_resolution;
            let radiance_texel_update_workgroups_x = round_up_div(radiance_texel_update_threads_x, 8);

            let visibility_texel_update_probes_y = texel_update_threads_y / settings.probe_visibility_resolution;
            let visibility_texel_update_probes_x = round_up_div(probe_update_count, visibility_texel_update_probes_y);
            let visibility_texel_update_threads_x = visibility_texel_update_probes_x * settings.probe_visibility_resolution;
            let visibility_texel_update_workgroups_x = round_up_div(visibility_texel_update_threads_x, 8);

            push.attach.probe_indirections.probe_update_dispatch = DispatchIndirectStruct(probe_update_workgroups_x,1,1);
            push.attach.probe_indirections.probe_trace_dispatch = DispatchIndirectStruct(probe_update_count * settings.probe_trace_resolution * settings.probe_trace_resolution, 1, 1);
            push.attach.probe_indirections.probe_radiance_update_dispatch = DispatchIndirectStruct(radiance_texel_update_workgroups_x, texel_update_workgroups_y, 1);
            push.attach.probe_indirections.probe_visibility_update_dispatch = DispatchIndirectStruct(visibility_texel_update_workgroups_x, texel_update_workgroups_y, 1);
            push.attach.probe_indirections.probe_debug_draw_dispatch = DrawIndexedIndirectStruct(
                960*3,
                (settings.debug_probe_draw_mode != 0) ? detailed_probe_count : 0,
                0,0,0
            );
        }
    }
}