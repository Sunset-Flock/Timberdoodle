#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(CullLightsH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32vec4>, light_mask_volume)
DAXA_DECL_TASK_HEAD_END

struct CullLightsPush
{
    CullLightsH::AttachmentShaderBlob at;
    daxa_BufferPtr(GPUPointLight) point_lights;
    daxa_BufferPtr(GPUSpotLight) spot_lights;
};

#define CULL_LIGHTS_XYZ 4

#if DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG
#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/lights.hlsl"

[[vk::push_constant]] CullLightsPush cull_lights_push;

func intersect_sphere_vs_aabb(float3 sphere_center, float sphere_radius, float3 aabb_min, float3 aabb_max) -> bool
{
    float3 closest = clamp(sphere_center, aabb_min, aabb_max);
    float3 diff = sphere_center - closest;
    float distSq = dot(diff, diff);
    return distSq <= square(sphere_radius);
}

func intersect_cone_frustum_vs_aabb(
    float3 frustum_origin,
    float3 view_direction,     // normalized
    float vertical_angle,      // in radians
    float horizontal_angle,    // in radians
    float max_range,
    float3 aabb_min,
    float3 aabb_max
) -> bool {
    float3 forward = normalize(view_direction);
    float3 world_up = float3(0.0, 0.0, 1.0);

    float3 right = normalize(cross(world_up, forward));
    if (length(right) < 1e-5) {
        world_up = float3(0.0, 1.0, 0.0);
        right = normalize(cross(world_up, forward));
    }

    float3 up = cross(forward, right);

    float tan_h = tan(horizontal_angle);
    float tan_v = tan(vertical_angle);

    // Frustum edge directions
    float3 dir_left   = normalize(forward - right * tan_h);
    float3 dir_right  = normalize(forward + right * tan_h);
    float3 dir_top    = normalize(forward - up * tan_v);
    float3 dir_bottom = normalize(forward + up * tan_v);

    // Plane normals (SAT axes)
    float3 axes[5];
    axes[0] = normalize(cross(dir_left, up));     // left plane
    axes[1] = normalize(cross(up, dir_right));    // right plane
    axes[2] = normalize(cross(dir_top, right));   // top plane
    axes[3] = normalize(cross(right, dir_bottom)); // bottom plane
    axes[4] = -forward;                            // far plane

    // Frustum corners (only need far corners for SAT)
    float3 far_center = frustum_origin + forward * max_range;
    float3 far_up = up * max_range * tan_v;
    float3 far_right = right * max_range * tan_h;

    float3 frustum_points[5] = {
        frustum_origin,
        far_center - far_up - far_right,
        far_center - far_up + far_right,
        far_center + far_up + far_right,
        far_center + far_up - far_right
    };

    // AABB corners
    float3 aabb_points[8] = {
        float3(aabb_min.x, aabb_min.y, aabb_min.z),
        float3(aabb_max.x, aabb_min.y, aabb_min.z),
        float3(aabb_min.x, aabb_max.y, aabb_min.z),
        float3(aabb_max.x, aabb_max.y, aabb_min.z),
        float3(aabb_min.x, aabb_min.y, aabb_max.z),
        float3(aabb_max.x, aabb_min.y, aabb_max.z),
        float3(aabb_min.x, aabb_max.y, aabb_max.z),
        float3(aabb_max.x, aabb_max.y, aabb_max.z)
    };

    // SAT test on all axes
    [unroll]
    for (int i = 0; i < 6; ++i) {
        float3 axis = axes[i];
        float frustum_min = dot(axis, frustum_points[0]);
        float frustum_max = frustum_min;

        [unroll]
        for (int j = 1; j < 5; ++j) {
            float d = dot(axis, frustum_points[j]);
            frustum_min = min(frustum_min, d);
            frustum_max = max(frustum_max, d);
        }

        float aabb_min_proj = dot(axis, aabb_points[0]);
        float aabb_max_proj = aabb_min_proj;

        [unroll]
        for (int j = 1; j < 8; ++j) {
            float d = dot(axis, aabb_points[j]);
            aabb_min_proj = min(aabb_min_proj, d);
            aabb_max_proj = max(aabb_max_proj, d);
        }

        if (aabb_max_proj < frustum_min || aabb_min_proj > frustum_max) {
            return false; // found a separating axis
        }
    }

    return true; // no separating axis found, must be intersecting
}

func draw_cone(float3 tip, float3 dir, float angle, float cutoff)
{
    let lines = 12;
    let sin_angle = sin(angle);
    let scale = cutoff;
    float3 tangent_side = normalize(cross(float3(0,0,1), dir));
    float3 tangent_up = normalize(cross(dir, tangent_side));
    uint line_draws_offset = debug_alloc_line_draws(cull_lights_push.at.globals.debug, lines * 2);
    if (line_draws_offset == ~0u)
    {
        // Allocation failure
        return;
    }
    for (uint i = 0; i < lines; ++i)
    {
        let c = cos( 3.14 * 2 * (float(i) * rcp(lines)));
        let s = sin( 3.14 * 2 * (float(i) * rcp(lines)));
        float3 end_pos = ((tangent_side * c + tangent_up * s) * sin_angle + dir) * scale + tip;

        let c_next = cos( 3.14 * 2 * fract(float(i + 1) * rcp(lines)));
        let s_next = sin( 3.14 * 2 * fract(float(i + 1) * rcp(lines)));
        float3 end_pos_next = ((tangent_side * c_next + tangent_up * s_next) * sin_angle + dir) * scale + tip;

        ShaderDebugLineDraw line;
        line.color = float3(1,0,0);
        line.start = tip;
        line.end = end_pos;
        cull_lights_push.at.globals.debug.line_draws.draws[line_draws_offset + i * 2] = line;
        
        ShaderDebugLineDraw line_next;
        line_next.color = float3(1,0,0);
        line_next.start = end_pos;
        line_next.end = end_pos_next;
        cull_lights_push.at.globals.debug.line_draws.draws[line_draws_offset + i * 2 + 1] = line_next;
    }
}

[shader("compute")]
[numthreads(CULL_LIGHTS_XYZ,CULL_LIGHTS_XYZ,CULL_LIGHTS_XYZ)]
func entry_cull_lights(uint3 dtid : SV_DispatchThreadID)
{
    let push = cull_lights_push;

    LightSettings light_settings = push.at.globals.light_settings;
    if (any(dtid >= light_settings.mask_volume_cell_count))
    {
        return;
    }

    uint4 pre_cull_mask = uint4(0,0,0,0);
    {
        float3 pre_cull_aabb_size = push.at.globals.light_settings.mask_volume_cell_size * CULL_LIGHTS_XYZ;
        float3 aabb_min = light_settings.mask_volume_min_pos + float3(dtid / CULL_LIGHTS_XYZ) * pre_cull_aabb_size;
        float3 aabb_max = aabb_min + pre_cull_aabb_size;

        for (uint wave_i = 0; wave_i < MAX_LIGHT_INSTANCES_PER_FRAME; wave_i += WARP_SIZE)
        {
            let light_index = wave_i + WaveGetLaneIndex();
            if (light_index >= light_settings.light_count)
            {
                break;
            }
            
            let light_frame_instance_index = light_index;
            let mask_uint = light_frame_instance_index / 32;
            let mask_bit = light_frame_instance_index - 32 * mask_uint;

            bool intersects = false;
            if (light_index < light_settings.point_light_count)
            {
                GPUPointLight point_light = push.point_lights[light_index];
                intersects = intersect_sphere_vs_aabb(point_light.position, point_light.cutoff, aabb_min, aabb_max);
            }
            else
            {
                let spot_index = light_index - light_settings.first_spot_light_instance;
                GPUSpotLight spot_light = push.spot_lights[spot_index];
                //draw_cone(spot_light.position, spot_light.direction, spot_light.outer_cone_angle, spot_light.cutoff);
                float3 center = spot_light.position + spot_light.direction * spot_light.cutoff * 0.5f;
                intersects = intersect_cone_frustum_vs_aabb(spot_light.position, spot_light.direction, spot_light.outer_cone_angle, spot_light.outer_cone_angle, spot_light.cutoff, aabb_min, aabb_max);
            }
            if (intersects)
            {
                pre_cull_mask[mask_uint] = pre_cull_mask[mask_uint] | (1u << mask_bit);
            }
        }
        pre_cull_mask = WaveActiveBitOr(pre_cull_mask);
    }

    
    float3 aabb_size = push.at.globals.light_settings.mask_volume_cell_size;
    float3 aabb_min = light_settings.mask_volume_min_pos + float3(dtid) * aabb_size;
    float3 aabb_max = aabb_min + aabb_size;
    uint4 mask = (uint4)0;
    while (any(pre_cull_mask != uint4(0,0,0,0)))
    {
        let l = lights_iterate_mask(light_settings, pre_cull_mask);

        let light_frame_instance_index = l;
        let mask_uint = light_frame_instance_index / 32;
        let mask_bit = light_frame_instance_index - 32 * mask_uint;
        
        bool intersects = false;
        if (l < light_settings.point_light_count)
        {
            GPUPointLight point_light = push.point_lights[l];
            intersects = intersect_sphere_vs_aabb(point_light.position, point_light.cutoff, aabb_min, aabb_max);
        }
        else
        {
            let spot_index = l - light_settings.first_spot_light_instance;
            GPUSpotLight spot_light = push.spot_lights[spot_index];
            // intersects = intersect_sphere_vs_aabb(spot_light.position, spot_light.cutoff, aabb_min, aabb_max);
            // float3 center = spot_light.position + spot_light.direction * spot_light.cutoff * 0.5f;
            // intersects = intersect_sphere_vs_aabb(center, spot_light.cutoff * 0.5f, aabb_min, aabb_max);
            intersects = intersect_cone_frustum_vs_aabb(spot_light.position, spot_light.direction, spot_light.outer_cone_angle, spot_light.outer_cone_angle, spot_light.cutoff, aabb_min, aabb_max);
            if (intersects)
            {
                // ShaderDebugAABBDraw aabb;
                // aabb.color = float3(1,0,0);
                // aabb.position = aabb_min + aabb_size * 0.5f;
                // aabb.size = aabb_size.xxx;
                // debug_draw_aabb(push.at.globals.debug, aabb);
                // 
                // ShaderDebugLineDraw line;
                // line.color = float3(1,0,0);
                // line.start = aabb_min + aabb_size * 0.5f;
                // line.end = spot_light.position;
                // debug_draw_line(push.at.globals.debug, line);
            }
        }
        if (intersects)
        {
            mask[mask_uint] = mask[mask_uint] | (1u << mask_bit);
        }
    }

    push.at.light_mask_volume.get()[dtid] = mask;
}


#endif