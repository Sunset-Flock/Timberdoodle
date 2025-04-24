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
};

#define CULL_LIGHTS_XYZ 4

#if DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG
#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/lights.hlsl"

[[vk::push_constant]] CullLightsPush cull_lights_push;

func intersect_sphere_aabb(float3 sphere_center, float sphere_radius, float3 aabb_min, float3 aabb_max) -> bool
{
    float3 closest = clamp(sphere_center, aabb_min, aabb_max);
    float3 diff = sphere_center - closest;
    float distSq = dot(diff, diff);
    return distSq <= square(sphere_radius);
}

#define PRE_CULL 1

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
        float3 pre_cull_aabb_size = push.at.globals.light_settings.mask_volume_cell_size * 4;
        float3 aabb_min = light_settings.mask_volume_min_pos + float3(dtid / 4) * pre_cull_aabb_size;
        float3 aabb_max = aabb_min + pre_cull_aabb_size;

        for (uint wave_i = 0; wave_i < MAX_LIGHT_INSTANCES_PER_FRAME; wave_i += WARP_SIZE)
        {
            let light_index = wave_i + WaveGetLaneIndex();
            if (light_index >= light_settings.point_light_count)
            {
                break;
            }
            GPUPointLight point_light = push.point_lights[light_index];
            let light_frame_instance_index = light_index;
            let intersects = intersect_sphere_aabb(point_light.position, point_light.cutoff, aabb_min, aabb_max);
            let mask_uint = light_frame_instance_index / 32;
            let mask_bit = light_frame_instance_index - 32 * mask_uint;
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
        GPUPointLight point_light = push.point_lights[l];
        let light_frame_instance_index = l;
        let intersects = intersect_sphere_aabb(point_light.position, point_light.cutoff, aabb_min, aabb_max);
        let mask_uint = light_frame_instance_index / 32;
        let mask_bit = light_frame_instance_index - 32 * mask_uint;
        if (intersects)
        {
            mask[mask_uint] = mask[mask_uint] | (1u << mask_bit);
        }
    }

    push.at.light_mask_volume.get()[dtid] = mask;
}


#endif