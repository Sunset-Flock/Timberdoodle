#pragma once

#include "cull_lights.hlsl"

#include <daxa/utils/pipeline_manager.hpp>
#include "../scene_renderer_context.hpp"
#include "../../daxa_helper.hpp"

inline MAKE_COMPUTE_COMPILE_INFO(cull_lights_compile_info, "./src/rendering/tasks/cull_lights.hlsl", "entry_cull_lights")

inline auto create_light_mask_volume(daxa::TaskGraph& tg, RenderContext& render_context)
{
    return tg.create_transient_image({
        .format = daxa::Format::R32G32B32A32_UINT,
        .size = { s_cast<u32>(render_context.render_data.light_settings.mask_volume_cell_count.x), s_cast<u32>(render_context.render_data.light_settings.mask_volume_cell_count.y), 1 },
        .array_layer_count = s_cast<u32>(render_context.render_data.light_settings.mask_volume_cell_count.z),
        .name = "light mask volume",
    });
}

inline void lights_resolve_settings(RenderGlobalData & render_data)
{
    LightSettings& settings = render_data.light_settings;
    settings.mask_volume_cell_size = {
        settings.mask_volume_size.x / settings.mask_volume_cell_count.x,
        settings.mask_volume_size.y / settings.mask_volume_cell_count.y,
        settings.mask_volume_size.z / settings.mask_volume_cell_count.z,
    };
    settings.mask_volume_min_pos = {
        std::round(render_data.main_camera.position.x / settings.mask_volume_cell_size.x) * settings.mask_volume_cell_size.x - settings.mask_volume_size.x * 0.5f,
        std::round(render_data.main_camera.position.y / settings.mask_volume_cell_size.y) * settings.mask_volume_cell_size.y - settings.mask_volume_size.y * 0.5f,
        std::round(render_data.main_camera.position.z / settings.mask_volume_cell_size.z) * settings.mask_volume_cell_size.z - settings.mask_volume_size.z * 0.5f,
    };
    settings.point_light_count = render_data.vsm_settings.point_light_count;
    settings.spot_light_count = render_data.vsm_settings.spot_light_count;
    settings.light_count = render_data.vsm_settings.point_light_count + render_data.vsm_settings.spot_light_count;
    settings.point_light_mask = {};
    for (u32 i = 0; i < settings.point_light_count; ++i)
    {
        u32 l = i + 0;
        u32 uint_idx = i / 32;
        u32 bit_idx = i - 32 * uint_idx;
        (&settings.point_light_mask.x)[uint_idx] |= (1u << bit_idx);
    }    
    settings.spot_light_mask = {};
    settings.first_spot_light_instance = settings.point_light_count;
    for (u32 i = 0; i < settings.spot_light_count; ++i)
    {
        u32 l = i + settings.first_spot_light_instance;
        u32 uint_idx = l / 32;
        u32 bit_idx = l - 32 * uint_idx;
        (&settings.spot_light_mask.x)[uint_idx] |= (1u << bit_idx);
    }
    if (settings.cull_all_point_lights)
    {
        settings.point_light_mask = {};
    }
    if (settings.cull_all_spot_lights)
    {
        settings.spot_light_mask = {};
    }
}

inline auto lights_significant_settings_change(LightSettings const & prev, LightSettings const & curr) -> bool
{
    return 
        std::memcmp(&prev.mask_volume_size, &curr.mask_volume_size, sizeof(daxa_f32vec3)) != 0 || 
        std::memcmp(&prev.mask_volume_cell_count, &curr.mask_volume_cell_count, sizeof(daxa_f32vec3)) != 0;
}

void cull_lights_task(daxa::TaskInterface ti, RenderContext* render_context)
{
    auto const & AT = CullLightsH::AT;
    render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"MISC","CULL_LIGHTS">());
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(cull_lights_compile_info().name));

    CullLightsPush push = {};
    push.at = ti.attachment_shader_blob;
    push.point_lights = render_context->render_data.scene.point_lights;
    push.spot_lights = render_context->render_data.scene.spot_lights;
    ti.recorder.push_constant(push);

    auto const mask_volume_size = ti.info(AT.light_mask_volume).value().size;
    auto const x = round_up_div(mask_volume_size.x, CULL_LIGHTS_XYZ);
    auto const y = round_up_div(mask_volume_size.y, CULL_LIGHTS_XYZ);
    auto const z = round_up_div(mask_volume_size.y, CULL_LIGHTS_XYZ);
    ti.recorder.dispatch({x,y,z});
    render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"MISC","CULL_LIGHTS">());
}