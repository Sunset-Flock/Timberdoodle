#include "rasterize_visbuffer.hpp"

namespace raster_visbuf
{
    auto create_visbuffer(daxa::TaskGraph & tg, RenderContext const & render_context, char const * name) -> daxa::TaskImageView
    {
        return tg.create_transient_image({
            .format = daxa::Format::R32_UINT,
            .size = {
                render_context.render_data.settings.render_target_size.x,
                render_context.render_data.settings.render_target_size.y,
                1,
            },
            .name = name,
        });
    }

    auto create_depth(daxa::TaskGraph & tg, RenderContext const & render_context, char const * name) -> daxa::TaskImageView
    {
        daxa::Format format = daxa::Format::D32_SFLOAT;
        return tg.create_transient_image({
            .format = format,
            .size = {
                render_context.render_data.settings.render_target_size.x,
                render_context.render_data.settings.render_target_size.y,
                1,
            },
            .name = name,
        });
    }

    auto task_draw_visbuffer_all(TaskDrawVisbufferAllInfo const info) -> TaskDrawVisbufferAllOut
    {
        TaskDrawVisbufferAllOut ret = {};

        // === Create Render Targets ===

        ret.view_camera_overdraw = daxa::NullTaskImage;
        if (info.render_context->render_data.settings.debug_draw_mode == DEBUG_DRAW_MODE_OVERDRAW)
        {
            ret.view_camera_overdraw = info.tg.create_transient_image({
                .format = daxa::Format::R32_UINT,
                .size = {
                    info.render_context->render_data.settings.render_target_size.x,
                    info.render_context->render_data.settings.render_target_size.y,
                    1,
                },
                .name = "view_camera_overdraw",
            });
            info.tg.clear_image({ret.view_camera_overdraw, std::array{0, 0, 0, 0}});
        }
        ret.main_camera_visbuffer = raster_visbuf::create_visbuffer(info.tg, *info.render_context);
        ret.main_camera_depth = raster_visbuf::create_depth(info.tg, *info.render_context);
        daxa::TaskImageView renderable_depth = ret.main_camera_depth;

        // === Render Visbuffer ===

        // Clear out counters for current meshlet instance lists.
        info.tg.add_task(daxa::Task("clear meshlet instance buffer")
                .transfer.writes(info.meshlet_instances)
                .executes(
                    [=](daxa::TaskInterface ti)
                    {
                        auto mesh_instances_address = ti.device_address(info.meshlet_instances).value();
                        MeshletInstancesBufferHead mesh_instances_reset = make_meshlet_instance_buffer_head(mesh_instances_address);
                        allocate_fill_copy(ti, mesh_instances_reset, ti.get(info.meshlet_instances));
                    }));

        daxa::TaskBufferView first_pass_meshlet_bitfield = info.tg.create_transient_buffer({
            .size = sizeof(FirstPassMeshletBitfield),
            .name = "first_pass_meshlet_bitfield",
        });
        info.tg.clear_buffer({
            .buffer = first_pass_meshlet_bitfield,
            .name = "clear first_pass_meshlet_bitfield"
        });

        // First Pass
        {
            daxa::TaskImageView first_pass_hiz = {};
            task_gen_hiz_single_pass({info.render_context.get(), info.tg, info.depth_history, info.render_context->tgpu_render_data, info.debug_image, &first_pass_hiz, RenderTimes::index<"VISBUFFER","FIRST_PASS_GEN_HIZ">()});

            std::array<daxa::TaskBufferView, PREPASS_DRAW_LIST_TYPE_COUNT> opaque_meshlet_expansions = {};
            tasks_expand_meshes_to_meshlets(TaskExpandMeshesToMeshletsInfo{
                .render_context = info.render_context.get(),
                .tg = info.tg,
                .cull_meshes = true,
                .cull_against_last_frame = true,
                .render_time_index = RenderTimes::index<"VISBUFFER","FIRST_PASS_CULL_MESHES">(),
                .hiz = first_pass_hiz,
                .globals = info.render_context->tgpu_render_data,
                .first_pass_meshlet_bitfield = first_pass_meshlet_bitfield,
                .mesh_instances = info.scene->mesh_instances_buffer,
                .meshlet_expansions = opaque_meshlet_expansions,
            });

            task_cull_and_draw_visbuffer({
                .render_context = info.render_context.get(),
                .tg = info.tg,
                .first_pass = true,
                .clear_render_targets = true,
                .first_pass_meshlet_bitfield = first_pass_meshlet_bitfield,
                .meshlet_cull_po2expansion = opaque_meshlet_expansions,
                .hiz = first_pass_hiz,
                .meshlet_instances = info.meshlet_instances,
                .mesh_instances = info.scene->mesh_instances_buffer,
                .vis_image = ret.main_camera_visbuffer,
                .debug_image = info.debug_image,
                .depth_image = renderable_depth,
                .overdraw_image = ret.view_camera_overdraw,
            });
        }

        // Second Pass
        {
            daxa::TaskImageView hiz = {};
            task_gen_hiz_single_pass({info.render_context.get(), info.tg, ret.main_camera_depth, info.render_context->tgpu_render_data, info.debug_image, &hiz, RenderTimes::index<"VISBUFFER","SECOND_PASS_GEN_HIZ">()});

            std::array<daxa::TaskBufferView, PREPASS_DRAW_LIST_TYPE_COUNT> opaque_meshlet_expansions = {};
            tasks_expand_meshes_to_meshlets(TaskExpandMeshesToMeshletsInfo{
                .render_context = info.render_context.get(),
                .tg = info.tg,
                .cull_meshes = true,
                .cull_against_last_frame = false,
                .render_time_index = RenderTimes::index<"VISBUFFER","SECOND_PASS_CULL_MESHES">(),
                .hiz = hiz,
                .globals = info.render_context->tgpu_render_data,
                .mesh_instances = info.scene->mesh_instances_buffer,
                .meshlet_expansions = opaque_meshlet_expansions,
            });

            task_cull_and_draw_visbuffer({
                .render_context = info.render_context.get(),
                .tg = info.tg,
                .first_pass_meshlet_bitfield = first_pass_meshlet_bitfield,
                .meshlet_cull_po2expansion = opaque_meshlet_expansions,
                .hiz = hiz,
                .meshlet_instances = info.meshlet_instances,
                .mesh_instances = info.scene->mesh_instances_buffer,
                .vis_image = ret.main_camera_visbuffer,
                .debug_image = info.debug_image,
                .depth_image = renderable_depth,
                .overdraw_image = ret.view_camera_overdraw,
            });
        }

        info.tg.clear_buffer({.buffer = info.visible_meshlet_instances, .size = 4, .clear_value = 0});
        // unused for now, will be used later for perfect f+ drawing.
        if (false) {
            auto visible_meshlets_bitfield = info.tg.create_transient_buffer({
                sizeof(daxa_u32) * MAX_MESHLET_INSTANCES,
                "visible meshlets bitfield",
            });
            auto visible_meshes_bitfield = daxa::NullTaskBuffer;
            info.tg.clear_buffer({.buffer = visible_meshlets_bitfield, .clear_value = 0});
            info.tg.add_task(AnalyzeVisBufferTask2{
                .views = AnalyzeVisBufferTask2::Views{
                    .globals = info.render_context->tgpu_render_data.view(),
                    .visbuffer = ret.main_camera_visbuffer,
                    .meshlet_instances = info.meshlet_instances,
                    .mesh_instances = info.scene->mesh_instances_buffer.view(),
                    .meshlet_visibility_bitfield = visible_meshlets_bitfield,
                    .visible_meshlets = info.visible_meshlet_instances,
                    .mesh_visibility_bitfield = visible_meshes_bitfield,
                    .debug_image = info.debug_image,
                },
                .render_context = info.render_context.get(),
            });
        }

        if (info.render_context->render_data.settings.draw_from_observer)
        {
            auto const observer_depth = info.tg.create_transient_image({
                .format = daxa::Format::D32_SFLOAT,
                .size = {
                    info.render_context->render_data.settings.render_target_size.x,
                    info.render_context->render_data.settings.render_target_size.y,
                    1,
                },
                .name = "observer view_camera_depth",
            });
            auto const observer_visbuffer = raster_visbuf::create_visbuffer(info.tg, *info.render_context, "observer view_camera_visbuffer");

            info.tg.clear_image({observer_depth, daxa::DepthValue{0.0f, 0u}});
            info.tg.clear_image({observer_visbuffer, std::array{INVALID_TRIANGLE_ID, 0u, 0u, 0u}});

            if (info.render_context->render_data.settings.observer_draw_first_pass)
            {
                task_draw_visbuffer({
                    .render_context = info.render_context.get(),
                    .tg = info.tg,
                    .pass = VISBUF_FIRST_PASS,
                    .observer = true,
                    .meshlet_instances = info.meshlet_instances,
                    .meshes = info.scene->_gpu_mesh_manifest,
                    .material_manifest = info.scene->_gpu_material_manifest,
                    .combined_transforms = info.scene->_gpu_entity_combined_transforms,
                    .vis_image = observer_visbuffer,
                    .depth_image = observer_depth,
                    .overdraw_image = ret.view_camera_overdraw,
                });
            }
            if (info.render_context->render_data.settings.observer_draw_second_pass)
            {
                task_draw_visbuffer({
                    .render_context = info.render_context.get(),
                    .tg = info.tg,
                    .pass = VISBUF_SECOND_PASS,
                    .observer = true,
                    .meshlet_instances = info.meshlet_instances,
                    .meshes = info.scene->_gpu_mesh_manifest,
                    .material_manifest = info.scene->_gpu_material_manifest,
                    .combined_transforms = info.scene->_gpu_entity_combined_transforms,
                    .vis_image = observer_visbuffer,
                    .depth_image = observer_depth,
                    .overdraw_image = ret.view_camera_overdraw,
                });
            }

            ret.view_camera_visbuffer = observer_visbuffer;
            ret.view_camera_depth = observer_depth;
        }
        else
        {
            ret.view_camera_visbuffer = ret.main_camera_visbuffer;
            ret.view_camera_depth = ret.main_camera_depth;
        }

        return ret;
    }
} // namespace raster_visbuf