#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"

#include "draw_visbuffer.inl"
#include "cull_meshes.inl"
#include "analyze_visbuffer.inl"
#include "gen_hiz.inl"
#include "select_first_pass_meshlets.inl"

#include "../scene_renderer_context.hpp"

namespace raster_visbuf
{
    inline auto create_visbuffer(daxa::TaskGraph tg, RenderContext const & render_context) -> daxa::TaskImageView
    {
        return tg.create_transient_image({
            .format = daxa::Format::R32_UINT,
            .size = { 
                render_context.render_data.settings.render_target_size.x,
                render_context.render_data.settings.render_target_size.y,
                1,
            },
            .name = "visbuffer",
        });
    }

    inline auto create_atomic_visbuffer(daxa::TaskGraph tg, RenderContext const & render_context) -> daxa::TaskImageView
    {
        return tg.create_transient_image({
            .format = daxa::Format::R64_UINT,
            .size = { 
                render_context.render_data.settings.render_target_size.x,
                render_context.render_data.settings.render_target_size.y,
                1,
            },
            .name = "atomic visbuffer",
        });
    }

    inline auto create_depth(daxa::TaskGraph tg, RenderContext const & render_context, char const * name = "depth") -> daxa::TaskImageView
    {
        daxa::Format format = render_context.render_data.settings.enable_atomic_visbuffer ? daxa::Format::R32_SFLOAT : daxa::Format::D32_SFLOAT;
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

    struct TaskDrawVisbufferAllInfo
    {
        daxa::TaskGraph& tg;
        std::unique_ptr<RenderContext>& render_context;
        Scene* scene;
        daxa::TaskBufferView meshlet_instances;
        daxa::TaskBufferView meshlet_instances_last_frame;
        daxa::TaskBufferView visible_meshlet_instances;
        daxa::TaskImageView debug_image;
        daxa::TaskImageView depth_history;
    };
    struct TaskDrawVisbufferAllOut
    {
        daxa::TaskImageView depth;
        daxa::TaskImageView visbuffer;
        daxa::TaskImageView overdraw_image;
    };
    inline auto task_draw_visbuffer_all(TaskDrawVisbufferAllInfo const info) -> TaskDrawVisbufferAllOut
    {
        TaskDrawVisbufferAllOut ret = {};

        bool visbuffer_culled_first_pass = (info.render_context->render_data.settings.enable_visbuffer_two_pass_culling != 0);
        bool hiz_culled_first_pass = !visbuffer_culled_first_pass;

        // === Create Render Targets ===

        ret.overdraw_image = daxa::NullTaskImage;
        if (info.render_context->render_data.settings.debug_draw_mode == DEBUG_DRAW_MODE_OVERDRAW)
        {
            ret.overdraw_image = info.tg.create_transient_image({
                .format = daxa::Format::R32_UINT,
                .size = {
                    info.render_context->render_data.settings.render_target_size.x,
                    info.render_context->render_data.settings.render_target_size.y,
                    1,
                },
                .name = "overdraw_image",
            });
            info.tg.clear_image({ret.overdraw_image, std::array{0, 0, 0, 0}});
        }
        daxa::TaskImageView atomic_visbuffer = daxa::NullTaskImage;
        if (info.render_context->render_data.settings.enable_atomic_visbuffer)
        {
            atomic_visbuffer = raster_visbuf::create_atomic_visbuffer(info.tg, *info.render_context);
        }
        ret.visbuffer = raster_visbuf::create_visbuffer(info.tg, *info.render_context);
        if (visbuffer_culled_first_pass)
        {
            ret.depth = raster_visbuf::create_depth(info.tg, *info.render_context);
        }
        else
        {
            ret.depth = info.depth_history;
        }

        // === Render Visbuffer ===

        // Clear out counters for current meshlet instance lists.
        info.tg.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, info.meshlet_instances),
            },
            .task = [=](daxa::TaskInterface ti)
            {
                auto mesh_instances_address = ti.device.buffer_device_address(ti.get(info.meshlet_instances).ids[0]).value();
                MeshletInstancesBufferHead mesh_instances_reset = make_meshlet_instance_buffer_head(mesh_instances_address);
                allocate_fill_copy(ti, mesh_instances_reset, ti.get(info.meshlet_instances));
            },
            .name = "clear meshlet instance buffer",
        });

        daxa::TaskBufferView first_pass_meshlets_bitfield_arena = {};
        task_select_first_pass_meshlets(SelectFirstPassMeshletsInfo{
            .render_context = info.render_context.get(),
            .tg = info.tg,
            .mesh_instances = info.scene->mesh_instances_buffer,
            .meshes = info.scene->_gpu_mesh_manifest,
            .materials = info.scene->_gpu_material_manifest,
            .entity_mesh_groups = info.scene->_gpu_entity_mesh_groups,
            .mesh_group_manifest = info.scene->_gpu_mesh_group_manifest,
            .visible_meshlets_prev = info.visible_meshlet_instances,
            .meshlet_instances_last_frame = info.meshlet_instances_last_frame,
            .meshlet_instances = info.meshlet_instances,
            .first_pass_meshlets_bitfield_arena = first_pass_meshlets_bitfield_arena,
        });

        if (visbuffer_culled_first_pass)
        {
            task_draw_visbuffer({
                .render_context = info.render_context.get(),
                .tg = info.tg,
                .pass = PASS0_DRAW_FIRST_PASS,
                .meshlet_instances = info.meshlet_instances,
                .meshes = info.scene->_gpu_mesh_manifest,
                .material_manifest = info.scene->_gpu_material_manifest,
                .combined_transforms = info.scene->_gpu_entity_combined_transforms,
                .vis_image = ret.visbuffer,
                .atomic_visbuffer = atomic_visbuffer,
                .debug_image = info.debug_image,
                .depth_image = ret.depth,
                .overdraw_image = ret.overdraw_image,
            });
        }
        else
        {
            daxa::TaskImageView first_pass_hiz = {};
            task_gen_hiz_single_pass({info.render_context.get(), info.tg, ret.depth, info.render_context->tgpu_render_data, info.debug_image, &first_pass_hiz, RenderTimes::VISBUFFER_FIRST_PASS_GEN_HIZ});

            std::array<daxa::TaskBufferView, PREPASS_DRAW_LIST_TYPE_COUNT> meshlet_cull_po2expansion = {};
            tasks_expand_meshes_to_meshlets(TaskExpandMeshesToMeshletsInfo{
                .render_context = info.render_context.get(),
                .tg = info.tg,
                .cull_meshes = true,
                .cull_against_last_frame = true,
                .render_time_index = RenderTimes::VISBUFFER_FIRST_PASS_CULL_MESHES,
                .hiz = first_pass_hiz,
                .globals = info.render_context->tgpu_render_data,
                .mesh_instances = info.scene->mesh_instances_buffer,
                .meshes = info.scene->_gpu_mesh_manifest,
                .materials = info.scene->_gpu_material_manifest,
                .entity_meta = info.scene->_gpu_entity_meta,
                .entity_meshgroup_indices = info.scene->_gpu_entity_mesh_groups,
                .meshgroups = info.scene->_gpu_mesh_group_manifest,
                .entity_transforms = info.scene->_gpu_entity_transforms,
                .entity_combined_transforms = info.scene->_gpu_entity_combined_transforms,
                .opaque_meshlet_cull_po2expansions = meshlet_cull_po2expansion,
            });        

            task_cull_and_draw_visbuffer({
                .render_context = info.render_context.get(),
                .tg = info.tg,
                .first_pass = true,
                .clear_render_targets = true,
                .meshlet_cull_po2expansion = meshlet_cull_po2expansion,
                .entity_meta_data = info.scene->_gpu_entity_meta,
                .entity_meshgroups = info.scene->_gpu_entity_mesh_groups,
                .entity_combined_transforms = info.scene->_gpu_entity_combined_transforms,
                .mesh_groups = info.scene->_gpu_mesh_group_manifest,
                .meshes = info.scene->_gpu_mesh_manifest,
                .material_manifest = info.scene->_gpu_material_manifest,
                .first_pass_meshlets_bitfield_arena = first_pass_meshlets_bitfield_arena,
                .hiz = first_pass_hiz,
                .meshlet_instances = info.meshlet_instances,
                .mesh_instances = info.scene->mesh_instances_buffer,
                .vis_image = ret.visbuffer,
                .atomic_visbuffer = atomic_visbuffer,
                .debug_image = info.debug_image,
                .depth_image = ret.depth,
                .overdraw_image = ret.overdraw_image,
            });
        }


        if (info.render_context->render_data.settings.enable_atomic_visbuffer != 0)
        {
            info.tg.add_task(SplitAtomicVisbufferTask{
                .views = std::array{
                    SplitAtomicVisbufferH::AT.atomic_visbuffer | atomic_visbuffer,
                    SplitAtomicVisbufferH::AT.visbuffer | ret.visbuffer,
                    SplitAtomicVisbufferH::AT.depth | ret.depth,
                },
                .gpu_context = info.render_context->gpu_context,
                .push = SplitAtomicVisbufferPush{.size = info.render_context->render_data.settings.render_target_size},
                .dispatch_callback = [=]()
                {
                    return daxa::DispatchInfo{
                        round_up_div(info.render_context->render_data.settings.render_target_size.x, SPLIT_ATOMIC_VISBUFFER_X),
                        round_up_div(info.render_context->render_data.settings.render_target_size.y, SPLIT_ATOMIC_VISBUFFER_Y),
                        1,
                    };
                },
            });
        }

        daxa::TaskImageView hiz = {};
        task_gen_hiz_single_pass({info.render_context.get(), info.tg, ret.depth, info.render_context->tgpu_render_data, info.debug_image, &hiz, RenderTimes::VISBUFFER_SECOND_PASS_GEN_HIZ});

        std::array<daxa::TaskBufferView, PREPASS_DRAW_LIST_TYPE_COUNT> meshlet_cull_po2expansion = {};
        tasks_expand_meshes_to_meshlets(TaskExpandMeshesToMeshletsInfo{
            .render_context = info.render_context.get(),
            .tg = info.tg,
            .cull_meshes = true,
            .cull_against_last_frame = false,
            .render_time_index = RenderTimes::VISBUFFER_SECOND_PASS_CULL_MESHES,
            .hiz = hiz,
            .globals = info.render_context->tgpu_render_data,
            .mesh_instances = info.scene->mesh_instances_buffer,
            .meshes = info.scene->_gpu_mesh_manifest,
            .materials = info.scene->_gpu_material_manifest,
            .entity_meta = info.scene->_gpu_entity_meta,
            .entity_meshgroup_indices = info.scene->_gpu_entity_mesh_groups,
            .meshgroups = info.scene->_gpu_mesh_group_manifest,
            .entity_transforms = info.scene->_gpu_entity_transforms,
            .entity_combined_transforms = info.scene->_gpu_entity_combined_transforms,
            .opaque_meshlet_cull_po2expansions = meshlet_cull_po2expansion,
        });

        task_cull_and_draw_visbuffer({
            .render_context = info.render_context.get(),
            .tg = info.tg,
            .meshlet_cull_po2expansion = meshlet_cull_po2expansion,
            .entity_meta_data = info.scene->_gpu_entity_meta,
            .entity_meshgroups = info.scene->_gpu_entity_mesh_groups,
            .entity_combined_transforms = info.scene->_gpu_entity_combined_transforms,
            .mesh_groups = info.scene->_gpu_mesh_group_manifest,
            .meshes = info.scene->_gpu_mesh_manifest,
            .material_manifest = info.scene->_gpu_material_manifest,
            .first_pass_meshlets_bitfield_arena = first_pass_meshlets_bitfield_arena,
            .hiz = hiz,
            .meshlet_instances = info.meshlet_instances,
            .mesh_instances = info.scene->mesh_instances_buffer,
            .vis_image = ret.visbuffer,
            .atomic_visbuffer = atomic_visbuffer,
            .debug_image = info.debug_image,
            .depth_image = ret.depth,
            .overdraw_image = ret.overdraw_image,
        });

        info.tg.clear_buffer({.buffer = info.visible_meshlet_instances, .size = 4, .clear_value = 0});
        if (visbuffer_culled_first_pass)
        {
            auto visible_meshlets_bitfield = info.tg.create_transient_buffer({
                sizeof(daxa_u32) * MAX_MESHLET_INSTANCES,
                "visible meshlets bitfield",
            });
            auto visible_meshes_bitfield = daxa::NullTaskBuffer;
            info.tg.clear_buffer({.buffer = visible_meshlets_bitfield, .clear_value = 0});
            info.tg.add_task(AnalyzeVisBufferTask2{
                .views = std::array{
                    AnalyzeVisbuffer2H::AT.globals | info.render_context->tgpu_render_data,
                    AnalyzeVisbuffer2H::AT.visbuffer | (info.render_context->render_data.settings.enable_atomic_visbuffer != 0 ? atomic_visbuffer : ret.visbuffer),
                    AnalyzeVisbuffer2H::AT.meshlet_instances | info.meshlet_instances,
                    AnalyzeVisbuffer2H::AT.mesh_instances | info.scene->mesh_instances_buffer,
                    AnalyzeVisbuffer2H::AT.meshlet_visibility_bitfield | visible_meshlets_bitfield,
                    AnalyzeVisbuffer2H::AT.visible_meshlets | info.visible_meshlet_instances,
                    AnalyzeVisbuffer2H::AT.mesh_visibility_bitfield | visible_meshes_bitfield,
                    AnalyzeVisbuffer2H::AT.debug_image | info.debug_image,
                },
                .render_context = info.render_context.get(),
            });
        }

        if (info.render_context->render_data.settings.draw_from_observer)
        {
            ret.depth = raster_visbuf::create_depth(info.tg, *info.render_context, "observer depth");

            task_draw_visbuffer({
                .render_context = info.render_context.get(),
                .tg = info.tg,
                .pass = PASS4_OBSERVER_DRAW_ALL,
                .hiz = hiz,
                .meshlet_instances = info.meshlet_instances,
                .meshes = info.scene->_gpu_mesh_manifest,
                .material_manifest = info.scene->_gpu_material_manifest,
                .combined_transforms = info.scene->_gpu_entity_combined_transforms,
                .vis_image = ret.visbuffer,
                .atomic_visbuffer = atomic_visbuffer,
                .depth_image = ret.depth,
                .overdraw_image = ret.overdraw_image,
            });
        }
        if (info.render_context->render_data.settings.enable_atomic_visbuffer != 0)
        {
           info.tg.add_task(SplitAtomicVisbufferTask{
                .views = std::array{
                    SplitAtomicVisbufferH::AT.atomic_visbuffer | atomic_visbuffer,
                    SplitAtomicVisbufferH::AT.visbuffer | ret.visbuffer,
                    SplitAtomicVisbufferH::AT.depth | ret.depth,
                },
                .gpu_context = info.render_context->gpu_context,
                .push = SplitAtomicVisbufferPush{.size = info.render_context->render_data.settings.render_target_size},
                .dispatch_callback = [=]()
                {
                    return daxa::DispatchInfo{
                        round_up_div(info.render_context->render_data.settings.render_target_size.x, SPLIT_ATOMIC_VISBUFFER_X),
                        round_up_div(info.render_context->render_data.settings.render_target_size.y, SPLIT_ATOMIC_VISBUFFER_Y),
                        1,
                    };
                },
            });
        }

        return ret;
    }
}