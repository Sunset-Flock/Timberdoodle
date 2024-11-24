#pragma once
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../gpu_context.hpp"
#include "../../shader_shared/geometry_pipeline.inl"
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/gpu_work_expansion.inl"
#include "../tasks/misc.hpp"
#include "../../scene/scene.hpp"

struct VSMState
{
    // Persistent state
    daxa::TaskBuffer globals = {};

    daxa::TaskImage memory_block = {};
    daxa::TaskImage meta_memory_table = {};
    daxa::TaskImage page_table = {};
    daxa::TaskImage page_view_pos_row = {};
    daxa::TaskImage point_page_tables = {};

    // Transient state
    daxa::TaskBufferView vsm_point_lights = {};
    daxa::TaskBufferView allocation_requests = {};
    daxa::TaskBufferView free_wrapped_pages_info = {};
    daxa::TaskBufferView free_page_buffer = {};
    daxa::TaskBufferView not_visited_page_buffer = {};
    daxa::TaskBufferView find_free_pages_header = {};
    daxa::TaskBufferView clip_projections = {};
    daxa::TaskBufferView dirty_page_masks = {};
    daxa::TaskImageView dirty_pages_hiz = {};
    daxa::TaskImageView overdraw_debug_image = {};

    daxa::TaskBufferView allocate_indirect = {};
    daxa::TaskBufferView clear_indirect = {};
    daxa::TaskBufferView clear_dirty_bit_indirect = {};

    std::array<VSMClipProjection, VSM_CLIP_LEVELS> clip_projections_cpu = {};
    std::array<FreeWrappedPagesInfo, VSM_CLIP_LEVELS> free_wrapped_pages_info_cpu = {};
    std::array<i32vec2, VSM_CLIP_LEVELS> last_frame_offsets = {};
    std::array<VSMPointLight, MAX_POINT_LIGHTS> point_lights_cpu = {};

    VSMGlobals globals_cpu = {};
    void update_vsm_lights(const std::vector<ActivePointLight> & active_lights) 
    {
        // TODO(msakmary) Might be broken idk how cubemaps actually work
        constexpr std::array<f32vec3, 6> cubemap_dirs = {
            f32vec3{ 1.0f,  0.0f,  0.0f},
            f32vec3{-1.0f,  0.0f,  0.0f},
            f32vec3{ 0.0f,  1.0f,  0.0f},
            f32vec3{ 0.0f, -1.0f,  0.0f},
            f32vec3{ 0.0f,  0.0f,  1.0f},
            f32vec3{ 0.0f,  0.0f, -1.0f},
        };
        constexpr std::array<f32vec3, 6> cubemap_ups = {
            f32vec3{ 0.0f,  0.0f,  1.0f},
            f32vec3{ 0.0f,  0.0f,  1.0f},
            f32vec3{ 0.0f,  0.0f,  1.0f},
            f32vec3{ 0.0f,  0.0f,  1.0f},
            f32vec3{ 0.0f,  1.0f,  0.0f},
            f32vec3{ 0.0f,  1.0f,  0.0f},
        };

        DBG_ASSERT_TRUE_M(active_lights.size() == MAX_POINT_LIGHTS, "FIXME(msakmary)");
        for(int point_light_idx = 0; point_light_idx < MAX_POINT_LIGHTS; ++point_light_idx)
        {
            auto & vsm_point_light = point_lights_cpu.at(point_light_idx);
            auto & active_light = active_lights.at(point_light_idx);
            vsm_point_light.light = active_light.point_light_ptr;
            vsm_point_light.page_table = point_page_tables.get_state().images[point_light_idx].default_view();

            for(i32 i = 0; i < 6; ++i){
                vsm_point_light.view_matrices[i] = glm::lookAt(active_light.position, active_light.position + cubemap_dirs.at(i), cubemap_ups.at(i));
                vsm_point_light.inverse_view_matrices[i] = glm::inverse(vsm_point_light.view_matrices[i]);
            }
        }
    }

    void initialize_persitent_state(GPUContext * gpu_context)
    {
        auto inf_depth_reverse_z_perspective = [](auto fov_rads, auto aspect, auto zNear)
        {
            assert(abs(aspect - std::numeric_limits<f32>::epsilon()) > 0.0f);

            f32 const tanHalfFovy = 1.0f / std::tan(fov_rads * 0.5f);

            glm::mat4x4 ret(0.0f);
            ret[0][0] = tanHalfFovy / aspect;
            ret[1][1] = -tanHalfFovy;
            ret[2][2] = 0.0f;
            ret[2][3] = -1.0f;
            ret[3][2] = zNear;
            return ret;
        };

        globals_cpu.point_light_projection_matrix = inf_depth_reverse_z_perspective(glm::radians(90.0f), 1.0f, 0.001f);
        globals_cpu.inverse_point_light_projection_matrix = glm::inverse(globals_cpu.point_light_projection_matrix);

        globals = daxa::TaskBuffer({
            .initial_buffers = {
                .buffers = std::array{
                    gpu_context->device.create_buffer({
                        .size = static_cast<daxa_u32>(sizeof(VSMGlobals)),
                        .name = "vsm globals physical buffer",
                    }),
                },
            },
            .name = "vsm globals buffer",
        });

        memory_block = daxa::TaskImage({
            .initial_images = {
                .images = std::array{
                    gpu_context->device.create_image({
                        .flags = daxa::ImageCreateFlagBits::ALLOW_MUTABLE_FORMAT,
                        .format = daxa::Format::R32_SFLOAT,
                        .size = {VSM_MEMORY_RESOLUTION, VSM_MEMORY_RESOLUTION, 1},
                        .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
                        .name = "vsm memory block physical image",
                    }),
                },
            },
            .name = "vsm memory block",
        });

        meta_memory_table = daxa::TaskImage({
            .initial_images = {
                .images = std::array{
                    gpu_context->device.create_image({
                        .flags = daxa::ImageCreateFlagBits::ALLOW_MUTABLE_FORMAT,
                        .format = daxa::Format::R32_UINT,
                        .size = {VSM_META_MEMORY_TABLE_RESOLUTION, VSM_META_MEMORY_TABLE_RESOLUTION, 1},
                        .usage =
                            daxa::ImageUsageFlagBits::SHADER_SAMPLED |
                            daxa::ImageUsageFlagBits::SHADER_STORAGE |
                            daxa::ImageUsageFlagBits::TRANSFER_DST,
                        .name = "vsm meta memory table physical image",
                    }),
                },
            },
            .name = "vsm meta memory table",
        });

        page_table = daxa::TaskImage({
            .initial_images = {
                .images = std::array{
                    gpu_context->device.create_image({
                        .format = daxa::Format::R32_UINT,
                        .size = {VSM_PAGE_TABLE_RESOLUTION, VSM_PAGE_TABLE_RESOLUTION, 1},
                        .array_layer_count = VSM_CLIP_LEVELS,
                        .usage =
                            daxa::ImageUsageFlagBits::SHADER_SAMPLED |
                            daxa::ImageUsageFlagBits::SHADER_STORAGE |
                            daxa::ImageUsageFlagBits::TRANSFER_DST,
                        .name = "vsm page table physical image",
                    }),
                },
            },
            .name = "vsm page table",
        });

        page_view_pos_row = daxa::TaskImage({
            .initial_images = {
                .images = std::array{
                    gpu_context->device.create_image({
                        .format = daxa::Format::R32G32B32A32_SFLOAT,
                        .size = {VSM_PAGE_TABLE_RESOLUTION, VSM_PAGE_TABLE_RESOLUTION, 1},
                        .array_layer_count = VSM_CLIP_LEVELS,
                        .usage =
                            daxa::ImageUsageFlagBits::SHADER_SAMPLED |
                            daxa::ImageUsageFlagBits::SHADER_STORAGE |
                            daxa::ImageUsageFlagBits::TRANSFER_DST,
                        .name = "vsm page height offsets physical image",
                    }),
                },
            },
            .name = "vsm page height offsets",
        });

        std::array<daxa::ImageId, MAX_POINT_LIGHTS> page_image_ids{};

        const u32 mip_levels = s_cast<u32>(std::log2(VSM_PAGE_TABLE_RESOLUTION)) + 1u;
        for(int point_light_vsm = 0; point_light_vsm < MAX_POINT_LIGHTS; point_light_vsm++)
        {
            page_image_ids.at(point_light_vsm) = gpu_context->device.create_image({
                .flags = daxa::ImageCreateFlagBits::COMPATIBLE_CUBE,
                .format = daxa::Format::R32_UINT,
                .size = {VSM_PAGE_TABLE_RESOLUTION, VSM_PAGE_TABLE_RESOLUTION, 1},
                .mip_level_count = mip_levels,
                .array_layer_count = 6,
                .usage = 
                    daxa::ImageUsageFlagBits::SHADER_SAMPLED |
                    daxa::ImageUsageFlagBits::SHADER_STORAGE |
                    daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = fmt::format("vsm point table {} phys image", point_light_vsm)
            });
        }

        point_page_tables = daxa::TaskImage({
            .initial_images = { .images = page_image_ids },
            .name = "vsm point tables"
        });

        auto upload_task_graph = daxa::TaskGraph({
            .device = gpu_context->device,
            .name = "upload task graph",
        });
        upload_task_graph.use_persistent_image(page_table);
        upload_task_graph.use_persistent_image(meta_memory_table);
        upload_task_graph.use_persistent_image(point_page_tables);

        auto const page_table_array_view = page_table.view().view({.base_array_layer = 0, .layer_count = VSM_CLIP_LEVELS});
        auto const point_table_array_view = point_page_tables.view().view({
            .base_mip_level = 0,
            .level_count = mip_levels - 1,
            .base_array_layer = 0,
            .layer_count = 6
        });

        upload_task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageViewType::REGULAR_2D_ARRAY, page_table_array_view),
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageViewType::REGULAR_2D_ARRAY, point_table_array_view),
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, meta_memory_table),
            },
            .task = [&](daxa::TaskInterface ti)
            {
                ti.recorder.clear_image({
                    .clear_value = std::array<daxa_u32, 4>{0u, 0u, 0u, 0u},
                    .dst_image = ti.get(page_table_array_view).ids[0],
                    .dst_slice = daxa::ImageMipArraySlice{
                        .base_array_layer = 0,
                        .layer_count = VSM_CLIP_LEVELS},
                });

                ti.recorder.clear_image({
                    .clear_value = std::array<daxa_u32, 4>{0u, 0u, 0u, 0u},
                    .dst_image = ti.get(meta_memory_table).ids[0],
                });

                for(int point_light_vsm = 0; point_light_vsm < MAX_POINT_LIGHTS; ++point_light_vsm)
                {
                    ti.recorder.clear_image({
                        .clear_value = std::array<daxa_u32, 4>{0u, 0u, 0u, 0u},
                        .dst_image = ti.get(point_table_array_view).ids[point_light_vsm],
                        .dst_slice = daxa::ImageMipArraySlice{
                            .base_mip_level = 0,
                            .level_count = mip_levels - 1,
                            .base_array_layer = 0,
                            .layer_count = 6,
                        },
                    });
                }
            },
        });
        upload_task_graph.submit({});
        upload_task_graph.complete({});
        upload_task_graph.execute({});
    }

    void cleanup_persistent_state(GPUContext * gpu_context)
    {
        gpu_context->device.destroy_buffer(globals.get_state().buffers[0]);
        gpu_context->device.destroy_image(memory_block.get_state().images[0]);
        gpu_context->device.destroy_image(meta_memory_table.get_state().images[0]);
        gpu_context->device.destroy_image(page_table.get_state().images[0]);
        gpu_context->device.destroy_image(page_view_pos_row.get_state().images[0]);
        for(i32 i = 0; i < MAX_POINT_LIGHTS; ++i)
        {
            gpu_context->device.destroy_image(point_page_tables.get_state().images[i]);
        }
    }

    void initialize_transient_state(daxa::TaskGraph & tg, RenderGlobalData const& rgd)
    {
        free_wrapped_pages_info = tg.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(FreeWrappedPagesInfo) * VSM_CLIP_LEVELS),
            .name = "vsm free wrapped pages info",
        });

        allocation_requests = tg.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(VSMAllocationRequestsHeader)),
            .name = "vsm allocation requests",
        });

        free_page_buffer = tg.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(PageCoordBuffer) * MAX_VSM_ALLOC_REQUESTS),
            .name = "vsm free page buffer",
        });

        not_visited_page_buffer = tg.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(PageCoordBuffer) * MAX_VSM_ALLOC_REQUESTS),
            .name = "vsm not visited page buffer",
        });

        find_free_pages_header = tg.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(FindFreePagesHeader)),
            .name = "find free pages header",
        });

        clip_projections = tg.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(VSMClipProjection) * VSM_CLIP_LEVELS),
            .name = "vsm clip projections",
        });

        allocate_indirect = tg.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(DispatchIndirectStruct)),
            .name = "vsm allocate indirect",
        });

        clear_indirect = tg.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(DispatchIndirectStruct)),
            .name = "vsm clear indirect",
        });

        clear_dirty_bit_indirect = tg.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(DispatchIndirectStruct)),
            .name = "vsm clear dirty bit indirect",
        });

        vsm_point_lights = tg.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(VSMPointLight) * MAX_POINT_LIGHTS),
            .name = "vsm point light infos",
        });

        auto const hiz_size = daxa::Extent3D{VSM_PAGE_TABLE_RESOLUTION, VSM_PAGE_TABLE_RESOLUTION, 1};

        dirty_pages_hiz = tg.create_transient_image({
            .dimensions = 2,
            .format = daxa::Format::R8_UINT,
            .size = hiz_size,
            .mip_level_count = s_cast<u32>(std::log2(VSM_PAGE_TABLE_RESOLUTION)) + 1,
            .array_layer_count = VSM_CLIP_LEVELS,
            .name = "vsm dirty hiz",
        });

        overdraw_debug_image = daxa::NullTaskImage;
        if (rgd.settings.debug_draw_mode == DEBUG_DRAW_MODE_VSM_OVERDRAW)
        {
            overdraw_debug_image = tg.create_transient_image({
                .dimensions = 2,
                .format = daxa::Format::R32_UINT,
                .size = {VSM_MEMORY_RESOLUTION, VSM_MEMORY_RESOLUTION, 1},
                .name = "vsm overdraw debug image",
            }); 
        }

        tg.clear_buffer({.buffer = allocation_requests, .clear_value = 0});
        tg.clear_buffer({.buffer = find_free_pages_header, .clear_value = 0});
    }

    void zero_out_transient_state(daxa::TaskGraph & tg, RenderGlobalData const& rgd)
    {
        free_wrapped_pages_info = daxa::NullTaskBuffer;
        allocation_requests = daxa::NullTaskBuffer;
        free_page_buffer = daxa::NullTaskBuffer;
        not_visited_page_buffer = daxa::NullTaskBuffer;
        find_free_pages_header = daxa::NullTaskBuffer;
        clip_projections = daxa::NullTaskBuffer;
        allocate_indirect = daxa::NullTaskBuffer;
        clear_indirect = daxa::NullTaskBuffer;
        clear_dirty_bit_indirect = daxa::NullTaskBuffer;
        dirty_pages_hiz = daxa::NullTaskImage;
        overdraw_debug_image = daxa::NullTaskImage;
    }
};