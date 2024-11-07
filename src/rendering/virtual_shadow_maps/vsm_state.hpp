#pragma once
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../gpu_context.hpp"
#include "../../shader_shared/geometry_pipeline.inl"
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/gpu_work_expansion.inl"
#include "../tasks/misc.hpp"

struct VSMState
{
    // Persistent state
    daxa::TaskBuffer globals = {};

    daxa::TaskImage memory_block = {};
    daxa::TaskImage memory_block64 = {};
    daxa::TaskImage meta_memory_table = {};
    daxa::TaskImage page_table = {};
    daxa::TaskImage page_view_pos_row = {};

    // Transient state
    daxa::TaskBufferView allocation_count = {};
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
    std::array<daxa::TaskBufferView, PREPASS_DRAW_LIST_TYPE_COUNT> meshlet_cull_po2expansions = {};

    daxa::TimelineQueryPool vsm_timeline_query_pool = {};

    std::array<VSMClipProjection, VSM_CLIP_LEVELS> clip_projections_cpu = {};
    std::array<FreeWrappedPagesInfo, VSM_CLIP_LEVELS> free_wrapped_pages_info_cpu = {};
    std::array<i32vec2, VSM_CLIP_LEVELS> last_frame_offsets = {};
    VSMGlobals globals_cpu = {};
    static constexpr u32 VSM_TASK_COUNT = 11;
    static constexpr u32 PER_FRAME_TIMESTAMP_COUNT = VSM_TASK_COUNT * 2;


    void initialize_persitent_state(GPUContext * gpu_context)
    {
        vsm_timeline_query_pool = gpu_context->device.create_timeline_query_pool({
            .query_count = PER_FRAME_TIMESTAMP_COUNT * s_cast<u32>(gpu_context->swapchain.info().max_allowed_frames_in_flight + 1),
            .name = "vsm_timestamp_query_pool"
        });

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
        memory_block64 = daxa::TaskImage({
            .initial_images = {
                .images = std::array{
                    gpu_context->device.create_image({
                        .flags = daxa::ImageCreateFlagBits::ALLOW_MUTABLE_FORMAT,
                        .format = daxa::Format::R64_UINT,
                        .size = {VSM_MEMORY_RESOLUTION, VSM_MEMORY_RESOLUTION, 1},
                        .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE,
                        .name = "vsm memory block physical image 64bit",
                    }),
                },
            },
            .name = "vsm memory block 64bit",
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


        auto upload_task_graph = daxa::TaskGraph({
            .device = gpu_context->device,
            .name = "upload task graph",
        });
        upload_task_graph.use_persistent_image(page_table);
        upload_task_graph.use_persistent_image(meta_memory_table);

        auto page_table_array_view = page_table.view().view({.base_array_layer = 0, .layer_count = VSM_CLIP_LEVELS});
        upload_task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageViewType::REGULAR_2D_ARRAY, page_table_array_view),
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
            },
        });
    }

    void cleanup_persistent_state(GPUContext * gpu_context)
    {
        gpu_context->device.destroy_buffer(globals.get_state().buffers[0]);
        gpu_context->device.destroy_image(memory_block.get_state().images[0]);
        gpu_context->device.destroy_image(memory_block64.get_state().images[0]);
        gpu_context->device.destroy_image(meta_memory_table.get_state().images[0]);
        gpu_context->device.destroy_image(page_table.get_state().images[0]);
        gpu_context->device.destroy_image(page_view_pos_row.get_state().images[0]);
    }

    void initialize_transient_state(daxa::TaskGraph & tg, RenderGlobalData const& rgd)
    {
        free_wrapped_pages_info = tg.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(FreeWrappedPagesInfo) * VSM_CLIP_LEVELS),
            .name = "vsm free wrapped pages info",
        });

        allocation_count = tg.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(AllocationCount)),
            .name = "vsm allocation count",
        });

        allocation_requests = tg.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(AllocationRequest) * MAX_VSM_ALLOC_REQUESTS),
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

        meshlet_cull_po2expansions = std::array{
            tg.create_transient_buffer({
                .size = static_cast<u32>(sizeof(Po2WorkExpansionBufferHead)),
                .name = "vsm meshlett po2expansion",
            }),
            tg.create_transient_buffer({
                .size = static_cast<u32>(sizeof(Po2WorkExpansionBufferHead)),
                .name = "vsm meshlett masked po2expansion",
            }),
        };


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

        tg.clear_buffer({.buffer = allocation_count, .clear_value = 0});
        tg.clear_buffer({.buffer = find_free_pages_header, .clear_value = 0});
    }

    void zero_out_transient_state(daxa::TaskGraph & tg, RenderGlobalData const& rgd)
    {
        free_wrapped_pages_info = daxa::NullTaskBuffer;
        allocation_count = daxa::NullTaskBuffer;
        allocation_requests = daxa::NullTaskBuffer;
        free_page_buffer = daxa::NullTaskBuffer;
        not_visited_page_buffer = daxa::NullTaskBuffer;
        find_free_pages_header = daxa::NullTaskBuffer;
        clip_projections = daxa::NullTaskBuffer;
        allocate_indirect = daxa::NullTaskBuffer;
        clear_indirect = daxa::NullTaskBuffer;
        clear_dirty_bit_indirect = daxa::NullTaskBuffer;
        meshlet_cull_po2expansions = std::array{ daxa::NullTaskBuffer, daxa::NullTaskBuffer };
        dirty_pages_hiz = daxa::NullTaskImage;
        overdraw_debug_image = daxa::NullTaskImage;
    }
};