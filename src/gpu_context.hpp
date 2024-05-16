#pragma once

#include "timberdoodle.hpp"
#include "window.hpp"
#include "shader_shared/shared.inl"
#include "shader_shared/globals.inl"
#include "shader_shared/vsm_shared.inl"

struct ShaderDebugDrawContext
{
    u32 max_circle_draws = 256'000;
    u32 max_rectangle_draws = 256'000;
    u32 max_aabb_draws = 256'000;
    u32 max_box_draws = 64'000;
    u32 max_line_draws = 512'000;
    u32 circle_vertices = 64; // Uses line strip
    u32 rectangle_vertices = 5; // Uses line strip
    u32 aabb_vertices = 24; // Uses line list
    u32 box_vertices = 24;
    u32 line_vertices = 2;
    daxa::BufferId buffer = {};
    ShaderDebugInput shader_debug_input = {};
    ShaderDebugOutput shader_debug_output = {};
    daxa_i32vec2 detector_window_position = {};
    i32 detector_window_size = 15;
    i32 old_detector_rt_size = 0;
    bool draw_magnified_area_rect = true;
    
    daxa::ImageInfo debug_lens_image_create_info = { 
        .format = daxa::Format::R16G16B16A16_SFLOAT, 
        .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | 
            daxa::ImageUsageFlagBits::SHADER_STORAGE | 
            daxa::ImageUsageFlagBits::TRANSFER_DST,
        .name = "debug detector image",
    };
    daxa::ImageId debug_lens_image = {};
    daxa::TaskImage tdebug_lens_image = {};
    daxa::TaskImage vsm_debug_meta_memory_table = {};
    daxa::TaskImage vsm_debug_page_table = {};
    daxa::BufferId readback_queue = {};

    std::vector<ShaderDebugCircleDraw> cpu_debug_circle_draws = {};
    std::vector<ShaderDebugRectangleDraw> cpu_debug_rectangle_draws = {};
    std::vector<ShaderDebugAABBDraw> cpu_debug_aabb_draws = {};
    std::vector<ShaderDebugBoxDraw> cpu_debug_box_draws = {};
    std::vector<ShaderDebugLineDraw> cpu_debug_line_draws = {};

    u32 frame_index = 0;

    void init(daxa::Device & device)
    {
        usize size = sizeof(ShaderDebugBufferHead);
        size += sizeof(ShaderDebugCircleDraw) * max_circle_draws;
        size += sizeof(ShaderDebugRectangleDraw) * max_rectangle_draws;
        size += sizeof(ShaderDebugAABBDraw) * max_aabb_draws;
        size += sizeof(ShaderDebugBoxDraw) * max_box_draws;
        size += sizeof(ShaderDebugLineDraw) * max_line_draws;
        buffer = device.create_buffer({
            .size = size,
            .name = "shader debug buffer",
        });
        debug_lens_image_create_info.size = { static_cast<u32>(detector_window_size), static_cast<u32>(detector_window_size), 1 };
        debug_lens_image = device.create_image(debug_lens_image_create_info);
        tdebug_lens_image = daxa::TaskImage({
            .initial_images = {.images = std::array{debug_lens_image}}, 
            .name = "debug detector image",
        });

        vsm_debug_page_table = daxa::TaskImage({
            .initial_images = {
                .images = std::array{
                    device.create_image({
                        .format = daxa::Format::R8G8B8A8_UNORM,
                        .size = {VSM_PAGE_TABLE_RESOLUTION, VSM_PAGE_TABLE_RESOLUTION, 1},
                        .usage =
                            daxa::ImageUsageFlagBits::SHADER_SAMPLED |
                            daxa::ImageUsageFlagBits::SHADER_STORAGE |
                            daxa::ImageUsageFlagBits::TRANSFER_DST,
                        .name = "vsm debug page table physical image",
                    }),
                },
            },
            .name = "vsm debug page table",
        });

        vsm_debug_meta_memory_table = daxa::TaskImage({
            .initial_images = {
                .images = std::array{
                    device.create_image({
                        .format = daxa::Format::R8G8B8A8_UNORM,
                        .size = {VSM_META_MEMORY_TABLE_RESOLUTION, VSM_META_MEMORY_TABLE_RESOLUTION, 1},
                        .usage =
                            daxa::ImageUsageFlagBits::SHADER_SAMPLED |
                            daxa::ImageUsageFlagBits::SHADER_STORAGE |
                            daxa::ImageUsageFlagBits::TRANSFER_DST,
                        .name = "vsm debug meta memory table physical image",
                    }),
                },
            },
            .name = "vsm debug meta memory table",
        });

        readback_queue = device.create_buffer({
            .size = sizeof(ShaderDebugOutput) * 4 /*4 is a save value for all kinds of frames in flight setups*/,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM, // cpu side buffer.
            .name = "shader debug readback queue",
        });
    }

    void update(daxa::Device & device, daxa_u32vec2 render_target_size, i32vec2 window_size)
    {
        i32 const res_factor = static_cast<u32>(render_target_size.y) / window_size.y;
        i32 const actual_detector_size = res_factor * detector_window_size;
        if (actual_detector_size != old_detector_rt_size)
        {
            if (device.is_id_valid(debug_lens_image))
            {
                device.destroy_image(debug_lens_image);
            }
            debug_lens_image_create_info.size = { static_cast<u32>(actual_detector_size), static_cast<u32>(actual_detector_size), 1 };
            debug_lens_image = device.create_image(debug_lens_image_create_info);
            tdebug_lens_image.set_images({.images=std::array{debug_lens_image}});
            old_detector_rt_size = actual_detector_size;
        }
        if (draw_magnified_area_rect)
        {
            auto u = (static_cast<f32>(detector_window_position.x) + 0.5f) / static_cast<f32>(window_size.x);
            auto v = (static_cast<f32>(detector_window_position.y) + 0.5f) / static_cast<f32>(window_size.y);
            auto span_u = (static_cast<f32>(detector_window_size + 2)) / static_cast<f32>(window_size.x);
            auto span_v = (static_cast<f32>(detector_window_size + 2)) / static_cast<f32>(window_size.y);
            
            cpu_debug_aabb_draws.push_back(ShaderDebugAABBDraw{
                .position = {u * 2.0f - 1.0f, v * 2.0f - 1.0f, 0.5},
                .size = {span_u * 2.0f, span_v * 2.0f, 0.99999999 },
                .color = daxa_f32vec3(1,0,0),
                .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER,
            });
        }
        shader_debug_input.texel_detector_window_half_size = actual_detector_size / 2;
        shader_debug_input.texel_detector_pos.x = detector_window_position.x * res_factor;
        shader_debug_input.texel_detector_pos.y = detector_window_position.y * res_factor;
        frame_index += 1;
    }
    
    void update_debug_buffer(daxa::Device & device, daxa::CommandRecorder & recorder, daxa::TransferMemoryPool & allocator)
    {
        u32 const circle_buffer_offset = sizeof(ShaderDebugBufferHead);
        u32 const rectangle_buffer_offset = circle_buffer_offset + sizeof(ShaderDebugCircleDraw) * max_circle_draws;
        u32 const aabb_buffer_offset = rectangle_buffer_offset + sizeof(ShaderDebugRectangleDraw) * max_rectangle_draws;
        u32 const box_buffer_offset = aabb_buffer_offset + sizeof(ShaderDebugAABBDraw) * max_aabb_draws;
        u32 const line_buffer_offset = aabb_buffer_offset + sizeof(ShaderDebugBoxDraw) * max_box_draws;
        
        auto head = ShaderDebugBufferHead{
            .circle_draw_indirect_info = {
                .vertex_count = circle_vertices,
                .instance_count = std::min(static_cast<u32>(cpu_debug_circle_draws.size()), max_circle_draws),
                .first_vertex = 0,
                .first_instance = 0,
            },
            .rectangle_draw_indirect_info = {
                .vertex_count = rectangle_vertices,
                .instance_count = std::min(static_cast<u32>(cpu_debug_rectangle_draws.size()), max_rectangle_draws),
                .first_vertex = 0,
                .first_instance = 0,
            },
            .aabb_draw_indirect_info = {
                .vertex_count = aabb_vertices,
                .instance_count = std::min(static_cast<u32>(cpu_debug_aabb_draws.size()), max_aabb_draws),
                .first_vertex = 0,
                .first_instance = 0,
            },
            .box_draw_indirect_info = {
                .vertex_count = box_vertices,
                .instance_count = std::min(static_cast<u32>(cpu_debug_box_draws.size()), max_box_draws),
                .first_vertex = 0,
                .first_instance = 0
            },
            .line_draw_indirect_info = {
                .vertex_count = line_vertices,
                .instance_count = std::min(static_cast<u32>(cpu_debug_line_draws.size()), max_line_draws),
                .first_vertex = 0,
                .first_instance = 0
            },
            .circle_draw_capacity = max_circle_draws,
            .rectangle_draw_capacity = max_rectangle_draws,
            .aabb_draw_capacity = max_aabb_draws,
            .box_draw_capacity = max_box_draws,
            .line_draw_capacity = max_line_draws,
            .cpu_input = shader_debug_input,
            .gpu_output = {},
            .circle_draws = device.get_device_address(buffer).value() + circle_buffer_offset,
            .rectangle_draws = device.get_device_address(buffer).value() + rectangle_buffer_offset,
            .aabb_draws = device.get_device_address(buffer).value() + aabb_buffer_offset,
            .box_draws = device.get_device_address(buffer).value() + box_buffer_offset,
            .line_draws = device.get_device_address(buffer).value() + line_buffer_offset,
        };
        auto alloc = allocator.allocate_fill(head).value();
        recorder.copy_buffer_to_buffer({
            .src_buffer = allocator.buffer(),
            .dst_buffer = buffer,
            .src_offset = alloc.buffer_offset,
            .dst_offset = 0,
            .size = sizeof(ShaderDebugBufferHead),
        });
        
        auto stage_circle_draws_size = sizeof(ShaderDebugCircleDraw) * head.circle_draw_indirect_info.instance_count;
        if (stage_circle_draws_size > 0)
        {
            auto stage_circle_draws = allocator.allocate(stage_circle_draws_size,4).value();
            std::memcpy(stage_circle_draws.host_address, cpu_debug_circle_draws.data(), stage_circle_draws_size);
            recorder.copy_buffer_to_buffer({
                .src_buffer = allocator.buffer(),
                .dst_buffer = buffer,
                .src_offset = stage_circle_draws.buffer_offset,
                .dst_offset = circle_buffer_offset,
                .size = stage_circle_draws_size,
            });
            cpu_debug_circle_draws.clear();
        }
        
        auto stage_rectangle_draws_size = sizeof(ShaderDebugRectangleDraw) * head.rectangle_draw_indirect_info.instance_count;
        if (stage_rectangle_draws_size > 0)
        {
            auto stage_rectangle_draws = allocator.allocate(stage_rectangle_draws_size,4).value();
            std::memcpy(stage_rectangle_draws.host_address, cpu_debug_rectangle_draws.data(), stage_rectangle_draws_size);
            recorder.copy_buffer_to_buffer({
                .src_buffer = allocator.buffer(),
                .dst_buffer = buffer,
                .src_offset = stage_rectangle_draws.buffer_offset,
                .dst_offset = rectangle_buffer_offset,
                .size = stage_rectangle_draws_size,
            });
            cpu_debug_rectangle_draws.clear();
        }
        
        auto stage_aabb_draws_size = sizeof(ShaderDebugAABBDraw) * head.aabb_draw_indirect_info.instance_count;
        if (stage_aabb_draws_size > 0)
        {
            auto stage_aabb_draws = allocator.allocate(stage_aabb_draws_size,4).value();
            std::memcpy(stage_aabb_draws.host_address, cpu_debug_aabb_draws.data(), stage_aabb_draws_size);
            recorder.copy_buffer_to_buffer({
                .src_buffer = allocator.buffer(),
                .dst_buffer = buffer,
                .src_offset = stage_aabb_draws.buffer_offset,
                .dst_offset = aabb_buffer_offset,
                .size = stage_aabb_draws_size,
            });
            cpu_debug_aabb_draws.clear();
        }

        auto stage_box_draws_size = sizeof(ShaderDebugBoxDraw) * head.box_draw_indirect_info.instance_count;
        if (stage_box_draws_size > 0)
        {
            auto stage_box_draws = allocator.allocate(stage_box_draws_size,4).value();
            std::memcpy(stage_box_draws.host_address, cpu_debug_box_draws.data(), stage_box_draws_size);
            recorder.copy_buffer_to_buffer({
                .src_buffer = allocator.buffer(),
                .dst_buffer = buffer,
                .src_offset = stage_box_draws.buffer_offset,
                .dst_offset = box_buffer_offset,
                .size = stage_box_draws_size,
            });
            cpu_debug_box_draws.clear();
        }
        auto stage_line_draws_size = sizeof(ShaderDebugLineDraw) * head.line_draw_indirect_info.instance_count;
        if (stage_line_draws_size > 0)
        {
            auto stage_line_draws = allocator.allocate(stage_line_draws_size,4).value();
            std::memcpy(stage_line_draws.host_address, cpu_debug_line_draws.data(), stage_line_draws_size);
            recorder.copy_buffer_to_buffer({
                .src_buffer = allocator.buffer(),
                .dst_buffer = buffer,
                .src_offset = stage_line_draws.buffer_offset,
                .dst_offset = line_buffer_offset,
                .size = stage_line_draws_size,
            });
            cpu_debug_line_draws.clear();
        }
    }
};

DAXA_DECL_TASK_HEAD_BEGIN(ReadbackH)
DAXA_TH_BUFFER(TRANSFER_READ, globals); // Use globals as fake dependency for shader debug data.
DAXA_DECL_TASK_HEAD_END

struct ReadbackTask : ReadbackH::Task
{
    AttachmentViews views = {};
    ShaderDebugDrawContext * shader_debug_context = {};
    void callback(daxa::TaskInterface ti)
    {
        // Copy out the debug output from 4 frames ago
        std::memcpy(&shader_debug_context->shader_debug_output, ti.device.get_host_address(shader_debug_context->readback_queue).value(), sizeof(ShaderDebugOutput));
        // Set the currently recording frame to write its debug output to the slot we just read from.
        ti.recorder.copy_buffer_to_buffer({
            .src_buffer = shader_debug_context->buffer,
            .dst_buffer = shader_debug_context->readback_queue,
            .src_offset = offsetof(ShaderDebugBufferHead, gpu_output),
            .dst_offset = sizeof(ShaderDebugOutput) * (shader_debug_context->frame_index % 4),
            .size = sizeof(ShaderDebugOutput),
        });
    }
};

struct GPUContext
{
    GPUContext(Window const & window);
    GPUContext(GPUContext &&) = default;
    ~GPUContext();

    // common unique:
    daxa::Instance context = {};
    daxa::Device device = {};
    daxa::Swapchain swapchain = {};
    daxa::PipelineManager pipeline_manager = {};

    ShaderDebugDrawContext shader_debug_context = {};

    // Pipelines:
    std::unordered_map<std::string, std::shared_ptr<daxa::RasterPipeline>> raster_pipelines = {};
    std::unordered_map<std::string, std::shared_ptr<daxa::ComputePipeline>> compute_pipelines = {};

    // TODO(msakmary) REMOVE
    daxa::SamplerId lin_clamp_sampler = {};

    u32 counter = {};
    auto dummy_string() -> std::string;
};