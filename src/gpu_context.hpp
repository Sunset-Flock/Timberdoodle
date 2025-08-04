#pragma once

#include "timberdoodle.hpp"
#include "window.hpp"
#include "shader_shared/shared.inl"
#include "shader_shared/globals.inl"
#include "shader_shared/vsm_shared.inl"
#include "daxa_helper.hpp"

template<typename T>
struct CPUDebugDraws
{
    u32 max_draws = {};
    u32 vertices = {};
    std::vector<T> cpu_draws = {};

    void draw(T const & draw)
    {
        if (cpu_draws.size() < max_draws)
        {
            cpu_draws.push_back(draw);
        }
    }
};

struct ShaderDebugDrawContext
{
    CPUDebugDraws<ShaderDebugLineDraw> line_draws = { .max_draws = 1u << 16u, .vertices = 2 };
    CPUDebugDraws<ShaderDebugCircleDraw> circle_draws = { .max_draws = 1u << 16u, .vertices = 128 };
    CPUDebugDraws<ShaderDebugRectangleDraw> rectangle_draws = { .max_draws = 1u << 16u, .vertices = 8 };
    CPUDebugDraws<ShaderDebugAABBDraw> aabb_draws = { .max_draws = 1u << 16u, .vertices = 24 };
    CPUDebugDraws<ShaderDebugBoxDraw> box_draws = { .max_draws = 1u << 14u, .vertices = 24 };
    CPUDebugDraws<ShaderDebugConeDraw> cone_draws = { .max_draws = 1u << 14u, .vertices = 64 };
    CPUDebugDraws<ShaderDebugSphereDraw> sphere_draws = { .max_draws = 1u << 14u, .vertices = (5*64) };
    daxa::BufferId buffer = {};
    ShaderDebugInput shader_debug_input = {};
    ShaderDebugOutput shader_debug_output = {};
    daxa_i32vec2 detector_window_position = {};
    i32 detector_window_size = 15;
    i32 old_detector_rt_size = 0;
    
    daxa::TaskImage vsm_debug_meta_memory_table = {};
    daxa::TaskImage vsm_recreated_shadowmap_memory_table = {};
    daxa::TaskImage vsm_debug_page_table = {};
    daxa::BufferId readback_queue = {};

    u32 frame_index = 0;

    void init(daxa::Device & device)
    {
        usize size = sizeof(ShaderDebugBufferHead);
        size += sizeof(ShaderDebugLineDraw) * line_draws.max_draws;
        size += sizeof(ShaderDebugCircleDraw) * circle_draws.max_draws;
        size += sizeof(ShaderDebugRectangleDraw) * rectangle_draws.max_draws;
        size += sizeof(ShaderDebugAABBDraw) * aabb_draws.max_draws;
        size += sizeof(ShaderDebugBoxDraw) * box_draws.max_draws;
        size += sizeof(ShaderDebugConeDraw) * cone_draws.max_draws;
        size += sizeof(ShaderDebugSphereDraw) * sphere_draws.max_draws;
        buffer = device.create_buffer({
            .size = size,
            .name = "shader debug buffer",
        });

        vsm_debug_page_table = daxa::TaskImage({
            .initial_images = {
                .images = std::array{
                    device.create_image({
                        .format = daxa::Format::R8G8B8A8_UNORM,
                        .size = {VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION, VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION, 1},
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

        vsm_recreated_shadowmap_memory_table = daxa::TaskImage({
            .initial_images = {
                .images = std::array{
                    device.create_image({
                        .format = daxa::Format::R8G8B8A8_SRGB,
                        .size = {VSM_DIRECTIONAL_TEXTURE_RESOLUTION, VSM_DIRECTIONAL_TEXTURE_RESOLUTION, 1},
                        .usage = 
                            daxa::ImageUsageFlagBits::SHADER_SAMPLED |
                            daxa::ImageUsageFlagBits::SHADER_STORAGE |
                            daxa::ImageUsageFlagBits::TRANSFER_DST,
                        .name = "vsm recreated shadowmap physical image",
                    }),
                },
            },
        });

        readback_queue = device.create_buffer({
            .size = sizeof(ShaderDebugOutput) * (MAX_GPU_FRAMES_IN_FLIGHT + 1),
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM, // cpu side buffer.
            .name = "shader debug readback queue",
        });
    }

    void update(daxa::Device & device, daxa_u32vec2 render_target_size, i32vec2 window_size, u32 renderer_frame_index)
    {
        frame_index = renderer_frame_index;

        // Readback
        {
            u32 const readback_index = frame_index % (MAX_GPU_FRAMES_IN_FLIGHT+1);
            auto& readback_buffer_ref = device.buffer_host_address_as<ShaderDebugOutput>(readback_queue).value()[readback_index];
            shader_debug_output = readback_buffer_ref;  // read back from previous frame
            readback_buffer_ref = {};                   // clear for next frame
        }
    }
    
    void update_debug_buffer(daxa::Device & device, daxa::CommandRecorder & recorder, daxa::TransferMemoryPool & allocator)
    {
        u32 const readback_index = frame_index % (MAX_GPU_FRAMES_IN_FLIGHT+1);
        u32 buffer_mem_offset = sizeof(ShaderDebugBufferHead);
        auto head = ShaderDebugBufferHead{
            .cpu_input = shader_debug_input,
            .gpu_output = device.device_address(readback_queue).value() + sizeof(ShaderDebugOutput) * readback_index,
        };

        auto update_debug_draws = [&](auto& draws, auto& cpu_draws)
        {
            u32 const offset = buffer_mem_offset;
            buffer_mem_offset += cpu_draws.max_draws * sizeof(decltype(cpu_draws.cpu_draws[0]));
            draws.draw_indirect = {
                .vertex_count = cpu_draws.vertices,
                .instance_count = std::min(static_cast<u32>(cpu_draws.cpu_draws.size()), cpu_draws.max_draws),
                .first_vertex = 0,
                .first_instance = 0,
            };
            draws.draw_capacity = cpu_draws.max_draws;
            draws.draw_requests = std::min(static_cast<u32>(cpu_draws.cpu_draws.size()), cpu_draws.max_draws);
            draws.draws = device.device_address(buffer).value() + offset;
        };
        
        update_debug_draws(head.line_draws, line_draws);
        update_debug_draws(head.circle_draws, circle_draws);
        update_debug_draws(head.rectangle_draws, rectangle_draws);
        update_debug_draws(head.aabb_draws, aabb_draws);
        update_debug_draws(head.box_draws, box_draws);
        update_debug_draws(head.cone_draws, cone_draws);
        update_debug_draws(head.sphere_draws, sphere_draws);

        auto alloc = allocator.allocate_fill(head).value();
        recorder.copy_buffer_to_buffer({
            .src_buffer = allocator.buffer(),
            .dst_buffer = buffer,
            .src_offset = alloc.buffer_offset,
            .dst_offset = 0,
            .size = sizeof(ShaderDebugBufferHead),
        });

        auto upload_debug_draws = [&](auto& draws, auto& cpu_draws){
            u32 const upload_size = sizeof(decltype(cpu_draws.cpu_draws[0])) * draws.draw_indirect.instance_count;
            if (upload_size > 0)
            {
                u32 const buffer_offset = draws.draws - device.device_address(buffer).value();
                auto stage_line_draws = allocator.allocate(upload_size).value();
                std::memcpy(stage_line_draws.host_address, cpu_draws.cpu_draws.data(), upload_size);
                recorder.copy_buffer_to_buffer({
                    .src_buffer = allocator.buffer(),
                    .dst_buffer = buffer,
                    .src_offset = stage_line_draws.buffer_offset,
                    .dst_offset = buffer_offset,
                    .size = upload_size,
                });
                cpu_draws.cpu_draws.clear();
            }
        };

        upload_debug_draws(head.line_draws, line_draws);
        upload_debug_draws(head.circle_draws, circle_draws);
        upload_debug_draws(head.rectangle_draws, rectangle_draws);
        upload_debug_draws(head.aabb_draws, aabb_draws);
        upload_debug_draws(head.box_draws, box_draws);
        upload_debug_draws(head.cone_draws, cone_draws);
        upload_debug_draws(head.sphere_draws, sphere_draws);
    }
};

DAXA_DECL_TRANSFER_TASK_HEAD_BEGIN(ReadbackH)
DAXA_TH_BUFFER(READ, globals); // Use globals as fake dependency for shader debug data.
DAXA_DECL_TASK_HEAD_END

struct ReadbackTask : ReadbackH::Task
{
    AttachmentViews views = {};
    ShaderDebugDrawContext * shader_debug_context = {};
    void callback(daxa::TaskInterface ti)
    {
        u32 const index = ((shader_debug_context->frame_index-1) % (MAX_GPU_FRAMES_IN_FLIGHT+1));
        std::memcpy(&shader_debug_context->shader_debug_output, ti.device.buffer_host_address(shader_debug_context->readback_queue).value(), sizeof(ShaderDebugOutput));
        // Set the currently recording frame to write its debug output to the slot we just read from.
        ti.recorder.copy_buffer_to_buffer({
            .src_buffer = shader_debug_context->buffer,
            .dst_buffer = shader_debug_context->readback_queue,
            .src_offset = offsetof(ShaderDebugBufferHead, gpu_output),
            .dst_offset = sizeof(ShaderDebugOutput) * index,
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
    daxa::Instance instance = {};
    daxa::Device device = {};
    daxa::Swapchain swapchain = {};
    daxa::PipelineManager pipeline_manager = {};

    ShaderDebugDrawContext shader_debug_context = {};

    struct RayTracingPipelineInfo
    {
        std::shared_ptr<daxa::RayTracingPipeline> pipeline = {};
        daxa::RayTracingShaderBindingTable sbt = {};
        daxa::BufferId sbt_buffer_id = {};
    };
    // Pipelines:
    std::unordered_map<std::string, std::shared_ptr<daxa::RasterPipeline>> raster_pipelines = {};
    std::unordered_map<std::string, std::shared_ptr<daxa::ComputePipeline>> compute_pipelines = {};
    std::unordered_map<std::string, RayTracingPipelineInfo> ray_tracing_pipelines = {};

    // TODO(msakmary) REMOVE
    daxa::SamplerId lin_clamp_sampler = {};
    daxa::SamplerId nearest_clamp_sampler = {};

    u32 counter = {};
    auto dummy_string() -> std::string;
};