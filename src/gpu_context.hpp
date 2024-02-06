#pragma once

#include "timberdoodle.hpp"
#include "window.hpp"
#include "shader_shared/shared.inl"
#include "shader_shared/globals.inl"

struct ShaderDebugDrawContext
{
    u32 max_circle_draws = 16'000;
    u32 circle_vertices = 64;
    daxa::RasterPipeline circle_draw_pipeline = {};
    daxa::BufferId buffer = {};

    void init(daxa::Device & device)
    {
        usize size = {};
        usize circle_draw_array_offset = (size += sizeof(ShaderDebugBufferHead));
        size += sizeof(ShaderDebugCircleDraw) * max_circle_draws;
        buffer = device.create_buffer({
            .size = size,
            .name = "shader debug buffer",
        });
    }

    void update_debug_buffer(daxa::Device & device, daxa::CommandRecorder & recorder, daxa::TransferMemoryPool & allocator)
    {
        auto head = ShaderDebugBufferHead{
            .draw_indirect_info = {
                .vertex_count = circle_vertices,
                .instance_count = 0,
                .first_vertex = 0,
                .first_instance = 0,
            },
            .circle_draw_capacity = max_circle_draws,
            .circle_draws = device.get_device_address(buffer).value() + sizeof(ShaderDebugBufferHead),
        };
        auto alloc = allocator.allocate_fill(head).value();
        recorder.copy_buffer_to_buffer({
            .src_buffer = allocator.buffer(),
            .dst_buffer = buffer,
            .src_offset = alloc.buffer_offset,
            .dst_offset = 0,
            .size = sizeof(ShaderDebugBufferHead),
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
    daxa::TransferMemoryPool transient_mem;

    ShaderGlobals shader_globals = {};
    daxa::BufferId shader_globals_buffer = {};
    daxa::TaskBuffer shader_globals_task_buffer = {};
    daxa::types::DeviceAddress shader_globals_address = {};

    ShaderDebugDrawContext debug_draw_info = {};

    // Pipelines:
    std::unordered_map<std::string, std::shared_ptr<daxa::RasterPipeline>> raster_pipelines = {};
    std::unordered_map<std::string_view, std::shared_ptr<daxa::ComputePipeline>> compute_pipelines = {};

    // Data
    Settings prev_settings = {};
    Settings settings = {};

    u32 counter = {};
    auto dummy_string() -> std::string;
};