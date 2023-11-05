#pragma once

#include "../sandbox.hpp"
#include "../window.hpp"

struct GPUContext
{
    GPUContext(Window const& window);
    GPUContext(GPUContext&&) = default;
    ~GPUContext();

    // common unique:
    daxa::Instance context = {};
    daxa::Device device = {};
    daxa::Swapchain swapchain = {};
    daxa::PipelineManager pipeline_manager = {};
    daxa::TransferMemoryPool transient_mem;
    
    ShaderGlobalsBlock shader_globals = {};
    daxa::BufferId shader_globals_buffer = {};
    daxa::types::BufferDeviceAddress shader_globals_ptr = {};
    daxa::SetUniformBufferInfo shader_globals_set_info = {};

    // Pipelines:
    std::unordered_map<std::string_view, std::shared_ptr<daxa::RasterPipeline>> raster_pipelines = {};
    std::unordered_map<std::string_view, std::shared_ptr<daxa::ComputePipeline>> compute_pipelines = {};

    // Data
    Settings prev_settings = {};
    Settings settings = {};

    u32 counter = {};
    auto dummy_string() -> std::string;
};