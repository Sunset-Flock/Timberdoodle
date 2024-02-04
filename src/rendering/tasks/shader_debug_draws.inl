#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/debug.inl"

#if __cplusplus

struct ShaderDebugDrawContext
{
    u32 max_circle_draws = 16'000;
    daxa::RasterPipeline circle_draw_pipeline = {};
    daxa::BufferId buffer = {};

    void init() 
    {
        // usize size = {};
        // usize circle_draw_array_offset = (size += sizeof(ShaderDebugBufferHead));
        // size += sizeof(ShaderDebugCircleDraw) * max_circle_draws;
        // auto buffer = context->device.create_buffer({
        //     .size = size,
        //     .name = "shader debug buffer",
        // });
    }

    void init_debug_buffer(daxa::Device & device, daxa::CommandRecorder & recorder, daxa::TransferMemoryPool & allocator)
    {
       //allocator.allocate_fill(ShaderDebugBufferHead{
       //    .circle_draw_capacity = max_circle_draws,
       //    .circle_draw_pipeline = 0,
       //    .circle_draws = device.get_device_address(buffer).value() + sizeof(ShaderDebugBufferHead),
       //}).value();

    }
};

#endif // #if __cplusplus