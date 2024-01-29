#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include "../../gpu_context.hpp"

template <typename T_USES_BASE, char const * T_FILE_PATH, typename T_PUSH>
inline daxa::ComputePipelineCompileInfo write_indirect_dispatch_args_base_compile_pipeline_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{T_FILE_PATH},
            .compile_options = {.defines = {{std::string(T_USES_BASE{}.name()) + std::string("_COMMAND"), "1"}}},
        },
        .push_constant_size = s_cast<u32>(sizeof(T_PUSH) + T_USES_BASE::attachment_shader_data_size()),
        .name = std::string{T_USES_BASE{}.name()},
    };
}

template <typename T_USES_BASE, char const * T_FILE_PATH, typename T_PUSH>
struct WriteIndirectDispatchArgsPushBaseTask : T_USES_BASE
{
    T_USES_BASE::AttachmentViews views = {};
    GPUContext * context = {};
    T_PUSH push = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(T_USES_BASE{}.name()));
        u32 volatile debug = T_USES_BASE::attachment_shader_data_size();
        push.globals = context->shader_globals_address;
        ti.recorder.push_constant(push);
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
            .offset = sizeof(T_PUSH),
        });
        ti.recorder.dispatch({.x = 1, .y = 1, .z = 1});
    }
};

#define CLEAR_REST -1

inline void task_clear_buffer(daxa::TaskGraph & tg, daxa::TaskBufferView buffer, u32 value, i32 range = CLEAR_REST)
{
    tg.add_task({
        .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffer)},
        .task = [=](daxa::TaskInterface ti)
        {
            ti.recorder.clear_buffer({
                .buffer = ti.get(buffer).ids[0],
                .offset = 0,
                .size = (range == CLEAR_REST) ? ti.device.info_buffer(ti.get(buffer).ids[0]).value().size : static_cast<daxa_u32>(range),
                .clear_value = value,
            });
        },
        .name = "clear task buffer",
    });
}

struct ClearRange
{
    u32 value = {};
    u32 offset = {};
    i32 size = {};
};
template <size_t N>
void task_multi_clear_buffer(daxa::TaskGraph & tg, daxa::TaskBufferView buffer, std::array<ClearRange, N> clear_ranges)
{
    tg.add_task({
        .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffer)},
        .task = [=](daxa::TaskInterface ti)
        {
            auto buffer_size = ti.device.info_buffer(ti.get(buffer).ids[0]).value().size;
            for (auto range : clear_ranges)
            {
                auto copy_size = (range.size == CLEAR_REST) ? (buffer_size - range.offset) : static_cast<u32>(range.size);
                ti.recorder.clear_buffer({
                    .buffer = ti.get(buffer).ids[0],
                    .offset = range.offset,
                    .size = copy_size,
                    .clear_value = range.value,
                });
            }
        },
        .name = "multi clear task buffer",
    });
}