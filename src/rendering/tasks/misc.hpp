#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include "../../gpu_context.hpp"

template <typename T_USES_BASE, char const *T_FILE_PATH, typename T_PUSH>
struct WriteIndirectDispatchArgsPushBaseTask : T_USES_BASE
{
    static inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{T_FILE_PATH},
            .compile_options = {
                .defines = {{std::string(T_USES_BASE{}.name()) + std::string("_COMMAND"), "1"}},
            },
        },
        .push_constant_size = sizeof(T_PUSH),
        .name = std::string{T_USES_BASE{}.name()},
    };
    GPUContext * context = {};
    T_PUSH push = {};
    virtual void callback(daxa::TaskInterface ti) const override
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(T_USES_BASE{}.name()));
        // Cope because push is passed as a parameter when constructing the task and the callback 
        // has to be const
        T_PUSH real_push = push;
        real_push.globals = context->shader_globals_address;
        real_push.uses = span_to_array<DAXA_TH_BLOB(T_USES_BASE){}.size()>(ti.attachment_shader_data_blob);
        ti.recorder.push_constant(real_push);
        ti.recorder.dispatch({.x = 1, .y = 1, .z = 1});
    }
};

#define CLEAR_REST -1

void task_clear_buffer(daxa::TaskGraph & tg, daxa::TaskBufferView buffer, u32 value, i32 range = CLEAR_REST)
{
    tg.add_task({
        .attachments = {daxa::TaskBufferAttachment{.access=daxa::TaskBufferAccess::TRANSFER_WRITE, .view=buffer}},
        .task = [=](daxa::TaskInterface ti){
            ti.recorder.clear_buffer({
                .buffer = ti.buf_attach(buffer).ids[0],
                .offset = 0,
                .size = (range == CLEAR_REST) ? ti.device.info_buffer(ti.buf_attach(buffer).ids[0]).value().size : static_cast<daxa_u32>(range),
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
template<size_t N>
void task_multi_clear_buffer(daxa::TaskGraph & tg, daxa::TaskBufferView buffer, std::array<ClearRange, N> clear_ranges)
{
    tg.add_task({
        .attachments = {daxa::TaskBufferAttachment{.access=daxa::TaskBufferAccess::TRANSFER_WRITE, .view = buffer}},
        .task = [=](daxa::TaskInterface ti){
            auto buffer_size = ti.device.info_buffer(ti.buf_attach(buffer).ids[0]).value().size;
            for (auto range : clear_ranges)
            {
                auto copy_size = (range.size == CLEAR_REST) ? (buffer_size - range.offset) : static_cast<u32>(range.size);
                ti.recorder.clear_buffer({
                    .buffer = ti.buf_attach(buffer).ids[0],
                    .offset = range.offset,
                    .size = copy_size,
                    .clear_value = range.value,
                });
            }
        },
        .name = "multi clear task buffer",
    });
}