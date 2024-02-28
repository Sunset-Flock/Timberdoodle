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
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        ti.recorder.push_constant(push, T_USES_BASE::attachment_shader_data_size());
        ti.recorder.dispatch({.x = 1, .y = 1, .z = 1});
    }
};

#define CLEAR_REST -1

inline void task_clear_buffer(daxa::TaskGraph & tg, daxa::TaskBufferView buffer, u32 value, i32 range = CLEAR_REST, u32 offset = 0)
{
    tg.add_task({
        .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffer)},
        .task = [=](daxa::TaskInterface ti)
        {
            ti.recorder.clear_buffer({
                .buffer = ti.get(buffer).ids[0],
                .offset = offset,
                .size = (range == CLEAR_REST) ? (ti.device.info_buffer(ti.get(buffer).ids[0]).value().size - offset) : static_cast<daxa_u32>(range),
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
inline void task_multi_clear_buffer(daxa::TaskGraph & tg, daxa::TaskBufferView buffer, std::array<ClearRange, N> clear_ranges)
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

inline void task_clear_image(daxa::TaskGraph & tg, daxa::TaskImageView image, daxa::ClearValue clear_value)
{
    tg.add_task({
        .attachments = { daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, image)},
        .task = [=](daxa::TaskInterface ti)
        {
            auto image_id = ti.get(image).ids[0];
            ti.recorder.clear_image({
                .clear_value = clear_value,
                .dst_image = image_id,
                .dst_slice = ti.get(image).view.slice,
            });
        },
        .name = "clear image",
    });
}

template<typename T>
inline void task_fill_buffer(daxa::TaskGraph & tg, daxa::TaskBufferView buffer, T clear_value, u32 offset = 0)
{
    tg.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffer),
        },
        .task = [=](daxa::TaskInterface ti)
        {
            auto alloc = ti.allocator->allocate_fill(clear_value).value();
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = ti.allocator->buffer(),
                .dst_buffer = ti.get(buffer).ids[0],
                .src_offset = alloc.buffer_offset,
                .dst_offset = offset,
                .size = sizeof(T),
            });
        },
        .name = "fill buffer",
    });
}

template<typename T>
inline void allocate_fill_copy(daxa::TaskInterface ti, T value, daxa::TaskBufferAttachmentInfo dst, u32 dst_offset = 0)
{
    auto address = ti.device.get_device_address(dst.ids[0]).value();
    auto alloc = ti.allocator->allocate_fill(value).value();
    ti.recorder.copy_buffer_to_buffer({
        .src_buffer = ti.allocator->buffer(),
        .dst_buffer = dst.ids[0],
        .src_offset = alloc.buffer_offset,
        .dst_offset = dst_offset,
        .size = sizeof(T),
    });
}

template<typename HeadT, typename PushT, daxa::StringLiteral shader_path>
struct SimpleIndirectComputeTask : HeadT
{
    HeadT::AttachmentViews views = {};
    GPUContext * context = {};
    PushT push = {};
    static auto pipeline_compile_info() -> daxa::ComputePipelineCompileInfo {
        return daxa::ComputePipelineCompileInfo{
            .shader_info = daxa::ShaderCompileInfo{
                daxa::ShaderFile{ std::string_view{shader_path.value, shader_path.SIZE}},
                {
                    .defines = {{ std::string(HeadT::name()) + "_SHADER", "1"}},
                },
            },
            .push_constant_size =
                s_cast<u32>(sizeof(PushT) + HeadT::attachment_shader_data_size()),
            .name = std::string(HeadT::name()),
        };
    }
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(HeadT::name()));
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        ti.recorder.push_constant(push, ti.attachment_shader_data.size());
        ti.recorder.dispatch_indirect({
            .indirect_buffer = ti.get(this->command).ids[0],
        });
    }
};

template<typename HeadT, typename PushT, daxa::StringLiteral shader_path>
struct SimpleComputeTask : HeadT
{
    HeadT::AttachmentViews views = {};
    GPUContext * context = {};
    std::function<daxa::DispatchInfo(void)> dispatch_callback = {};
    PushT push = {};
    static auto pipeline_compile_info() -> daxa::ComputePipelineCompileInfo {
        return {
            .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{shader_path.c_str()},
                {.defines = {{ std::string(HeadT::name()) + "_SHADER", "1"}}}},
            .push_constant_size =
                s_cast<u32>(sizeof(PushT) + HeadT::attachment_shader_data_size()),
            .name = HeadT::name(),
        };
    }
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(HeadT::name()));
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        ti.recorder.push_constant(push, ti.attachment_shader_data.size());
        ti.recorder.dispatch(dispatch_callback());
    }
};