#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include "../../gpu_context.hpp"

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

void assign_blob(auto & arr, auto const & span)
{
    std::memcpy(arr.value.data(), span.data(), span.size());
}

template <typename T_USES_BASE, char const * T_FILE_PATH, typename T_PUSH>
struct WriteIndirectDispatchArgsPushBaseTask : T_USES_BASE
{
    T_USES_BASE::AttachmentViews views = {};
    GPUContext * context = {};
    T_PUSH push = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(std::string{T_USES_BASE{}.name()}));
        assign_blob(push.uses, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = 1, .y = 1, .z = 1});
    }
};

template<typename HeadTaskT, typename PushT, daxa::StringLiteral shader_path, daxa::StringLiteral entry_point>
auto make_simple_compile_info() -> daxa::ComputePipelineCompileInfo
{
    auto const shader_path_sv = std::string_view(shader_path.value, shader_path.SIZE);
    auto const entry_point_sv = std::string_view(entry_point.value, entry_point.SIZE);
    auto const lang = shader_path_sv.ends_with(".glsl") ? daxa::ShaderLanguage::GLSL : daxa::ShaderLanguage::SLANG;
    auto const shader_comp_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{ std::filesystem::path(shader_path_sv) },
        .compile_options = {
            .entry_point = std::string(entry_point_sv),
            .language = lang,
            .defines = {{ std::string(HeadTaskT::name()) + "_SHADER", "1"}},
        },
    };
    auto const value = daxa::ComputePipelineCompileInfo{
        .shader_info = shader_comp_info,
        .push_constant_size = s_cast<u32>(sizeof(PushT)),
        .name = std::string(HeadTaskT::name()),
    };
    return value;
}

template<typename HeadTaskT, typename PushT, daxa::StringLiteral shader_path, daxa::StringLiteral entry_point>
struct SimpleIndirectComputeTask : HeadTaskT
{
    HeadTaskT::AttachmentViews views = {};
    GPUContext * context = {};
    PushT push = {};
    static inline const daxa::ComputePipelineCompileInfo pipeline_compile_info = make_simple_compile_info<HeadTaskT, PushT, shader_path, entry_point>();
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(std::string{HeadTaskT::name()}));
        assign_blob(push.uses, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch_indirect({
            .indirect_buffer = ti.get(this->AT.command).ids[0],
        });
    }
};

template<typename HeadTaskT, typename PushT, daxa::StringLiteral shader_path, daxa::StringLiteral entry_point>
struct SimpleComputeTask : HeadTaskT
{
    HeadTaskT::AttachmentViews views = {};
    GPUContext * context = {};
    PushT push = {};
    std::function<daxa::DispatchInfo(void)> dispatch_callback = [](){ return daxa::DispatchInfo{1,1,1}; };
    static inline const daxa::ComputePipelineCompileInfo pipeline_compile_info = make_simple_compile_info<HeadTaskT, PushT, shader_path, entry_point>();
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(std::string{HeadTaskT::name()}));
        assign_blob(push.uses, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch(dispatch_callback());
    }
};

template<typename HeadTaskT, typename PushT, daxa::StringLiteral shader_path, daxa::StringLiteral entry_point>
struct SimpleComputeTaskPushless : HeadTaskT
{
    HeadTaskT::AttachmentViews views = {};
    GPUContext * context = {};
    std::function<daxa::DispatchInfo(void)> dispatch_callback = [](){ return daxa::DispatchInfo{1,1,1}; };
    static inline const daxa::ComputePipelineCompileInfo pipeline_compile_info = make_simple_compile_info<HeadTaskT, PushT, shader_path, entry_point>();
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(std::string{HeadTaskT::name()}));
        PushT push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch(dispatch_callback());
    }
};