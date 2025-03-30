#pragma once

#define DAXA_REMOVE_DEPRECATED 0
#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include "../../gpu_context.hpp"

template <typename T>
inline void task_fill_buffer(daxa::TaskGraph & tg, daxa::TaskBufferView buffer, T clear_value, u32 offset = 0)
{
    tg.add_task(daxa::InlineTask{"fill buffer"}
            .tf.writes(buffer)
            .executes(
                [=](daxa::TaskInterface ti)
                {
                    auto alloc = ti.allocator->allocate_fill(clear_value).value();
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = ti.allocator->buffer(),
                        .dst_buffer = ti.id(daxa::TaskBufferAttachmentIndex{0}),
                        .src_offset = alloc.buffer_offset,
                        .dst_offset = offset,
                        .size = sizeof(T),
                    });
                }));
}

template <typename T>
inline void allocate_fill_copy(daxa::TaskInterface ti, T value, daxa::TaskBufferAttachmentInfo dst, u32 dst_offset = 0)
{
    auto alloc = ti.allocator->allocate_fill(value).value();
    ti.recorder.copy_buffer_to_buffer({
        .src_buffer = ti.allocator->buffer(),
        .dst_buffer = dst.ids[0],
        .src_offset = alloc.buffer_offset,
        .dst_offset = dst_offset,
        .size = sizeof(T),
    });
}

template <typename T_USES_BASE, char const * T_FILE_PATH, typename T_PUSH>
struct WriteIndirectDispatchArgsPushBaseTask : T_USES_BASE
{
    T_USES_BASE::AttachmentViews views = {};
    GPUContext * gpu_context = {};
    T_PUSH push = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*gpu_context->compute_pipelines.at(std::string{T_USES_BASE{}.name()}));
        push.attach = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = 1, .y = 1, .z = 1});
    }
};

template <typename HeadTaskT, typename PushT, daxa::StringLiteral shader_path, daxa::StringLiteral entry_point>
auto make_simple_compile_info() -> daxa::ComputePipelineCompileInfo
{
    auto const shader_path_sv = std::string_view(shader_path.value, shader_path.SIZE);
    auto const entry_point_sv = std::string_view(entry_point.value, entry_point.SIZE);
    auto const lang = shader_path_sv.ends_with(".glsl") ? daxa::ShaderLanguage::GLSL : daxa::ShaderLanguage::SLANG;
    auto const shader_comp_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{std::filesystem::path(shader_path_sv)},
        .compile_options = {
            .entry_point = std::string(entry_point_sv),
            .language = lang,
            .defines = {{std::string(HeadTaskT::name()) + "_SHADER", "1"}},
        },
    };
    auto const value = daxa::ComputePipelineCompileInfo{
        .shader_info = shader_comp_info,
        .push_constant_size = s_cast<u32>(sizeof(PushT)),
        .name = std::string(HeadTaskT::name()),
    };
    return value;
}

template <typename HeadTaskT, typename PushT, daxa::StringLiteral shader_path, daxa::StringLiteral entry_point>
struct SimpleIndirectComputeTask : HeadTaskT
{
    HeadTaskT::AttachmentViews views = {};
    GPUContext * gpu_context = {};
    PushT push = {};
    static inline daxa::ComputePipelineCompileInfo const pipeline_compile_info = make_simple_compile_info<HeadTaskT, PushT, shader_path, entry_point>();
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*gpu_context->compute_pipelines.at(std::string{HeadTaskT::name()}));
        push.attach = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch_indirect({
            .indirect_buffer = ti.id(this->AT.command),
        });
    }
};

template <typename HeadTaskT, typename PushT, daxa::StringLiteral shader_path, daxa::StringLiteral entry_point>
struct SimpleComputeTask : HeadTaskT
{
    HeadTaskT::AttachmentViews views = {};
    GPUContext * gpu_context = {};
    PushT push = {};
    std::function<daxa::DispatchInfo(void)> dispatch_callback = []()
    {
        return daxa::DispatchInfo{1, 1, 1};
    };
    static inline daxa::ComputePipelineCompileInfo const pipeline_compile_info = make_simple_compile_info<HeadTaskT, PushT, shader_path, entry_point>();
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*gpu_context->compute_pipelines.at(std::string{HeadTaskT::name()}));
        push.attach = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch(dispatch_callback());
    }
};

template <typename HeadTaskT, typename PushT, daxa::StringLiteral shader_path, daxa::StringLiteral entry_point>
struct SimpleComputeTaskPushless : HeadTaskT
{
    HeadTaskT::AttachmentViews views = {};
    GPUContext * gpu_context = {};
    std::function<daxa::DispatchInfo(void)> dispatch_callback = []()
    {
        return daxa::DispatchInfo{1, 1, 1};
    };
    static inline daxa::ComputePipelineCompileInfo const pipeline_compile_info = make_simple_compile_info<HeadTaskT, PushT, shader_path, entry_point>();
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*gpu_context->compute_pipelines.at(std::string{HeadTaskT::name()}));
        PushT push = {};
        push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch(dispatch_callback());
    }
};