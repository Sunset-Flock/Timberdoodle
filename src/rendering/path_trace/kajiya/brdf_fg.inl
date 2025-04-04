#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(BrdfFgH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, output_tex)
DAXA_DECL_TASK_HEAD_END

#if defined(__cplusplus)

#include "../../scene_renderer_context.hpp"

inline MAKE_COMPUTE_COMPILE_INFO(brdf_fg_compute_pipeline_info, "./src/rendering/path_trace/kajiya/brdf_fg.hlsl", "main")

struct BrdfFgTask : BrdfFgH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        // render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::RAY_TRACED_AMBIENT_OCCLUSION);
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(brdf_fg_compute_pipeline_info().name));
        BrdfFgH::AttachmentShaderBlob push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = 64 / 8, .y = 64 / 8});
        // render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::RAY_TRACED_AMBIENT_OCCLUSION);
    }
};

#endif