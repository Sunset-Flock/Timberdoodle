#pragma once

#include "copy_depth.hlsl"

#include <daxa/utils/pipeline_manager.hpp>
#include "../scene_renderer_context.hpp"

inline MAKE_COMPUTE_COMPILE_INFO(copy_depth_pipeline_compile_info, "./src/rendering/tasks/copy_depth.hlsl", "entry_copy_depth")

struct CopyDepthTask : CopyDepthH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(copy_depth_pipeline_compile_info().name));
        
        auto const info = ti.info(AT.depth_src).value();
        ti.recorder.push_constant(CopyDepthPush{
            .attachments = ti.attachment_shader_blob,
            .size = {static_cast<f32>(info.size.x), static_cast<f32>(info.size.y)},
        });

        auto const dispatch_dim = round_up_div({info.size.x,info.size.y,1}, {8,8,1});
        ti.recorder.dispatch({dispatch_dim.x, dispatch_dim.y, dispatch_dim.z});
    }
};