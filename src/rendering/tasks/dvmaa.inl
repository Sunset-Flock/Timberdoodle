#pragma once

#include "daxa/daxa.inl"
#include "daxa/utils/task_graph.inl"

#include "../../shader_shared/shared.inl"

#define DVM_WG_X 16
#define DVM_WG_Y 16

struct DvmState
{
    daxa_u32 samples;
    daxa_u32 current_sample;
};

DAXA_DECL_TASK_HEAD_BEGIN(DVMResolveVisImageH)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, dvm_vis_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, dvm_depth_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, vis_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

struct DVMResolveVisImagePush
{
    DAXA_TH_BLOB(DVMResolveVisImageH, attachments)
    daxa_u32vec2 resolution;
    daxa_u32 resolve_sample;
};

#if defined(__cplusplus)

#include "../scene_renderer_context.hpp"

static inline daxa::ComputePipelineCompileInfo dvm_resolve_visbuffer_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/tasks/dvmaa.slang"},
            .compile_options = {.entry_point = "entry_resolve_vis_image", .language = daxa::ShaderLanguage::SLANG},
        },
        .push_constant_size = sizeof(DVMResolveVisImagePush),
        .name = std::string{DVMResolveVisImageH::NAME},
    };
};

struct DVMResolveVisImageTask : DVMResolveVisImageH::Task
{
    AttachmentViews views = {};
    RenderContext * rctx = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*rctx->gpuctx->compute_pipelines.at(DVMResolveVisImageH::NAME));
        auto const image_id = ti.get(AT.dvm_vis_image).ids[0];
        auto const image_info = ti.device.info_image(image_id).value();
        DVMResolveVisImagePush push = {
            .resolution = rctx->render_data.settings.render_target_size,
            .resolve_sample = rctx->render_data.frame_index % 4,
        };
        assign_blob(push.attachments, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        u32 const dispatch_x = round_up_div(image_info.size.x, DVM_WG_X);
        u32 const dispatch_y = round_up_div(image_info.size.y, DVM_WG_Y);
        ti.recorder.dispatch({dispatch_x, dispatch_y, 1});
    }
};

#endif