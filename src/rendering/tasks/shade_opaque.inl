#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"

DAXA_DECL_TASK_HEAD_BEGIN(ShadeOpaqueH, 13)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE_CONCURRENT, REGULAR_2D, debug_lens_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, color_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_ONLY,  REGULAR_2D, vis_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_ONLY,  REGULAR_2D, debug_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, CUBE, sky_ibl)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMaterial), material_manifest)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), instantiated_meshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), combined_transforms)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32), luminance_average)
DAXA_DECL_TASK_HEAD_END

struct ShadeOpaquePush
{
    DAXA_TH_BLOB(ShadeOpaqueH, attachments)
    daxa_f32vec2 size;
    daxa_f32vec2 inv_size;
};

#define SHADE_OPAQUE_WG_X 16
#define SHADE_OPAQUE_WG_Y 8

#if __cplusplus

#include "../../gpu_context.hpp"

inline daxa::ComputePipelineCompileInfo shade_opaque_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/tasks/shade_opaque.glsl"}},
        .push_constant_size = s_cast<u32>(sizeof(ShadeOpaquePush)),
        .name = std::string{ShadeOpaqueH::NAME},
    };
};
struct ShadeOpaqueTask : ShadeOpaqueH::Task
{
    AttachmentViews views = {};
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(shade_opaque_pipeline_compile_info().name));
        auto const color_image_id = ti.get(AT.color_image).ids[0];
        auto const color_image_info = ti.device.info_image(color_image_id).value();
        ShadeOpaquePush push = {
            .size = { static_cast<f32>(color_image_info.size.x), static_cast<f32>(color_image_info.size.y) },
            .inv_size = { 1.0f / static_cast<f32>(color_image_info.size.x), 1.0f / static_cast<f32>(color_image_info.size.y) },
        };
        assign_blob(push.attachments, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        u32 const dispatch_x = round_up_div(color_image_info.size.x, SHADE_OPAQUE_WG_X);
        u32 const dispatch_y = round_up_div(color_image_info.size.y, SHADE_OPAQUE_WG_Y);
        ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
    }
};
#endif