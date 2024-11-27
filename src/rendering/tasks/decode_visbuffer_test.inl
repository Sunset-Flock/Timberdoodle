#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"

DAXA_DECL_TASK_HEAD_BEGIN(DecodeVisbufferTestH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_ONLY, REGULAR_2D, vis_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE_CONCURRENT, REGULAR_2D, debug_image)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMaterial), material_manifest)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), combined_transforms)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), instantiated_meshlets)
DAXA_DECL_TASK_HEAD_END

struct DecodeVisbufferTestPush
{
    DAXA_TH_BLOB(DecodeVisbufferTestH, attachments)
    daxa_f32vec2 size;
    daxa_f32vec2 inv_size;
};

#define DECODE_VISBUFFER_TEST_X 8
#define DECODE_VISBUFFER_TEST_Y 8

#if __cplusplus

#include "../../gpu_context.hpp"

inline daxa::ComputePipelineCompileInfo2 decode_visbuffer_test_pipeline_info2()
{
    return daxa::ComputePipelineCompileInfo2{
        .source = daxa::ShaderFile{"./src/rendering/tasks/decode_visbuffer_test.hlsl"},
        .entry_point = "entry_decode_visbuffer",
        .language = daxa::ShaderLanguage::SLANG,
        .push_constant_size = s_cast<u32>(sizeof(DecodeVisbufferTestPush)),
        .name = std::string{DecodeVisbufferTestH::NAME},
    };
};
struct DecodeVisbufferTestTask : DecodeVisbufferTestH::Task
{
    AttachmentViews views = {};
    GPUContext * gpu_context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*gpu_context->compute_pipelines.at(decode_visbuffer_test_pipeline_info2().name));
        auto const image_id = ti.get(AT.vis_image).ids[0];
        auto const image_info = ti.device.image_info(image_id).value();
        DecodeVisbufferTestPush push = {
            .attachments = ti.attachment_shader_blob,
            .size = { static_cast<f32>(image_info.size.x), static_cast<f32>(image_info.size.y) },
            .inv_size = { 1.0f / static_cast<f32>(image_info.size.x), 1.0f / static_cast<f32>(image_info.size.y) },
        };
        ti.recorder.push_constant(push);
        u32 const dispatch_x = round_up_div(image_info.size.x, DECODE_VISBUFFER_TEST_X);
        u32 const dispatch_y = round_up_div(image_info.size.y, DECODE_VISBUFFER_TEST_Y);
        ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
    }
};
#endif