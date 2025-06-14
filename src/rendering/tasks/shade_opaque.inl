#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/vsm_shared.inl"

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(ShadeOpaqueH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(READ_WRITE_CONCURRENT, REGULAR_2D, debug_lens_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec3>, color_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32>, selected_mark_image)
DAXA_TH_IMAGE_TYPED(READ, daxa::RWTexture2DId<daxa_f32vec4>, ao_image)
DAXA_TH_IMAGE_TYPED(READ, daxa::RWTexture2DId<daxa_u32>, vis_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, pgi_screen_irrdiance)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, depth)
DAXA_TH_IMAGE_ID(READ_WRITE, REGULAR_2D, debug_image)
DAXA_TH_IMAGE_TYPED(READ, daxa::RWTexture2DId<daxa_u32>, clocks_image)
DAXA_TH_IMAGE_ID(READ, REGULAR_2D, vsm_overdraw_debug)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, transmittance)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, sky)
DAXA_TH_IMAGE_ID(SAMPLED, CUBE, sky_ibl)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D_ARRAY, vsm_page_view_pos_row)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, vsm_memory_block)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, overdraw_image)  // OPTIONAL
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ, daxa::RWTexture2DArrayId<daxa_u32>, vsm_point_spot_page_table, 8)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GPUMaterial), material_manifest)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshletInstancesBufferHead), instantiated_meshlets)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_f32mat4x3), combined_transforms)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_f32), exposure)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMPointLight), vsm_point_lights)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMSpotLight), vsm_spot_lights)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(FreeWrappedPagesInfo), vsm_wrapped_pages)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GPUPointLight), point_lights)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GPUSpotLight), spot_lights)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_u32vec4>, light_mask_volume)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, pgi_irradiance)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_f32vec2>, pgi_visibility)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, pgi_info)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::RWTexture2DArrayId<daxa_u32>, pgi_requests)
DAXA_TH_TLAS_ID(READ, tlas)
DAXA_DECL_TASK_HEAD_END

struct ShadeOpaqueAttachments
{
    DAXA_TH_BLOB(ShadeOpaqueH, attachments)
};
DAXA_DECL_BUFFER_PTR(ShadeOpaqueAttachments);
struct ShadeOpaquePush
{
    daxa_BufferPtr(ShadeOpaqueAttachments) attachments;
    daxa_f32vec2 size;
    daxa_f32vec2 inv_size;
};

#define SHADE_OPAQUE_WG_X 8
#define SHADE_OPAQUE_WG_Y 8

#if defined(__cplusplus)

#include "../../gpu_context.hpp"
#include "../scene_renderer_context.hpp"
#include "shader_debug_draws.inl"

inline daxa::ComputePipelineCompileInfo shade_opaque_pipeline_compile_info()
{
    return {
        // .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/tasks/shade_opaque.glsl"}},
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/tasks/shade_opaque.hlsl"},
            .compile_options = {
                .entry_point = "entry_main_cs",
                .language = daxa::ShaderLanguage::SLANG,
            },
        },
        .push_constant_size = s_cast<u32>(sizeof(ShadeOpaquePush)),
        .name = std::string{ShadeOpaqueH::NAME},
    };
};
struct ShadeOpaqueTask : ShadeOpaqueH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(shade_opaque_pipeline_compile_info().name));
        auto const color_image_id = ti.id(AT.color_image);
        auto const color_image_info = ti.device.image_info(color_image_id).value();

        auto alloc = ti.allocator->allocate(sizeof(ShadeOpaqueAttachments));
        std::memcpy(alloc->host_address, ti.attachment_shader_blob.data(), sizeof(ShadeOpaqueH::AttachmentShaderBlob));
        ShadeOpaquePush push = {
            .attachments = alloc->device_address,
            .size = {static_cast<f32>(color_image_info.size.x), static_cast<f32>(color_image_info.size.y)},
            .inv_size = {1.0f / static_cast<f32>(color_image_info.size.x), 1.0f / static_cast<f32>(color_image_info.size.y)},
        };

        ti.recorder.push_constant(push);
        u32 const dispatch_x = round_up_div(color_image_info.size.x, SHADE_OPAQUE_WG_X);
        u32 const dispatch_y = round_up_div(color_image_info.size.y, SHADE_OPAQUE_WG_Y);

        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"SHADE_OPAQUE","SHADE_OPAQUE">());
        ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"SHADE_OPAQUE","SHADE_OPAQUE">());
    }
};
#endif