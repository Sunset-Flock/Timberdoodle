#pragma once

#include "daxa/daxa.inl"
#include "daxa/utils/task_graph.inl"

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry_pipeline.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/gpu_work_expansion.inl"

#define COMPUTE_RASTERIZE_WORKGROUP_X 64

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(DrawVisbuffer_WriteCommandH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(daxa_u32), draw_commands)
DAXA_DECL_TASK_HEAD_END

// When drawing triangles, this draw command has triangle ids appended to the end of the command.
DAXA_DECL_RASTER_TASK_HEAD_BEGIN(DrawVisbufferH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER(INDIRECT_COMMAND_READ, draw_commands)
// Used by observer to cull:
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, hiz)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_IMAGE_ID(READ_WRITE, REGULAR_2D, overdraw_image)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vis_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

// Used as compute or raster task depending on if its used for atomic or normal attachment
DAXA_DECL_TASK_HEAD_BEGIN(CullMeshletsDrawVisbufferH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
// Cull Attachments:
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, hiz)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, po2expansion)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, masked_po2expansion)
DAXA_TH_BUFFER_PTR(READ_WRITE, SFPMBitfieldRef, first_pass_meshlets_bitfield_arena)
// Draw Attachments:
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_IMAGE_ID(READ_WRITE, REGULAR_2D, overdraw_image)   // Optional
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vis_image)     // Optional
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)   // Optional
DAXA_DECL_TASK_HEAD_END

struct DrawVisbufferDrawData
{
    daxa_u32 pass_index;
    daxa_u32 draw_list_section_index;
    daxa_b32 observer;
};

struct DrawVisbufferPush_WriteCommand
{
    DrawVisbuffer_WriteCommandH::AttachmentShaderBlob attach;
    daxa_u32 pass;
};

struct DrawVisbufferPush
{
    DrawVisbufferH::AttachmentShaderBlob attach;
    DrawVisbufferDrawData draw_data;
    daxa_BufferPtr(GPUMesh) meshes;
    daxa_BufferPtr(GPUMaterial) materials;
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms;
};

#if DAXA_LANGUAGE != DAXA_LANGUAGE_GLSL
struct CullMeshletsDrawVisbufferPush
{
    CullMeshletsDrawVisbufferH::AttachmentShaderBlob attach;
    DrawVisbufferDrawData draw_data;
    daxa_BufferPtr(GPUMesh) meshes;
    daxa_BufferPtr(GPUMaterial) materials;
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms;
};
#endif