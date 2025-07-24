#pragma once

#include "daxa/daxa.inl"
#include "daxa/utils/task_graph.inl"

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/geometry_pipeline.inl"
#include "../../shader_shared/gpu_work_expansion.inl"

#define CULL_MESHES_WORKGROUP_X 64

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(ExpandMeshesToMeshletsH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, hiz)                                                              // OPTIONAL
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D_ARRAY, hip)                                                        // OPTIONAL
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D_ARRAY, point_hip)                                                  // OPTIONAL
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(FirstPassMeshletBitfield), first_pass_meshlet_bitfield) // OPTIONAL
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(uint), opaque_expansion)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(uint), masked_expansion)
// TODO REMOVE, PUT IN VSM GLOBALS
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMPointLight), vsm_point_lights)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMSpotLight), vsm_spot_lights)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_DECL_TASK_HEAD_END

struct ExpandMeshesToMeshletsAttachments
{
    ExpandMeshesToMeshletsH::AttachmentShaderBlob attachments;
} DAXA_DECL_BUFFER_PTR(ExpandMeshesToMeshletsAttachments);

struct ExpandMeshesToMeshletsPush
{
    daxa_BufferPtr(ExpandMeshesToMeshletsAttachments) attachments;
    daxa::b32 cull_meshes;
    daxa::b32 cull_against_last_frame; /// WARNING: only supported for non vsm path!
    daxa_BufferPtr(GPUMesh) meshes;
    daxa_BufferPtr(GPUMeshLodGroup) mesh_lod_groups;
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms;
    daxa::i32 mip_level;
};