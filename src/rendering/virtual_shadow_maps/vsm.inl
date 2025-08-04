#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry_pipeline.inl"
#if DAXA_LANGUAGE != DAXA_LANGUAGE_GLSL
#include "../../shader_shared/gpu_work_expansion.inl"
#endif

#define INVALIDATE_PAGES_X_DISPATCH 256
#define FORCE_ALWAYS_PRESENT_PAGES_X_DISPATCH 256
#define MARK_REQUIRED_PAGES_X_DISPATCH 16
#define MARK_REQUIRED_PAGES_Y_DISPATCH 16
#define FIND_FREE_PAGES_X_DISPATCH 32
#define ALLOCATE_PAGES_X_DISPATCH 32
#define CLEAR_DIRTY_BIT_X_DISPATCH 32
#define CLEAR_PAGES_X_DISPATCH 16
#define CLEAR_PAGES_Y_DISPATCH 16
#define GEN_DIRTY_BIT_HIZ_X_DISPATCH 16
#define GEN_DIRTY_BIT_HIZ_Y_DISPATCH 16
#define GEN_DIRTY_BIT_HIZ_X_WINDOW 64
#define GEN_DIRTY_BIT_HIZ_Y_WINDOW 64
#define DEBUG_PAGE_TABLE_X_DISPATCH 16
#define DEBUG_PAGE_TABLE_Y_DISPATCH 16
#define DEBUG_META_MEMORY_TABLE_X_DISPATCH 16
#define DEBUG_META_MEMORY_TABLE_Y_DISPATCH 16
#define RECREATE_SHADOW_MAP_X_DISPATCH 16
#define RECREATE_SHADOW_MAP_Y_DISPATCH 16
#define GET_DEBUG_STATISTICS_X_DISPATCH 8
#define GET_DEBUG_STATISTICS_Y_DISPATCH 8

#if (DAXA_LANGUAGE != DAXA_LANGUAGE_GLSL)
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(InvalidatePagesH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
// Draw Attachments:
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
// VSM Attachments:
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_BufferPtr(FreeWrappedPagesInfo), free_wrapped_pages_info)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_page_table)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DId<daxa_u64>, vsm_meta_memory_table)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(FreeWrappedPagesH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(FreeWrappedPagesInfo), free_wrapped_pages_info)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_page_table)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DId<daxa_u64>, vsm_meta_memory_table)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(ForceAlwaysResidentPagesH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_BufferPtr(VSMAllocationRequestsHeader), vsm_allocation_requests)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_point_spot_page_table)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DId<daxa_u64>, vsm_meta_memory_table)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(MarkRequiredPagesH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_BufferPtr(VSMAllocationRequestsHeader), vsm_allocation_requests)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMPointLight), vsm_point_lights)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMSpotLight), vsm_spot_lights)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, g_buffer_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, g_buffer_geo_normal)
DAXA_TH_IMAGE_TYPED(READ, daxa::RWTexture2DArrayId<daxa_f32vec4>, vsm_page_view_pos_row)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_page_table)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DId<daxa_u64>, vsm_meta_memory_table)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_point_spot_page_table, 8)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_u32vec4>, light_mask_volume)
DAXA_DECL_TASK_HEAD_END

struct MarkRequiredPagesAttachments
{
    DAXA_TH_BLOB(MarkRequiredPagesH, attachments);
};
DAXA_DECL_BUFFER_PTR(MarkRequiredPagesAttachments);

struct MarkRequiredPagesPush
{
    daxa_BufferPtr(MarkRequiredPagesAttachments) attachments;
};
#endif

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(FindFreePagesH)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_BufferPtr(FindFreePagesHeader), vsm_find_free_pages_header)
DAXA_TH_BUFFER_PTR(WRITE, daxa_BufferPtr(PageCoordBuffer), vsm_free_pages_buffer)
DAXA_TH_BUFFER_PTR(WRITE, daxa_BufferPtr(PageCoordBuffer), vsm_not_visited_pages_buffer)
DAXA_TH_BUFFER_PTR(WRITE, daxa_BufferPtr(DispatchIndirectStruct), vsm_allocate_indirect)
DAXA_TH_BUFFER_PTR(WRITE, daxa_BufferPtr(DispatchIndirectStruct), vsm_clear_indirect)
DAXA_TH_BUFFER_PTR(WRITE, daxa_BufferPtr(DispatchIndirectStruct), vsm_clear_dirty_bit_indirect)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMAllocationRequestsHeader), vsm_allocation_requests)
DAXA_TH_IMAGE_ID(READ_WRITE, REGULAR_2D, vsm_meta_memory_table)
DAXA_DECL_TASK_HEAD_END

#if (DAXA_LANGUAGE != DAXA_LANGUAGE_GLSL)
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(AllocatePagesH)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(FindFreePagesHeader), vsm_find_free_pages_header)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMAllocationRequestsHeader), vsm_allocation_requests)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PageCoordBuffer), vsm_free_pages_buffer)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PageCoordBuffer), vsm_not_visited_pages_buffer)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(DispatchIndirectStruct), vsm_allocate_indirect)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_page_table)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, vsm_page_view_pos_row)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DId<daxa_u64>, vsm_meta_memory_table)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_point_spot_page_table, 8)
DAXA_DECL_TASK_HEAD_END

struct AllocatePagesAttachments
{
    DAXA_TH_BLOB(AllocatePagesH, attachments);
};
DAXA_DECL_BUFFER_PTR(AllocatePagesAttachments);

struct AllocatePagesPush
{
    daxa_BufferPtr(AllocatePagesAttachments) attachments;
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(ClearPagesH)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMAllocationRequestsHeader), vsm_allocation_requests)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(DispatchIndirectStruct), vsm_clear_indirect)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_page_table)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DId<daxa_f32>, vsm_memory_block)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_point_spot_page_table, 8)
DAXA_DECL_TASK_HEAD_END

// TODO: Fix the hardcoded constant 8
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(GenDirtyBitHizH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_TH_IMAGE_ID_MIP_ARRAY(READ_WRITE, REGULAR_2D_ARRAY, vsm_dirty_bit_hiz, 8)
DAXA_DECL_TASK_HEAD_END
struct GenDirtyBitHizPush
{
    DAXA_TH_BLOB(GenDirtyBitHizH, attachments)
    daxa_u32 mip_count;
};

// TODO: Fix the hardcoded constant 6
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(GenPointDirtyBitHizH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_u32>, vsm_point_spot_page_table)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip0, 7)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip1, 6)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip2, 5)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip3, 4)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip4, 3)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip5, 2)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip6, 1)
DAXA_DECL_TASK_HEAD_END

struct GenPointDirtyBitHizPush
{
    daxa_BufferPtr(GenPointDirtyBitHizH::AttachmentShaderBlob) attachments;
};
DAXA_DECL_RASTER_TASK_HEAD_BEGIN(CullAndDrawPagesH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, po2expansion)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, masked_po2expansion)
// Draw Attachments:
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GPUMaterial), material_manifest)
// Vsm Attachments:
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D_ARRAY, vsm_dirty_bit_hiz)
DAXA_TH_IMAGE_ID(READ, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_TH_IMAGE_ID(READ_WRITE_CONCURRENT, REGULAR_2D, vsm_memory_block)
DAXA_TH_IMAGE_ID(READ_WRITE, REGULAR_2D, vsm_overdraw_debug)
DAXA_DECL_TASK_HEAD_END
struct CullAndDrawPagesPush
{
    daxa_BufferPtr(CullAndDrawPagesH::AttachmentShaderBlob) attachments;
    daxa_u32 draw_list_type;
    daxa::RWTexture2DId<daxa::u32> daxa_uint_vsm_memory_view;
};

DAXA_DECL_RASTER_TASK_HEAD_BEGIN(CullAndDrawPointPagesH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, po2expansion_mip0)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, masked_po2expansion_mip0)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, po2expansion_mip1)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, masked_po2expansion_mip1)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, po2expansion_mip2)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, masked_po2expansion_mip2)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, po2expansion_mip3)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, masked_po2expansion_mip3)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, po2expansion_mip4)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, masked_po2expansion_mip4)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, po2expansion_mip5)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, masked_po2expansion_mip5)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, po2expansion_mip6)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, masked_po2expansion_mip6)
// Draw Attachments:
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GPUMaterial), material_manifest)
// Vsm Attachments:
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMPointLight), vsm_point_lights)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMSpotLight), vsm_spot_lights)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ, daxa::RWTexture2DArrayId<daxa_u32>, vsm_point_spot_page_table, 8)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32>, vsm_memory_block)
// Hpb Attachments:
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip0)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip1)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip2)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip3)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip4)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip5)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_u32>, vsm_dirty_bit_hiz_mip6)
DAXA_DECL_TASK_HEAD_END

struct CullAndDrawPointPagesPush
{
    daxa_BufferPtr(CullAndDrawPointPagesH::AttachmentShaderBlob) attachments;
    daxa_u32 draw_list_type;
    daxa_u32 mip_level;
    daxa::RWTexture2DId<daxa::u32> daxa_uint_vsm_memory_view;
    daxa::ImageViewId hpb_view;
};
#endif

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(ClearDirtyBitH)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMAllocationRequestsHeader), vsm_allocation_requests)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(DispatchIndirectStruct), vsm_clear_dirty_bit_indirect)
DAXA_TH_IMAGE_ID(READ_WRITE, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_DECL_TASK_HEAD_END

#if (DAXA_LANGUAGE != DAXA_LANGUAGE_GLSL)
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(DebugVirtualPageTableH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_TH_IMAGE_TYPED(READ, daxa::RWTexture2DArrayId<daxa_u32>, vsm_page_table)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, vsm_debug_page_table)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(DebugMetaMemoryTableH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_page_table)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DId<daxa_u64>, vsm_meta_memory_table)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, vsm_debug_meta_memory_table)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, vsm_point_spot_page_table, 8)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RecreateShadowMapH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_u32>, vsm_page_table)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, vsm_memory_block)
DAXA_TH_IMAGE_ID(READ, REGULAR_2D, vsm_overdraw_debug)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, vsm_recreated_shadow_map)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(GetDebugStatisticsH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMAllocationRequestsHeader), vsm_allocation_requests)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(FindFreePagesHeader), vsm_find_free_pages_header)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::RWTexture2DId<daxa_u64>, vsm_meta_memory_table)
DAXA_DECL_TASK_HEAD_END
#endif
