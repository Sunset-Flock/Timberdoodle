#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry_pipeline.inl"
#if DAXA_LANGUAGE != DAXA_LANGUAGE_GLSL
    #include "../../shader_shared/po2_expansion.inl"
    #include "../rasterize_visbuffer/cull_meshes.inl"
#endif

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

DAXA_DECL_TASK_HEAD_BEGIN(FreeWrappedPagesH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(FreeWrappedPagesInfo), free_wrapped_pages_info)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, vsm_meta_memory_table)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(MarkRequiredPagesH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_BufferPtr(AllocationCount), vsm_allocation_count)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(AllocationRequest), vsm_allocation_requests)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_ONLY, REGULAR_2D_ARRAY, vsm_page_height_offsets)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, vsm_meta_memory_table)
DAXA_DECL_TASK_HEAD_END

#if DAXA_LANGUAGE != DAXA_LANGUAGE_GLSL
    DAXA_DECL_TASK_HEAD_BEGIN(CullAndDrawPages_WriteCommandH)
    DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(Po2WorkExpansionBufferHead), meshlet_cull_po2expansion)
    DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_meshlet_cull_po2expansion)
    DAXA_DECL_TASK_HEAD_END
#endif

DAXA_DECL_TASK_HEAD_BEGIN(FindFreePagesH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_BufferPtr(FindFreePagesHeader), vsm_find_free_pages_header)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(PageCoordBuffer), vsm_free_pages_buffer)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(PageCoordBuffer), vsm_not_visited_pages_buffer)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(DispatchIndirectStruct), vsm_allocate_indirect)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(DispatchIndirectStruct), vsm_clear_indirect)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(DispatchIndirectStruct), vsm_clear_dirty_bit_indirect)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(AllocationCount), vsm_allocation_count)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, vsm_meta_memory_table)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(AllocatePagesH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(FindFreePagesHeader), vsm_find_free_pages_header)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(AllocationCount), vsm_allocation_count)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(AllocationRequest), vsm_allocation_requests)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(PageCoordBuffer), vsm_free_pages_buffer)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(PageCoordBuffer), vsm_not_visited_pages_buffer)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectStruct), vsm_allocate_indirect)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D_ARRAY, vsm_page_height_offsets)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, vsm_meta_memory_table)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(ClearPagesH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(AllocationRequest), vsm_allocation_requests)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectStruct), vsm_clear_indirect)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, vsm_memory)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, vsm_memory64)
DAXA_DECL_TASK_HEAD_END
struct ClearPagesPush
{
    DAXA_TH_BLOB(ClearPagesH, attachments)
    daxa_u32 use64bit;
};

DAXA_DECL_TASK_HEAD_BEGIN(GenDirtyBitHizH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_TH_IMAGE_ID_MIP_ARRAY(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D_ARRAY, vsm_dirty_bit_hiz, 8)
DAXA_DECL_TASK_HEAD_END
struct GenDirtyBitHizPush
{
    DAXA_TH_BLOB(GenDirtyBitHizH, attachments)
    daxa_u32 mip_count;
};


#if DAXA_LANGUAGE != DAXA_LANGUAGE_GLSL
    DAXA_DECL_TASK_HEAD_BEGIN(CullAndDrawPagesH)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion0)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion0)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion1)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion1)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion2)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion2)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion3)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion3)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion4)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion4)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion5)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion5)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion6)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion6)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion7)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion7)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion8)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion8)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion9)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion9)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion10)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion10)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion11)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion11)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion12)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion12)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion13)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion13)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion14)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion14)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion15)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion15)
    // Draw Attachments:
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
    DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMaterial), material_manifest)
    // Vsm Attachments:
    DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
    DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_SAMPLED, REGULAR_2D_ARRAY, vsm_dirty_bit_hiz)
    DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_STORAGE_READ_ONLY, REGULAR_2D_ARRAY, vsm_page_table)
    DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_STORAGE_READ_WRITE, REGULAR_2D, vsm_memory_block)
    DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_STORAGE_READ_WRITE, REGULAR_2D, vsm_memory_block64)
    DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_STORAGE_READ_WRITE, REGULAR_2D, vsm_overdraw_debug)
    DAXA_DECL_TASK_HEAD_END
    struct CullAndDrawPagesPush
    {
        #if !(DAXA_LANGUAGE == DAXA_LANGUAGE_GLSL)
            daxa_BufferPtr(CullAndDrawPagesH::AttachmentShaderBlob) attachments;
        #endif
        daxa_u32 draw_list_type;
        daxa_u32 bucket_index;
        daxa_u32 cascade;
        daxa::RWTexture2DId<daxa::u32> daxa_uint_vsm_memory_view;
    };
#endif

DAXA_DECL_TASK_HEAD_BEGIN(ClearDirtyBitH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(AllocationRequest), vsm_allocation_requests)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(AllocationCount), vsm_allocation_count)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectStruct), vsm_clear_dirty_bit_indirect)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(DebugVirtualPageTableH)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_ONLY, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, vsm_debug_page_table)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(DebugMetaMemoryTableH)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, vsm_meta_memory_table)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, vsm_debug_meta_memory_table)
DAXA_DECL_TASK_HEAD_END

#if defined(__cplusplus)
#include "../tasks/misc.hpp"
#include "vsm_state.hpp"
#include <glm/gtx/vector_angle.hpp>
#include "../scene_renderer_context.hpp"

inline daxa::ComputePipelineCompileInfo vsm_free_wrapped_pages_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/virtual_shadow_maps/free_wrapped_pages.glsl"}},
        .push_constant_size = static_cast<u32>(sizeof(FreeWrappedPagesH::AttachmentShaderBlob)),
        .name = std::string{FreeWrappedPagesH::NAME},
    };
}

inline daxa::ComputePipelineCompileInfo vsm_mark_required_pages_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/virtual_shadow_maps/mark_required_pages.glsl"}},
        .push_constant_size = static_cast<u32>(sizeof(MarkRequiredPagesH::AttachmentShaderBlob)),
        .name = std::string{MarkRequiredPagesH::NAME},
    };
}

inline daxa::ComputePipelineCompileInfo vsm_find_free_pages_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/virtual_shadow_maps/find_free_pages.glsl"}},
        .push_constant_size = static_cast<u32>(sizeof(FindFreePagesH::AttachmentShaderBlob)),
        .name = std::string{FindFreePagesH::NAME},
    };
}

inline daxa::ComputePipelineCompileInfo vsm_allocate_pages_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/virtual_shadow_maps/allocate_pages.glsl"}},
        .push_constant_size = static_cast<u32>(sizeof(AllocatePagesH::AttachmentShaderBlob)),
        .name = std::string{AllocatePagesH::NAME},
    };
}

inline daxa::ComputePipelineCompileInfo vsm_clear_pages_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/virtual_shadow_maps/clear_pages.glsl"}},
        .push_constant_size = static_cast<u32>(sizeof(ClearPagesPush)),
        .name = std::string{ClearPagesH::NAME},
    };
}

inline daxa::ComputePipelineCompileInfo vsm_gen_dirty_bit_hiz_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/virtual_shadow_maps/gen_dirty_bit_hiz.hlsl"},
            .compile_options = {.language = daxa::ShaderLanguage::SLANG}},
        .push_constant_size = static_cast<u32>(sizeof(GenDirtyBitHizPush)),
        .name = std::string{GenDirtyBitHizH::NAME},
    };
}

static constexpr inline char const CULL_AND_DRAW_PAGES_SHADER_PATH[] = "./src/rendering/virtual_shadow_maps/cull_and_draw_pages.hlsl";
inline daxa::RasterPipelineCompileInfo vsm_cull_and_draw_pages_base_pipeline_compile_info()
{
    return {
        .mesh_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{CULL_AND_DRAW_PAGES_SHADER_PATH},
            .compile_options = {.language = daxa::ShaderLanguage::SLANG},
        },
        .task_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{CULL_AND_DRAW_PAGES_SHADER_PATH},
            .compile_options = {.language = daxa::ShaderLanguage::SLANG},
        },
        .raster = {
            .depth_clamp_enable = true,
            .depth_bias_enable = true,
            .depth_bias_constant_factor = 10.0f,
            .depth_bias_slope_factor = 2.0f,
        },
        .push_constant_size = s_cast<u32>(sizeof(CullAndDrawPagesPush)),
    };
}

inline daxa::RasterPipelineCompileInfo vsm_cull_and_draw_pages_opaque_pipeline_compile_info()
{
    auto ret = vsm_cull_and_draw_pages_base_pipeline_compile_info();
    ret.mesh_shader_info.value().compile_options.entry_point = "vsm_entry_mesh_opaque";
    ret.task_shader_info.value().compile_options.entry_point = "vsm_entry_task";
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{CULL_AND_DRAW_PAGES_SHADER_PATH},
        .compile_options = {
            .entry_point = "vsm_entry_fragment_opaque",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "VsmCullAndDrawPagesOpaque";
    return ret;
}

inline daxa::RasterPipelineCompileInfo vsm_cull_and_draw_pages_masked_pipeline_compile_info()
{
    auto ret = vsm_cull_and_draw_pages_base_pipeline_compile_info();
    ret.mesh_shader_info.value().compile_options.entry_point = "vsm_entry_mesh_masked";
    ret.task_shader_info.value().compile_options.entry_point = "vsm_entry_task";
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{CULL_AND_DRAW_PAGES_SHADER_PATH},
        .compile_options = {
            .entry_point = "vsm_entry_fragment_masked",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "VsmCullAndDrawPagesMasked";
    return ret;
}

inline std::array<daxa::RasterPipelineCompileInfo, 2> cull_and_draw_pages_pipelines = {
    vsm_cull_and_draw_pages_opaque_pipeline_compile_info(),
    vsm_cull_and_draw_pages_masked_pipeline_compile_info()};

inline daxa::ComputePipelineCompileInfo vsm_clear_dirty_bit_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/virtual_shadow_maps/clear_dirty_bit.glsl"}},
        .push_constant_size = static_cast<u32>(sizeof(ClearDirtyBitH::AttachmentShaderBlob)),
        .name = std::string{ClearDirtyBitH::NAME},
    };
}

inline daxa::ComputePipelineCompileInfo vsm_debug_virtual_page_table_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/virtual_shadow_maps/draw_debug_textures.glsl"},
            .compile_options = {.defines = {{"DEBUG_PAGE_TABLE", "1"}}},
        },
        .push_constant_size = static_cast<u32>(sizeof(DebugVirtualPageTableH::AttachmentShaderBlob)),
        .name = std::string{DebugVirtualPageTableH::NAME},
    };
}

inline daxa::ComputePipelineCompileInfo vsm_debug_meta_memory_table_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/virtual_shadow_maps/draw_debug_textures.glsl"},
            .compile_options = {.defines = {{"DEBUG_META_MEMORY_TABLE", "1"}}},
        },
        .push_constant_size = static_cast<u32>(sizeof(DebugMetaMemoryTableH::AttachmentShaderBlob)),
        .name = std::string{DebugMetaMemoryTableH::NAME},
    };
}

using CullAndDrawPages_WriteCommandTask = SimpleComputeTaskPushless<
    CullAndDrawPages_WriteCommandH::Task,
    CullAndDrawPages_WriteCommandH::AttachmentShaderBlob,
    CULL_AND_DRAW_PAGES_SHADER_PATH,
    "vsm_entry_write_commands">;

struct FreeWrappedPagesTask : FreeWrappedPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    daxa::TimelineQueryPool timeline_pool = {};
    u32 const per_frame_timestamp_count = {};

    void callback(daxa::TaskInterface ti)
    {
        u32 const fif_index = render_context->render_data.frame_index % (render_context->gpuctx->swapchain.info().max_allowed_frames_in_flight + 1);
        u32 const timestamp_start_index = per_frame_timestamp_count * fif_index;

        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_free_wrapped_pages_pipeline_compile_info().name));
        FreeWrappedPagesH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::BOTTOM_OF_PIPE, .query_index = 0 + timestamp_start_index});
        ti.recorder.dispatch({1, VSM_PAGE_TABLE_RESOLUTION, VSM_CLIP_LEVELS});
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 1 + timestamp_start_index});
    }
};

struct MarkRequiredPagesTask : MarkRequiredPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    daxa::TimelineQueryPool timeline_pool = {};
    u32 const per_frame_timestamp_count = {};

    void callback(daxa::TaskInterface ti)
    {
        u32 const fif_index = render_context->render_data.frame_index % (render_context->gpuctx->swapchain.info().max_allowed_frames_in_flight + 1);
        u32 const timestamp_start_index = per_frame_timestamp_count * fif_index;

        auto const depth_resolution = render_context->gpuctx->device.image_info(ti.get(AT.depth).ids[0]).value().size;
        auto const dispatch_size = u32vec2{
            (depth_resolution.x + MARK_REQUIRED_PAGES_X_DISPATCH - 1) / MARK_REQUIRED_PAGES_X_DISPATCH,
            (depth_resolution.y + MARK_REQUIRED_PAGES_Y_DISPATCH - 1) / MARK_REQUIRED_PAGES_Y_DISPATCH,
        };
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_mark_required_pages_pipeline_compile_info().name));
        MarkRequiredPagesH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 2 + timestamp_start_index});
        ti.recorder.dispatch({.x = dispatch_size.x, .y = dispatch_size.y});
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 3 + timestamp_start_index});
    }
};

struct FindFreePagesTask : FindFreePagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    daxa::TimelineQueryPool timeline_pool = {};
    u32 const per_frame_timestamp_count = {};

    static constexpr i32 meta_memory_pix_count = VSM_META_MEMORY_TABLE_RESOLUTION * VSM_META_MEMORY_TABLE_RESOLUTION;
    static constexpr i32 dispatch_x_size = (meta_memory_pix_count + FIND_FREE_PAGES_X_DISPATCH - 1) / FIND_FREE_PAGES_X_DISPATCH;

    void callback(daxa::TaskInterface ti)
    {
        u32 const fif_index = render_context->render_data.frame_index % (render_context->gpuctx->swapchain.info().max_allowed_frames_in_flight + 1);
        u32 const timestamp_start_index = per_frame_timestamp_count * fif_index;

        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_find_free_pages_pipeline_compile_info().name));
        FindFreePagesH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 4 + timestamp_start_index});
        ti.recorder.dispatch({dispatch_x_size, 1, 1});
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 5 + timestamp_start_index});
    }
};

struct AllocatePagesTask : AllocatePagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    daxa::TimelineQueryPool timeline_pool = {};
    u32 const per_frame_timestamp_count = {};

    void callback(daxa::TaskInterface ti)
    {
        u32 const fif_index = render_context->render_data.frame_index % (render_context->gpuctx->swapchain.info().max_allowed_frames_in_flight + 1);
        u32 const timestamp_start_index = per_frame_timestamp_count * fif_index;

        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_allocate_pages_pipeline_compile_info().name));
        AllocatePagesH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 6 + timestamp_start_index});
        ti.recorder.dispatch_indirect({
            .indirect_buffer = ti.get(AT.vsm_allocate_indirect).ids[0],
            .offset = 0u,
        });
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 7 + timestamp_start_index});
    }
};

struct ClearPagesTask : ClearPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    daxa::TimelineQueryPool timeline_pool = {};
    u32 const per_frame_timestamp_count = {};

    void callback(daxa::TaskInterface ti)
    {
        u32 const fif_index = render_context->render_data.frame_index % (render_context->gpuctx->swapchain.info().max_allowed_frames_in_flight + 1);
        u32 const timestamp_start_index = per_frame_timestamp_count * fif_index;
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_clear_pages_pipeline_compile_info().name));
        ClearPagesPush push = {};
        push.use64bit = render_context->render_data.vsm_settings.use64bit;
        assign_blob(push.attachments, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 8 + timestamp_start_index});
        ti.recorder.dispatch_indirect({
            .indirect_buffer = ti.get(AT.vsm_clear_indirect).ids[0],
            .offset = 0u,
        });
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 9 + timestamp_start_index});
    }
};

struct GenDirtyBitHizTask : GenDirtyBitHizH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    daxa::TimelineQueryPool timeline_pool = {};
    u32 const per_frame_timestamp_count = {};

    void callback(daxa::TaskInterface ti)
    {
        u32 const fif_index = render_context->render_data.frame_index % (render_context->gpuctx->swapchain.info().max_allowed_frames_in_flight + 1);
        u32 const timestamp_start_index = per_frame_timestamp_count * fif_index;

        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_gen_dirty_bit_hiz_pipeline_compile_info().name));
        auto const dispatch_x = round_up_div(VSM_PAGE_TABLE_RESOLUTION, 64);
        auto const dispatch_y = round_up_div(VSM_PAGE_TABLE_RESOLUTION, 64);
        GenDirtyBitHizPush push = {
            .mip_count = ti.get(AT.vsm_dirty_bit_hiz).view.slice.level_count,
        };
        assign_blob(push.attachments, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 10 + timestamp_start_index});
        ti.recorder.dispatch({dispatch_x, dispatch_y, VSM_CLIP_LEVELS});
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 11 + timestamp_start_index});
    }
};

struct CullAndDrawPagesTask : CullAndDrawPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    daxa::TimelineQueryPool timeline_pool = {};
    u32 const per_frame_timestamp_count = {};

    void callback(daxa::TaskInterface ti)
    {
        u32 const fif_index = render_context->render_data.frame_index % (render_context->gpuctx->swapchain.info().max_allowed_frames_in_flight + 1);
        u32 const timestamp_start_index = per_frame_timestamp_count * fif_index;

        auto const memory_block_view = render_context->gpuctx->device.create_image_view({
            .type = daxa::ImageViewType::REGULAR_2D,
            .format = render_context->render_data.vsm_settings.use64bit ? daxa::Format::R64_UINT : daxa::Format::R32_UINT,
            .image = render_context->render_data.vsm_settings.use64bit ? ti.get(AT.vsm_memory_block64).ids[0] : ti.get(AT.vsm_memory_block).ids[0],
            .name = "vsm memory daxa integer view",
        });

        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 12 + timestamp_start_index});
        auto render_cmd = std::move(ti.recorder).begin_renderpass({
            .render_area = daxa::Rect2D{.width = VSM_TEXTURE_RESOLUTION, .height = VSM_TEXTURE_RESOLUTION},
        });

        render_cmd.set_depth_bias({
            .constant_factor = render_context->render_data.vsm_settings.constant_bias,
            .clamp = 0.0,
            .slope_factor = render_context->render_data.vsm_settings.slope_bias,
        });
        auto attachment_alloc = ti.allocator->allocate(sizeof(CullAndDrawPagesH::AttachmentShaderBlob)).value();
        ti.assign_attachment_shader_blob(reinterpret_cast<CullAndDrawPagesH::AttachmentShaderBlob*>(attachment_alloc.host_address)->value);
        for (u32 cascade = 0; cascade < 16; ++cascade)
        {
            daxa::BufferId po2expansion;
            daxa::BufferId masked_po2expansion;
            switch(cascade)
            {
                case 0: po2expansion = ti.get(AT.po2expansion0).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion0).ids[0]; break;
                case 1: po2expansion = ti.get(AT.po2expansion1).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion1).ids[0]; break;
                case 2: po2expansion = ti.get(AT.po2expansion2).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion2).ids[0]; break;
                case 3: po2expansion = ti.get(AT.po2expansion3).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion3).ids[0]; break;
                case 4: po2expansion = ti.get(AT.po2expansion4).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion4).ids[0]; break;
                case 5: po2expansion = ti.get(AT.po2expansion5).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion5).ids[0]; break;
                case 6: po2expansion = ti.get(AT.po2expansion6).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion6).ids[0]; break;
                case 7: po2expansion = ti.get(AT.po2expansion7).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion7).ids[0]; break;
                case 8: po2expansion = ti.get(AT.po2expansion8).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion8).ids[0]; break;
                case 9: po2expansion = ti.get(AT.po2expansion9).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion9).ids[0]; break;
                case 10: po2expansion = ti.get(AT.po2expansion10).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion10).ids[0]; break;
                case 11: po2expansion = ti.get(AT.po2expansion11).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion11).ids[0]; break;
                case 12: po2expansion = ti.get(AT.po2expansion12).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion12).ids[0]; break;
                case 13: po2expansion = ti.get(AT.po2expansion13).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion13).ids[0]; break;
                case 14: po2expansion = ti.get(AT.po2expansion14).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion14).ids[0]; break;
                case 15: po2expansion = ti.get(AT.po2expansion15).ids[0]; masked_po2expansion = ti.get(AT.masked_po2expansion15).ids[0]; break;
            }
            for (u32 opaque_draw_list_type = 0; opaque_draw_list_type < 2; ++opaque_draw_list_type)
            {
                auto buffer = opaque_draw_list_type == PREPASS_DRAW_LIST_OPAQUE ? po2expansion : masked_po2expansion;
                render_cmd.set_pipeline(*render_context->gpuctx->raster_pipelines.at(cull_and_draw_pages_pipelines[opaque_draw_list_type].name));
                for (u32 i = 0; i < 32; ++i)
                {
                    CullAndDrawPagesPush push = {
                        .attachments = attachment_alloc.device_address,
                        .draw_list_type = opaque_draw_list_type,
                        .bucket_index = i,
                        .cascade = cascade,
                        .daxa_uint_vsm_memory_view = static_cast<daxa_ImageViewId>(memory_block_view),
                    };
                    render_cmd.push_constant(push);
                    render_cmd.draw_mesh_tasks_indirect({
                        .indirect_buffer = buffer,
                        .offset = sizeof(DispatchIndirectStruct) * i,
                        .draw_count = 1,
                        .stride = sizeof(DispatchIndirectStruct),
                    });
                }
            }
        }
        ti.recorder = std::move(render_cmd).end_renderpass();
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::ALL_GRAPHICS, .query_index = 13 + timestamp_start_index});
        ti.recorder.destroy_image_view_deferred(memory_block_view);
    }
};

struct ClearDirtyBitTask : ClearDirtyBitH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    daxa::TimelineQueryPool timeline_pool = {};
    u32 const per_frame_timestamp_count = {};

    void callback(daxa::TaskInterface ti)
    {
        u32 const fif_index = render_context->render_data.frame_index % (render_context->gpuctx->swapchain.info().max_allowed_frames_in_flight + 1);
        u32 const timestamp_start_index = per_frame_timestamp_count * fif_index;

        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_clear_dirty_bit_pipeline_compile_info().name));
        ClearDirtyBitH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::ALL_GRAPHICS, .query_index = 14 + timestamp_start_index});
        ti.recorder.dispatch_indirect({
            .indirect_buffer = ti.get(AT.vsm_clear_dirty_bit_indirect).ids[0],
            .offset = 0u,
        });
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 15 + timestamp_start_index});
    }
};

struct DebugVirtualPageTableTask : DebugVirtualPageTableH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    daxa::TimelineQueryPool timeline_pool = {};
    u32 const per_frame_timestamp_count = {};
    static constexpr auto dispatch_size = u32vec2{
        (VSM_PAGE_TABLE_RESOLUTION + DEBUG_PAGE_TABLE_X_DISPATCH - 1) / DEBUG_PAGE_TABLE_X_DISPATCH,
        (VSM_PAGE_TABLE_RESOLUTION + DEBUG_PAGE_TABLE_Y_DISPATCH - 1) / DEBUG_PAGE_TABLE_Y_DISPATCH,
    };

    void callback(daxa::TaskInterface ti)
    {
        u32 const fif_index = render_context->render_data.frame_index % (render_context->gpuctx->swapchain.info().max_allowed_frames_in_flight + 1);
        u32 const timestamp_start_index = per_frame_timestamp_count * fif_index;

        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_debug_virtual_page_table_pipeline_compile_info().name));
        DebugVirtualPageTableH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 16 + timestamp_start_index});
        ti.recorder.dispatch({dispatch_size.x, dispatch_size.y});
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 17 + timestamp_start_index});
    }
};

struct DebugMetaMemoryTableTask : DebugMetaMemoryTableH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    daxa::TimelineQueryPool timeline_pool = {};
    u32 const per_frame_timestamp_count = {};
    static constexpr auto dispatch_size = u32vec2{
        (VSM_META_MEMORY_TABLE_RESOLUTION + DEBUG_META_MEMORY_TABLE_X_DISPATCH - 1) / DEBUG_META_MEMORY_TABLE_X_DISPATCH,
        (VSM_META_MEMORY_TABLE_RESOLUTION + DEBUG_META_MEMORY_TABLE_Y_DISPATCH - 1) / DEBUG_META_MEMORY_TABLE_Y_DISPATCH,
    };

    void callback(daxa::TaskInterface ti)
    {
        u32 const fif_index = render_context->render_data.frame_index % (render_context->gpuctx->swapchain.info().max_allowed_frames_in_flight + 1);
        u32 const timestamp_start_index = per_frame_timestamp_count * fif_index;

        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_debug_meta_memory_table_pipeline_compile_info().name));
        DebugMetaMemoryTableH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 18 + timestamp_start_index});
        ti.recorder.dispatch({dispatch_size.x, dispatch_size.y});
        ti.recorder.write_timestamp({.query_pool = timeline_pool, .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER, .query_index = 19 + timestamp_start_index});
    }
};

struct TaskDrawVSMsInfo
{
    Scene * scene = {};
    RenderContext * render_context = {};
    daxa::TaskGraph * tg = {};
    VSMState * vsm_state = {};
    std::array<daxa::TaskBufferView, PREPASS_DRAW_LIST_TYPES> meshlet_cull_po2expansions = {};
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskBufferView mesh_instances = {};
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView entity_combined_transforms = {};
    daxa::TaskBufferView material_manifest = {};
    daxa::TaskImageView depth = {};
};

inline void task_draw_vsms(TaskDrawVSMsInfo const & info)
{
    auto const vsm_page_table_view = info.vsm_state->page_table.view().view({.base_array_layer = 0, .layer_count = VSM_CLIP_LEVELS});
    auto const vsm_page_height_offsets_view = info.vsm_state->page_height_offsets.view().view({.base_array_layer = 0, .layer_count = VSM_CLIP_LEVELS});
    auto const vsm_dirty_bit_hiz_view = info.vsm_state->dirty_pages_hiz.view({
        .base_mip_level = 0,
        .level_count = s_cast<u32>(std::log2(VSM_PAGE_TABLE_RESOLUTION)) + 1,
        .base_array_layer = 0,
        .layer_count = VSM_CLIP_LEVELS,
    });
    info.tg->add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, info.vsm_state->clip_projections),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, info.vsm_state->free_wrapped_pages_info),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, info.vsm_state->globals),
        },
        .task = [info](daxa::TaskInterface ti)
        {
            u32 const fif_index = info.render_context->render_data.frame_index % (info.render_context->gpuctx->swapchain.info().max_allowed_frames_in_flight + 1);
            u32 const timestamp_start_index = info.vsm_state->PER_FRAME_TIMESTAMP_COUNT * fif_index;
            ti.recorder.reset_timestamps({
                .query_pool = info.vsm_state->vsm_timeline_query_pool,
                .start_index = timestamp_start_index,
                .count = info.vsm_state->PER_FRAME_TIMESTAMP_COUNT,
            });
            allocate_fill_copy(ti, info.vsm_state->clip_projections_cpu, ti.get(info.vsm_state->clip_projections));
            allocate_fill_copy(ti, info.vsm_state->free_wrapped_pages_info_cpu, ti.get(info.vsm_state->free_wrapped_pages_info));
            allocate_fill_copy(ti, info.vsm_state->globals_cpu, ti.get(info.vsm_state->globals));
        },
        .name = "vsm setup task",
    });

    info.tg->add_task(FreeWrappedPagesTask{
        .views = std::array{
            daxa::attachment_view(FreeWrappedPagesH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(FreeWrappedPagesH::AT.free_wrapped_pages_info, info.vsm_state->free_wrapped_pages_info),
            daxa::attachment_view(FreeWrappedPagesH::AT.vsm_clip_projections, info.vsm_state->clip_projections),
            daxa::attachment_view(FreeWrappedPagesH::AT.vsm_page_table, vsm_page_table_view),
            daxa::attachment_view(FreeWrappedPagesH::AT.vsm_meta_memory_table, info.vsm_state->meta_memory_table),
        },
        .render_context = info.render_context,
        .timeline_pool = info.vsm_state->vsm_timeline_query_pool,
        .per_frame_timestamp_count = info.vsm_state->PER_FRAME_TIMESTAMP_COUNT,
    });

    info.tg->add_task(MarkRequiredPagesTask{
        .views = std::array{
            daxa::attachment_view(MarkRequiredPagesH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_globals, info.vsm_state->globals),
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_allocation_count, info.vsm_state->allocation_count),
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_allocation_requests, info.vsm_state->allocation_requests),
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_clip_projections, info.vsm_state->clip_projections),
            daxa::attachment_view(MarkRequiredPagesH::AT.depth, info.depth),
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_page_table, vsm_page_table_view),
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_page_height_offsets, vsm_page_height_offsets_view),
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_meta_memory_table, info.vsm_state->meta_memory_table),
        },
        .render_context = info.render_context,
        .timeline_pool = info.vsm_state->vsm_timeline_query_pool,
        .per_frame_timestamp_count = info.vsm_state->PER_FRAME_TIMESTAMP_COUNT,
    });

    info.tg->add_task(FindFreePagesTask{
        .views = std::array{
            daxa::attachment_view(FindFreePagesH::AT.vsm_free_pages_buffer, info.vsm_state->free_page_buffer),
            daxa::attachment_view(FindFreePagesH::AT.vsm_not_visited_pages_buffer, info.vsm_state->not_visited_page_buffer),
            daxa::attachment_view(FindFreePagesH::AT.vsm_find_free_pages_header, info.vsm_state->find_free_pages_header),
            daxa::attachment_view(FindFreePagesH::AT.vsm_allocate_indirect, info.vsm_state->allocate_indirect),
            daxa::attachment_view(FindFreePagesH::AT.vsm_clear_indirect, info.vsm_state->clear_indirect),
            daxa::attachment_view(FindFreePagesH::AT.vsm_clear_dirty_bit_indirect, info.vsm_state->clear_dirty_bit_indirect),
            daxa::attachment_view(FindFreePagesH::AT.vsm_globals, info.vsm_state->globals),
            daxa::attachment_view(FindFreePagesH::AT.vsm_allocation_count, info.vsm_state->allocation_count),
            daxa::attachment_view(FindFreePagesH::AT.vsm_meta_memory_table, info.vsm_state->meta_memory_table),
        },
        .render_context = info.render_context,
        .timeline_pool = info.vsm_state->vsm_timeline_query_pool,
        .per_frame_timestamp_count = info.vsm_state->PER_FRAME_TIMESTAMP_COUNT,
    });

    info.tg->add_task(AllocatePagesTask{
        .views = std::array{
            daxa::attachment_view(AllocatePagesH::AT.vsm_find_free_pages_header, info.vsm_state->find_free_pages_header),
            daxa::attachment_view(AllocatePagesH::AT.vsm_globals, info.vsm_state->globals),
            daxa::attachment_view(AllocatePagesH::AT.vsm_allocation_count, info.vsm_state->allocation_count),
            daxa::attachment_view(AllocatePagesH::AT.vsm_allocation_requests, info.vsm_state->allocation_requests),
            daxa::attachment_view(AllocatePagesH::AT.vsm_free_pages_buffer, info.vsm_state->free_page_buffer),
            daxa::attachment_view(AllocatePagesH::AT.vsm_not_visited_pages_buffer, info.vsm_state->not_visited_page_buffer),
            daxa::attachment_view(AllocatePagesH::AT.vsm_allocate_indirect, info.vsm_state->allocate_indirect),
            daxa::attachment_view(AllocatePagesH::AT.vsm_clip_projections, info.vsm_state->clip_projections),
            daxa::attachment_view(AllocatePagesH::AT.vsm_page_table, vsm_page_table_view),
            daxa::attachment_view(AllocatePagesH::AT.vsm_page_height_offsets, vsm_page_height_offsets_view),
            daxa::attachment_view(AllocatePagesH::AT.vsm_meta_memory_table, info.vsm_state->meta_memory_table),
        },
        .render_context = info.render_context,
        .timeline_pool = info.vsm_state->vsm_timeline_query_pool,
        .per_frame_timestamp_count = info.vsm_state->PER_FRAME_TIMESTAMP_COUNT,
    });

    if (!info.vsm_state->overdraw_debug_image.is_null())
    {
        task_clear_image(*info.tg, info.vsm_state->overdraw_debug_image, std::array{0,0,0,0});
    }
    info.tg->add_task(ClearPagesTask{
        .views = std::array{
            daxa::attachment_view(ClearPagesH::AT.vsm_allocation_requests, info.vsm_state->allocation_requests),
            daxa::attachment_view(ClearPagesH::AT.vsm_clear_indirect, info.vsm_state->clear_indirect),
            daxa::attachment_view(ClearPagesH::AT.vsm_page_table, vsm_page_table_view),
            daxa::attachment_view(ClearPagesH::AT.vsm_memory, info.vsm_state->memory_block),
            daxa::attachment_view(ClearPagesH::AT.vsm_memory64, info.vsm_state->memory_block64),
        },
        .render_context = info.render_context,
        .timeline_pool = info.vsm_state->vsm_timeline_query_pool,
        .per_frame_timestamp_count = info.vsm_state->PER_FRAME_TIMESTAMP_COUNT,
    });

    info.tg->add_task(GenDirtyBitHizTask{
        .views = std::array{
            daxa::attachment_view(GenDirtyBitHizH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(GenDirtyBitHizH::AT.vsm_clip_projections, info.vsm_state->clip_projections),
            daxa::attachment_view(GenDirtyBitHizH::AT.vsm_page_table, vsm_page_table_view),
            daxa::attachment_view(GenDirtyBitHizH::AT.vsm_dirty_bit_hiz, vsm_dirty_bit_hiz_view),
        },
        .render_context = info.render_context,
        .timeline_pool = info.vsm_state->vsm_timeline_query_pool,
        .per_frame_timestamp_count = info.vsm_state->PER_FRAME_TIMESTAMP_COUNT,
    });

    std::array<std::array<daxa::TaskBufferView, 2>, 16> cascade_meshlet_expansions = {};
    for (u32 cascade = 0; cascade < 16; ++cascade)
    {
        tasks_expand_meshes_to_meshlets(TaskExpandMeshesToMeshletsInfo{
            .render_context = info.render_context,
            .task_list = *info.tg,
            .cull_meshes = true,
            .vsm_hip = info.vsm_state->dirty_pages_hiz,
            .vsm_cascade = cascade,
            .vsm_clip_projections = info.vsm_state->clip_projections,
            .globals = info.render_context->tgpu_render_data,
            .mesh_instances = info.mesh_instances,
            .meshes = info.scene->_gpu_mesh_manifest,
            .materials = info.scene->_gpu_material_manifest,
            .entity_meta = info.scene->_gpu_entity_meta,
            .entity_meshgroup_indices = info.scene->_gpu_entity_mesh_groups,
            .meshgroups = info.scene->_gpu_mesh_group_manifest,
            .entity_transforms = info.scene->_gpu_entity_transforms,
            .entity_combined_transforms = info.scene->_gpu_entity_combined_transforms,
            .opaque_meshlet_cull_po2expansions = cascade_meshlet_expansions[cascade],
            .dispatch_clear = {0,1,1},
            .buffer_name_prefix = std::string("vsm cascade ") + std::to_string(cascade) + ' ',
        });
    }

    info.tg->add_task(CullAndDrawPagesTask{
        .views = std::array{
            daxa::attachment_view(CullAndDrawPagesH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion0, cascade_meshlet_expansions[0][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion0, cascade_meshlet_expansions[0][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion1, cascade_meshlet_expansions[1][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion1, cascade_meshlet_expansions[1][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion2, cascade_meshlet_expansions[2][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion2, cascade_meshlet_expansions[2][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion3, cascade_meshlet_expansions[3][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion3, cascade_meshlet_expansions[3][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion4, cascade_meshlet_expansions[4][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion4, cascade_meshlet_expansions[4][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion5, cascade_meshlet_expansions[5][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion5, cascade_meshlet_expansions[5][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion6, cascade_meshlet_expansions[6][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion6, cascade_meshlet_expansions[6][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion7, cascade_meshlet_expansions[7][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion7, cascade_meshlet_expansions[7][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion8, cascade_meshlet_expansions[8][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion8, cascade_meshlet_expansions[8][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion9, cascade_meshlet_expansions[9][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion9, cascade_meshlet_expansions[9][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion10, cascade_meshlet_expansions[10][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion10, cascade_meshlet_expansions[10][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion11, cascade_meshlet_expansions[11][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion11, cascade_meshlet_expansions[11][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion12, cascade_meshlet_expansions[12][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion12, cascade_meshlet_expansions[12][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion13, cascade_meshlet_expansions[13][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion13, cascade_meshlet_expansions[13][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion14, cascade_meshlet_expansions[14][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion14, cascade_meshlet_expansions[14][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.po2expansion15, cascade_meshlet_expansions[15][0]),
            daxa::attachment_view(CullAndDrawPagesH::AT.masked_po2expansion15, cascade_meshlet_expansions[15][1]),
            daxa::attachment_view(CullAndDrawPagesH::AT.meshlet_instances, info.meshlet_instances),
            daxa::attachment_view(CullAndDrawPagesH::AT.mesh_instances, info.mesh_instances),
            daxa::attachment_view(CullAndDrawPagesH::AT.meshes, info.meshes),
            daxa::attachment_view(CullAndDrawPagesH::AT.entity_combined_transforms, info.entity_combined_transforms),
            daxa::attachment_view(CullAndDrawPagesH::AT.material_manifest, info.material_manifest),
            daxa::attachment_view(CullAndDrawPagesH::AT.vsm_clip_projections, info.vsm_state->clip_projections),
            daxa::attachment_view(CullAndDrawPagesH::AT.vsm_dirty_bit_hiz, vsm_dirty_bit_hiz_view),
            daxa::attachment_view(CullAndDrawPagesH::AT.vsm_page_table, vsm_page_table_view),
            daxa::attachment_view(CullAndDrawPagesH::AT.vsm_memory_block, info.vsm_state->memory_block),
            daxa::attachment_view(CullAndDrawPagesH::AT.vsm_memory_block64, info.vsm_state->memory_block64),
            daxa::attachment_view(CullAndDrawPagesH::AT.vsm_overdraw_debug, info.vsm_state->overdraw_debug_image),
        },
        .render_context = info.render_context,
        .timeline_pool = info.vsm_state->vsm_timeline_query_pool,
        .per_frame_timestamp_count = info.vsm_state->PER_FRAME_TIMESTAMP_COUNT,
    });

    task_clear_image(*info.tg, info.render_context->gpuctx->shader_debug_context.vsm_debug_page_table, std::array{0.0f, 0.0f, 0.0f, 0.0f});
    info.tg->add_task(DebugVirtualPageTableTask{
        .views = std::array{
            daxa::attachment_view(DebugVirtualPageTableH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(DebugVirtualPageTableH::AT.vsm_globals, info.vsm_state->globals),
            daxa::attachment_view(DebugVirtualPageTableH::AT.vsm_page_table, vsm_page_table_view),
            daxa::attachment_view(DebugVirtualPageTableH::AT.vsm_debug_page_table, info.render_context->gpuctx->shader_debug_context.vsm_debug_page_table),
        },
        .render_context = info.render_context,
        .timeline_pool = info.vsm_state->vsm_timeline_query_pool,
        .per_frame_timestamp_count = info.vsm_state->PER_FRAME_TIMESTAMP_COUNT,
    });

    task_clear_image(*info.tg, info.render_context->gpuctx->shader_debug_context.vsm_debug_meta_memory_table, std::array{0.0f, 0.0f, 0.0f, 0.0f});
    info.tg->add_task(DebugMetaMemoryTableTask{
        .views = std::array{
            daxa::attachment_view(DebugMetaMemoryTableH::AT.vsm_page_table, vsm_page_table_view),
            daxa::attachment_view(DebugMetaMemoryTableH::AT.vsm_meta_memory_table, info.vsm_state->meta_memory_table),
            daxa::attachment_view(DebugMetaMemoryTableH::AT.vsm_debug_meta_memory_table, info.render_context->gpuctx->shader_debug_context.vsm_debug_meta_memory_table),
        },
        .render_context = info.render_context,
        .timeline_pool = info.vsm_state->vsm_timeline_query_pool,
        .per_frame_timestamp_count = info.vsm_state->PER_FRAME_TIMESTAMP_COUNT,
    });

    info.tg->add_task(ClearDirtyBitTask{
        .views = std::array{
            daxa::attachment_view(ClearDirtyBitH::AT.vsm_allocation_requests, info.vsm_state->allocation_requests),
            daxa::attachment_view(ClearDirtyBitH::AT.vsm_allocation_count, info.vsm_state->allocation_count),
            daxa::attachment_view(ClearDirtyBitH::AT.vsm_clear_dirty_bit_indirect, info.vsm_state->clear_dirty_bit_indirect),
            daxa::attachment_view(ClearDirtyBitH::AT.vsm_page_table, vsm_page_table_view),
        },
        .render_context = info.render_context,
        .timeline_pool = info.vsm_state->vsm_timeline_query_pool,
        .per_frame_timestamp_count = info.vsm_state->PER_FRAME_TIMESTAMP_COUNT,
    });
}

struct CameraController;
struct GetVSMProjectionsInfo
{
    CameraInfo const * camera_info = {};
    f32vec3 sun_direction = {};
    f32 sun_x_angle = {};
    f32 clip_0_scale = {};
    f32 clip_0_near = {};
    f32 clip_0_far = {};
    f64 clip_0_height_offset = {};
    bool use_simplified_light_matrix = false;

    ShaderDebugDrawContext * debug_context = {};
};

inline auto get_vsm_projections(GetVSMProjectionsInfo const & info) -> std::array<VSMClipProjection, VSM_CLIP_LEVELS>
{
    std::array<VSMClipProjection, VSM_CLIP_LEVELS> clip_projections = {};
    auto const default_vsm_pos = glm::vec3{0.0, 0.0, 0.0};
    auto const default_vsm_up = glm::vec3{0.0, 0.0, 1.0};
    auto const default_vsm_forward = -info.sun_direction;
    auto const default_vsm_view = glm::lookAt(default_vsm_pos, default_vsm_forward, default_vsm_up);

    auto const xy_forward = f32vec4(glm::normalize(f32vec3(default_vsm_forward.x, default_vsm_forward.y, 0.0)), 0.0);
    auto const xy_right = f32vec4(-xy_forward.y, xy_forward.x, 0.0, 0.0);

    auto to_rotation_inv_matrix = glm::identity<f32mat4x4>();
    to_rotation_inv_matrix[0] = f32vec4(xy_forward.x, xy_right.x, 0.0, 0.0);
    to_rotation_inv_matrix[1] = f32vec4(xy_forward.y, xy_right.y, 0.0, 0.0);

    f32vec3 const rotation_inv_vsm_forward = to_rotation_inv_matrix * f32vec4(default_vsm_forward, 0.0f);
    auto const rotation_inv_vsm_view = glm::lookAt(default_vsm_pos, rotation_inv_vsm_forward, default_vsm_up);

    auto calculate_clip_projection = [&info](i32 clip) -> glm::mat4x4
    {
        auto const clip_scale = std::pow(2.0f, s_cast<f32>(clip));
        auto clip_projection = glm::ortho(
            -info.clip_0_scale * clip_scale, // left
            info.clip_0_scale * clip_scale,  // right
            -info.clip_0_scale * clip_scale, // bottom
            info.clip_0_scale * clip_scale,  // top
            info.use_simplified_light_matrix ? -1000.0f : info.clip_0_near * clip_scale,   // near
            info.use_simplified_light_matrix ?  1000.0f : info.clip_0_far * clip_scale     // far
        );
        // Switch from OpenGL default to Vulkan default (invert the Y clip coordinate)
        clip_projection[1][1] *= -1.0;
        return clip_projection;
    };
    auto const target_camera_position = glm::vec4(info.camera_info->position, 1.0);
    auto const uv_page_size = s_cast<f32>(VSM_PAGE_SIZE) / s_cast<f32>(VSM_TEXTURE_RESOLUTION);
    // NDC space is [-1, 1] but uv space is [0, 1], PAGE_SIZE / TEXTURE_RESOLUTION gives us the page size in uv space
    // thus we need to multiply by two to get the page size in ndc coordinates
    auto const ndc_page_size = uv_page_size * 2.0f;

    // 0 == x, 2 == z
    f32 const elevation_angle = glm::asin(rotation_inv_vsm_forward.z / std::sqrt(std::pow(rotation_inv_vsm_forward.x, 2.0f) + std::pow(rotation_inv_vsm_forward.z, 2.0f)));
    auto const align_axis = (90.0f + glm::degrees(elevation_angle)) > 45.0f ? PAGE_ALIGN_AXIS_X : PAGE_ALIGN_AXIS_Z;
    for (i32 clip = 0; clip < VSM_CLIP_LEVELS; clip++)
    {
        auto const curr_clip_proj = calculate_clip_projection(clip);
        auto const clip_projection_view = curr_clip_proj * default_vsm_view;
        auto const curr_clip_scale = std::pow(2.0f, s_cast<f32>(clip));

        // Project the target position into VSM ndc coordinates and calculate a page alligned position
        auto const clip_projected_target_pos = clip_projection_view * target_camera_position;
        auto const ndc_target_pos = glm::vec3(clip_projected_target_pos) / clip_projected_target_pos.w;
        auto const ndc_page_scaled_target_pos = glm::vec2(ndc_target_pos) / ndc_page_size;
        auto const ndc_page_scaled_aligned_target_pos = glm::vec2(glm::ceil(ndc_page_scaled_target_pos));

        // Here we calculate the offsets that will be applied per page in the clip level
        // This is used to virtually offset the depth of each page so that we can actually snap the vsm position to the camera position
        auto const inv_rot_clip_projection_view = glm::inverse(curr_clip_proj * rotation_inv_vsm_view);
        auto const near_offset_ndc_u_in_world = inv_rot_clip_projection_view * glm::vec4(ndc_page_size, 0.0, 0.0, 1.0);
        auto const near_offset_ndc_v_in_world = inv_rot_clip_projection_view * glm::vec4(0.0, ndc_page_size, 0.0, 1.0);

        // Inverse projection from ndc -> world does not account for near plane offset, thus we need to add it manually
        // we simply shift the position in the oppposite of view direction by near plane distance
        auto const curr_clip_near = info.clip_0_near * curr_clip_scale;
        auto const ndc_u_in_world = glm::vec3(near_offset_ndc_u_in_world) + curr_clip_near * -rotation_inv_vsm_forward;
        auto const ndc_v_in_world = glm::vec3(near_offset_ndc_v_in_world) + curr_clip_near * -rotation_inv_vsm_forward;

        // Calculate the actual per page world space offsets
        f32 const u_offset_scale = ndc_u_in_world[align_axis] / rotation_inv_vsm_forward[align_axis];
        auto const u_offset_vector = u_offset_scale * -rotation_inv_vsm_forward;

        f32 const v_offset_scale = ndc_v_in_world[align_axis] / rotation_inv_vsm_forward[align_axis];
        auto const v_offset_vector = v_offset_scale * -rotation_inv_vsm_forward;

        // Get the per page offsets on a world space xy plane
        f32vec3 const xy_plane_ndc_u_in_world = f32vec4(ndc_u_in_world + u_offset_vector, 0.0);
        f32vec3 const xy_plane_ndc_v_in_world = f32vec4(ndc_v_in_world + v_offset_vector, 0.0);

        auto const rot_clip_xy_plane_world_position = glm::vec3(
            ndc_page_scaled_aligned_target_pos.x * xy_plane_ndc_u_in_world +
            ndc_page_scaled_aligned_target_pos.y * xy_plane_ndc_v_in_world);

        f32vec3 const clip_xy_plane_world_position = f32vec4(rot_clip_xy_plane_world_position, 0.0);
        // Clip offset from the xy plane - essentially clip_xy_plane_world_position gives us the position on a world xy plane positioned
        // at the height 0. We want to shift the clip camera up so that it observes the player position from the above. The height from
        // which the camera observes this player should be set according to the info.height_offset
        f32 const height_offset_scaling_factor = std::min(1.0f / -rotation_inv_vsm_forward[align_axis], 100.0f);
        f32vec3 const rot_camera_position = to_rotation_inv_matrix * f32vec4(info.camera_info->position, 0.0f);
        auto const unclamped_view_offset_scale = s_cast<i32>(
            std::floor(rot_camera_position[align_axis] * height_offset_scaling_factor) +
            (std::floor(info.clip_0_height_offset * (align_axis == 0 ? -1.0f : 1.0f) * height_offset_scaling_factor) * curr_clip_scale));

        auto const view_offset_scale = std::min(unclamped_view_offset_scale, 10000);
        auto const view_offset = s_cast<f32>(view_offset_scale) * -rotation_inv_vsm_forward;
        f32vec3 const clip_position = glm::inverse(to_rotation_inv_matrix) * f32vec4(clip_xy_plane_world_position + view_offset, 0.0);

        auto const rotation_invar_inv_proj_view = glm::inverse(curr_clip_proj * rotation_inv_vsm_view);
        auto const near_offset_inv_ndc_v_in_world = rotation_invar_inv_proj_view * glm::vec4(0.0, ndc_page_size, 0.0, 1.0);
        auto const inv_ndc_v_in_world = glm::vec3(near_offset_inv_ndc_v_in_world) + curr_clip_near * -rotation_inv_vsm_forward;
        f32 const v_inv_offset_scale = inv_ndc_v_in_world[align_axis] / rotation_inv_vsm_forward[align_axis];
        auto const v_inv_offset_vector = v_inv_offset_scale * -rotation_inv_vsm_forward;

        auto const final_rot_inv_clip_view = glm::lookAt(clip_position, clip_position + glm::normalize(rotation_inv_vsm_forward), default_vsm_up);
        auto const final_rot_inv_clip_projection_view = curr_clip_proj * final_rot_inv_clip_view;
        auto const inv_origin_shift = (final_rot_inv_clip_projection_view * glm::vec4(0.0, 0.0, 0.0, 1.0)).z;
        auto const page_depth_offset = (final_rot_inv_clip_projection_view * glm::vec4(v_inv_offset_vector, 1.0)).z - inv_origin_shift;

        auto const final_clip_view = [&]
        {
            if (info.use_simplified_light_matrix)
            {
                // Find the offset from the un-translated view matrix
                auto const ndcShift = 2.0f * ndc_page_scaled_aligned_target_pos / s_cast<float>(VSM_PAGE_TABLE_RESOLUTION);

                // Shift rendering projection matrix by opposite of page offset in clip space, then apply *only* that shift to the view matrix
                auto const shiftedProjection = glm::translate(glm::mat4(1), glm::vec3(-ndcShift, 0)) * curr_clip_proj;
                return glm::inverse(curr_clip_proj) * shiftedProjection * default_vsm_view;
            }

            return glm::lookAt(clip_position, clip_position + glm::normalize(default_vsm_forward), default_vsm_up);
        }();

        auto clip_camera = CameraInfo{
            .view = final_clip_view,
            .inv_view = glm::inverse(final_clip_view),
            .proj = curr_clip_proj,
            .inv_proj = glm::inverse(curr_clip_proj),
            .view_proj = curr_clip_proj * final_clip_view,
            .inv_view_proj = glm::inverse(curr_clip_proj * final_clip_view),
            .position = clip_position,
            .up = default_vsm_up,
            .screen_size = {VSM_PAGE_TABLE_RESOLUTION << 1, VSM_PAGE_TABLE_RESOLUTION << 1},
            .inv_screen_size = {
                1.0f / s_cast<f32>(VSM_PAGE_TABLE_RESOLUTION << 1),
                1.0f / s_cast<f32>(VSM_PAGE_TABLE_RESOLUTION << 1),
            },
        };

        glm::vec3 ws_ndc_corners[2][2][2];
        for (u32 z = 0; z < 2; ++z)
        {
            for (u32 y = 0; y < 2; ++y)
            {
                for (u32 x = 0; x < 2; ++x)
                {
                    glm::vec3 corner = glm::vec3((glm::vec2(x, y) - 0.5f) * 2.0f, 1.0f - z * 0.5f);
                    glm::vec4 proj_corner = clip_camera.inv_view_proj * glm::vec4(corner, 1);
                    ws_ndc_corners[x][y][z] = glm::vec3(proj_corner) / proj_corner.w;
                }
            }
        }
        clip_camera.orthogonal_half_ws_width = curr_clip_scale * info.clip_0_scale;
        clip_camera.is_orthogonal = 1u;
        clip_camera.near_plane_normal = glm::normalize(
            glm::cross(ws_ndc_corners[0][1][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0]));
        clip_camera.right_plane_normal = glm::normalize(
            glm::cross(ws_ndc_corners[1][1][0] - ws_ndc_corners[1][0][0], ws_ndc_corners[1][0][1] - ws_ndc_corners[1][0][0]));
        clip_camera.left_plane_normal = glm::normalize(
            glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][0][1], ws_ndc_corners[0][0][0] - ws_ndc_corners[0][0][1]));
        clip_camera.top_plane_normal = glm::normalize(
            glm::cross(ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[0][0][1] - ws_ndc_corners[0][0][0]));
        clip_camera.bottom_plane_normal = glm::normalize(
            glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][1][0], ws_ndc_corners[1][1][0] - ws_ndc_corners[0][1][0]));

        const f32 near_plane = info.clip_0_near * curr_clip_scale;
        const f32 far_plane = info.clip_0_far * curr_clip_scale;
        const f32 near_to_far_range = far_plane - near_plane;
        clip_projections.at(clip) = VSMClipProjection{
            .height_offset = info.use_simplified_light_matrix ? 0 : view_offset_scale,
            .depth_page_offset = info.use_simplified_light_matrix ? 0 : page_depth_offset,
            .page_offset = {
                (-s_cast<daxa_i32>(ndc_page_scaled_aligned_target_pos.x)),
                (-s_cast<daxa_i32>(ndc_page_scaled_aligned_target_pos.y)),
            },
            .near_to_far_range = near_to_far_range,
            .near_dist = near_plane,
            .camera = clip_camera,
        };
    }
    return clip_projections;
}

struct DebugDrawClipFrustiInfo
{
    GetVSMProjectionsInfo const * proj_info = {};
    std::array<VSMClipProjection, VSM_CLIP_LEVELS> * clip_projections = {};
    std::array<bool, VSM_CLIP_LEVELS> * draw_clip_frustum = {};
    std::array<bool, VSM_CLIP_LEVELS> * draw_clip_frustum_pages = {};
    ShaderDebugDrawContext * debug_context = {};
    f32vec3 vsm_view_direction = {};
};

inline void debug_draw_clip_fusti(DebugDrawClipFrustiInfo const & info)
{

    auto hsv2rgb = [](f32vec3 c) -> f32vec3
    {
        f32vec4 k = f32vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        f32vec3 p = f32vec3(
            std::abs(glm::fract(c.x + k.x) * 6.0 - k.w),
            std::abs(glm::fract(c.x + k.y) * 6.0 - k.w),
            std::abs(glm::fract(c.x + k.z) * 6.0 - k.w));
        return c.z * glm::mix(f32vec3(k.x), glm::clamp(p - f32vec3(k.x), f32vec3(0.0), f32vec3(1.0)), f32vec3(c.y));
    };

    static constexpr std::array offsets = {
        glm::ivec2(-1, 1), glm::ivec2(-1, -1), glm::ivec2(1, -1), glm::ivec2(1, 1),
        glm::ivec2(-1, 1), glm::ivec2(-1, -1), glm::ivec2(1, -1), glm::ivec2(1, 1)};

    for (i32 clip = 0; clip < VSM_CLIP_LEVELS; clip++)
    {
        auto const & clip_projection = info.clip_projections->at(clip);
        if (!info.draw_clip_frustum->at(clip)) { continue; }

        if (info.draw_clip_frustum_pages->at(clip))
        {
            auto const left_right_size = std::abs((1.0f / std::bit_cast<glm::mat4x4>(clip_projection.camera.proj)[0][0])) * 2.0f;
            auto const top_bottom_size = std::abs((1.0f / std::bit_cast<glm::mat4x4>(clip_projection.camera.proj)[1][1])) * 2.0f;
            auto const near_far_size = (1.0f / std::bit_cast<glm::mat4x4>(clip_projection.camera.proj)[2][2]) * 2.0f;
            auto const page_size = glm::vec2(left_right_size / s_cast<f32>(VSM_PAGE_TABLE_RESOLUTION), top_bottom_size / s_cast<f32>(VSM_PAGE_TABLE_RESOLUTION));
            auto const page_proj = glm::ortho(
                -page_size.x / 2.0f,
                page_size.x / 2.0f,
                -page_size.y / 2.0f,
                page_size.y / 2.0f,
                info.proj_info->clip_0_near * s_cast<i32>(std::pow(2.0f, clip)),
                info.proj_info->clip_0_far * s_cast<i32>(std::pow(2.0f, clip)));
            auto const uv_page_size = s_cast<f32>(VSM_PAGE_SIZE) / s_cast<f32>(VSM_TEXTURE_RESOLUTION);
            for (i32 page_u_index = 0; page_u_index < VSM_PAGE_TABLE_RESOLUTION; page_u_index++)
            {
                for (i32 page_v_index = 0; page_v_index < VSM_PAGE_TABLE_RESOLUTION; page_v_index++)
                {
                    auto const corner_virtual_uv = uv_page_size * glm::vec2(page_u_index, page_v_index);
                    auto const page_center_virtual_uv_offset = glm::vec2(uv_page_size * 0.5f);
                    auto const virtual_uv = corner_virtual_uv + page_center_virtual_uv_offset;

                    auto const page_index = glm::ivec2(virtual_uv * s_cast<f32>(VSM_PAGE_TABLE_RESOLUTION));
                    f32 depth;
                    if (clip_projection.page_align_axis == PAGE_ALIGN_AXIS_X)
                    {
                        depth = -page_index.y * clip_projection.depth_page_offset;
                    }
                    else
                    {
                        depth = ((VSM_PAGE_TABLE_RESOLUTION - 1) - page_index.y) * clip_projection.depth_page_offset;
                    }
                    auto const virtual_page_ndc = (virtual_uv * 2.0f) - glm::vec2(1.0f);
                    auto const page_ndc_position = glm::vec4(virtual_page_ndc, -depth, 1.0);
                    auto const new_position = std::bit_cast<glm::mat4x4>(clip_projection.camera.inv_view_proj) * page_ndc_position;

                    auto const page_view = glm::lookAt(glm::vec3(new_position), glm::vec3(new_position) + info.vsm_view_direction, {0.0, 0.0, 1.0});
                    auto const page_inv_projection_view = glm::inverse(page_proj * page_view);

                    ShaderDebugBoxDraw box_draw = {};
                    box_draw.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
                    box_draw.color = std::bit_cast<daxa_f32vec3>(hsv2rgb(f32vec3(std::pow(s_cast<f32>(clip) / (VSM_CLIP_LEVELS - 1), 0.5f), 1.0f, 1.0f)));
                    for (i32 i = 0; i < 8; i++)
                    {
                        auto const ndc_pos = glm::vec4(offsets[i], i < 4 ? 0.0f : 1.0f, 1.0f);
                        auto const world_pos = page_inv_projection_view * ndc_pos;
                        box_draw.vertices[i] = {world_pos.x, world_pos.y, world_pos.z};
                    }
                    info.debug_context->cpu_debug_box_draws.push_back(box_draw);
                }
            }
        }
        else
        {
            ShaderDebugBoxDraw box_draw = {};
            box_draw.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
            box_draw.color = std::bit_cast<daxa_f32vec3>(hsv2rgb(f32vec3(std::pow(s_cast<f32>(clip) / (VSM_CLIP_LEVELS - 1), 0.5f), 1.0f, 1.0f)));
            for (i32 i = 0; i < 8; i++)
            {
                auto const ndc_pos = glm::vec4(offsets[i], i < 4 ? 0.0f : 1.0f, 1.0f);
                auto const world_pos = std::bit_cast<glm::mat4x4>(clip_projection.camera.inv_view_proj) * ndc_pos;
                box_draw.vertices[i] = {world_pos.x, world_pos.y, world_pos.z};
            }
            info.debug_context->cpu_debug_box_draws.push_back(box_draw);
        }
    }
}

inline void fill_vsm_invalidation_mask(std::vector<DynamicMesh> const & meshes, VSMState & state, ShaderDebugDrawContext & context)
{
    ShaderDebugAABBDraw aabb_draw = {};
    aabb_draw.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
    static constexpr std::array<f32vec3, 8> base_positions =
        {
            f32vec3{-1, -1, -1},
            f32vec3{1, -1, -1},
            f32vec3{-1, 1, -1},
            f32vec3{1, 1, -1},
            f32vec3{-1, -1, 1},
            f32vec3{1, -1, 1},
            f32vec3{-1, 1, 1},
            f32vec3{1, 1, 1},
        };

    for (i32 clip = 0; clip < VSM_CLIP_LEVELS; clip++)
    {
        std::memset(&state.free_wrapped_pages_info_cpu.at(clip).mask, 0, sizeof(state.free_wrapped_pages_info_cpu.at(0).mask));
    }
    for (auto const & mesh : meshes)
    {
        for (auto const & aabb : mesh.meshlet_aabbs)
        {
            // for (i32 i = 0; i < 2; i++)
            // {
            //     auto const transform = i == 0 ? mesh.curr_transform : mesh.prev_transform;
            //     auto const ws_center = transform * f32vec4(aabb.center.x, aabb.center.y, aabb.center.z, 1.0f);
            //     auto const ws_size = transform * f32vec4(aabb.size.x, aabb.size.y, aabb.size.z, 0.0f);
            //     aabb_draw.position = {ws_center.x, ws_center.y, ws_center.z};
            //     aabb_draw.size = {ws_size.x, ws_size.y, ws_size.z};
            //     aabb_draw.color = i == 0 ? daxa_f32vec3{0.0, 1.0, 0.0} : daxa_f32vec3{0.0, 0.0, 1.0};
            //     context.cpu_debug_aabb_draws.push_back(aabb_draw);
            // }

            for (i32 clip_level = 0; clip_level < VSM_CLIP_LEVELS; clip_level++)
            {
                for (i32 i = 0; i < 2; i++)
                {
                    f32vec2 min_bounds = f32vec2(std::numeric_limits<f32>::infinity());
                    f32vec2 max_bounds = f32vec2(-std::numeric_limits<f32>::infinity());
                    auto & curr_page_mask = state.free_wrapped_pages_info_cpu.at(clip_level).mask;
                    for (auto const & base_position : base_positions)
                    {
                        auto const model_matrix = i == 0 ? mesh.curr_transform : mesh.prev_transform;
                        f32vec4 const model_position = f32vec4(base_position * std::bit_cast<f32vec3>(aabb.size) * 0.5f + std::bit_cast<f32vec3>(aabb.center), 1.0f);
                        auto const world_position = model_matrix * model_position;
                        auto const clip_pos = state.clip_projections_cpu.at(clip_level).camera.view_proj * world_position;
                        auto const uv_pos = (f32vec2(clip_pos.x, clip_pos.y) + 1.0f) * 0.5f;
                        min_bounds = glm::min(min_bounds, uv_pos);
                        max_bounds = glm::max(max_bounds, uv_pos);
                    }
                    i32vec2 const min_page_bounds = i32vec2(floor(min_bounds * f32(VSM_PAGE_TABLE_RESOLUTION)));
                    i32vec2 const max_page_bounds = i32vec2(ceil(max_bounds * f32(VSM_PAGE_TABLE_RESOLUTION)));

                    bool const is_oustide_frustum =
                        glm::any(glm::greaterThanEqual(min_page_bounds, i32vec2(VSM_PAGE_TABLE_RESOLUTION))) ||
                        glm::any(glm::lessThanEqual(max_page_bounds, i32vec2(0)));

                    if (is_oustide_frustum)
                    {
                        continue;
                    }
                    i32vec2 const clamped_min_bounds = glm::clamp(min_page_bounds, 0, VSM_PAGE_TABLE_RESOLUTION - 1);
                    i32vec2 const clamped_max_bounds = glm::clamp(max_page_bounds, 0, VSM_PAGE_TABLE_RESOLUTION - 1);

                    for (i32 x = clamped_min_bounds.x; x < clamped_max_bounds.x; x++)
                    {
                        for (i32 y = clamped_min_bounds.y; y < clamped_max_bounds.y; y++)
                        {
                            auto const linear_index = x * VSM_PAGE_TABLE_RESOLUTION + y;
                            auto const index_offset = linear_index / 32;
                            auto const in_uint_offset = linear_index % 32;
                            curr_page_mask[index_offset] |= 1u << in_uint_offset;
                        }
                    }
                }
            }
        }
    }
}
#endif //__cplusplus