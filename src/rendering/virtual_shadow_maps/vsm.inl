#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry_pipeline.inl"

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
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, vsm_meta_memory_table)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(CullAndDrawPages_WriteCommandH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(MeshletCullArgBucketsBufferHead), vsm_meshlets_cull_arg_buckets)
DAXA_DECL_TASK_HEAD_END

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
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(GenDirtyBitHizH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_TH_IMAGE_ID_MIP_ARRAY(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D_ARRAY, vsm_dirty_bit_hiz, 8)
DAXA_DECL_TASK_HEAD_END
struct GenDirtyBitHizPush
{
    DAXA_TH_BLOB(GenDirtyBitHizH, attachments)
    daxa_u32 mip_count;
};

DAXA_DECL_TASK_HEAD_BEGIN(CullAndDrawPagesH)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(MeshletCullArgBucketsBufferHead), meshlets_cull_arg_buckets)
// Draw Attachments:
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMaterial), material_manifest)
// Vsm Attachments:
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_SAMPLED, REGULAR_2D_ARRAY, vsm_dirty_bit_hiz)
DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_STORAGE_READ_ONLY, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_STORAGE_READ_WRITE, REGULAR_2D_ARRAY, vsm_memory_block)
DAXA_DECL_TASK_HEAD_END
struct CullAndDrawPagesPush
{
    DAXA_TH_BLOB(CullAndDrawPagesH, attachments)
    daxa_u32 draw_list_type;
    daxa_u32 bucket_index;
    daxa_ImageViewId daxa_u32_vsm_memory_view;
};

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
        .push_constant_size = static_cast<u32>(sizeof(ClearPagesH::AttachmentShaderBlob)),
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
        .raster = {.depth_clamp_enable = true},
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

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_free_wrapped_pages_pipeline_compile_info().name));
        FreeWrappedPagesH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({1, VSM_PAGE_TABLE_RESOLUTION, VSM_CLIP_LEVELS});
    }
};

struct MarkRequiredPagesTask : MarkRequiredPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const depth_resolution = render_context->gpuctx->device.info_image(ti.get(AT.depth).ids[0]).value().size;
        auto const dispatch_size = u32vec2{
            (depth_resolution.x + MARK_REQUIRED_PAGES_X_DISPATCH - 1) / MARK_REQUIRED_PAGES_X_DISPATCH,
            (depth_resolution.y + MARK_REQUIRED_PAGES_Y_DISPATCH - 1) / MARK_REQUIRED_PAGES_Y_DISPATCH,
        };
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_mark_required_pages_pipeline_compile_info().name));
        MarkRequiredPagesH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = dispatch_size.x, .y = dispatch_size.y});
    }
};

struct FindFreePagesTask : FindFreePagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    static constexpr i32 meta_memory_pix_count = VSM_META_MEMORY_TABLE_RESOLUTION * VSM_META_MEMORY_TABLE_RESOLUTION;
    static constexpr i32 dispatch_x_size = (meta_memory_pix_count + FIND_FREE_PAGES_X_DISPATCH - 1) / FIND_FREE_PAGES_X_DISPATCH;

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_find_free_pages_pipeline_compile_info().name));
        FindFreePagesH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({dispatch_x_size, 1, 1});
    }
};

struct AllocatePagesTask : AllocatePagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_allocate_pages_pipeline_compile_info().name));
        AllocatePagesH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch_indirect({
            .indirect_buffer = ti.get(AT.vsm_allocate_indirect).ids[0],
            .offset = 0u,
        });
    }
};

struct ClearPagesTask : ClearPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_clear_pages_pipeline_compile_info().name));
        ClearPagesH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch_indirect({
            .indirect_buffer = ti.get(AT.vsm_clear_indirect).ids[0],
            .offset = 0u,
        });
    }
};

struct GenDirtyBitHizTask : GenDirtyBitHizH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_gen_dirty_bit_hiz_pipeline_compile_info().name));
        auto const dispatch_x = round_up_div(VSM_PAGE_TABLE_RESOLUTION, 64);
        auto const dispatch_y = round_up_div(VSM_PAGE_TABLE_RESOLUTION, 64);
        GenDirtyBitHizPush push = {
            .mip_count = ti.get(AT.vsm_dirty_bit_hiz).view.slice.level_count,
        };
        assign_blob(push.attachments, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({dispatch_x, dispatch_y, VSM_CLIP_LEVELS});
    }
};

struct CullAndDrawPagesTask : CullAndDrawPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const memory_block_view = render_context->gpuctx->device.create_image_view({
            .type = daxa::ImageViewType::REGULAR_2D,
            .format = daxa::Format::R32_UINT,
            .image = ti.get(AT.vsm_memory_block).ids[0],
            .name = "vsm memory daxa_u32 view",
        });

        auto render_cmd = std::move(ti.recorder).begin_renderpass({
            .render_area = daxa::Rect2D{.width = VSM_TEXTURE_RESOLUTION, .height = VSM_TEXTURE_RESOLUTION},
        });

        for (u32 opaque_draw_list_type = 0; opaque_draw_list_type < 2; ++opaque_draw_list_type)
        {
            render_cmd.set_pipeline(*render_context->gpuctx->raster_pipelines.at(cull_and_draw_pages_pipelines[opaque_draw_list_type].name));
            for (u32 i = 0; i < 32; ++i)
            {
                CullAndDrawPagesPush push = {
                    .draw_list_type = opaque_draw_list_type,
                    .bucket_index = i,
                    .daxa_u32_vsm_memory_view = memory_block_view,
                };
                ti.assign_attachment_shader_blob(push.attachments.value);
                render_cmd.push_constant(push);
                render_cmd.draw_mesh_tasks_indirect({
                    .indirect_buffer = ti.get(AT.meshlets_cull_arg_buckets).ids[0],
                    .offset = sizeof(DispatchIndirectStruct) * i + sizeof(CullMeshletsArgBuckets) * opaque_draw_list_type,
                    .draw_count = 1,
                    .stride = sizeof(DispatchIndirectStruct),
                });
            }
        }
        ti.recorder = std::move(render_cmd).end_renderpass();
        ti.recorder.destroy_image_view_deferred(memory_block_view);
    }
};

struct ClearDirtyBitTask : ClearDirtyBitH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_clear_dirty_bit_pipeline_compile_info().name));
        ClearDirtyBitH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch_indirect({
            .indirect_buffer = ti.get(AT.vsm_clear_dirty_bit_indirect).ids[0],
            .offset = 0u,
        });
    }
};

struct DebugVirtualPageTableTask : DebugVirtualPageTableH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    static constexpr auto dispatch_size = u32vec2{
        (VSM_PAGE_TABLE_RESOLUTION + DEBUG_PAGE_TABLE_X_DISPATCH - 1) / DEBUG_PAGE_TABLE_X_DISPATCH,
        (VSM_PAGE_TABLE_RESOLUTION + DEBUG_PAGE_TABLE_Y_DISPATCH - 1) / DEBUG_PAGE_TABLE_Y_DISPATCH,
    };

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_debug_virtual_page_table_pipeline_compile_info().name));
        DebugVirtualPageTableH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({dispatch_size.x, dispatch_size.y});
    }
};

struct DebugMetaMemoryTableTask : DebugMetaMemoryTableH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    static constexpr auto dispatch_size = u32vec2{
        (VSM_META_MEMORY_TABLE_RESOLUTION + DEBUG_META_MEMORY_TABLE_X_DISPATCH - 1) / DEBUG_META_MEMORY_TABLE_X_DISPATCH,
        (VSM_META_MEMORY_TABLE_RESOLUTION + DEBUG_META_MEMORY_TABLE_Y_DISPATCH - 1) / DEBUG_META_MEMORY_TABLE_Y_DISPATCH,
    };

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_debug_meta_memory_table_pipeline_compile_info().name));
        DebugMetaMemoryTableH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({dispatch_size.x, dispatch_size.y});
    }
};

struct TaskDrawVSMsInfo
{
    RenderContext * render_context = {};
    daxa::TaskGraph * tg = {};
    VSMState * vsm_state = {};
    daxa::TaskBufferView meshlets_cull_arg_buckets_buffers = {};
    daxa::TaskBufferView meshlet_instances = {};
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
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, info.vsm_state->meshlet_cull_arg_buckets_buffer_head),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, info.meshlets_cull_arg_buckets_buffers),
        },
        .task = [info](daxa::TaskInterface ti)
        {
            allocate_fill_copy(ti, info.vsm_state->clip_projections_cpu, ti.get(info.vsm_state->clip_projections));
            allocate_fill_copy(ti, info.vsm_state->free_wrapped_pages_info_cpu, ti.get(info.vsm_state->free_wrapped_pages_info));
            allocate_fill_copy(ti, info.vsm_state->globals_cpu, ti.get(info.vsm_state->globals));
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = ti.get(info.meshlets_cull_arg_buckets_buffers).ids[0],
                .dst_buffer = ti.get(info.vsm_state->meshlet_cull_arg_buckets_buffer_head).ids[0],
                .size = sizeof(MeshletCullArgBucketsBufferHead),
            });
        },
    });

    info.tg->add_task(CullAndDrawPages_WriteCommandTask{
        .views = std::array{
            daxa::attachment_view(CullAndDrawPages_WriteCommandH::AT.vsm_meshlets_cull_arg_buckets, info.vsm_state->meshlet_cull_arg_buckets_buffer_head),
        },
        .context = info.render_context->gpuctx,
        .dispatch_callback = []()
        { return daxa::DispatchInfo{DRAW_LIST_TYPES, 1, 1}; },
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
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_meta_memory_table, info.vsm_state->meta_memory_table),
        },
        .render_context = info.render_context,
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
    });

    info.tg->add_task(ClearPagesTask{
        .views = std::array{
            daxa::attachment_view(ClearPagesH::AT.vsm_allocation_requests, info.vsm_state->allocation_requests),
            daxa::attachment_view(ClearPagesH::AT.vsm_clear_indirect, info.vsm_state->clear_indirect),
            daxa::attachment_view(ClearPagesH::AT.vsm_page_table, vsm_page_table_view),
            daxa::attachment_view(ClearPagesH::AT.vsm_memory, info.vsm_state->memory_block),
        },
        .render_context = info.render_context,
    });

    info.tg->add_task(GenDirtyBitHizTask{
        .views = std::array{
            daxa::attachment_view(GenDirtyBitHizH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(GenDirtyBitHizH::AT.vsm_page_table, vsm_page_table_view),
            daxa::attachment_view(GenDirtyBitHizH::AT.vsm_dirty_bit_hiz, vsm_dirty_bit_hiz_view),
        },
        .render_context = info.render_context,
    });

    info.tg->add_task(CullAndDrawPagesTask{
        .views = std::array{
            daxa::attachment_view(CullAndDrawPagesH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(CullAndDrawPagesH::AT.meshlets_cull_arg_buckets, info.vsm_state->meshlet_cull_arg_buckets_buffer_head),
            daxa::attachment_view(CullAndDrawPagesH::AT.meshlet_instances, info.meshlet_instances),
            daxa::attachment_view(CullAndDrawPagesH::AT.meshes, info.meshes),
            daxa::attachment_view(CullAndDrawPagesH::AT.entity_combined_transforms, info.entity_combined_transforms),
            daxa::attachment_view(CullAndDrawPagesH::AT.material_manifest, info.material_manifest),
            daxa::attachment_view(CullAndDrawPagesH::AT.vsm_clip_projections, info.vsm_state->clip_projections),
            daxa::attachment_view(CullAndDrawPagesH::AT.vsm_dirty_bit_hiz, vsm_dirty_bit_hiz_view),
            daxa::attachment_view(CullAndDrawPagesH::AT.vsm_page_table, vsm_page_table_view),
            daxa::attachment_view(CullAndDrawPagesH::AT.vsm_memory_block, info.vsm_state->memory_block),
        },
        .render_context = info.render_context,
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
    });

    task_clear_image(*info.tg, info.render_context->gpuctx->shader_debug_context.vsm_debug_meta_memory_table, std::array{0.0f, 0.0f, 0.0f, 0.0f});
    info.tg->add_task(DebugMetaMemoryTableTask{
        .views = std::array{
            daxa::attachment_view(DebugMetaMemoryTableH::AT.vsm_page_table, vsm_page_table_view),
            daxa::attachment_view(DebugMetaMemoryTableH::AT.vsm_meta_memory_table, info.vsm_state->meta_memory_table),
            daxa::attachment_view(DebugMetaMemoryTableH::AT.vsm_debug_meta_memory_table, info.render_context->gpuctx->shader_debug_context.vsm_debug_meta_memory_table),
        },
        .render_context = info.render_context,
    });

    info.tg->add_task(ClearDirtyBitTask{
        .views = std::array{
            daxa::attachment_view(ClearDirtyBitH::AT.vsm_allocation_requests, info.vsm_state->allocation_requests),
            daxa::attachment_view(ClearDirtyBitH::AT.vsm_allocation_count, info.vsm_state->allocation_count),
            daxa::attachment_view(ClearDirtyBitH::AT.vsm_clear_dirty_bit_indirect, info.vsm_state->clear_dirty_bit_indirect),
            daxa::attachment_view(ClearDirtyBitH::AT.vsm_page_table, vsm_page_table_view),
        },
        .render_context = info.render_context,
    });
}

struct CameraController;
struct GetVSMProjectionsInfo
{
    CameraController const * camera_info = {};
    f32vec3 sun_direction = {};
    f32 clip_0_scale = {};
    f32 clip_0_near = {};
    f32 clip_0_far = {};
    f32 clip_0_height_offset = {};

    ShaderDebugDrawContext * debug_context = {};
};

inline auto get_vsm_projections(GetVSMProjectionsInfo const & info) -> std::array<VSMClipProjection, VSM_CLIP_LEVELS>
{
    std::array<VSMClipProjection, VSM_CLIP_LEVELS> clip_projections = {};
    auto const default_vsm_pos = glm::vec3{0.0, 0.0, 0.0};
    auto const default_vsm_up = glm::vec3{0.0, 0.0, 1.0};
    auto const default_vsm_forward = -info.sun_direction;
    auto const default_vsm_view = glm::lookAt(default_vsm_pos, default_vsm_forward, default_vsm_up);

    auto calculate_clip_projection = [&info](i32 clip) -> glm::mat4x4
    {
        auto const clip_scale = std::pow(2.0f, s_cast<f32>(clip));
        auto clip_projection = glm::ortho(
            -info.clip_0_scale * clip_scale, // left
            info.clip_0_scale * clip_scale,  // right
            -info.clip_0_scale * clip_scale, // bottom
            info.clip_0_scale * clip_scale,  // top
            info.clip_0_near * clip_scale,   // near
            info.clip_0_far * clip_scale     // far
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

    for (i32 clip = 0; clip < VSM_CLIP_LEVELS; clip++)
    {
        auto const curr_clip_proj = calculate_clip_projection(clip);
        auto const clip_projection_view = curr_clip_proj * default_vsm_view;

        // Project the target position into VSM ndc coordinates and calculate a page alligned position
        auto const clip_projected_target_pos = clip_projection_view * target_camera_position;
        auto const ndc_target_pos = glm::vec3(clip_projected_target_pos) / clip_projected_target_pos.w;
        auto const ndc_page_scaled_target_pos = glm::vec2(ndc_target_pos) / ndc_page_size;
        auto const ndc_page_scaled_aligned_target_pos = glm::vec2(glm::ceil(ndc_page_scaled_target_pos));
        // auto const ndc_page_scaled_aligned_target_pos = glm::vec2(glm::ivec2(ndc_page_scaled_target_pos));

        // Here we calculate the offsets that will be applied per page in the clip level
        // This is used to virtually offset the depth of each page so that we can actually snap the vsm position to the camera position
        auto const near_offset_ndc_u_in_world = glm::inverse(clip_projection_view) * glm::vec4(ndc_page_size, 0.0, 0.0, 1.0);
        auto const near_offset_ndc_v_in_world = glm::inverse(clip_projection_view) * glm::vec4(0.0, ndc_page_size, 0.0, 1.0);

        // Inverse projection from ndc -> world does not account for near plane offset, thus we need to add it manually
        // we simply shift the position in the oppposite of view direction by near plane distance
        auto const curr_clip_scale = std::pow(2.0f, s_cast<f32>(clip));
        auto const curr_clip_near = info.clip_0_near * curr_clip_scale;
        auto const ndc_u_in_world = glm::vec3(near_offset_ndc_u_in_world) + curr_clip_near * -default_vsm_forward;
        auto const ndc_v_in_world = glm::vec3(near_offset_ndc_v_in_world) + curr_clip_near * -default_vsm_forward;

        // Calculate the actual per page world space offsets
        f32 const u_offset_scale = ndc_u_in_world.z / default_vsm_forward.z;
        auto const u_offset_vector = u_offset_scale * -default_vsm_forward;

        f32 const v_offset_scale = ndc_v_in_world.z / default_vsm_forward.z;
        auto const v_offset_vector = v_offset_scale * -default_vsm_forward;

        // Get the per page offsets on a world space xy plane
        auto const xy_plane_ndc_u_in_world = ndc_u_in_world + u_offset_vector;
        auto const xy_plane_ndc_v_in_world = ndc_v_in_world + v_offset_vector;

        // Clip position on the xy world plane
        auto const clip_xy_plane_world_position = glm::vec3(
            ndc_page_scaled_aligned_target_pos.x * xy_plane_ndc_u_in_world +
            ndc_page_scaled_aligned_target_pos.y * xy_plane_ndc_v_in_world);

        // Clip offset from the xy plane - essentially clip_xy_plane_world_position gives us the position on a world xy plane positioned
        // at the height 0. We want to shift the clip camera up so that it observes the player position from the above. The height from
        // which the camera observes this player should be set according to the info.height_offset
        auto const view_offset_scale = s_cast<i32>(std::floor(info.camera_info->position.z / -default_vsm_forward.z) + info.clip_0_height_offset * curr_clip_scale);
        auto const view_offset = s_cast<f32>(view_offset_scale) * -default_vsm_forward;
        auto const clip_position = clip_xy_plane_world_position + view_offset;

        auto const final_clip_view = glm::lookAt(clip_position, clip_position + glm::normalize(default_vsm_forward), default_vsm_up);
        auto const final_clip_projection_view = curr_clip_proj * final_clip_view;

        auto const origin_shift = (final_clip_projection_view * glm::vec4(0.0, 0.0, 0.0, 1.0)).z;
        auto const page_u_depth_offset = (final_clip_projection_view * glm::vec4(u_offset_vector, 1.0)).z - origin_shift;
        auto const page_v_depth_offset = (final_clip_projection_view * glm::vec4(v_offset_vector, 1.0)).z - origin_shift;

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

        clip_projections.at(clip) = VSMClipProjection{
            .height_offset = view_offset_scale,
            .depth_page_offset = {page_u_depth_offset, page_v_depth_offset},
            .page_offset = {
                (-s_cast<daxa_i32>(ndc_page_scaled_aligned_target_pos.x)),
                (-s_cast<daxa_i32>(ndc_page_scaled_aligned_target_pos.y)),
            },
            .camera = clip_camera,
        };
    }
    return clip_projections;
}

struct DebugDrawClipFrustiInfo
{
    std::span<VSMClipProjection const> clip_projections;
    bool draw_individual_pages = {};
    ShaderDebugDrawContext * debug_context = {};
    f32vec3 vsm_view_direction = {};
};

inline void debug_draw_clip_fusti(DebugDrawClipFrustiInfo const & info)
{
    static constexpr std::array offsets = {
        glm::ivec2(-1, 1), glm::ivec2(-1, -1), glm::ivec2(1, -1), glm::ivec2(1, 1),
        glm::ivec2(-1, 1), glm::ivec2(-1, -1), glm::ivec2(1, -1), glm::ivec2(1, 1)};

    for (auto const & clip_projection : info.clip_projections)
    {
        auto const left_right_size = std::abs((1.0f / std::bit_cast<glm::mat4x4>(clip_projection.camera.proj)[0][0])) * 2.0f;
        auto const top_bottom_size = std::abs((1.0f / std::bit_cast<glm::mat4x4>(clip_projection.camera.proj)[1][1])) * 2.0f;
        auto const near_far_size = (1.0f / std::bit_cast<glm::mat4x4>(clip_projection.camera.proj)[2][2]) * 2.0f;
        auto const page_size = glm::vec2(left_right_size / VSM_PAGE_TABLE_RESOLUTION, top_bottom_size / VSM_PAGE_TABLE_RESOLUTION);

        auto const page_proj = glm::ortho(
            -page_size.x / 2.0f,
            page_size.x / 2.0f,
            -page_size.y / 2.0f,
            page_size.y / 2.0f,
            1.0f,
            100.0f);
        if (info.draw_individual_pages)
        {
            auto const uv_page_size = s_cast<f32>(VSM_PAGE_SIZE) / s_cast<f32>(VSM_TEXTURE_RESOLUTION);
            for (i32 page_u_index = 0; page_u_index < VSM_PAGE_TABLE_RESOLUTION; page_u_index++)
            {
                for (i32 page_v_index = 0; page_v_index < VSM_PAGE_TABLE_RESOLUTION; page_v_index++)
                {
                    auto const corner_virtual_uv = uv_page_size * glm::vec2(page_u_index, page_v_index);
                    auto const page_center_virtual_uv_offset = glm::vec2(uv_page_size * 0.5f);
                    auto const virtual_uv = corner_virtual_uv + page_center_virtual_uv_offset;

                    auto const page_index = glm::ivec2(virtual_uv * s_cast<f32>(VSM_PAGE_TABLE_RESOLUTION));
                    f32 const depth =
                        ((VSM_PAGE_TABLE_RESOLUTION - 1) - page_index.x) * clip_projection.depth_page_offset.x +
                        ((VSM_PAGE_TABLE_RESOLUTION - 1) - page_index.y) * clip_projection.depth_page_offset.y;
                    auto const virtual_page_ndc = (virtual_uv * 2.0f) - glm::vec2(1.0f);
                    auto const page_ndc_position = glm::vec4(virtual_page_ndc, -depth, 1.0);
                    auto const new_position = std::bit_cast<glm::mat4x4>(clip_projection.camera.inv_view_proj) * page_ndc_position;

                    auto const page_view = glm::lookAt(glm::vec3(new_position), glm::vec3(new_position) + info.vsm_view_direction, {0.0, 0.0, 1.0});
                    auto const page_inv_projection_view = glm::inverse(page_proj * page_view);

                    ShaderDebugBoxDraw box_draw = {};
                    box_draw.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
                    box_draw.color = daxa_f32vec3{0.0, 0.0, 1.0};
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
            box_draw.color = daxa_f32vec3{0.0, 0.0, 1.0};
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
#endif //__cplusplus