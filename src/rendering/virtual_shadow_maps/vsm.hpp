#pragma once

#include "vsm.inl"

#include "../tasks/misc.hpp"
#include "vsm_state.hpp"
#include <glm/gtx/vector_angle.hpp>
#include "../scene_renderer_context.hpp"
#include "../rasterize_visbuffer/cull_meshes.hpp"


MAKE_COMPUTE_COMPILE_INFO(vsm_invalidate_pages_pipeline_compile_info, "./src/rendering/virtual_shadow_maps/invalidate_pages.hlsl", "main")
MAKE_COMPUTE_COMPILE_INFO(vsm_free_wrapped_pages_pipeline_compile_info, "./src/rendering/virtual_shadow_maps/free_wrapped_pages.hlsl", "main")
MAKE_COMPUTE_COMPILE_INFO(vsm_force_always_resident_pages_pipeline_compile_info, "./src/rendering/virtual_shadow_maps/force_always_resident_pages.hlsl", "main")
MAKE_COMPUTE_COMPILE_INFO(vsm_mark_required_pages_pipeline_compile_info, "./src/rendering/virtual_shadow_maps/mark_required_pages.hlsl", "main")
MAKE_COMPUTE_COMPILE_INFO(vsm_find_free_pages_pipeline_compile_info, "./src/rendering/virtual_shadow_maps/find_free_pages.glsl", "main")
MAKE_COMPUTE_COMPILE_INFO(vsm_allocate_pages_pipeline_compile_info, "./src/rendering/virtual_shadow_maps/allocate_pages.hlsl", "main")
MAKE_COMPUTE_COMPILE_INFO(vsm_clear_pages_pipeline_compile_info, "./src/rendering/virtual_shadow_maps/clear_pages.hlsl", "main")
MAKE_COMPUTE_COMPILE_INFO(vsm_gen_dirty_bit_hiz_pipeline_compile_info, "./src/rendering/virtual_shadow_maps/gen_dirty_bit_hiz.hlsl", "main")
MAKE_COMPUTE_COMPILE_INFO(vsm_clear_dirty_bit_pipeline_compile_info, "./src/rendering/virtual_shadow_maps/clear_dirty_bit.glsl", "main")
MAKE_COMPUTE_COMPILE_INFO(vsm_debug_virtual_page_table_pipeline_compile_info, "./src/rendering/virtual_shadow_maps/draw_debug_textures.hlsl", "debug_virtual_main")
MAKE_COMPUTE_COMPILE_INFO(vsm_debug_meta_memory_table_pipeline_compile_info, "./src/rendering/virtual_shadow_maps/draw_debug_textures.hlsl", "debug_meta_main")
MAKE_COMPUTE_COMPILE_INFO(vsm_get_debug_statistics_pipeline_compile_info, "./src/rendering/virtual_shadow_maps/get_debug_statistics.hlsl", "main")
MAKE_COMPUTE_COMPILE_INFO(vsm_gen_point_dirty_bit_hiz_pipeline_compile_info, "./src/rendering/virtual_shadow_maps/gen_point_dirty_bit_hiz.hlsl", "main")

static constexpr inline char const CULL_AND_DRAW_DIRECTIONAL_PAGES_SHADER_PATH[] = "./src/rendering/virtual_shadow_maps/cull_and_draw_directional_pages.hlsl";

inline daxa::RasterPipelineCompileInfo vsm_cull_and_draw_directional_pages_base_pipeline_compile_info()
{
    return {
        .mesh_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{CULL_AND_DRAW_DIRECTIONAL_PAGES_SHADER_PATH},
            .compile_options = {.language = daxa::ShaderLanguage::SLANG},
        },
        .task_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{CULL_AND_DRAW_DIRECTIONAL_PAGES_SHADER_PATH},
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

inline daxa::RasterPipelineCompileInfo vsm_cull_and_draw_directional_pages_opaque_pipeline_compile_info()
{
    auto ret = vsm_cull_and_draw_directional_pages_base_pipeline_compile_info();
    ret.mesh_shader_info.value().compile_options.entry_point = "directional_vsm_entry_mesh_opaque";
    ret.task_shader_info.value().compile_options.entry_point = "directional_vsm_entry_task";
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{CULL_AND_DRAW_DIRECTIONAL_PAGES_SHADER_PATH},
        .compile_options = {
            .entry_point = "directional_vsm_entry_fragment_opaque",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "VsmCullAndDrawDirectionalPagesOpaque";
    return ret;
}

inline daxa::RasterPipelineCompileInfo vsm_cull_and_draw_directional_pages_masked_pipeline_compile_info()
{
    auto ret = vsm_cull_and_draw_directional_pages_base_pipeline_compile_info();
    ret.mesh_shader_info.value().compile_options.entry_point = "directional_vsm_entry_mesh_masked";
    ret.task_shader_info.value().compile_options.entry_point = "directional_vsm_entry_task";
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{CULL_AND_DRAW_DIRECTIONAL_PAGES_SHADER_PATH},
        .compile_options = {
            .entry_point = "directional_vsm_entry_fragment_masked",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "VsmCullAndDrawDirectionalPagesMasked";
    return ret;
}

inline std::array<daxa::RasterPipelineCompileInfo, 2> cull_and_draw_directional_pages_pipelines = {
    vsm_cull_and_draw_directional_pages_opaque_pipeline_compile_info(),
    vsm_cull_and_draw_directional_pages_masked_pipeline_compile_info()};

static constexpr inline char const CULL_AND_DRAW_POINT_PAGES_SHADER_PATH[] = "./src/rendering/virtual_shadow_maps/cull_and_draw_point_pages.hlsl";

inline daxa::RasterPipelineCompileInfo vsm_cull_and_draw_point_pages_base_pipeline_compile_info()
{
    return {
        .mesh_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{CULL_AND_DRAW_POINT_PAGES_SHADER_PATH},
            .compile_options = {.language = daxa::ShaderLanguage::SLANG},
        },
        .task_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{CULL_AND_DRAW_POINT_PAGES_SHADER_PATH},
            .compile_options = {.language = daxa::ShaderLanguage::SLANG},
        },
        .raster = {
            // .depth_clamp_enable = true,
            // .depth_bias_enable = true,
            // .depth_bias_constant_factor = 10.0f,
            // .depth_bias_slope_factor = 2.0f,
        },
        .push_constant_size = s_cast<u32>(sizeof(CullAndDrawPointPagesPush)),
    };
}

inline daxa::RasterPipelineCompileInfo vsm_cull_and_draw_point_pages_opaque_pipeline_compile_info()
{
    auto ret = vsm_cull_and_draw_point_pages_base_pipeline_compile_info();
    ret.mesh_shader_info.value().compile_options.entry_point = "point_vsm_entry_mesh_opaque";
    ret.task_shader_info.value().compile_options.entry_point = "point_vsm_entry_task";
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{CULL_AND_DRAW_POINT_PAGES_SHADER_PATH},
        .compile_options = {
            .entry_point = "point_vsm_entry_fragment_opaque",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "VsmCullAndDrawPointPagesOpaque";
    return ret;
}

inline daxa::RasterPipelineCompileInfo vsm_cull_and_draw_point_pages_masked_pipeline_compile_info()
{
    auto ret = vsm_cull_and_draw_point_pages_base_pipeline_compile_info();
    ret.mesh_shader_info.value().compile_options.entry_point = "point_vsm_entry_mesh_masked";
    ret.task_shader_info.value().compile_options.entry_point = "point_vsm_entry_task";
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{CULL_AND_DRAW_POINT_PAGES_SHADER_PATH},
        .compile_options = {
            .entry_point = "point_vsm_entry_fragment_masked",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "VsmCullAndDrawPointPagesMasked";
    return ret;
}

inline std::array<daxa::RasterPipelineCompileInfo, 2> cull_and_draw_point_pages_pipelines = {
    vsm_cull_and_draw_point_pages_opaque_pipeline_compile_info(),
    vsm_cull_and_draw_point_pages_masked_pipeline_compile_info()};
struct InvalidatePagesTask : InvalidatePagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(vsm_invalidate_pages_pipeline_compile_info().name));
        InvalidatePagesH::AttachmentShaderBlob push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        u32 const x_dispatch = round_up_div(render_context->mesh_instance_counts.vsm_invalidate_instance_count, INVALIDATE_PAGES_X_DISPATCH);
        u32 const y_dispatch = (VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION * VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION) / (VSM_INVALIDATE_PAGE_BLOCK_RESOLUTION * VSM_INVALIDATE_PAGE_BLOCK_RESOLUTION);
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "INVALIDATE_PAGES">());
        ti.recorder.dispatch({x_dispatch, y_dispatch, VSM_CLIP_LEVELS});
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "INVALIDATE_PAGES">());
    }
};

struct GetDebugStatisticsTask : GetDebugStatisticsH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    static constexpr auto dispatch_size = u32vec2{
        (VSM_META_MEMORY_TABLE_RESOLUTION + GET_DEBUG_STATISTICS_X_DISPATCH - 1) / GET_DEBUG_STATISTICS_X_DISPATCH,
        (VSM_META_MEMORY_TABLE_RESOLUTION + GET_DEBUG_STATISTICS_Y_DISPATCH - 1) / GET_DEBUG_STATISTICS_Y_DISPATCH,
    };

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(vsm_get_debug_statistics_pipeline_compile_info().name));
        GetDebugStatisticsH::AttachmentShaderBlob push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({dispatch_size.x, dispatch_size.y});
    }
};

struct FreeWrappedPagesTask : FreeWrappedPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(vsm_free_wrapped_pages_pipeline_compile_info().name));
        FreeWrappedPagesH::AttachmentShaderBlob push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "FREE_WRAPPED_PAGES">());
        ti.recorder.dispatch({1, VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION, VSM_CLIP_LEVELS});
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "FREE_WRAPPED_PAGES">());
    }
};

struct ForceAlwaysResidentPagesTask : ForceAlwaysResidentPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(vsm_force_always_resident_pages_pipeline_compile_info().name));

        ForceAlwaysResidentPagesH::AttachmentShaderBlob push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        u32 const array_layer_count =
            (render_context->render_data.vsm_settings.point_light_count * 6) + // 6 cubemap faces
            render_context->render_data.vsm_settings.spot_light_count;

        ti.recorder.dispatch({round_up_div(array_layer_count, FORCE_ALWAYS_PRESENT_PAGES_X_DISPATCH)});
    }
};

struct MarkRequiredPagesTask : MarkRequiredPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const depth_resolution = ti.info(AT.g_buffer_depth).value().size;
        auto const dispatch_size = u32vec2{
            (depth_resolution.x + MARK_REQUIRED_PAGES_X_DISPATCH - 1) / MARK_REQUIRED_PAGES_X_DISPATCH,
            (depth_resolution.y + MARK_REQUIRED_PAGES_Y_DISPATCH - 1) / MARK_REQUIRED_PAGES_Y_DISPATCH,
        };
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(vsm_mark_required_pages_pipeline_compile_info().name));

        auto alloc = ti.allocator->allocate(sizeof(MarkRequiredPagesH::AttachmentShaderBlob));
        std::memcpy(alloc->host_address, ti.attachment_shader_blob.data(), sizeof(MarkRequiredPagesH::AttachmentShaderBlob));
        MarkRequiredPagesPush push = {.attachments = alloc->device_address};
        ti.recorder.push_constant(push);

        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "MARK_REQUIRED_PAGES">());
        ti.recorder.dispatch({.x = dispatch_size.x, .y = dispatch_size.y});
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "MARK_REQUIRED_PAGES">());
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
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(vsm_find_free_pages_pipeline_compile_info().name));
        FindFreePagesH::AttachmentShaderBlob push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "FIND_FREE_PAGES">());
        ti.recorder.dispatch({dispatch_x_size, 1, 1});
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "FIND_FREE_PAGES">());
    }
};

struct AllocatePagesTask : AllocatePagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(vsm_allocate_pages_pipeline_compile_info().name));

        auto alloc = ti.allocator->allocate(sizeof(AllocatePagesH::AttachmentShaderBlob));
        std::memcpy(alloc->host_address, ti.attachment_shader_blob.data(), sizeof(AllocatePagesH::AttachmentShaderBlob));
        AllocatePagesPush push = {.attachments = alloc->device_address};
        ti.recorder.push_constant(push);

        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "ALLOCATE_PAGES">());
        ti.recorder.dispatch_indirect({
            .indirect_buffer = ti.id(AT.vsm_allocate_indirect),
            .offset = 0u,
        });
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "ALLOCATE_PAGES">());
    }
};

struct ClearPagesTask : ClearPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(vsm_clear_pages_pipeline_compile_info().name));
        ClearPagesH::AttachmentShaderBlob push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "CLEAR_PAGES">());
        ti.recorder.dispatch_indirect({
            .indirect_buffer = ti.id(AT.vsm_clear_indirect),
            .offset = 0u,
        });
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "CLEAR_PAGES">());
    }
};

struct GenDirtyBitHizTask : GenDirtyBitHizH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "GEN_DIRY_BIT_HIZ_DIRECTIONAL">());
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(vsm_gen_dirty_bit_hiz_pipeline_compile_info().name));
        auto const dispatch_x = round_up_div(VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION, GEN_DIRTY_BIT_HIZ_X_WINDOW);
        auto const dispatch_y = round_up_div(VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION, GEN_DIRTY_BIT_HIZ_Y_WINDOW);
        GenDirtyBitHizPush push = {
            .mip_count = ti.get(AT.vsm_dirty_bit_hiz).view.slice.level_count,
        };
        push.attachments = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({dispatch_x, dispatch_y, VSM_CLIP_LEVELS});
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "GEN_DIRY_BIT_HIZ_DIRECTIONAL">());
    }
};

struct GenPointDirtyBitHizTask : GenPointDirtyBitHizH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "GEN_DIRY_BIT_HIZ_POINT_SPOT">());
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(vsm_gen_point_dirty_bit_hiz_pipeline_compile_info().name));
        auto const dispatch_x = round_up_div(VSM_POINT_SPOT_PAGE_TABLE_RESOLUTION, GEN_DIRTY_BIT_HIZ_X_WINDOW);
        auto const dispatch_y = round_up_div(VSM_POINT_SPOT_PAGE_TABLE_RESOLUTION, GEN_DIRTY_BIT_HIZ_Y_WINDOW);

        auto attachment_alloc = ti.allocator->allocate(sizeof(GenPointDirtyBitHizH::AttachmentShaderBlob)).value();
        *reinterpret_cast<GenPointDirtyBitHizH::AttachmentShaderBlob *>(attachment_alloc.host_address) = ti.attachment_shader_blob;
        GenPointDirtyBitHizPush push = {.attachments = attachment_alloc.device_address};
        ti.recorder.push_constant(push);
        auto const dispatch_z =
            (render_context->render_data.vsm_settings.point_light_count * 7 * 6) + // MAX_POINT_LIGHTS * MIP_LEVELS * CUBE_FACES
            (render_context->render_data.vsm_settings.spot_light_count * 7);       // MAX_SPOT_LIGHTS  * MIP_LEVELS
        ti.recorder.dispatch({dispatch_x, dispatch_y, dispatch_z});
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "GEN_DIRY_BIT_HIZ_POINT_SPOT">());
    }
};

struct CullAndDrawPagesTask : CullAndDrawPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const memory_block_view = render_context->gpu_context->device.create_image_view({
            .type = daxa::ImageViewType::REGULAR_2D,
            .format = daxa::Format::R32_UINT,
            .image = ti.id(AT.vsm_memory_block),
            .name = "vsm memory daxa integer view",
        });

        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "CULL_AND_DRAW_PAGES_DIRECTIONAL">());
        auto render_cmd = std::move(ti.recorder).begin_renderpass({
            .render_area = daxa::Rect2D{.width = VSM_DIRECTIONAL_TEXTURE_RESOLUTION, .height = VSM_DIRECTIONAL_TEXTURE_RESOLUTION},
        });

        render_cmd.set_depth_bias({
            .constant_factor = render_context->render_data.vsm_settings.constant_bias,
            .clamp = 0.0,
            .slope_factor = render_context->render_data.vsm_settings.slope_bias,
        });
        auto attachment_alloc = ti.allocator->allocate(sizeof(CullAndDrawPagesH::AttachmentShaderBlob)).value();
        *reinterpret_cast<CullAndDrawPagesH::AttachmentShaderBlob *>(attachment_alloc.host_address) = ti.attachment_shader_blob;
        for (u32 opaque_draw_list_type = 0; opaque_draw_list_type < 2; ++opaque_draw_list_type)
        {
            auto buffer = opaque_draw_list_type == PREPASS_DRAW_LIST_OPAQUE ? ti.id(AT.po2expansion) : ti.id(AT.masked_po2expansion);
            render_cmd.set_pipeline(*render_context->gpu_context->raster_pipelines.at(cull_and_draw_directional_pages_pipelines[opaque_draw_list_type].name));
            CullAndDrawPagesPush push = {
                .attachments = attachment_alloc.device_address,
                .draw_list_type = opaque_draw_list_type,
                .daxa_uint_vsm_memory_view = static_cast<daxa_ImageViewId>(memory_block_view),
            };
            render_cmd.push_constant(push);
            render_cmd.draw_mesh_tasks_indirect({
                .indirect_buffer = buffer,
                .offset = 0,
                .draw_count = 1,
                .stride = sizeof(DispatchIndirectStruct),
            });
        }

        ti.recorder = std::move(render_cmd).end_renderpass();
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "CULL_AND_DRAW_PAGES_DIRECTIONAL">());
        ti.recorder.destroy_image_view_deferred(memory_block_view);
    }
};

struct CullAndDrawPointPagesTask : CullAndDrawPointPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "CULL_AND_DRAW_PAGES_POINT_SPOT">());
        auto const memory_block_view = render_context->gpu_context->device.create_image_view({
            .type = daxa::ImageViewType::REGULAR_2D,
            .format = daxa::Format::R32_UINT,
            .image = ti.get(AT.vsm_memory_block).ids[0],
            .name = "vsm memory daxa integer view",
        });

        for (i32 mip = 0; mip < 7; ++mip)
        {
            u32 const render_resolution = VSM_POINT_SPOT_TEXTURE_RESOLUTION / (1 << mip);
            auto render_cmd = std::move(ti.recorder).begin_renderpass({
                .render_area = daxa::Rect2D{.width = render_resolution, .height = render_resolution},
            });

            render_cmd.set_depth_bias({
                .constant_factor = render_context->render_data.vsm_settings.constant_bias,
                .clamp = 0.0,
                .slope_factor = render_context->render_data.vsm_settings.slope_bias,
            });
            auto attachment_alloc = ti.allocator->allocate(sizeof(CullAndDrawPointPagesH::AttachmentShaderBlob)).value();
            *reinterpret_cast<CullAndDrawPointPagesH::AttachmentShaderBlob *>(attachment_alloc.host_address) = ti.attachment_shader_blob;

            daxa::BufferId po2expansion;
            daxa::BufferId masked_po2expansion;
            daxa::ImageViewId hpb;
            switch (mip)
            {
                case 0:
                    po2expansion = ti.get(AT.po2expansion_mip0).ids[0];
                    masked_po2expansion = ti.get(AT.masked_po2expansion_mip0).ids[0];
                    hpb = ti.get(AT.vsm_dirty_bit_hiz_mip0).view_ids[0];
                    break;
                case 1:
                    po2expansion = ti.get(AT.po2expansion_mip1).ids[0];
                    masked_po2expansion = ti.get(AT.masked_po2expansion_mip1).ids[0];
                    hpb = ti.get(AT.vsm_dirty_bit_hiz_mip1).view_ids[0];
                    break;
                case 2:
                    po2expansion = ti.get(AT.po2expansion_mip2).ids[0];
                    masked_po2expansion = ti.get(AT.masked_po2expansion_mip2).ids[0];
                    hpb = ti.get(AT.vsm_dirty_bit_hiz_mip2).view_ids[0];
                    break;
                case 3:
                    po2expansion = ti.get(AT.po2expansion_mip3).ids[0];
                    masked_po2expansion = ti.get(AT.masked_po2expansion_mip3).ids[0];
                    hpb = ti.get(AT.vsm_dirty_bit_hiz_mip3).view_ids[0];
                    break;
                case 4:
                    po2expansion = ti.get(AT.po2expansion_mip4).ids[0];
                    masked_po2expansion = ti.get(AT.masked_po2expansion_mip4).ids[0];
                    hpb = ti.get(AT.vsm_dirty_bit_hiz_mip4).view_ids[0];
                    break;
                case 5:
                    po2expansion = ti.get(AT.po2expansion_mip5).ids[0];
                    masked_po2expansion = ti.get(AT.masked_po2expansion_mip5).ids[0];
                    hpb = ti.get(AT.vsm_dirty_bit_hiz_mip5).view_ids[0];
                    break;
                case 6:
                    po2expansion = ti.get(AT.po2expansion_mip6).ids[0];
                    masked_po2expansion = ti.get(AT.masked_po2expansion_mip6).ids[0];
                    hpb = ti.get(AT.vsm_dirty_bit_hiz_mip6).view_ids[0];
                    break;
            }

            for (u32 opaque_draw_list_type = 0; opaque_draw_list_type < 2; ++opaque_draw_list_type)
            {
                auto buffer = opaque_draw_list_type == PREPASS_DRAW_LIST_OPAQUE ? po2expansion : masked_po2expansion;
                render_cmd.set_pipeline(*render_context->gpu_context->raster_pipelines.at(cull_and_draw_point_pages_pipelines[opaque_draw_list_type].name));
                CullAndDrawPointPagesPush push = {
                    .attachments = attachment_alloc.device_address,
                    .draw_list_type = opaque_draw_list_type,
                    .mip_level = s_cast<u32>(mip),
                    .daxa_uint_vsm_memory_view = static_cast<daxa_ImageViewId>(memory_block_view),
                    .hpb_view = hpb};
                render_cmd.push_constant(push);
                render_cmd.draw_mesh_tasks_indirect({
                    .indirect_buffer = buffer,
                    .offset = 0,
                    .draw_count = 1,
                    .stride = sizeof(DispatchIndirectStruct),
                });
            }

            ti.recorder = std::move(render_cmd).end_renderpass();
        }
        ti.recorder.destroy_image_view_deferred(memory_block_view);
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "CULL_AND_DRAW_PAGES_POINT_SPOT">());
    }
};

struct ClearDirtyBitTask : ClearDirtyBitH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(vsm_clear_dirty_bit_pipeline_compile_info().name));
        ClearDirtyBitH::AttachmentShaderBlob push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "CLEAR_DIRY_BITS">());
        ti.recorder.dispatch_indirect({
            .indirect_buffer = ti.id(AT.vsm_clear_dirty_bit_indirect),
            .offset = 0u,
        });
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"VSM", "CLEAR_DIRY_BITS">());
    }
};

struct DebugVirtualPageTableTask : DebugVirtualPageTableH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    static constexpr auto dispatch_size = u32vec2{
        (VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION + DEBUG_PAGE_TABLE_X_DISPATCH - 1) / DEBUG_PAGE_TABLE_X_DISPATCH,
        (VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION + DEBUG_PAGE_TABLE_Y_DISPATCH - 1) / DEBUG_PAGE_TABLE_Y_DISPATCH,
    };

    void callback(daxa::TaskInterface ti)
    {

        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(vsm_debug_virtual_page_table_pipeline_compile_info().name));
        DebugVirtualPageTableH::AttachmentShaderBlob push = ti.attachment_shader_blob;
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
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(vsm_debug_meta_memory_table_pipeline_compile_info().name));
        DebugMetaMemoryTableH::AttachmentShaderBlob push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({dispatch_size.x, dispatch_size.y});
    }
};

struct TaskDrawVSMsInfo
{
    Scene * scene = {};
    RenderContext * render_context = {};
    daxa::TaskGraph * tg = {};
    VSMState * vsm_state = {};
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskBufferView mesh_instances = {};
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView entity_combined_transforms = {};
    daxa::TaskBufferView material_manifest = {};
    daxa::TaskImageView g_buffer_depth = {};
    daxa::TaskImageView g_buffer_geo_normal = {};
    daxa::TaskImageView light_mask_volume = {};
};

inline void task_draw_vsms(TaskDrawVSMsInfo const & info)
{
    auto const vsm_page_table_view = info.vsm_state->page_table.view().layers(0, VSM_CLIP_LEVELS);
    auto const vsm_page_view_pos_row_view = info.vsm_state->page_view_pos_row.view().layers(0, VSM_CLIP_LEVELS);
    auto const vsm_dirty_bit_hiz_view = info.vsm_state->dirty_pages_hiz.mips(0, s_cast<u32>(std::log2(VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION)) + 1).layers(0, VSM_CLIP_LEVELS);

    info.tg->add_task(daxa::InlineTask::Transfer("vsm setup task")
            .writes(info.vsm_state->clip_projections)
            .writes(info.vsm_state->free_wrapped_pages_info)
            .writes(info.vsm_state->globals)
            .writes(info.vsm_state->vsm_point_lights)
            .writes(info.vsm_state->vsm_spot_lights)
            .executes(
                [info](daxa::TaskInterface ti)
                {
                    allocate_fill_copy(ti, info.vsm_state->clip_projections_cpu, ti.get(info.vsm_state->clip_projections));
                    allocate_fill_copy(ti, info.vsm_state->free_wrapped_pages_info_cpu, ti.get(info.vsm_state->free_wrapped_pages_info));
                    allocate_fill_copy(ti, info.vsm_state->globals_cpu, ti.get(info.vsm_state->globals));
                    allocate_fill_copy(ti, info.vsm_state->point_lights_cpu, ti.get(info.vsm_state->vsm_point_lights));
                    allocate_fill_copy(ti, info.vsm_state->spot_lights_cpu, ti.get(info.vsm_state->vsm_spot_lights));
                }));

    info.tg->add_task(InvalidatePagesTask{
        .views = InvalidatePagesTask::Views{
            .globals = info.render_context->tgpu_render_data.view(),
            .mesh_instances = info.mesh_instances,
            .meshes = info.meshes,
            .entity_combined_transforms = info.entity_combined_transforms,
            .vsm_clip_projections = info.vsm_state->clip_projections,
            .free_wrapped_pages_info = info.vsm_state->free_wrapped_pages_info,
            .vsm_page_table = vsm_page_table_view,
            .vsm_meta_memory_table = info.vsm_state->meta_memory_table.view(),
        },
        .render_context = info.render_context,
    });

    info.tg->add_task(FreeWrappedPagesTask{
        .views = FreeWrappedPagesTask::Views{
            .globals = info.render_context->tgpu_render_data.view(),
            .free_wrapped_pages_info = info.vsm_state->free_wrapped_pages_info,
            .vsm_clip_projections = info.vsm_state->clip_projections,
            .vsm_page_table = vsm_page_table_view,
            .vsm_meta_memory_table = info.vsm_state->meta_memory_table.view(),
        },
        .render_context = info.render_context,
    });

    auto const vsm_point_spot_page_table_view =
        info.vsm_state->point_spot_page_tables.view()
            .mips(0, s_cast<u32>(std::log2(VSM_POINT_SPOT_PAGE_TABLE_RESOLUTION)) + 1)
            .layers(0, (6 * MAX_POINT_LIGHTS) + MAX_SPOT_LIGHTS);

    auto const vsm_last_mip_point_spot_page_table_view =
        info.vsm_state->point_spot_page_tables.view()
            .mips(VSM_FORCED_MIP_LEVEL, 1)
            .layers(0, (6 * MAX_POINT_LIGHTS) + MAX_SPOT_LIGHTS);

    info.tg->add_task(ForceAlwaysResidentPagesTask{
        .views = ForceAlwaysResidentPagesTask::Views{
            .globals = info.render_context->tgpu_render_data.view(),
            .vsm_allocation_requests = info.vsm_state->allocation_requests,
            .vsm_point_spot_page_table = vsm_last_mip_point_spot_page_table_view,
            .vsm_meta_memory_table = info.vsm_state->meta_memory_table.view(),
        },
        .render_context = info.render_context,
    });

    info.tg->add_task(MarkRequiredPagesTask{
        .views = MarkRequiredPagesTask::Views{
            .globals = info.render_context->tgpu_render_data.view(),
            .vsm_allocation_requests = info.vsm_state->allocation_requests,
            .vsm_globals = info.vsm_state->globals.view(),
            .vsm_clip_projections = info.vsm_state->clip_projections,
            .vsm_point_lights = info.vsm_state->vsm_point_lights,
            .vsm_spot_lights = info.vsm_state->vsm_spot_lights,
            .g_buffer_depth = info.g_buffer_depth,
            .g_buffer_geo_normal = info.g_buffer_geo_normal,
            .vsm_page_view_pos_row = vsm_page_view_pos_row_view,
            .vsm_page_table = vsm_page_table_view,
            .vsm_meta_memory_table = info.vsm_state->meta_memory_table.view(),
            .vsm_point_spot_page_table = vsm_point_spot_page_table_view,
            .light_mask_volume = info.light_mask_volume,
        },
        .render_context = info.render_context,
    });

    info.tg->add_task(FindFreePagesTask{
        .views = FindFreePagesTask::Views{
            .vsm_find_free_pages_header = info.vsm_state->find_free_pages_header,
            .vsm_free_pages_buffer = info.vsm_state->free_page_buffer,
            .vsm_not_visited_pages_buffer = info.vsm_state->not_visited_page_buffer,
            .vsm_allocate_indirect = info.vsm_state->allocate_indirect,
            .vsm_clear_indirect = info.vsm_state->clear_indirect,
            .vsm_clear_dirty_bit_indirect = info.vsm_state->clear_dirty_bit_indirect,
            .vsm_globals = info.vsm_state->globals.view(),
            .vsm_allocation_requests = info.vsm_state->allocation_requests,
            .vsm_meta_memory_table = info.vsm_state->meta_memory_table.view(),
        },
        .render_context = info.render_context,
    });

    info.tg->add_task(GetDebugStatisticsTask{
        .views = GetDebugStatisticsTask::Views{
            .globals = info.render_context->tgpu_render_data.view(),
            .vsm_allocation_requests = info.vsm_state->allocation_requests,
            .vsm_find_free_pages_header = info.vsm_state->find_free_pages_header,
            .vsm_meta_memory_table = info.vsm_state->meta_memory_table.view(),
        },
        .render_context = info.render_context,
    });

    info.tg->add_task(AllocatePagesTask{
        .views = AllocatePagesTask::Views{
            .vsm_globals = info.vsm_state->globals.view(),
            .vsm_find_free_pages_header = info.vsm_state->find_free_pages_header,
            .vsm_allocation_requests = info.vsm_state->allocation_requests,
            .vsm_free_pages_buffer = info.vsm_state->free_page_buffer,
            .vsm_not_visited_pages_buffer = info.vsm_state->not_visited_page_buffer,
            .vsm_allocate_indirect = info.vsm_state->allocate_indirect,
            .vsm_clip_projections = info.vsm_state->clip_projections,
            .vsm_page_table = vsm_page_table_view,
            .vsm_page_view_pos_row = vsm_page_view_pos_row_view,
            .vsm_meta_memory_table = info.vsm_state->meta_memory_table.view(),
            .vsm_point_spot_page_table = vsm_point_spot_page_table_view,
        },
        .render_context = info.render_context,
    });

    if (!info.vsm_state->overdraw_debug_image.is_null())
    {
        info.tg->clear_image({info.vsm_state->overdraw_debug_image, std::array{0u, 0u, 0u, 0u}});
    }
    info.tg->add_task(ClearPagesTask{
        .views = ClearPagesTask::Views{
            .vsm_allocation_requests = info.vsm_state->allocation_requests,
            .vsm_clear_indirect = info.vsm_state->clear_indirect,
            .vsm_page_table = vsm_page_table_view,
            .vsm_memory_block = info.vsm_state->memory_block.view(),
            .vsm_point_spot_page_table = vsm_point_spot_page_table_view,
        },
        .render_context = info.render_context,
    });

    info.tg->add_task(GenDirtyBitHizTask{
        .views = GenDirtyBitHizTask::Views{
            .globals = info.render_context->tgpu_render_data.view(),
            .vsm_clip_projections = info.vsm_state->clip_projections,
            .vsm_page_table = vsm_page_table_view,
            .vsm_dirty_bit_hiz = vsm_dirty_bit_hiz_view,
        },
        .render_context = info.render_context,
    });

    std::array<daxa::TaskImageView, 7> hpb_mip_views;
    for (i32 mip = 0; mip < 7; ++mip)
    {
        hpb_mip_views.at(mip) =
            info.vsm_state->point_dirty_pages_hiz_mips.at(mip)
                .mips(0, s_cast<u32>(7 - mip))
                .layers(0, (6 * MAX_POINT_LIGHTS) + MAX_SPOT_LIGHTS);
    }

    info.tg->add_task(GenPointDirtyBitHizTask{
        .views = GenPointDirtyBitHizTask::Views{
            .globals = info.render_context->tgpu_render_data.view(),
            .vsm_point_spot_page_table = vsm_point_spot_page_table_view,
            .vsm_dirty_bit_hiz_mip0 = hpb_mip_views.at(0),
            .vsm_dirty_bit_hiz_mip1 = hpb_mip_views.at(1),
            .vsm_dirty_bit_hiz_mip2 = hpb_mip_views.at(2),
            .vsm_dirty_bit_hiz_mip3 = hpb_mip_views.at(3),
            .vsm_dirty_bit_hiz_mip4 = hpb_mip_views.at(4),
            .vsm_dirty_bit_hiz_mip5 = hpb_mip_views.at(5),
            .vsm_dirty_bit_hiz_mip6 = hpb_mip_views.at(6),
        },
        .render_context = info.render_context});

    std::array<daxa::TaskBufferView, 2> directional_cascade_meshlet_expansions = {};
    tasks_expand_meshes_to_meshlets(TaskExpandMeshesToMeshletsInfo{
        .render_context = info.render_context,
        .tg = *info.tg,
        .cull_meshes = true,
        .vsm_hip = info.vsm_state->dirty_pages_hiz,
        .is_directional_light = true,
        .vsm_clip_projections = info.vsm_state->clip_projections,
        .globals = info.render_context->tgpu_render_data,
        .mesh_instances = info.mesh_instances,
        .meshlet_expansions = directional_cascade_meshlet_expansions,
        .dispatch_clear = {0, 1, 1},
        .buffer_name_prefix = std::string("vsm directional"),
    });

    info.tg->add_task(CullAndDrawPagesTask{
        .views = CullAndDrawPagesTask::Views{
            .globals = info.render_context->tgpu_render_data.view(),
            .po2expansion = directional_cascade_meshlet_expansions[0],
            .masked_po2expansion = directional_cascade_meshlet_expansions[1],
            .meshlet_instances = info.meshlet_instances,
            .mesh_instances = info.mesh_instances,
            .meshes = info.meshes,
            .entity_combined_transforms = info.entity_combined_transforms,
            .material_manifest = info.material_manifest,
            .vsm_clip_projections = info.vsm_state->clip_projections,
            .vsm_dirty_bit_hiz = vsm_dirty_bit_hiz_view,
            .vsm_page_table = vsm_page_table_view,
            .vsm_memory_block = info.vsm_state->memory_block.view(),
            .vsm_overdraw_debug = info.vsm_state->overdraw_debug_image,
        },
        .render_context = info.render_context,
    });

    std::array<std::array<daxa::TaskBufferView, 2>, 7> point_meshlet_mip_expansion = {};
    for (i32 mip = 0; mip < 7; ++mip)
    {
        tasks_expand_meshes_to_meshlets(TaskExpandMeshesToMeshletsInfo{
            .render_context = info.render_context,
            .tg = *info.tg,
            .cull_meshes = true,
            .vsm_hip = daxa::NullTaskImage,
            .vsm_point_hip = hpb_mip_views.at(mip),
            .is_point_spot_light = true,
            .mip_level = mip,
            .vsm_point_lights = info.vsm_state->vsm_point_lights,
            .vsm_spot_lights = info.vsm_state->vsm_spot_lights,
            .globals = info.render_context->tgpu_render_data,
            .mesh_instances = info.mesh_instances,
            .meshlet_expansions = point_meshlet_mip_expansion[mip],
            .buffer_name_prefix = std::string("vsm point/spot light ") + std::to_string(mip) + ' ',
        });
    }

    info.tg->add_task(CullAndDrawPointPagesTask{
        .views = CullAndDrawPointPagesTask::Views{
            .globals = info.render_context->tgpu_render_data.view(),
            .po2expansion_mip0 = point_meshlet_mip_expansion[0][0],
            .masked_po2expansion_mip0 = point_meshlet_mip_expansion[0][1],
            .po2expansion_mip1 = point_meshlet_mip_expansion[1][0],
            .masked_po2expansion_mip1 = point_meshlet_mip_expansion[1][1],
            .po2expansion_mip2 = point_meshlet_mip_expansion[2][0],
            .masked_po2expansion_mip2 = point_meshlet_mip_expansion[2][1],
            .po2expansion_mip3 = point_meshlet_mip_expansion[3][0],
            .masked_po2expansion_mip3 = point_meshlet_mip_expansion[3][1],
            .po2expansion_mip4 = point_meshlet_mip_expansion[4][0],
            .masked_po2expansion_mip4 = point_meshlet_mip_expansion[4][1],
            .po2expansion_mip5 = point_meshlet_mip_expansion[5][0],
            .masked_po2expansion_mip5 = point_meshlet_mip_expansion[5][1],
            .po2expansion_mip6 = point_meshlet_mip_expansion[6][0],
            .masked_po2expansion_mip6 = point_meshlet_mip_expansion[6][1],
            .meshlet_instances = info.meshlet_instances,
            .mesh_instances = info.mesh_instances,
            .meshes = info.meshes,
            .entity_combined_transforms = info.entity_combined_transforms,
            .material_manifest = info.material_manifest,
            .vsm_point_lights = info.vsm_state->vsm_point_lights,
            .vsm_spot_lights = info.vsm_state->vsm_spot_lights,
            .vsm_point_spot_page_table = vsm_point_spot_page_table_view,
            .vsm_memory_block = info.vsm_state->memory_block.view(),
            .vsm_dirty_bit_hiz_mip0 = hpb_mip_views.at(0),
            .vsm_dirty_bit_hiz_mip1 = hpb_mip_views.at(1),
            .vsm_dirty_bit_hiz_mip2 = hpb_mip_views.at(2),
            .vsm_dirty_bit_hiz_mip3 = hpb_mip_views.at(3),
            .vsm_dirty_bit_hiz_mip4 = hpb_mip_views.at(4),
            .vsm_dirty_bit_hiz_mip5 = hpb_mip_views.at(5),
            .vsm_dirty_bit_hiz_mip6 = hpb_mip_views.at(6),
        },
        .render_context = info.render_context,
    });

    info.tg->add_task(ClearDirtyBitTask{
        .views = ClearDirtyBitTask::Views{
            .vsm_allocation_requests = info.vsm_state->allocation_requests,
            .vsm_clear_dirty_bit_indirect = info.vsm_state->clear_dirty_bit_indirect,
            .vsm_page_table = vsm_page_table_view,
        },
        .render_context = info.render_context,
    });
}

struct CameraController;
struct GetVSMProjectionsInfo
{
    CameraInfo const * camera_info = {};
    f32vec3 sun_direction = {};
    f32 clip_0_scale = {};
    f32 clip_0_near = {};
    f32 clip_0_far = {};
    f64 clip_0_height_offset = {};
    bool use_fixed_near_far = true;

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
        auto const near_far_clip_scale = info.use_fixed_near_far ? 1.0f : clip_scale;
        auto clip_projection = glm::ortho(
            -info.clip_0_scale * clip_scale,        // left
            info.clip_0_scale * clip_scale,         // right
            -info.clip_0_scale * clip_scale,        // bottom
            info.clip_0_scale * clip_scale,         // top
            info.clip_0_near * near_far_clip_scale, // near
            info.clip_0_far * near_far_clip_scale   // far
        );
        // Switch from OpenGL default to Vulkan default (invert the Y clip coordinate)
        clip_projection[1][1] *= -1.0;
        return clip_projection;
    };
    auto const target_camera_position = glm::vec4(info.camera_info->position, 1.0);
    auto const uv_page_size = s_cast<f32>(VSM_PAGE_SIZE) / s_cast<f32>(VSM_DIRECTIONAL_TEXTURE_RESOLUTION);
    // NDC space is [-1, 1] but uv space is [0, 1], PAGE_SIZE / TEXTURE_RESOLUTION gives us the page size in uv space
    // thus we need to multiply by two to get the page size in ndc coordinates
    auto const ndc_page_size = uv_page_size * 2.0f;

    for (i32 clip = 0; clip < VSM_CLIP_LEVELS; clip++)
    {
        auto const curr_clip_proj = calculate_clip_projection(clip);
        auto const clip_projection_view = curr_clip_proj * default_vsm_view;
        auto const curr_clip_scale = std::pow(2.0f, s_cast<f32>(clip));
        auto const curr_near_far_clip_scale = info.use_fixed_near_far ? 1.0f : curr_clip_scale;

        // Project the target position into VSM ndc coordinates and calculate a page alligned position
        auto const clip_projected_target_pos = clip_projection_view * target_camera_position;
        auto const ndc_target_pos = glm::vec3(clip_projected_target_pos) / clip_projected_target_pos.w;
        auto const ndc_page_scaled_target_pos = glm::vec2(ndc_target_pos) / ndc_page_size;
        auto const ndc_page_scaled_aligned_target_pos = glm::vec2(glm::ceil(ndc_page_scaled_target_pos));

        auto const ndc_page_aligned_target_pos = glm::vec4(ndc_page_scaled_aligned_target_pos * ndc_page_size, ndc_target_pos.z, 1.0f);
        auto const world_page_aligned_target_pos = glm::vec3(glm::inverse(clip_projection_view) * ndc_page_aligned_target_pos);

        auto const clip_position = world_page_aligned_target_pos + s_cast<float>(info.clip_0_height_offset) * curr_near_far_clip_scale * -default_vsm_forward;
        auto const final_clip_view = glm::lookAt(clip_position, clip_position + glm::normalize(default_vsm_forward), default_vsm_up);

        auto clip_camera = CameraInfo{
            .view = final_clip_view,
            .inv_view = glm::inverse(final_clip_view),
            .proj = curr_clip_proj,
            .inv_proj = glm::inverse(curr_clip_proj),
            .view_proj = curr_clip_proj * final_clip_view,
            .inv_view_proj = glm::inverse(curr_clip_proj * final_clip_view),
            .position = clip_position,
            .up = default_vsm_up,
            .screen_size = {VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION << 1, VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION << 1},
            .inv_screen_size = {
                1.0f / s_cast<f32>(VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION << 1),
                1.0f / s_cast<f32>(VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION << 1),
            },
        };

        glm::vec3 ws_ndc_corners[2][2][2];
        for (u32 z = 0; z < 2; ++z)
        {
            for (u32 y = 0; y < 2; ++y)
            {
                for (u32 x = 0; x < 2; ++x)
                {
                    glm::vec3 corner = glm::vec3((glm::vec2(x, y) - 0.5f) * 2.0f, z);
                    glm::vec4 proj_corner = clip_camera.inv_view_proj * glm::vec4(corner, 1);
                    ws_ndc_corners[x][y][z] = glm::vec3(proj_corner) / proj_corner.w;
                }
            }
        }
        clip_camera.orthogonal_half_ws_width = curr_clip_scale * info.clip_0_scale;
        clip_camera.is_orthogonal = 1u;
        clip_camera.near_plane_normal = glm::vec3(0, 0, 0); // Orthogonal doesn't cull against near

        clip_camera.right_plane_normal = glm::normalize(
            glm::cross(ws_ndc_corners[1][1][0] - ws_ndc_corners[1][0][0], ws_ndc_corners[1][0][1] - ws_ndc_corners[1][0][0]));
        clip_camera.left_plane_normal = glm::normalize(
            glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][0][1], ws_ndc_corners[0][0][0] - ws_ndc_corners[0][0][1]));
        clip_camera.top_plane_normal = glm::normalize(
            glm::cross(ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[0][0][1] - ws_ndc_corners[0][0][0]));
        clip_camera.bottom_plane_normal = glm::normalize(
            glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][1][0], ws_ndc_corners[1][1][0] - ws_ndc_corners[0][1][0]));

        f32 const near_plane = info.clip_0_near * curr_near_far_clip_scale;
        f32 const far_plane = info.clip_0_far * curr_near_far_clip_scale;
        f32 const near_to_far_range = far_plane - near_plane;
        clip_projections.at(clip) = VSMClipProjection{
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

        ShaderDebugBoxDraw box_draw = {};
        box_draw.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        box_draw.color = std::bit_cast<daxa_f32vec3>(hsv2rgb(f32vec3(std::pow(s_cast<f32>(clip) / (VSM_CLIP_LEVELS - 1), 0.5f), 1.0f, 1.0f)));
        for (i32 i = 0; i < 8; i++)
        {
            auto const ndc_pos = glm::vec4(offsets[i], i < 4 ? 0.0f : 1.0f, 1.0f);
            auto const world_pos = std::bit_cast<glm::mat4x4>(clip_projection.camera.inv_view_proj) * ndc_pos;
            box_draw.vertices[i] = {world_pos.x, world_pos.y, world_pos.z};
        }
        info.debug_context->box_draws.draw(box_draw);
    }
}

struct DebugDrawPointFrusiInfo
{
    VSMPointLight const * light;
    VSMState const * state;
    ShaderDebugDrawContext * debug_context;
};

inline void debug_draw_point_frusti(DebugDrawPointFrusiInfo const & info)
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

    auto const & inverse_projection = info.state->globals_cpu.inverse_point_light_projection_matrix;

    for (i32 cube_face = 0; cube_face < 5; ++cube_face)
    {
        auto const & inverse_view = info.light->face_cameras[cube_face].inv_view;
        ShaderDebugBoxDraw box_draw = {};
        box_draw.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        box_draw.color = std::bit_cast<daxa_f32vec3>(hsv2rgb(glm::vec3(cube_face / 6.0f, 1.0f, 1.0f)));
        for (i32 vertex = 0; vertex < 8; vertex++)
        {
            auto const ndc_pos = glm::vec4(offsets[vertex], vertex < 4 ? 1.0f : 0.0001f, 1.0f);
            auto const view_pos_unproj = inverse_projection * ndc_pos;
            auto const view_pos = glm::vec3(view_pos_unproj.x, view_pos_unproj.y, view_pos_unproj.z) / view_pos_unproj.w;
            auto const world_pos = inverse_view * glm::vec4(view_pos, 1.0f);
            box_draw.vertices[vertex] = {world_pos.x, world_pos.y, world_pos.z};
        }
        info.debug_context->box_draws.draw(box_draw);
    }
}

struct DebugDrawSpotFrustumInfo
{
    VSMSpotLight const * light;
    VSMState const * state;
    ShaderDebugDrawContext * debug_context;
};

inline void debug_draw_spot_frustum(DebugDrawSpotFrustumInfo const & info)
{
    static constexpr std::array offsets = {
        glm::ivec2(-1, 1), glm::ivec2(-1, -1), glm::ivec2(1, -1), glm::ivec2(1, 1),
        glm::ivec2(-1, 1), glm::ivec2(-1, -1), glm::ivec2(1, -1), glm::ivec2(1, 1)};

    auto const & inverse_projection = info.light->camera.inv_proj;
    auto const & inverse_view = info.light->camera.inv_view;

    ShaderDebugBoxDraw box_draw = {};
    box_draw.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
    box_draw.color = daxa_f32vec3{1.0f, 0.0f, 0.0f};
    for (i32 vertex = 0; vertex < 8; vertex++)
    {
        auto const ndc_pos = glm::vec4(offsets[vertex], vertex < 4 ? 1.0f : 0.0001f, 1.0f);
        auto const view_pos_unproj = inverse_projection * ndc_pos;
        auto const view_pos = glm::vec3(view_pos_unproj.x, view_pos_unproj.y, view_pos_unproj.z) / view_pos_unproj.w;
        auto const world_pos = inverse_view * glm::vec4(view_pos, 1.0f);
        box_draw.vertices[vertex] = {world_pos.x, world_pos.y, world_pos.z};
    }
    info.debug_context->box_draws.draw(box_draw);
}