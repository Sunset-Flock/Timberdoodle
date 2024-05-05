#pragma once

#include "daxa/daxa.inl"
#include "daxa/utils/task_graph.inl"

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry_pipeline.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/po2_expansion.inl"

DAXA_DECL_TASK_HEAD_BEGIN(DrawVisbuffer_WriteCommandH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), draw_commands)
DAXA_DECL_TASK_HEAD_END

// When drawing triangles, this draw command has triangle ids appended to the end of the command.
DAXA_DECL_TASK_HEAD_BEGIN(DrawVisbufferH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER(DRAW_INDIRECT_INFO_READ, draw_commands)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMaterial), material_manifest)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vis_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_STORAGE_READ_WRITE, REGULAR_2D, overdraw_image)
DAXA_DECL_TASK_HEAD_END

#if DAXA_SHADERLANG != DAXA_SHADERLANG_GLSL
DAXA_DECL_TASK_HEAD_BEGIN(CullMeshletsDrawVisbufferH)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
// Cull Attachments:
DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_SAMPLED, REGULAR_2D, hiz)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), po2expansion)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Po2WorkExpansionBufferHead), masked_po2expansion)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_u32), first_pass_meshlets_bitfield_offsets)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, U32ArenaBufferRef, first_pass_meshlets_bitfield_arena)
// Draw Attachments:
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(OpaqueMeshInstancesBufferHead), mesh_instances)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMaterial), material_manifest)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vis_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_STORAGE_READ_WRITE, REGULAR_2D, overdraw_image)
DAXA_DECL_TASK_HEAD_END
#endif

struct DrawVisbufferPush_WriteCommand
{
    DAXA_TH_BLOB(DrawVisbuffer_WriteCommandH, uses)
    daxa_u32 pass;
};

struct DrawVisbufferPush
{
    DAXA_TH_BLOB(DrawVisbufferH, uses)
    daxa_u32 pass;
};

#if DAXA_SHADERLANG != DAXA_SHADERLANG_GLSL
struct CullMeshletsDrawVisbufferPush
{
    DAXA_TH_BLOB(CullMeshletsDrawVisbufferH, uses)
    daxa_u32 draw_list_type;
    daxa_u32 bucket_index;
};
#endif

#if defined(__cplusplus)
#include "../../gpu_context.hpp"
#include "../tasks/misc.hpp"
#include "../tasks/dvmaa.hpp"

static constexpr inline char const SLANG_DRAW_VISBUFFER_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/draw_visbuffer.hlsl";

using DrawVisbuffer_WriteCommandTask2 = SimpleComputeTask<
    DrawVisbuffer_WriteCommandH::Task,
    DrawVisbufferPush_WriteCommand,
    SLANG_DRAW_VISBUFFER_SHADER_PATH,
    "entry_write_commands">;

static inline daxa::RasterPipelineCompileInfo draw_visbuffer_base_compile_info() {
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.depth_test = {
        .depth_attachment_format = daxa::Format::D32_SFLOAT,
        .enable_depth_write = true,
        .depth_test_compare_op = daxa::CompareOp::GREATER,
        .min_depth_bounds = 0.0f,
        .max_depth_bounds = 1.0f,
    };
    ret.color_attachments = {
        daxa::RenderAttachment{
            .format = daxa::Format::R32_UINT,
        },
        //daxa::RenderAttachment{
        //    .format = daxa::Format::R16G16B16A16_SFLOAT,
        //},
    };
    ret.name = "DrawVisbufferOpaque";
    ret.push_constant_size = s_cast<u32>(sizeof(DrawVisbufferPush));
    ret.raster.static_state_sample_count = daxa::None; // Set to use dynamic state for msaa.
    return ret;
};

inline daxa::RasterPipelineCompileInfo slang_draw_visbuffer_solid_pipeline_compile_info()
{
    auto ret = draw_visbuffer_base_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_fragment_opaque",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_vertex_opaque",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "SlangDrawVisbufferOpaque";
    return ret;
};

inline daxa::RasterPipelineCompileInfo slang_draw_visbuffer_masked_pipeline_compile_info()
{
    auto ret = draw_visbuffer_base_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_fragment_masked",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_vertex_masked",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "SlangDrawVisbufferMasked";
    return ret;
};

inline daxa::RasterPipelineCompileInfo slang_draw_visbuffer_mesh_shader_solid_pipeline_compile_info()
{
    auto ret = slang_draw_visbuffer_solid_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_mesh_fragment_opaque",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.vertex_shader_info = daxa::None;
    ret.mesh_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_mesh_opaque",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "SlangDrawVisbufferMeshShaderSolid";
    return ret;
};

inline daxa::RasterPipelineCompileInfo slang_draw_visbuffer_mesh_shader_masked_pipeline_compile_info()
{
    auto ret = slang_draw_visbuffer_masked_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_mesh_fragment_mask",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.vertex_shader_info = daxa::None;
    ret.mesh_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_mesh_mask",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "SlangDrawVisbufferMeshShaderMasked";
    return ret;
};

inline std::array<daxa::RasterPipelineCompileInfo, 2> slang_draw_visbuffer_mesh_shader_pipelines = {
    slang_draw_visbuffer_mesh_shader_solid_pipeline_compile_info(),
    slang_draw_visbuffer_mesh_shader_masked_pipeline_compile_info()
};

inline daxa::RasterPipelineCompileInfo slang_cull_meshlets_draw_visbuffer_opaque_pipeline_compile_info()
{
    auto ret = slang_draw_visbuffer_mesh_shader_solid_pipeline_compile_info();
    ret.task_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_task_cull_draw_opaque_and_mask",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.mesh_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_mesh_cull_draw_opaque",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_mesh_fragment_cull_draw_opaque",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "SlangCullMeshletsDrawVisbufferOpaque";
    ret.push_constant_size = s_cast<u32>(sizeof(CullMeshletsDrawVisbufferPush));
    return ret;
};

inline daxa::RasterPipelineCompileInfo slang_cull_meshlets_draw_visbuffer_masked_pipeline_compile_info()
{
    auto ret = slang_draw_visbuffer_mesh_shader_masked_pipeline_compile_info();
    ret.task_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_task_cull_draw_opaque_and_mask",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.mesh_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_mesh_cull_draw_mask",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_mesh_fragment_cull_draw_mask",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "SlangCullMeshletsDrawVisbufferMasked";
    ret.push_constant_size = s_cast<u32>(sizeof(CullMeshletsDrawVisbufferPush));
    return ret;
};

inline std::array<daxa::RasterPipelineCompileInfo, 2> slang_cull_meshlets_draw_visbuffer_pipelines = {
    slang_cull_meshlets_draw_visbuffer_opaque_pipeline_compile_info(),
    slang_cull_meshlets_draw_visbuffer_masked_pipeline_compile_info()
};

struct DrawVisbufferTask : DrawVisbufferH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    u32 pass = {};
    void callback(daxa::TaskInterface ti)
    {
        bool const dvmaa = render_context->render_data.settings.anti_aliasing_mode == AA_MODE_DVM;
        bool const clear_images = pass != PASS1_DRAW_POST_CULL;
        auto [x, y, z] = ti.device.info_image(ti.get(AT.depth_image).ids[0]).value().size;
        auto load_op = clear_images ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD;
        daxa::RenderPassBeginInfo render_pass_begin_info{
            .depth_attachment =
                daxa::RenderAttachmentInfo{
                    .image_view = ti.get(AT.depth_image).ids[0].default_view(),
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = load_op,
                    .store_op = daxa::AttachmentStoreOp::STORE,
                    .clear_value = daxa::DepthValue{0.0f, 0},
                },
            .render_area = daxa::Rect2D{.width = x, .height = y},
        };
        render_pass_begin_info.color_attachments = {
            daxa::RenderAttachmentInfo{
                .image_view = ti.get(AT.vis_image).ids[0].default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = std::array<u32, 4>{INVALID_TRIANGLE_ID, 0, 0, 0},
            },
            //daxa::RenderAttachmentInfo{
            //    .image_view = ti.get(AT::debug_image).ids[0].default_view(),
            //    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
            //    .load_op = load_op,
            //    .store_op = daxa::AttachmentStoreOp::STORE,
            //    .clear_value = daxa::ClearValue{std::array<u32, 4>{0, 0, 0, 0}},
            //},
        };
        auto render_cmd = std::move(ti.recorder).begin_renderpass(render_pass_begin_info);
        if (dvmaa)
        {
            render_cmd.set_rasterization_samples(daxa::RasterizationSamples::E4);
        }
        else
        {
            render_cmd.set_rasterization_samples(daxa::RasterizationSamples::E1);
        }
        for (u32 opaque_draw_list_type = 0; opaque_draw_list_type < 2; ++opaque_draw_list_type)
        {
            render_cmd.set_pipeline(*render_context->gpuctx->raster_pipelines.at(slang_draw_visbuffer_mesh_shader_pipelines[opaque_draw_list_type].name));
            DrawVisbufferPush push{ .pass = pass };
            assign_blob(push.uses, ti.attachment_shader_blob);
            render_cmd.push_constant(push);
            render_cmd.draw_mesh_tasks_indirect({
                .indirect_buffer = ti.get(AT.draw_commands).ids[0],
                .offset = sizeof(DispatchIndirectStruct) * opaque_draw_list_type,
                .draw_count = 1,
                .stride = sizeof(DispatchIndirectStruct),
            });
        }
        ti.recorder = std::move(render_cmd).end_renderpass();
    }
};

struct CullMeshletsDrawVisbufferTask : CullMeshletsDrawVisbufferH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    void callback(daxa::TaskInterface ti)
    {
        bool const clear_images = false;
        auto load_op = clear_images ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD;
        auto [x, y, z] = ti.device.info_image(ti.get(AT.depth_image).ids[0]).value().size;
        daxa::RenderPassBeginInfo render_pass_begin_info{
            .depth_attachment =
                daxa::RenderAttachmentInfo{
                    .image_view = ti.get(AT.depth_image).ids[0].default_view(),
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = load_op,
                    .store_op = daxa::AttachmentStoreOp::STORE,
                    .clear_value = daxa::ClearValue{daxa::DepthValue{0.0f, 0}},
                },
            .render_area = daxa::Rect2D{.width = x, .height = y},
        };
        render_pass_begin_info.color_attachments = {
            daxa::RenderAttachmentInfo{
                .image_view = ti.get(AT.vis_image).ids[0].default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::ClearValue{std::array<u32, 4>{INVALID_TRIANGLE_ID, 0, 0, 0}},
            },
            // daxa::RenderAttachmentInfo{
            //     .image_view = ti.get(AT.debug_image).ids[0].default_view(),
            //     .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
            //     .load_op = load_op,
            //     .store_op = daxa::AttachmentStoreOp::STORE,
            //     .clear_value = daxa::ClearValue{std::array<u32, 4>{0, 0, 0, 0}},
            // },
        };
        auto render_cmd = std::move(ti.recorder).begin_renderpass(render_pass_begin_info);
        for (u32 opaque_draw_list_type = 0; opaque_draw_list_type < DRAW_LIST_TYPES; ++opaque_draw_list_type)
        {
            auto buffer = ti.get(opaque_draw_list_type == DRAW_LIST_OPAQUE ? AT.po2expansion : AT.masked_po2expansion).ids[0];
            render_cmd.set_pipeline(*render_context->gpuctx->raster_pipelines.at(slang_cull_meshlets_draw_visbuffer_pipelines[opaque_draw_list_type].name));
            for (u32 i = 0; i < 32; ++i)
            {
                CullMeshletsDrawVisbufferPush push = {
                    .draw_list_type = opaque_draw_list_type,
                    .bucket_index = i,
                };
                assign_blob(push.uses, ti.attachment_shader_blob);
                render_cmd.push_constant(push);
                render_cmd.draw_mesh_tasks_indirect({
                    .indirect_buffer = buffer,
                    .offset = sizeof(DispatchIndirectStruct) * i,
                    .draw_count = 1,
                    .stride = sizeof(DispatchIndirectStruct),
                });
            }
        }
        ti.recorder = std::move(render_cmd).end_renderpass();
    }
};

struct TaskCullAndDrawVisbufferInfo
{
    RenderContext * render_context = {};
    daxa::TaskGraph & task_graph;
    std::array<daxa::TaskBufferView, DRAW_LIST_TYPES> meshlet_cull_po2expansion = {};
    daxa::TaskBufferView entity_meta_data = {};
    daxa::TaskBufferView entity_meshgroups = {};
    daxa::TaskBufferView entity_combined_transforms = {};
    daxa::TaskBufferView mesh_groups = {};
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView material_manifest = {};
    daxa::TaskBufferView first_pass_meshlets_bitfield_offsets = {};
    daxa::TaskBufferView first_pass_meshlets_bitfield_arena = {};
    daxa::TaskImageView hiz = {};
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskBufferView mesh_instances = {};
    daxa::TaskImageView vis_image = {};
    daxa::TaskImageView debug_image = {};
    daxa::TaskImageView depth_image = {};
    daxa::TaskImageView dvmaa_vis_image = {};
    daxa::TaskImageView dvmaa_depth_image = {};
    daxa::TaskImageView overdraw_image = {};
};
inline void task_cull_and_draw_visbuffer(TaskCullAndDrawVisbufferInfo const & info)
{
    bool const dvmaa = info.render_context->render_data.settings.anti_aliasing_mode == AA_MODE_DVM;
    info.task_graph.add_task(CullMeshletsDrawVisbufferTask{
        .views = std::array{
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.hiz, info.hiz),
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.po2expansion, info.meshlet_cull_po2expansion[0]),
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.masked_po2expansion, info.meshlet_cull_po2expansion[1]),
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.first_pass_meshlets_bitfield_offsets, info.first_pass_meshlets_bitfield_offsets),
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.first_pass_meshlets_bitfield_arena, info.first_pass_meshlets_bitfield_arena),
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.meshlet_instances, info.meshlet_instances),
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.mesh_instances, info.mesh_instances),
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.meshes, info.meshes),
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.entity_combined_transforms, info.entity_combined_transforms),
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.material_manifest, info.material_manifest),
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.vis_image, dvmaa ? info.dvmaa_vis_image : info.vis_image),
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.depth_image, dvmaa ? info.dvmaa_depth_image : info.depth_image),
            daxa::attachment_view(CullMeshletsDrawVisbufferH::AT.overdraw_image, info.overdraw_image),

        },
        .render_context = info.render_context,
    });
    if (dvmaa)
    {
        info.task_graph.add_task(DVMResolveVisImageTask{
            .views = std::array{
                daxa::attachment_view(DVMResolveVisImageH::AT.dvm_vis_image, info.dvmaa_vis_image),
                daxa::attachment_view(DVMResolveVisImageH::AT.vis_image, info.vis_image),
                daxa::attachment_view(DVMResolveVisImageH::AT.dvm_depth_image, info.dvmaa_depth_image),
                daxa::attachment_view(DVMResolveVisImageH::AT.depth_image, info.depth_image),
            },
            .rctx = info.render_context,
        });
    }
}

struct TaskDrawVisbufferInfo
{
    RenderContext * render_context = {};
    daxa::TaskGraph & task_graph;
    u32 const pass = {};
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView material_manifest = {};
    daxa::TaskBufferView combined_transforms = {};
    daxa::TaskImageView vis_image = {};
    daxa::TaskImageView debug_image = {};
    daxa::TaskImageView depth_image = {};
    daxa::TaskImageView dvmaa_vis_image = {};
    daxa::TaskImageView dvmaa_depth_image = {};
    daxa::TaskImageView overdraw_image = {};
};

inline void task_draw_visbuffer(TaskDrawVisbufferInfo const & info)
{
    auto draw_commands_array = info.task_graph.create_transient_buffer({
        .size = 2 * static_cast<u32>(std::max(sizeof(DrawIndirectStruct), sizeof(DispatchIndirectStruct))),
        .name = std::string("draw visbuffer command buffer array") + info.render_context->gpuctx->dummy_string(),
    });

    DrawVisbuffer_WriteCommandTask2 write_task = {
        .views = std::array{
            daxa::attachment_view(DrawVisbuffer_WriteCommandH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(DrawVisbuffer_WriteCommandH::AT.meshlet_instances, info.meshlet_instances),
            daxa::attachment_view(DrawVisbuffer_WriteCommandH::AT.draw_commands, draw_commands_array),
        },
        .context = info.render_context->gpuctx,
        .push = DrawVisbufferPush_WriteCommand{.pass = info.pass},
        .dispatch_callback = [](){ return daxa::DispatchInfo{1,1,1}; },
    };
    info.task_graph.add_task(write_task);

    bool dvmaa = info.render_context->render_data.settings.anti_aliasing_mode == AA_MODE_DVM;

    if (info.overdraw_image != daxa::NullTaskImage)
    {
        task_clear_image(info.task_graph, info.overdraw_image, std::array{0u,0u,0u,0u});
    }

    DrawVisbufferTask draw_task = {
        .views = std::array{
            DrawVisbufferH::AT.globals | info.render_context->tgpu_render_data,
            DrawVisbufferH::AT.draw_commands | draw_commands_array,
            DrawVisbufferH::AT.meshlet_instances | info.meshlet_instances,
            DrawVisbufferH::AT.meshes | info.meshes,
            DrawVisbufferH::AT.material_manifest | info.material_manifest,
            DrawVisbufferH::AT.entity_combined_transforms | info.combined_transforms,
            DrawVisbufferH::AT.vis_image | (dvmaa ? info.dvmaa_vis_image : info.vis_image),
            DrawVisbufferH::AT.depth_image | (dvmaa ? info.dvmaa_depth_image : info.depth_image),
            DrawVisbufferH::AT.overdraw_image | info.overdraw_image,
        },
        .render_context = info.render_context,
        .pass = info.pass,
    };
    info.task_graph.add_task(draw_task);
    
    if (dvmaa)
    {
        info.task_graph.add_task(DVMResolveVisImageTask{
            .views = std::array{
                daxa::attachment_view(DVMResolveVisImageH::AT.dvm_vis_image, info.dvmaa_vis_image),
                daxa::attachment_view(DVMResolveVisImageH::AT.vis_image, info.vis_image),
                daxa::attachment_view(DVMResolveVisImageH::AT.dvm_depth_image, info.dvmaa_depth_image),
                daxa::attachment_view(DVMResolveVisImageH::AT.depth_image, info.depth_image),
            },
            .rctx = info.render_context,
        });
    }
}
#endif // #if defined(__cplusplus)