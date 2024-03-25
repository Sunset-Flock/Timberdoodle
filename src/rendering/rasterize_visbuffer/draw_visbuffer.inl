#pragma once

#include "daxa/daxa.inl"
#include "daxa/utils/task_graph.inl"

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry_pipeline.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"

DAXA_DECL_TASK_HEAD_BEGIN(DrawVisbuffer_WriteCommandH, 3)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_u64, draw_commands)
DAXA_DECL_TASK_HEAD_END

// When drawing triangles, this draw command has triangle ids appended to the end of the command.
DAXA_DECL_TASK_HEAD_BEGIN(DrawVisbufferH, 8)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER(DRAW_INDIRECT_INFO_READ, draw_commands)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMaterial), material_manifest)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vis_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(DrawVisbuffer_MeshShaderH, 15)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(MeshletCullArgBucketsBufferHead), meshlets_cull_arg_buckets_buffer_opaque)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshgroups)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMeshGroup), meshgroups)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, U32ArenaBufferRef, entity_meshlet_visibility_bitfield_offsets)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshlet_visibility_bitfield_arena)
DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_SAMPLED, REGULAR_2D, hiz)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE, daxa_RWBufferPtr(MeshletInstancesBufferHead), instantiated_meshlets)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vis_image)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, debug_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

struct DrawVisbufferPush_WriteCommand
{
    DAXA_TH_BLOB(DrawVisbuffer_WriteCommandH, uses)
    daxa_u32 pass;
    daxa_u32 mesh_shader;
};

struct DrawVisbufferPush
{
    DAXA_TH_BLOB(DrawVisbufferH, uses)
    daxa_u32 pass;
};

struct DrawVisbufferPush_MeshShader
{
    DAXA_TH_BLOB(DrawVisbuffer_MeshShaderH, uses)
    daxa_u32 bucket_index;
};

#if defined(__cplusplus)
#include "../../gpu_context.hpp"
#include "../tasks/misc.hpp"
#include "cull_meshlets.inl"
#include "../tasks/dvmaa.hpp"

static constexpr inline char const DRAW_VISBUFFER_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/draw_visbuffer.glsl";
static constexpr inline char const SLANG_DRAW_VISBUFFER_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/draw_visbuffer.slang";

using DrawVisbuffer_WriteCommandTask = SimpleComputeTask<
    DrawVisbuffer_WriteCommandH::Task,
    DrawVisbufferPush_WriteCommand,
    DRAW_VISBUFFER_SHADER_PATH,
    "main">;

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

inline daxa::RasterPipelineCompileInfo draw_visbuffer_solid_pipeline_compile_info()
{
    auto ret = draw_visbuffer_base_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .defines = {{"NO_MESH_SHADER", "1"}, {"OPAQUE", "1"}},
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .defines = {{"NO_MESH_SHADER", "1"}, {"OPAQUE", "1"}},
        },
    };
    ret.name = "DrawVisbufferOpaque";
    return ret;
};

inline daxa::RasterPipelineCompileInfo draw_visbuffer_masked_pipeline_compile_info()
{
    auto ret = draw_visbuffer_base_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .defines = {{"NO_MESH_SHADER", "1"}, {"DISCARD", "1"}},
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .defines = {{"NO_MESH_SHADER", "1"}, {"DISCARD", "1"}},
        },
    };
    ret.name = "DrawVisbufferDiscard";
    return ret;
};

inline std::array<daxa::RasterPipelineCompileInfo, 2> draw_visbuffer_pipelines = {
    draw_visbuffer_solid_pipeline_compile_info(),
    draw_visbuffer_masked_pipeline_compile_info()
};

inline daxa::RasterPipelineCompileInfo slang_draw_visbuffer_solid_pipeline_compile_info()
{
    auto ret = draw_visbuffer_solid_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_fragment",
            .language = daxa::ShaderLanguage::SLANG,
            .defines = {{"NO_MESH_SHADER", "1"}, {"OPAQUE", "1"}},
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_vertex",
            .language = daxa::ShaderLanguage::SLANG,
            .defines = {{"NO_MESH_SHADER", "1"}, {"OPAQUE", "1"}},
        },
    };
    return ret;
};

inline daxa::RasterPipelineCompileInfo slang_draw_visbuffer_masked_pipeline_compile_info()
{
    auto ret = draw_visbuffer_masked_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_fragment",
            .language = daxa::ShaderLanguage::SLANG,
            .defines = {{"NO_MESH_SHADER", "1"}, {"DISCARD", "1"}},
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_vertex",
            .language = daxa::ShaderLanguage::SLANG,
            .defines = {{"NO_MESH_SHADER", "1"}, {"DISCARD", "1"}},
        },
    };
    return ret;
};

inline std::array<daxa::RasterPipelineCompileInfo, 2> slang_draw_visbuffer_pipelines = {
    slang_draw_visbuffer_solid_pipeline_compile_info(),
    slang_draw_visbuffer_masked_pipeline_compile_info()
};

inline daxa::RasterPipelineCompileInfo slang_draw_visbuffer_mesh_shader_solid_pipeline_compile_info()
{
    auto ret = slang_draw_visbuffer_solid_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_fragment",
            .language = daxa::ShaderLanguage::SLANG,
            .defines = {{"MESH_SHADER", "1"}, {"OPAQUE", "1"}},
        },
    };
    ret.vertex_shader_info = daxa::None;
    ret.mesh_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_mesh",
            .language = daxa::ShaderLanguage::SLANG,
            .defines = {{"MESH_SHADER", "1"}, {"OPAQUE", "1"}},
        },
    };
    ret.name = "DrawVisbufferMeshShaderSolid";
    return ret;
};

inline daxa::RasterPipelineCompileInfo slang_draw_visbuffer_mesh_shader_masked_pipeline_compile_info()
{
    auto ret = slang_draw_visbuffer_masked_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_mesh_fragment",
            .language = daxa::ShaderLanguage::SLANG,
            .defines = {{"MESH_SHADER", "1"}, {"DISCARD", "1"}},
        },
    };
    ret.vertex_shader_info = daxa::None;
    ret.mesh_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_mesh",
            .language = daxa::ShaderLanguage::SLANG,
            .defines = {{"MESH_SHADER", "1"}, {"DISCARD", "1"}},
        },
    };
    ret.name = "DrawVisbufferMeshShaderMasked";
    return ret;
};

inline std::array<daxa::RasterPipelineCompileInfo, 2> slang_draw_visbuffer_mesh_shader_pipelines = {
    slang_draw_visbuffer_mesh_shader_solid_pipeline_compile_info(),
    slang_draw_visbuffer_mesh_shader_masked_pipeline_compile_info()
};

// inline daxa::RasterPipelineCompileInfo draw_visbuffer_mesh_shader_cull_and_draw_pipeline_compile_info()
// {
//     auto ret = slang_draw_visbuffer_solid_pipeline_compile_info();
//     auto comp_info = daxa::ShaderCompileInfo{
//         .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
//         .compile_options = {.defines = {{"MESH_SHADER_CULL_AND_DRAW", "1"}}},
//     };
//     ret.fragment_shader_info = comp_info;
//     ret.mesh_shader_info = comp_info;
//     ret.task_shader_info = comp_info;
//     ret.name = "DrawVisbuffer_MeshShader";
//     ret.push_constant_size = s_cast<u32>(sizeof(DrawVisbufferPush_MeshShader));
//     return ret;
// };

struct DrawVisbufferTask : DrawVisbufferH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    u32 pass = {};
    bool mesh_shader = {};
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
            if (mesh_shader)
            {
                render_cmd.set_pipeline(*render_context->gpuctx->raster_pipelines.at(slang_draw_visbuffer_mesh_shader_pipelines[opaque_draw_list_type].name));
            }
            else
            {
                if (render_context->render_data.settings.use_slang_for_drawing)
                {
                    render_cmd.set_pipeline(*render_context->gpuctx->raster_pipelines.at(draw_visbuffer_pipelines[opaque_draw_list_type].name));
                }
                else
                {
                    render_cmd.set_pipeline(*render_context->gpuctx->raster_pipelines.at(draw_visbuffer_pipelines[opaque_draw_list_type].name));
                }
            }
            DrawVisbufferPush push{ .pass = pass };
            assign_blob(push.uses, ti.attachment_shader_blob);
            render_cmd.push_constant(push);
            if (mesh_shader)
            {
                render_cmd.draw_mesh_tasks_indirect({
                    .indirect_buffer = ti.get(AT.draw_commands).ids[0],
                    .offset = sizeof(DispatchIndirectStruct) * opaque_draw_list_type,
                    .draw_count = 1,
                    .stride = sizeof(DispatchIndirectStruct),
                });
            }
            else
            {
                render_cmd.draw_indirect({
                    .draw_command_buffer = ti.get(AT.draw_commands).ids[0],
                    .indirect_buffer_offset = sizeof(DrawIndirectStruct) * opaque_draw_list_type,
                    .draw_count = 1,
                    .draw_command_stride = sizeof(DrawIndirectStruct),
                });
            }
        }
        ti.recorder = std::move(render_cmd).end_renderpass();
    }
};

struct CullAndDrawVisbufferTask : DrawVisbuffer_MeshShaderH::Task
{
    AttachmentViews views = {};
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
#if 0
        bool const clear_images = false;
        auto load_op = clear_images ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD;
        auto [x, y, z] = ti.device.info_image(ti.get(DrawVisbuffer_MeshShader::depth_image).ids[0]).value().size;
        daxa::RenderPassBeginInfo render_pass_begin_info{
            .depth_attachment =
                daxa::RenderAttachmentInfo{
                    .image_view = ti.get(DrawVisbuffer_MeshShader::depth_image).ids[0].default_view(),
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = load_op,
                    .store_op = daxa::AttachmentStoreOp::STORE,
                    .clear_value = daxa::ClearValue{daxa::DepthValue{0.0f, 0}},
                },
            .render_area = daxa::Rect2D{.width = x, .height = y},
        };
        render_pass_begin_info.color_attachments = {
            daxa::RenderAttachmentInfo{
                .image_view = ti.get(DrawVisbuffer_MeshShader::vis_image).ids[0].default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::ClearValue{std::array<u32, 4>{INVALID_TRIANGLE_ID, 0, 0, 0}},
            },
            daxa::RenderAttachmentInfo{
                .image_view = ti.get(DrawVisbuffer_MeshShader::debug_image).ids[0].default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::ClearValue{std::array<u32, 4>{0, 0, 0, 0}},
            },
        };
        auto render_cmd = std::move(ti.recorder).begin_renderpass(render_pass_begin_info);
        render_cmd.set_pipeline(*context->raster_pipelines.at(draw_visbuffer_mesh_shader_cull_and_draw_pipeline_compile_info().name));
        for (u32 i = 0; i < 32; ++i)
        {
            render_cmd.push_constant_vptr({
                .data = ti.attachment_shader_data.data(),
                .size = ti.attachment_shader_data.size(),
                .offset = sizeof(DrawVisbufferPush_MeshShader),
            });
            render_cmd.push_constant(DrawVisbufferPush_MeshShader{ .bucket_index = i}, DrawVisbuffer_MeshShader::attachment_shader_data_size());
            render_cmd.draw_mesh_tasks_indirect({
                .indirect_buffer = ti.get(DrawVisbuffer_MeshShader::command).ids[0],
                .offset = sizeof(DispatchIndirectStruct) * i,
                .draw_count = 1,
                .stride = sizeof(DispatchIndirectStruct),
            });
        }
        ti.recorder = std::move(render_cmd).end_renderpass();
#endif
    }
};

struct TaskCullAndDrawVisbufferInfo
{
    RenderContext * render_context = {};
    daxa::TaskGraph & task_graph;
    bool const enable_mesh_shader = {};
    daxa::TaskBufferView meshlets_cull_arg_buckets_buffer_opaque = {};
    daxa::TaskBufferView meshlets_cull_arg_buckets_buffer_discard = {};
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
    daxa::TaskImageView vis_image = {};
    daxa::TaskImageView debug_image = {};
    daxa::TaskImageView depth_image = {};
    daxa::TaskImageView dvmaa_vis_image = {};
    daxa::TaskImageView dvmaa_depth_image = {};
};
inline void task_cull_and_draw_visbuffer(TaskCullAndDrawVisbufferInfo const & info)
{
#if 0
    if (info.enable_mesh_shader)
    {
        info.task_graph.add_task(CullAndDrawVisbufferTask{
            .views = std::array{
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::globals, info.context->tgpu_render_data}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::command, info.cull_meshlets_commands}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::meshlet_cull_indirect_args, info.meshlet_cull_indirect_args}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::instantiated_meshlets, info.meshlet_instances}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::meshes, info.meshes}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::entity_meta, info.entity_meta_data}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::entity_meshgroups, info.entity_meshgroups}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::meshgroups, info.mesh_groups}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::entity_combined_transforms, info.entity_combined_transforms}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::entity_meshlet_visibility_bitfield_offsets, info.entity_meshlet_visibility_bitfield_offsets}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::entity_meshlet_visibility_bitfield_arena, info.entity_meshlet_visibility_bitfield_arena}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::hiz, info.hiz}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::vis_image, info.vis_image}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::debug_image, info.debug_image}},
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::depth_image, info.depth_image}},
            },
            .context = info.context,
        });
    }
    else
#endif
    {
        auto draw_commands_array = info.task_graph.create_transient_buffer({
            .size = static_cast<u32>(std::max(sizeof(DrawIndirectStruct), sizeof(DispatchIndirectStruct)) * 2),
            .name = std::string("draw visbuffer command buffer array") + info.render_context->gpuctx->dummy_string(),
        });
        if (info.enable_mesh_shader)
        {
            task_fill_buffer(
                info.task_graph, 
                draw_commands_array,
                std::array{
                    DispatchIndirectStruct{ 1, 0, 1 },
                    DispatchIndirectStruct{ 1, 0, 1 },
                });
        }
        else
        {
            task_fill_buffer(
                info.task_graph, 
                draw_commands_array,
                std::array{
                    DrawIndirectStruct{
                        .vertex_count = MAX_TRIANGLES_PER_MESHLET * 3,
                        .instance_count = {},
                        .first_vertex = {},
                        .first_instance = {},
                    },
                    DrawIndirectStruct{
                        .vertex_count = MAX_TRIANGLES_PER_MESHLET * 3,
                        .instance_count = {},
                        .first_vertex = {},
                        .first_instance = {},
                    },
                });
        }
        info.task_graph.add_task(CullMeshletsTask{
            .views = std::array{
                daxa::attachment_view(CullMeshletsH::AT.globals, info.render_context->tgpu_render_data),
                daxa::attachment_view(CullMeshletsH::AT.hiz, info.hiz),
                daxa::attachment_view(CullMeshletsH::AT.meshlets_cull_arg_buckets_buffer, info.meshlets_cull_arg_buckets_buffer_opaque),
                daxa::attachment_view(CullMeshletsH::AT.entity_meta_data, info.entity_meta_data),
                daxa::attachment_view(CullMeshletsH::AT.entity_meshgroups, info.entity_meshgroups),
                daxa::attachment_view(CullMeshletsH::AT.meshgroups, info.mesh_groups),
                daxa::attachment_view(CullMeshletsH::AT.entity_combined_transforms, info.entity_combined_transforms),
                daxa::attachment_view(CullMeshletsH::AT.meshes, info.meshes),
                daxa::attachment_view(CullMeshletsH::AT.first_pass_meshlets_bitfield_offsets, info.first_pass_meshlets_bitfield_offsets),
                daxa::attachment_view(CullMeshletsH::AT.first_pass_meshlets_bitfield_arena, info.first_pass_meshlets_bitfield_arena),
                daxa::attachment_view(CullMeshletsH::AT.meshlet_instances, info.meshlet_instances),
                daxa::attachment_view(CullMeshletsH::AT.draw_commands, draw_commands_array),
            },
            .render_context = info.render_context,
            .opaque_or_discard = 0,
        });
        info.task_graph.add_task(CullMeshletsTask{
            .views = std::array{
                daxa::attachment_view(CullMeshletsH::AT.globals, info.render_context->tgpu_render_data),
                daxa::attachment_view(CullMeshletsH::AT.hiz, info.hiz),
                daxa::attachment_view(CullMeshletsH::AT.meshlets_cull_arg_buckets_buffer, info.meshlets_cull_arg_buckets_buffer_discard),
                daxa::attachment_view(CullMeshletsH::AT.entity_meta_data, info.entity_meta_data),
                daxa::attachment_view(CullMeshletsH::AT.entity_meshgroups, info.entity_meshgroups),
                daxa::attachment_view(CullMeshletsH::AT.meshgroups, info.mesh_groups),
                daxa::attachment_view(CullMeshletsH::AT.entity_combined_transforms, info.entity_combined_transforms),
                daxa::attachment_view(CullMeshletsH::AT.meshes, info.meshes),
                daxa::attachment_view(CullMeshletsH::AT.first_pass_meshlets_bitfield_offsets, info.first_pass_meshlets_bitfield_offsets),
                daxa::attachment_view(CullMeshletsH::AT.first_pass_meshlets_bitfield_arena, info.first_pass_meshlets_bitfield_arena),
                daxa::attachment_view(CullMeshletsH::AT.meshlet_instances, info.meshlet_instances),
                daxa::attachment_view(CullMeshletsH::AT.draw_commands, draw_commands_array),
            },
            .render_context = info.render_context,
            .opaque_or_discard = 1,
        });

        bool const dvmaa = info.render_context->render_data.settings.anti_aliasing_mode == AA_MODE_DVM;

        DrawVisbufferTask draw_task = {
            .views = std::array{
                daxa::attachment_view(DrawVisbufferH::AT.globals, info.render_context->tgpu_render_data),
                daxa::attachment_view(DrawVisbufferH::AT.draw_commands, draw_commands_array),
                daxa::attachment_view(DrawVisbufferH::AT.meshlet_instances, info.meshlet_instances),
                daxa::attachment_view(DrawVisbufferH::AT.meshes, info.meshes),
                daxa::attachment_view(DrawVisbufferH::AT.entity_combined_transforms, info.entity_combined_transforms),
                daxa::attachment_view(DrawVisbufferH::AT.material_manifest, info.material_manifest),
                daxa::attachment_view(DrawVisbufferH::AT.vis_image, dvmaa ? info.dvmaa_vis_image : info.vis_image),
                daxa::attachment_view(DrawVisbufferH::AT.depth_image, dvmaa ? info.dvmaa_depth_image : info.depth_image),
            },
            .render_context = info.render_context,
            .pass = PASS1_DRAW_POST_CULL,
            .mesh_shader = info.enable_mesh_shader,
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
}

struct TaskDrawVisbufferInfo
{
    RenderContext * render_context = {};
    daxa::TaskGraph & task_graph;
    bool const enable_mesh_shader = {};
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
        .push = DrawVisbufferPush_WriteCommand{.pass = info.pass, .mesh_shader = info.enable_mesh_shader ? 1u : 0u},
        .dispatch_callback = [](){ return daxa::DispatchInfo{1,1,1}; },
    };
    info.task_graph.add_task(write_task);

    bool dvmaa = info.render_context->render_data.settings.anti_aliasing_mode == AA_MODE_DVM;
    info.task_graph.add_task(write_task);

    DrawVisbufferTask draw_task = {
        .views = std::array{
            daxa::attachment_view(DrawVisbufferH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(DrawVisbufferH::AT.draw_commands, draw_commands_array),
            daxa::attachment_view(DrawVisbufferH::AT.meshlet_instances, info.meshlet_instances),
            daxa::attachment_view(DrawVisbufferH::AT.meshes, info.meshes),
            daxa::attachment_view(DrawVisbufferH::AT.material_manifest, info.material_manifest),
            daxa::attachment_view(DrawVisbufferH::AT.entity_combined_transforms, info.combined_transforms),
            daxa::attachment_view(DrawVisbufferH::AT.vis_image, dvmaa ? info.dvmaa_vis_image : info.vis_image),
            daxa::attachment_view(DrawVisbufferH::AT.depth_image, dvmaa ? info.dvmaa_depth_image : info.depth_image),
        },
        .render_context = info.render_context,
        .pass = info.pass,
        .mesh_shader = info.enable_mesh_shader,
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