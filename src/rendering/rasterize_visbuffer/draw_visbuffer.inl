#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"

DAXA_DECL_TASK_HEAD_BEGIN(DrawVisbuffer_WriteCommand, 3)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_u64, draw_commands)
DAXA_DECL_TASK_HEAD_END

// When drawing triangles, this draw command has triangle ids appended to the end of the command.
DAXA_DECL_TASK_HEAD_BEGIN(DrawVisbuffer, 10)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
DAXA_TH_BUFFER(DRAW_INDIRECT_INFO_READ, draw_commands)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMaterial), material_manifest)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(VisibleMeshletList), visible_meshlet_instances)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vis_image)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, debug_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(DrawVisbuffer_MeshShader, 15)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(MeshletCullArgBucketsBufferHead), meshlets_cull_arg_buckets_buffer_opaque)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshgroups)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMeshGroup), meshgroups)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, EntityMeshletVisibilityBitfieldOffsetsView, entity_meshlet_visibility_bitfield_offsets)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshlet_visibility_bitfield_arena)
DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_SAMPLED, REGULAR_2D, hiz)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE, daxa_RWBufferPtr(MeshletInstancesBufferHead), instantiated_meshlets)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vis_image)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, debug_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

struct DrawVisbufferPush_WriteCommand
{
    DAXA_TH_BLOB(DrawVisbuffer_WriteCommand, uses)
    daxa_u32 pass;
    daxa_u32 mesh_shader;
};

struct DrawVisbufferPush
{
    DAXA_TH_BLOB(DrawVisbuffer, uses)
    daxa_u32 pass;
};

struct DrawVisbufferPush_MeshShader
{
    DAXA_TH_BLOB(DrawVisbuffer_MeshShader, uses)
    daxa_u32 bucket_index;
};

#if __cplusplus
#include "../../gpu_context.hpp"
#include "../tasks/misc.hpp"
#include "cull_meshlets.inl"

static constexpr inline char const DRAW_VISBUFFER_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/draw_visbuffer.glsl";

static inline daxa::DepthTestInfo DRAW_VISBUFFER_DEPTH_TEST_INFO = {
    .depth_attachment_format = daxa::Format::D32_SFLOAT,
    .enable_depth_write = true,
    .depth_test_compare_op = daxa::CompareOp::GREATER,
    .min_depth_bounds = 0.0f,
    .max_depth_bounds = 1.0f,
};

static inline std::vector<daxa::RenderAttachment> DRAW_VISBUFFER_RENDER_ATTACHMENT_INFOS = {
    daxa::RenderAttachment{
        .format = daxa::Format::R32_UINT,
    },
    //daxa::RenderAttachment{
    //    .format = daxa::Format::R16G16B16A16_SFLOAT,
    //},
};

using DrawVisbuffer_WriteCommandTask =
    WriteIndirectDispatchArgsPushBaseTask<DrawVisbuffer_WriteCommand, DRAW_VISBUFFER_SHADER_PATH, DrawVisbufferPush_WriteCommand>;
auto draw_visbuffer_write_command_pipeline_compile_info()
{
    return write_indirect_dispatch_args_base_compile_pipeline_info<
        DrawVisbuffer_WriteCommand, DRAW_VISBUFFER_SHADER_PATH, DrawVisbufferPush_WriteCommand>();
}

inline daxa::RasterPipelineCompileInfo draw_visbuffer_no_mesh_shader_pipeline_opaque_compile_info()
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.depth_test = DRAW_VISBUFFER_DEPTH_TEST_INFO;
    ret.color_attachments = DRAW_VISBUFFER_RENDER_ATTACHMENT_INFOS;
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"NO_MESH_SHADER", "1"}, {"OPAQUE", "1"}}},
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"NO_MESH_SHADER", "1"}, {"OPAQUE", "1"}}},
    };
    ret.name = "DrawVisbufferOpaque";
    ret.push_constant_size = s_cast<u32>(sizeof(DrawVisbufferPush) + DrawVisbuffer::attachment_shader_data_size());
    return ret;
};

inline daxa::RasterPipelineCompileInfo draw_visbuffer_no_mesh_shader_pipeline_discard_compile_info()
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.depth_test = DRAW_VISBUFFER_DEPTH_TEST_INFO;
    ret.color_attachments = DRAW_VISBUFFER_RENDER_ATTACHMENT_INFOS;
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"NO_MESH_SHADER", "1"}, {"DISCARD", "1"}}},
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"NO_MESH_SHADER", "1"}, {"DISCARD", "1"}}},
    };
    ret.name = "DrawVisbufferDiscard";
    ret.push_constant_size = s_cast<u32>(sizeof(DrawVisbufferPush) + DrawVisbuffer::attachment_shader_data_size());
    return ret;
};

inline daxa::RasterPipelineCompileInfo draw_visbuffer_mesh_shader_pipeline_compile_info()
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.depth_test = DRAW_VISBUFFER_DEPTH_TEST_INFO;
    ret.color_attachments = DRAW_VISBUFFER_RENDER_ATTACHMENT_INFOS;
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"MESH_SHADER", "1"}}},
    };
    ret.mesh_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"MESH_SHADER", "1"}}},
    };
    ret.name = "DrawVisbufferMeshShader";
    // TODO(msakmary + pahrens) I have a very strong suspicion this is broken - why is mesh shader pipeline using the Draw Visbuffer push constant
    // and not a Mesh shader version of the push constant???
    ret.push_constant_size = s_cast<u32>(sizeof(DrawVisbufferPush) + DrawVisbuffer::attachment_shader_data_size());
    return ret;
};

inline daxa::RasterPipelineCompileInfo draw_visbuffer_mesh_shader_cull_and_draw_pipeline_compile_info()
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.depth_test = DRAW_VISBUFFER_DEPTH_TEST_INFO;
    ret.color_attachments = DRAW_VISBUFFER_RENDER_ATTACHMENT_INFOS;
    auto comp_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"MESH_SHADER_CULL_AND_DRAW", "1"}}},
    };
    ret.fragment_shader_info = comp_info;
    ret.mesh_shader_info = comp_info;
    ret.task_shader_info = comp_info;
    ret.name = "DrawVisbuffer_MeshShader";
    ret.push_constant_size = s_cast<u32>(sizeof(DrawVisbufferPush_MeshShader) + DrawVisbuffer_MeshShader::attachment_shader_data_size());
    return ret;
};

struct DrawVisbufferTask : DrawVisbuffer
{
    DrawVisbuffer::AttachmentViews views = {};
    GPUContext * context = {};
    u32 pass = {};
    bool mesh_shader = {};
    void callback(daxa::TaskInterface ti)
    {
        bool const clear_images = pass != PASS1_DRAW_POST_CULL;
        auto [x, y, z] = ti.device.info_image(ti.get(DrawVisbuffer::depth_image).ids[0]).value().size;
        auto load_op = clear_images ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD;
        daxa::RenderPassBeginInfo render_pass_begin_info{
            .depth_attachment =
                daxa::RenderAttachmentInfo{
                    .image_view = ti.get(DrawVisbuffer::depth_image).ids[0].default_view(),
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = load_op,
                    .store_op = daxa::AttachmentStoreOp::STORE,
                    .clear_value = daxa::DepthValue{0.0f, 0},
                },
            .render_area = daxa::Rect2D{.width = x, .height = y},
        };
        render_pass_begin_info.color_attachments = {
            daxa::RenderAttachmentInfo{
                .image_view = ti.get(DrawVisbuffer::vis_image).ids[0].default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = std::array<u32, 4>{INVALID_TRIANGLE_ID, 0, 0, 0},
            },
            //daxa::RenderAttachmentInfo{
            //    .image_view = ti.get(DrawVisbuffer::debug_image).ids[0].default_view(),
            //    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
            //    .load_op = load_op,
            //    .store_op = daxa::AttachmentStoreOp::STORE,
            //    .clear_value = daxa::ClearValue{std::array<u32, 4>{0, 0, 0, 0}},
            //},
        };
        auto render_cmd = std::move(ti.recorder).begin_renderpass(render_pass_begin_info);
        for (u32 opaque_or_discard = 0; opaque_or_discard < 2; ++opaque_or_discard)
        {
            if (mesh_shader)
            {
                render_cmd.set_pipeline(*context->raster_pipelines.at(draw_visbuffer_mesh_shader_pipeline_compile_info().name));
            }
            else
            {
                if (opaque_or_discard == 0)
                {
                    render_cmd.set_pipeline(*context->raster_pipelines.at(draw_visbuffer_no_mesh_shader_pipeline_opaque_compile_info().name));
                }
                else
                {
                    render_cmd.set_pipeline(*context->raster_pipelines.at(draw_visbuffer_no_mesh_shader_pipeline_discard_compile_info().name));
                }
            }

            render_cmd.push_constant_vptr({
                .data = ti.attachment_shader_data.data(),
                .size = ti.attachment_shader_data.size(),
            });
            render_cmd.push_constant(DrawVisbufferPush{.pass = pass}, DrawVisbuffer::attachment_shader_data_size());
            if (mesh_shader)
            {
                render_cmd.draw_mesh_tasks_indirect({
                    .indirect_buffer = ti.get(DrawVisbuffer::draw_commands).ids[0],
                    .offset = sizeof(DrawIndirectStruct) * opaque_or_discard,
                    .draw_count = 1,
                    .stride = sizeof(DispatchIndirectStruct),
                });
            }
            else
            {
                render_cmd.draw_indirect({
                    .draw_command_buffer = ti.get(DrawVisbuffer::draw_commands).ids[0],
                    .indirect_buffer_offset = sizeof(DrawIndirectStruct) * opaque_or_discard,
                    .draw_count = 1,
                    .draw_command_stride = sizeof(DrawIndirectStruct),
                });
            }
        }
        ti.recorder = std::move(render_cmd).end_renderpass();
    }
};

struct CullAndDrawVisbufferTask : DrawVisbuffer_MeshShader
{
    DrawVisbuffer_MeshShader::AttachmentViews views = {};
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
    GPUContext * context = {};
    daxa::TaskGraph & tg;
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
    daxa::TaskBufferView visible_meshlet_instances = {};
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskImageView vis_image = {};
    daxa::TaskImageView debug_image = {};
    daxa::TaskImageView depth_image = {};
};
inline void task_cull_and_draw_visbuffer(TaskCullAndDrawVisbufferInfo const & info)
{
    if (info.enable_mesh_shader)
    {
#if 0
        info.tg.add_task(CullAndDrawVisbufferTask{
            .views = std::array{
                daxa::TaskViewVariant{std::pair{CullAndDrawVisbufferTask::globals, info.context->tshader_globals_buffer}},
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
#endif
    }
    else
    {
        auto draw_commands_array = info.tg.create_transient_buffer({
            .size = static_cast<u32>(std::max(sizeof(DrawIndirectStruct), sizeof(DispatchIndirectStruct)) * 2),
            .name = std::string("draw visbuffer command buffer array") + info.context->dummy_string(),
        });
        task_fill_buffer(
            info.tg, 
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
        info.tg.add_task(CullMeshletsTask{
            .views = std::array{
                daxa::attachment_view(CullMeshletsTask::globals, info.context->tshader_globals_buffer),
                daxa::attachment_view(CullMeshletsTask::hiz, info.hiz),
                daxa::attachment_view(CullMeshletsTask::meshlets_cull_arg_buckets_buffer, info.meshlets_cull_arg_buckets_buffer_opaque),
                daxa::attachment_view(CullMeshletsTask::entity_meta_data, info.entity_meta_data),
                daxa::attachment_view(CullMeshletsTask::entity_meshgroups, info.entity_meshgroups),
                daxa::attachment_view(CullMeshletsTask::meshgroups, info.mesh_groups),
                daxa::attachment_view(CullMeshletsTask::entity_combined_transforms, info.entity_combined_transforms),
                daxa::attachment_view(CullMeshletsTask::meshes, info.meshes),
                daxa::attachment_view(CullMeshletsTask::first_pass_meshlets_bitfield_offsets, info.first_pass_meshlets_bitfield_offsets),
                daxa::attachment_view(CullMeshletsTask::first_pass_meshlets_bitfield_arena, info.first_pass_meshlets_bitfield_arena),
                daxa::attachment_view(CullMeshletsTask::meshlet_instances, info.meshlet_instances),
                daxa::attachment_view(CullMeshletsTask::draw_commands, draw_commands_array),
            },
            .context = info.context,
            .opaque_or_discard = 0,
        });
        info.tg.add_task(CullMeshletsTask{
            .views = std::array{
                daxa::attachment_view(CullMeshletsTask::globals, info.context->tshader_globals_buffer),
                daxa::attachment_view(CullMeshletsTask::hiz, info.hiz),
                daxa::attachment_view(CullMeshletsTask::meshlets_cull_arg_buckets_buffer, info.meshlets_cull_arg_buckets_buffer_discard),
                daxa::attachment_view(CullMeshletsTask::entity_meta_data, info.entity_meta_data),
                daxa::attachment_view(CullMeshletsTask::entity_meshgroups, info.entity_meshgroups),
                daxa::attachment_view(CullMeshletsTask::meshgroups, info.mesh_groups),
                daxa::attachment_view(CullMeshletsTask::entity_combined_transforms, info.entity_combined_transforms),
                daxa::attachment_view(CullMeshletsTask::meshes, info.meshes),
                daxa::attachment_view(CullMeshletsTask::first_pass_meshlets_bitfield_offsets, info.first_pass_meshlets_bitfield_offsets),
                daxa::attachment_view(CullMeshletsTask::first_pass_meshlets_bitfield_arena, info.first_pass_meshlets_bitfield_arena),
                daxa::attachment_view(CullMeshletsTask::meshlet_instances, info.meshlet_instances),
                daxa::attachment_view(CullMeshletsTask::draw_commands, draw_commands_array),
            },
            .context = info.context,
            .opaque_or_discard = 1,
        });
        DrawVisbufferTask draw_task = {
            .views = std::array{
                daxa::attachment_view(DrawVisbufferTask::globals, info.context->tshader_globals_buffer),
                daxa::attachment_view(DrawVisbufferTask::draw_commands, draw_commands_array),
                daxa::attachment_view(DrawVisbufferTask::meshlet_instances, info.meshlet_instances),
                daxa::attachment_view(DrawVisbufferTask::meshes, info.meshes),
                daxa::attachment_view(DrawVisbufferTask::entity_combined_transforms, info.entity_combined_transforms),
                daxa::attachment_view(DrawVisbufferTask::vis_image, info.vis_image),
                daxa::attachment_view(DrawVisbufferTask::debug_image, info.debug_image),
                daxa::attachment_view(DrawVisbufferTask::depth_image, info.depth_image),
                daxa::attachment_view(DrawVisbufferTask::material_manifest, info.material_manifest),
                daxa::attachment_view(DrawVisbufferTask::visible_meshlet_instances, info.visible_meshlet_instances),
            },
            .context = info.context,
            .pass = PASS1_DRAW_POST_CULL,
            .mesh_shader = false,
        };
        info.tg.add_task(draw_task);
    }
}

struct TaskDrawVisbufferInfo
{
    GPUContext * context = {};
    daxa::TaskGraph & tg;
    bool const enable_mesh_shader = {};
    u32 const pass = {};
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView material_manifest = {};
    daxa::TaskBufferView combined_transforms = {};
    daxa::TaskBufferView visible_meshlet_instances = {};
    daxa::TaskImageView vis_image = {};
    daxa::TaskImageView debug_image = {};
    daxa::TaskImageView depth_image = {};
};

inline void task_draw_visbuffer(TaskDrawVisbufferInfo const & info)
{
    auto draw_commands_array = info.tg.create_transient_buffer({
        .size = 2 * static_cast<u32>(std::max(sizeof(DrawIndirectStruct), sizeof(DispatchIndirectStruct))),
        .name = std::string("draw visbuffer command buffer array") + info.context->dummy_string(),
    });

    DrawVisbuffer_WriteCommandTask write_task = {
        .views = std::array{
            daxa::attachment_view(DrawVisbuffer_WriteCommandTask::globals, info.context->tshader_globals_buffer),
            daxa::attachment_view(DrawVisbuffer_WriteCommandTask::meshlet_instances, info.meshlet_instances),
            daxa::attachment_view(DrawVisbuffer_WriteCommandTask::draw_commands, draw_commands_array),
        },
        .context = info.context,
        .push = DrawVisbufferPush_WriteCommand{.pass = info.pass, .mesh_shader = info.enable_mesh_shader ? 1u : 0u},
    };
    info.tg.add_task(write_task);

    DrawVisbufferTask draw_task = {
        .views = std::array{
            daxa::attachment_view(DrawVisbufferTask::globals, info.context->tshader_globals_buffer),
            daxa::attachment_view(DrawVisbufferTask::draw_commands, draw_commands_array),
            daxa::attachment_view(DrawVisbufferTask::meshlet_instances, info.meshlet_instances),
            daxa::attachment_view(DrawVisbufferTask::meshes, info.meshes),
            daxa::attachment_view(DrawVisbufferTask::material_manifest, info.material_manifest),
            daxa::attachment_view(DrawVisbufferTask::entity_combined_transforms, info.combined_transforms),
            daxa::attachment_view(DrawVisbufferTask::vis_image, info.vis_image),
            daxa::attachment_view(DrawVisbufferTask::debug_image, info.debug_image),
            daxa::attachment_view(DrawVisbufferTask::depth_image, info.depth_image),
            daxa::attachment_view(DrawVisbufferTask::visible_meshlet_instances, info.visible_meshlet_instances),
        },
        .context = info.context,
        .pass = info.pass,
        .mesh_shader = info.enable_mesh_shader,
    };
    info.tg.add_task(draw_task);
}
#endif