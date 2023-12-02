#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"

DAXA_DECL_TASK_HEAD_BEGIN(DrawVisbuffer_WriteCommand)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstances), instantiated_meshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_u64, command)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(DrawVisbuffer)
// When drawing triangles, this draw command has triangle ids appended to the end of the command.
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_u64, command)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(MeshletInstances), instantiated_meshlets)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_IMAGE_ID(COLOR_ATTACHMENT, REGULAR_2D, vis_image)
DAXA_TH_IMAGE_NO_SHADER(COLOR_ATTACHMENT, debug_image)
DAXA_TH_IMAGE_NO_SHADER(DEPTH_ATTACHMENT, depth_image)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(DrawVisbuffer_MeshShader)
// When drawing triangles, this draw command has triangle ids appended to the end of the command.
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_u64, command)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(MeshletCullIndirectArgTable), meshlet_cull_indirect_args)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshgroups)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMeshGroup), meshgroups)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, EntityMeshletVisibilityBitfieldOffsetsView, entity_meshlet_visibility_bitfield_offsets)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshlet_visibility_bitfield_arena)
DAXA_TH_IMAGE_ID(GRAPHICS_SHADER_SAMPLED, REGULAR_2D, hiz)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE, daxa_RWBufferPtr(MeshletInstances), instantiated_meshlets)
DAXA_TH_IMAGE_NO_SHADER(COLOR_ATTACHMENT, vis_image)
DAXA_TH_IMAGE_NO_SHADER(COLOR_ATTACHMENT, debug_image)
DAXA_TH_IMAGE_NO_SHADER(DEPTH_ATTACHMENT, depth_image)
DAXA_DECL_TASK_HEAD_END

#define DRAW_VISBUFFER_PASS_ONE 0
#define DRAW_VISBUFFER_PASS_TWO 1
#define DRAW_VISBUFFER_PASS_OBSERVER 2

struct DrawVisbufferPush_WriteCommand
{
    daxa_BufferPtr(ShaderGlobals) globals;
    DrawVisbuffer_WriteCommand uses;
    daxa_u32 pass;
    daxa_u32 mesh_shader;
};

struct DrawVisbufferPush
{
    daxa_BufferPtr(ShaderGlobals) globals;
    DrawVisbuffer uses;
    daxa_u32 pass;
};

struct DrawVisbufferPush_MeshShader
{
    daxa_BufferPtr(ShaderGlobals) globals;
    DrawVisbuffer_MeshShader uses;
    daxa_u32 bucket_index;
};

#if __cplusplus
#include "../../gpu_context.hpp"
#include "../tasks/misc.hpp"
#include "cull_meshlets.inl"

static constexpr inline char const DRAW_VISBUFFER_SHADER_PATH[] =
    "./src/rendering/rasterize_visbuffer/draw_visbuffer.glsl";

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
    daxa::RenderAttachment{
        .format = daxa::Format::R16G16B16A16_SFLOAT,
    },
};

using DrawVisbuffer_WriteCommandTask = WriteIndirectDispatchArgsPushBaseTask<
    DrawVisbuffer_WriteCommand,
    DRAW_VISBUFFER_SHADER_PATH,
    DrawVisbufferPush_WriteCommand>;

inline static const daxa::RasterPipelineCompileInfo DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_NO_MESH_SHADER = []()
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.depth_test = DRAW_VISBUFFER_DEPTH_TEST_INFO;
    ret.color_attachments = DRAW_VISBUFFER_RENDER_ATTACHMENT_INFOS;
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"NO_MESH_SHADER", "1"}}},
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"NO_MESH_SHADER", "1"}}},
    };
    ret.name = "DrawVisbuffer";
    ret.push_constant_size = sizeof(DrawVisbufferPush);
    return ret;
}();

inline static const daxa::RasterPipelineCompileInfo DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER = []()
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
    ret.push_constant_size = sizeof(DrawVisbufferPush);
    return ret;
}();

inline static const daxa::RasterPipelineCompileInfo DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER_CULL_AND_DRAW = []()
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.depth_test = DRAW_VISBUFFER_DEPTH_TEST_INFO;
    ret.color_attachments = DRAW_VISBUFFER_RENDER_ATTACHMENT_INFOS;
    auto comp_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .write_out_preprocessed_code = "./preproc",
            .defines = {{"MESH_SHADER_CULL_AND_DRAW", "1"}},
        },
    };
    ret.fragment_shader_info = comp_info;
    ret.mesh_shader_info = comp_info;
    ret.task_shader_info = comp_info;
    ret.name = "DrawVisbuffer_MeshShader";
    ret.push_constant_size = sizeof(DrawVisbufferPush_MeshShader);
    return ret;
}();

struct DrawVisbufferTask
{
    USE_TASK_HEAD(DrawVisbuffer)
    inline static const daxa::RasterPipelineCompileInfo PIPELINE_COMPILE_INFO = DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_NO_MESH_SHADER;
    GPUContext *context = {};
    u32 pass = {};
    bool mesh_shader = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::ImageId vis_image = uses.vis_image.image();
        daxa::ImageId depth_image = uses.depth_image.image();
        daxa::ImageId debug_image = uses.debug_image.image();
        auto & cmd = ti.get_recorder();
        const bool clear_images = pass == DRAW_VISBUFFER_PASS_ONE || pass == DRAW_VISBUFFER_PASS_OBSERVER;
        auto load_op = clear_images ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD;
        daxa::RenderPassBeginInfo render_pass_begin_info{
            .depth_attachment = daxa::RenderAttachmentInfo{
                .image_view = depth_image.default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::DepthValue{0.0f, 0},
            },
            .render_area = daxa::Rect2D{
                .width = (ti.get_device().info_image(depth_image).value().size.x),
                .height = (ti.get_device().info_image(depth_image).value().size.y),
            },
        };
        render_pass_begin_info.color_attachments = {
            daxa::RenderAttachmentInfo{
                .image_view = vis_image.default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = std::array<u32, 4>{INVALID_TRIANGLE_ID, 0, 0, 0},
            },
            daxa::RenderAttachmentInfo{
                .image_view = debug_image.default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::ClearValue{std::array<u32, 4>{0, 0, 0, 0}},
            },
        };
        auto render_cmd = std::move(cmd).begin_renderpass(render_pass_begin_info);
        if (mesh_shader)
        {
            render_cmd.set_pipeline(*context->raster_pipelines.at(DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER.name));
        }
        else
        {
            render_cmd.set_pipeline(*context->raster_pipelines.at(PIPELINE_COMPILE_INFO.name));
        }
        DrawVisbufferPush push{
            .globals = context->shader_globals_address,
            .pass = pass,
        };
        ti.copy_task_head_to(&push.uses);
        render_cmd.push_constant(push);
        if (mesh_shader)
        {
            render_cmd.draw_mesh_tasks_indirect({
                .indirect_buffer = uses.command.buffer(),
                .offset = 0,
                .draw_count = 1,
                .stride = sizeof(DispatchIndirectStruct),
            });
        }
        else
        {
            render_cmd.draw_indirect({
                .draw_command_buffer = uses.command.buffer(),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
            });
        }
        cmd = std::move(render_cmd).end_renderpass();
    }
};

struct CullAndDrawVisbufferTask
{
    USE_TASK_HEAD(DrawVisbuffer_MeshShader)
    inline static const daxa::RasterPipelineCompileInfo PIPELINE_COMPILE_INFO = DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER_CULL_AND_DRAW;
    GPUContext *context = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::ImageId vis_image = uses.vis_image.image();
        daxa::ImageId depth_image = uses.depth_image.image();
        daxa::ImageId debug_image = uses.debug_image.image();
        auto & cmd = ti.get_recorder();
        const bool clear_images = false;
        auto load_op = clear_images ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD;
        daxa::RenderPassBeginInfo render_pass_begin_info{
            .depth_attachment = daxa::RenderAttachmentInfo{
                .image_view = depth_image.default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::ClearValue{daxa::DepthValue{0.0f, 0}},
            },
            .render_area = daxa::Rect2D{
                .width = (ti.get_device().info_image(depth_image).value().size.x),
                .height = (ti.get_device().info_image(depth_image).value().size.y),
            },
        };
        render_pass_begin_info.color_attachments = {
            daxa::RenderAttachmentInfo{
                .image_view = vis_image.default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::ClearValue{std::array<u32, 4>{INVALID_TRIANGLE_ID, 0, 0, 0}},
            },
            daxa::RenderAttachmentInfo{
                .image_view = debug_image.default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::ClearValue{std::array<u32, 4>{0, 0, 0, 0}},
            },
        };
        auto render_cmd = std::move(cmd).begin_renderpass(render_pass_begin_info);
        render_cmd.set_pipeline(*context->raster_pipelines.at(DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER_CULL_AND_DRAW.name));
        for (u32 i = 0; i < 32; ++i)
        {
            DrawVisbufferPush_MeshShader push = {
                .globals = context->shader_globals_address,
                .bucket_index = i
            };
            ti.copy_task_head_to(&push.uses);
            render_cmd.push_constant(push);
            render_cmd.draw_mesh_tasks_indirect({
                .indirect_buffer = uses.command.buffer(),
                .offset = sizeof(DispatchIndirectStruct) * i,
                .draw_count = 1,
                .stride = sizeof(DispatchIndirectStruct),
            });
        }
        cmd = std::move(render_cmd).end_renderpass();
    }
};

struct TaskCullAndDrawVisbufferInfo
{
    GPUContext *context = {};
    daxa::TaskGraph & tg;
    const bool enable_mesh_shader = {};
    daxa::TaskBufferView cull_meshlets_commands = {};
    daxa::TaskBufferView meshlet_cull_indirect_args = {};
    daxa::TaskBufferView entity_meta_data = {};
    daxa::TaskBufferView entity_meshgroups = {};
    daxa::TaskBufferView entity_combined_transforms = {};
    daxa::TaskBufferView mesh_groups = {};
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView entity_meshlet_visibility_bitfield_offsets = {};
    daxa::TaskBufferView entity_meshlet_visibility_bitfield_arena = {};
    daxa::TaskImageView hiz = {};
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskImageView vis_image = {};
    daxa::TaskImageView debug_image = {};
    daxa::TaskImageView depth_image = {};
};
inline void task_cull_and_draw_visbuffer(TaskCullAndDrawVisbufferInfo const & info)
{
    if (info.enable_mesh_shader)
    {
        info.tg.add_task(CullAndDrawVisbufferTask{
            .uses = {
                .command = info.cull_meshlets_commands,
                .meshlet_cull_indirect_args = info.meshlet_cull_indirect_args,
                .instantiated_meshlets = info.meshlet_instances,
                .meshes = info.meshes,
                .entity_meta = info.entity_meta_data,
                .entity_meshgroups = info.entity_meshgroups,
                .meshgroups = info.mesh_groups,
                .entity_combined_transforms = info.entity_combined_transforms,
                .entity_meshlet_visibility_bitfield_offsets = info.entity_meshlet_visibility_bitfield_offsets,
                .entity_meshlet_visibility_bitfield_arena = info.entity_meshlet_visibility_bitfield_arena,
                .hiz = info.hiz,
                .vis_image = info.vis_image,
                .debug_image = info.debug_image,
                .depth_image = info.depth_image,
            },
            .context = info.context,
        });
    }
    else
    {
        auto draw_command = info.tg.create_transient_buffer({
            .size = static_cast<u32>(std::max(sizeof(DrawIndirectStruct), sizeof(DispatchIndirectStruct))),
            .name = std::string("draw visbuffer command buffer") + info.context->dummy_string(),
        });
        // clear to zero, rest of values will be initialized by CullMeshletsTask.
        task_clear_buffer(info.tg, draw_command, 0);
        info.tg.add_task(CullMeshletsTask{
            .uses = {
                .hiz = info.hiz,
                .commands = info.cull_meshlets_commands,
                .meshlet_cull_indirect_args = info.meshlet_cull_indirect_args,
                .entity_meta_data = info.entity_meta_data,
                .entity_meshgroups = info.entity_meshgroups,
                .meshgroups = info.mesh_groups,
                .entity_combined_transforms = info.entity_combined_transforms,
                .meshes = info.meshes,
                .entity_meshlet_visibility_bitfield_offsets = info.entity_meshlet_visibility_bitfield_offsets,
                .entity_meshlet_visibility_bitfield_arena = info.entity_meshlet_visibility_bitfield_arena,
                .instantiated_meshlets = info.meshlet_instances,
                .draw_command = draw_command,
            },
            .context = info.context,
        });
        info.tg.add_task(DrawVisbufferTask{
            .uses = {
                .command = draw_command,
                .instantiated_meshlets = info.meshlet_instances,
                .meshes = info.meshes,
                .entity_combined_transforms = info.entity_combined_transforms,
                .vis_image = info.vis_image,
                .debug_image = info.debug_image,
                .depth_image = info.depth_image,
            },
            .context = info.context,
            .pass = DRAW_VISBUFFER_PASS_TWO,
            .mesh_shader = false,
        });
    }
}

struct TaskDrawVisbufferInfo
{
    GPUContext *context = {};
    daxa::TaskGraph & tg;
    DrawVisbuffer::Uses uses = {};
    const bool enable_mesh_shader = {};
    const u32 pass = {};
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView combined_transforms = {};
    daxa::TaskImageView vis_image = {};
    daxa::TaskImageView debug_image = {};
    daxa::TaskImageView depth_image = {};
};
inline void task_draw_visbuffer(TaskDrawVisbufferInfo const & info)
{
    auto draw_command = info.tg.create_transient_buffer({
        .size = static_cast<u32>(std::max(sizeof(DrawIndirectStruct), sizeof(DispatchIndirectStruct))),
        .name = std::string("draw visbuffer command buffer") + info.context->dummy_string(),
    });
    info.tg.add_task(DrawVisbuffer_WriteCommandTask{
        .uses = {
            .instantiated_meshlets = info.meshlet_instances,
            .command = draw_command,
        },
        .context = info.context,
        .push = DrawVisbufferPush_WriteCommand{.pass = info.pass, .mesh_shader = info.enable_mesh_shader ? 1u : 0u},
    });
    info.tg.add_task(DrawVisbufferTask{
        .uses = {
            .command = draw_command,
            .instantiated_meshlets = info.meshlet_instances,
            .meshes = info.meshes,
            .entity_combined_transforms = info.combined_transforms,
            .vis_image = info.vis_image,
            .debug_image = info.debug_image,
            .depth_image = info.depth_image,
        },
        .context = info.context,
        .pass = info.pass,
        .mesh_shader = info.enable_mesh_shader,
    });
}
#endif