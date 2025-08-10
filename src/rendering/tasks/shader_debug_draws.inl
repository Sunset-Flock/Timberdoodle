#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/debug.inl"

DAXA_DECL_RASTER_TASK_HEAD_BEGIN(DebugDrawH)
DAXA_TH_BUFFER_PTR(VS::READ_WRITE, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

#define DEBUG_OBJECT_DRAW_MODE_LINE 0
#define DEBUG_OBJECT_DRAW_MODE_CIRCLE 1
#define DEBUG_OBJECT_DRAW_MODE_RECTANGLE 2
#define DEBUG_OBJECT_DRAW_MODE_AABB 3
#define DEBUG_OBJECT_DRAW_MODE_BOX 4
#define DEBUG_OBJECT_DRAW_MODE_CONE 5
#define DEBUG_OBJECT_DRAW_MODE_SPHERE 6

#define DEBUG_OBJECT_LINE_VERTICES 2
#define DEBUG_OBJECT_CIRCLE_VERTICES 128
#define DEBUG_OBJECT_RECTANGLE_VERTICES 8
#define DEBUG_OBJECT_AABB_VERTICES 24
#define DEBUG_OBJECT_BOX_VERTICES 24
#define DEBUG_OBJECT_CONE_VERTICES 64
#define DEBUG_OBJECT_SPHERE_VERTICES (5*64)

struct DebugDrawPush
{
    DebugDrawH::AttachmentShaderBlob attachments;
    daxa::u32 draw_as_observer;
    daxa::u32 mode;
};

#if defined(__cplusplus)

#include "../scene_renderer_context.hpp"
#include "../../daxa_helper.hpp"

static constexpr inline char const DRAW_SHADER_DEBUG_PATH[] = "./src/rendering/tasks/shader_debug_draws.hlsl";

inline daxa::RasterPipelineCompileInfo draw_shader_debug_lines_pipeline_compile_info()
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.depth_test = {
        .depth_attachment_format = daxa::Format::D32_SFLOAT,
        .enable_depth_write = false,
        .depth_test_compare_op = daxa::CompareOp::GREATER,
        .min_depth_bounds = 0.0f,
        .max_depth_bounds = 1.0f,
    };
    ret.color_attachments = std::vector{
        daxa::RenderAttachment{
            .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
        },
    };
    ret.raster = {
        .primitive_topology = daxa::PrimitiveTopology::LINE_STRIP,
        .primitive_restart_enable = {},
        .polygon_mode = daxa::PolygonMode::FILL,
        .face_culling = daxa::FaceCullFlagBits::NONE,
        .front_face_winding = daxa::FrontFaceWinding::CLOCKWISE,
        .depth_clamp_enable = {},
        .rasterizer_discard_enable = {},
        .depth_bias_enable = {},
        .depth_bias_constant_factor = 0.0f,
        .depth_bias_clamp = 0.0f,
        .depth_bias_slope_factor = 0.0f,
        .line_width = 1.7f,
    };
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {
            .entry_point = "entry_fragment",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {
            .entry_point = "entry_vertex_line",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "DrawShaderDebugLines";
    ret.push_constant_size = sizeof(DebugDrawPush);
    ret.raster.primitive_topology = daxa::PrimitiveTopology::LINE_LIST;
    return ret;
};

struct DebugDrawTask : DebugDrawH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto const colorImageSize = ti.info(AT.color_image).value().size;
        daxa::RenderPassBeginInfo render_pass_begin_info{
            .depth_attachment =
                daxa::RenderAttachmentInfo{
                    .image_view = ti.get(AT.depth_image).view_ids[0],
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = daxa::AttachmentLoadOp::LOAD,
                    .store_op = daxa::AttachmentStoreOp::STORE,
                    .clear_value = daxa::DepthValue{0.0f, 0},
                },
            .render_area = daxa::Rect2D{.width = colorImageSize.x, .height = colorImageSize.y},
        };
        render_pass_begin_info.color_attachments = {
            daxa::RenderAttachmentInfo{
                .image_view = ti.get(AT.color_image).view_ids[0],
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = daxa::AttachmentLoadOp::LOAD,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::ClearValue{std::array<u32, 4>{0, 0, 0, 0}},
            },
        };
        auto render_cmd = std::move(ti.recorder).begin_renderpass(render_pass_begin_info);

        DebugDrawPush push{
            .attachments = ti.attachment_shader_blob,
            .draw_as_observer = render_context->render_data.settings.draw_from_observer,
        };

        render_cmd.set_pipeline(*render_context->gpu_context->raster_pipelines.at(draw_shader_debug_lines_pipeline_compile_info().name));
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_LINE;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, line_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_CIRCLE;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, circle_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_RECTANGLE;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, rectangle_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_AABB;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, aabb_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_BOX;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, box_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_CONE;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, cone_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_SPHERE;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, sphere_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }

        ti.recorder = std::move(render_cmd).end_renderpass();
    }
};

#endif // #if __cplusplus