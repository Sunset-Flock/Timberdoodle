#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/debug.inl"

DAXA_DECL_TASK_HEAD_BEGIN(DebugDrawH, 3)
DAXA_TH_BUFFER_PTR(VERTEX_SHADER_READ_WRITE, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

struct DebugDrawPush
{
    DAXA_TH_BLOB(DebugDrawH, attachments)
    daxa_u32 draw_as_observer;
};

#if __cplusplus

#include "../../gpu_context.hpp"

static constexpr inline char const DRAW_SHADER_DEBUG_PATH[] = "./src/rendering/tasks/shader_debug_draws.glsl";
inline daxa::RasterPipelineCompileInfo draw_shader_debug_common_pipeline_compile_info()
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
            .format = daxa::Format::R8G8B8A8_UNORM,
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
        .line_width = 1.0f,
    };
    return ret;
}

inline daxa::RasterPipelineCompileInfo draw_shader_debug_circles_pipeline_compile_info()
{
    auto ret = draw_shader_debug_common_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {.defines = {{"DRAW_CIRCLE", "1"}}},
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {.defines = {{"DRAW_CIRCLE", "1"}}},
    };
    ret.name = "DrawShaderDebugCircles";
    ret.push_constant_size = sizeof(DebugDrawPush);
    return ret;
};

inline daxa::RasterPipelineCompileInfo draw_shader_debug_rectangles_pipeline_compile_info()
{
    auto ret = draw_shader_debug_common_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {.defines = {{"DRAW_RECTANGLE", "1"}}},
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {.defines = {{"DRAW_RECTANGLE", "1"}}},
    };
    ret.name = "DrawShaderDebugRectangles";
    ret.push_constant_size = sizeof(DebugDrawPush);
    return ret;
};

inline daxa::RasterPipelineCompileInfo draw_shader_debug_aabb_pipeline_compile_info()
{
    auto ret = draw_shader_debug_common_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {.defines = {{"DRAW_AABB", "1"}}},
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {.defines = {{"DRAW_AABB", "1"}}},
    };
    ret.name = "DrawShaderDebugAABB";
    ret.push_constant_size = sizeof(DebugDrawPush);
    ret.raster.primitive_topology = daxa::PrimitiveTopology::LINE_LIST;
    return ret;
};

inline daxa::RasterPipelineCompileInfo draw_shader_debug_box_pipeline_compile_info()
{
    auto ret = draw_shader_debug_common_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {.defines = {{"DRAW_BOX", "1"}}},
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {.defines = {{"DRAW_BOX", "1"}}},
    };
    ret.name = "DrawShaderDebugBox";
    ret.push_constant_size = sizeof(DebugDrawPush);
    ret.raster.primitive_topology = daxa::PrimitiveTopology::LINE_LIST;
    return ret;
};

struct DebugDrawTask : DebugDrawH::Task
{
    AttachmentViews views = {};
    RenderContext * rctx = {};
    void callback(daxa::TaskInterface ti)
    {
        auto const colorImageSize = ti.device.info_image(ti.get(AT.color_image).ids[0]).value().size;
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

        render_cmd.set_pipeline(*rctx->gpuctx->raster_pipelines.at(draw_shader_debug_circles_pipeline_compile_info().name));

        DebugDrawPush push{.draw_as_observer = rctx->render_data.settings.draw_from_observer};
        assign_blob(push.attachments, ti.attachment_shader_blob);
        render_cmd.push_constant(push);

        render_cmd.draw_indirect({
            .draw_command_buffer = rctx->gpuctx->shader_debug_context.buffer,
            .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, circle_draw_indirect_info),
            .draw_count = 1,
            .draw_command_stride = sizeof(DrawIndirectStruct),
            .is_indexed = false,
        });
        render_cmd.set_pipeline(*rctx->gpuctx->raster_pipelines.at(draw_shader_debug_rectangles_pipeline_compile_info().name));
        render_cmd.draw_indirect({
            .draw_command_buffer = rctx->gpuctx->shader_debug_context.buffer,
            .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, rectangle_draw_indirect_info),
            .draw_count = 1,
            .draw_command_stride = sizeof(DrawIndirectStruct),
            .is_indexed = false,
        });
        render_cmd.set_pipeline(*rctx->gpuctx->raster_pipelines.at(draw_shader_debug_aabb_pipeline_compile_info().name));
        render_cmd.draw_indirect({
            .draw_command_buffer = rctx->gpuctx->shader_debug_context.buffer,
            .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, aabb_draw_indirect_info),
            .draw_count = 1,
            .draw_command_stride = sizeof(DrawIndirectStruct),
            .is_indexed = false,
        });
        render_cmd.set_pipeline(*rctx->gpuctx->raster_pipelines.at(draw_shader_debug_box_pipeline_compile_info().name));
        render_cmd.draw_indirect({
            .draw_command_buffer = rctx->gpuctx->shader_debug_context.buffer,
            .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, box_draw_indirect_info),
            .draw_count = 1,
            .draw_command_stride = sizeof(DrawIndirectStruct),
            .is_indexed = false,
        });

        ti.recorder = std::move(render_cmd).end_renderpass();
    }
};

#endif // #if __cplusplus