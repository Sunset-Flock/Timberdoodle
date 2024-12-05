#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"

DAXA_DECL_TASK_HEAD_BEGIN(DDGIDrawDebugProbesH)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image)
DAXA_TH_IMAGE_TYPED(FRAGMENT_SHADER_SAMPLED, daxa::Texture2DId<daxa_f32>, scene_depth_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, probes_depth_image)
DAXA_DECL_TASK_HEAD_END

struct DDGIDrawDebugProbesPush
{
    DDGIDrawDebugProbesH::AttachmentShaderBlob attach;
    daxa_f32vec3* probe_mesh_positions;
};

#if defined(__cplusplus)

#include "../scene_renderer_context.hpp"
#include "../../daxa_helper.hpp"
#include "ddgi.hpp"

static constexpr inline char const DDGI_SHADER_PATH[] = "./src/rendering/ddgi/ddgi_update.hlsl";
inline daxa::RasterPipelineCompileInfo ddgi_draw_debug_probes_compile_info()
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.color_attachments = std::vector{
        daxa::RenderAttachment{
            .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
        },
    };
    ret.depth_test = {
        .depth_attachment_format = daxa::Format::D32_SFLOAT,
        .enable_depth_write = true,
        .depth_test_compare_op = daxa::CompareOp::GREATER,
        .min_depth_bounds = 0.0f,
        .max_depth_bounds = 1.0f,
    };
    ret.raster = {
        .primitive_topology = daxa::PrimitiveTopology::TRIANGLE_LIST,
        .primitive_restart_enable = {},
        .polygon_mode = daxa::PolygonMode::FILL,
        .face_culling = daxa::FaceCullFlagBits::BACK_BIT,
        .front_face_winding = daxa::FrontFaceWinding::COUNTER_CLOCKWISE,
        .depth_clamp_enable = {},
        .rasterizer_discard_enable = {},
        .depth_bias_enable = {},
        .depth_bias_constant_factor = 0.0f,
        .depth_bias_clamp = 0.0f,
        .depth_bias_slope_factor = 0.0f,
        .line_width = 1.0f,
    };
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DDGI_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_fragment_draw_debug_probes",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DDGI_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_vertex_draw_debug_probes",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.push_constant_size = sizeof(DDGIDrawDebugProbesPush);
    ret.name = "DDGIDrawDebugProbes";
    return ret;
}

struct DDGIDrawDebugProbesTask : DDGIDrawDebugProbesH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    DDGIState* ddgi_state = {};
    void callback(daxa::TaskInterface ti)
    {
        auto const colorImageSize = ti.device.image_info(ti.get(AT.color_image).ids[0]).value().size;
        daxa::RenderPassBeginInfo render_pass_begin_info{
            .depth_attachment =
                daxa::RenderAttachmentInfo{
                    .image_view = ti.get(AT.probes_depth_image).view_ids[0],
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = daxa::AttachmentLoadOp::CLEAR,
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

        render_cmd.set_pipeline(*render_context->gpu_context->raster_pipelines.at(ddgi_draw_debug_probes_compile_info().name));

        DDGIDrawDebugProbesPush push{
            .attach = ti.attachment_shader_blob,
            .probe_mesh_positions = ddgi_state->debug_probe_mesh_vertex_positions_addr,
        };
        render_cmd.push_constant(push);

        render_cmd.set_index_buffer({
            .id = ddgi_state->debug_probe_mesh_buffer,
        }); 

        render_cmd.draw_indexed({
            .index_count = ddgi_state->debug_probe_mesh_triangles * 3,
            .instance_count = static_cast<u32>(render_context->render_data.ddgi_settings.probe_count.x * render_context->render_data.ddgi_settings.probe_count.y * render_context->render_data.ddgi_settings.probe_count.z),
        });

        ti.recorder = std::move(render_cmd).end_renderpass();
    }
};

struct TaskDDgiDrawDebugProbesInfo
{
    daxa::TaskGraph & tg;
    RenderContext * render_context = {};
    DDGIState * ddgi_state = {};
    daxa::TaskImageView color_image = {};
    daxa::TaskImageView depth_image = {};
};
void task_ddgi_draw_debug_probes(TaskDDgiDrawDebugProbesInfo const & info)
{
    daxa::TaskImageView debug_draw_depth = info.tg.create_transient_image({
        .format = daxa::Format::D32_SFLOAT,
        .size = { info.render_context->render_data.settings.render_target_size.x, info.render_context->render_data.settings.render_target_size.y, 1},
        .name = "ddgi draw debug probes depth image",
    });

    info.tg.add_task(DDGIDrawDebugProbesTask{
        .views = std::array{
            DDGIDrawDebugProbesH::AT.globals | info.render_context->tgpu_render_data,
            DDGIDrawDebugProbesH::AT.color_image | info.color_image,
            DDGIDrawDebugProbesH::AT.scene_depth_image | info.depth_image,
            DDGIDrawDebugProbesH::AT.probes_depth_image | debug_draw_depth,
        },
        .render_context = info.render_context,
        .ddgi_state = info.ddgi_state,
    });
}

#endif