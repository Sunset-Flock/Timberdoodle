#pragma once

#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"

DAXA_DECL_TASK_HEAD_BEGIN(DDGIDrawDebugProbesH)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_TYPED(GRAPHICS_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_radiance)
DAXA_DECL_TASK_HEAD_END

struct DDGIDrawDebugProbesPush
{
    DDGIDrawDebugProbesH::AttachmentShaderBlob attach;
    daxa_f32vec3* probe_mesh_positions;
};

#define DDGI_UPDATE_WG_XYZ 4

DAXA_DECL_TASK_HEAD_BEGIN(DDGIUpdateProbesH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, probe_radiance)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky_transmittance)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky)
DAXA_TH_TLAS_ID(COMPUTE_SHADER_READ, tlas)
DAXA_DECL_TASK_HEAD_END

struct DDGIUpdateProbesPush
{
    DDGIUpdateProbesH::AttachmentShaderBlob attach;
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



inline auto ddgi_update_probes_compute_compile_info() -> daxa::ComputePipelineCompileInfo2
{
    return daxa::ComputePipelineCompileInfo2{
        .source = daxa::ShaderFile{"./src/rendering/ddgi/ddgi_update.hlsl"},
        .entry_point = "entry_update_probes",
        .language = daxa::ShaderLanguage::SLANG,
        .push_constant_size = s_cast<u32>(sizeof(DDGIUpdateProbesPush)),
        .name = std::string{DDGIUpdateProbesH::NAME},
    };
}

struct DDGIUpdateProbesTask : DDGIUpdateProbesH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    DDGIState* ddgi_state = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(ddgi_update_probes_compute_compile_info().name));

        DDGIUpdateProbesPush push = {};
        push.attach = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);

        auto const x = render_context->render_data.ddgi_settings.probe_count.x * render_context->render_data.ddgi_settings.probe_surface_resolution;
        auto const y = render_context->render_data.ddgi_settings.probe_count.y;
        auto const z = render_context->render_data.ddgi_settings.probe_count.z;
        auto const dispatch_x = round_up_div(x, DDGI_UPDATE_WG_XYZ);
        auto const dispatch_y = round_up_div(y, DDGI_UPDATE_WG_XYZ);
        auto const dispatch_z = round_up_div(z, DDGI_UPDATE_WG_XYZ);
        ti.recorder.dispatch({dispatch_x,dispatch_y,dispatch_z});
    }
};


#endif