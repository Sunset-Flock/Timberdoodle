#pragma once
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../gpu_context.hpp"
#include "../../shader_shared/geometry_pipeline.inl"
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/gpu_work_expansion.inl"
#include "../tasks/misc.hpp"
#include "../../scene/scene.hpp"
#include"pgi_update.inl"

// PGI Process:
// - Pre Update All possible probes
//   - clear new probes
//   - invalidate timed out probes
//   - build dispatch indirect structs and indirections for other passes
// - Trace Probe Rays
// - Update Probes (analyze rays)
//   - reposition probes
//   - update validity
// - Update Probe Texels
//   - Using the validity, update probes radiance and visibility texels

struct PGIState
{
    daxa::BufferId debug_probe_mesh_buffer = {};
    daxa::u32 debug_probe_mesh_triangles = {};
    daxa::u32 debug_probe_mesh_vertices = {};
    daxa_f32vec3* debug_probe_mesh_vertex_positions_addr = {};

    daxa::TaskImage probe_radiance = daxa::TaskImage(daxa::TaskImageInfo{.name = "default init pgi probe radiance texture"});
    daxa::TaskImageView probe_radiance_view = daxa::NullTaskImage;
    daxa::TaskImage probe_visibility = daxa::TaskImage(daxa::TaskImageInfo{.name = "default init pgi probe visibility texture"});
    daxa::TaskImageView probe_visibility_view = daxa::NullTaskImage;
    daxa::TaskImage probe_info = daxa::TaskImage(daxa::TaskImageInfo{.name = "default init pgi probe info texture"});
    daxa::TaskImageView probe_info_view = daxa::NullTaskImage;
    daxa::TaskImage cell_requests = daxa::TaskImage(daxa::TaskImageInfo{.name = "default init pgi cell requests texture"});
    daxa::TaskImageView cell_requests_view = daxa::NullTaskImage;

    void initialize(daxa::Device& device);
    void recreate_resources(daxa::Device& device, PGISettings const & settings);
    void cleanup(daxa::Device& device);
};

#include "../scene_renderer_context.hpp"
#include "../../daxa_helper.hpp"

static constexpr inline char const PGI_SHADER_PATH[] = "./src/rendering/pgi/pgi_debug_draw_probes.hlsl";
inline daxa::RasterPipelineCompileInfo pgi_draw_debug_probes_compile_info()
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
        .source = daxa::ShaderFile{PGI_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_fragment_draw_debug_probes",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{PGI_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_vertex_draw_debug_probes",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.push_constant_size = sizeof(PGIDrawDebugProbesPush);
    ret.name = "PGIDrawDebugProbes";
    return ret;
}

struct PGIDrawDebugProbesTask : PGIDrawDebugProbesH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    PGIState* pgi_state = {};
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

        render_cmd.set_pipeline(*render_context->gpu_context->raster_pipelines.at(pgi_draw_debug_probes_compile_info().name));

        PGIDrawDebugProbesPush push{
            .attach = ti.attachment_shader_blob,
            .probe_mesh_positions = pgi_state->debug_probe_mesh_vertex_positions_addr,
        };
        render_cmd.push_constant(push);

        render_cmd.set_index_buffer({
            .id = pgi_state->debug_probe_mesh_buffer,
        }); 

        if (render_context->render_data.pgi_settings.enable_indirect_sparse)
        {
            render_cmd.draw_indirect({
                .draw_command_buffer = ti.get(AT.probe_indirections).ids[0],
                .indirect_buffer_offset = offsetof(PGIIndirections, probe_debug_draw_dispatch),
                .draw_count = 1,
                .is_indexed = true,
            });
        }
        else
        {
            render_cmd.draw_indexed({
                .index_count = pgi_state->debug_probe_mesh_triangles * 3,
                .instance_count = static_cast<u32>(render_context->render_data.pgi_settings.probe_count.x * render_context->render_data.pgi_settings.probe_count.y * render_context->render_data.pgi_settings.probe_count.z),
            });
        }

        ti.recorder = std::move(render_cmd).end_renderpass();
    }
};

inline auto pgi_update_probes_compute_compile_info() -> daxa::ComputePipelineCompileInfo2
{
    return daxa::ComputePipelineCompileInfo2{
        .source = daxa::ShaderFile{"./src/rendering/pgi/pgi_update.hlsl"},
        .entry_point = "entry_update_probe_irradiance",
        .language = daxa::ShaderLanguage::SLANG,
        .push_constant_size = s_cast<u32>(sizeof(PGIUpdateProbeTexelsPush)),
        .name = std::string{"entry_update_probe_irradiance"},
    };
}

inline auto pgi_update_probes_compute_compile_info2() -> daxa::ComputePipelineCompileInfo2
{
    return daxa::ComputePipelineCompileInfo2{
        .source = daxa::ShaderFile{"./src/rendering/pgi/pgi_update.hlsl"},
        .entry_point = "entry_update_probe_visibility",
        .language = daxa::ShaderLanguage::SLANG,
        .push_constant_size = s_cast<u32>(sizeof(PGIUpdateProbeTexelsPush)),
        .name = std::string{"entry_update_probe_visibility"},
    };
}

struct PGIUpdateProbeTexelsTask : PGIUpdateProbeTexelsH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    PGIState* pgi_state = {};
    void callback(daxa::TaskInterface ti)
    {
        PGIUpdateProbeTexelsPush push = {};
        push.attach = ti.attachment_shader_blob;
        {
            ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(pgi_update_probes_compute_compile_info().name));
            ti.recorder.push_constant(push);

            if (render_context->render_data.pgi_settings.enable_indirect_sparse)
            {
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(AT.probe_indirections).ids[0],
                    .offset = offsetof(PGIIndirections, probe_radiance_update_dispatch),
                });
            }
            else
            {
                auto const x = render_context->render_data.pgi_settings.probe_count.x * render_context->render_data.pgi_settings.probe_radiance_resolution;
                auto const y = render_context->render_data.pgi_settings.probe_count.y * render_context->render_data.pgi_settings.probe_radiance_resolution;
                auto const z = render_context->render_data.pgi_settings.probe_count.z;
                auto const dispatch_x = round_up_div(x, PGI_UPDATE_WG_XY);
                auto const dispatch_y = round_up_div(y, PGI_UPDATE_WG_XY);
                auto const dispatch_z = round_up_div(z, PGI_UPDATE_WG_Z);
                ti.recorder.dispatch({dispatch_x,dispatch_y,dispatch_z});
            }
        }
        {
            ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(pgi_update_probes_compute_compile_info2().name));
            ti.recorder.push_constant(push);
            if (render_context->render_data.pgi_settings.enable_indirect_sparse)
            {
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(AT.probe_indirections).ids[0],
                    .offset = offsetof(PGIIndirections, probe_visibility_update_dispatch),
                });
            }
            else
            {
                auto const x = render_context->render_data.pgi_settings.probe_count.x * render_context->render_data.pgi_settings.probe_visibility_resolution;
                auto const y = render_context->render_data.pgi_settings.probe_count.y * render_context->render_data.pgi_settings.probe_visibility_resolution;
                auto const z = render_context->render_data.pgi_settings.probe_count.z;
                auto const dispatch_x = round_up_div(x, PGI_UPDATE_WG_XY);
                auto const dispatch_y = round_up_div(y, PGI_UPDATE_WG_XY);
                auto const dispatch_z = round_up_div(z, PGI_UPDATE_WG_Z);
                ti.recorder.dispatch({dispatch_x,dispatch_y,dispatch_z});
            }
        }
    }
};

inline auto pgi_update_probes_compute_compile_info3() -> daxa::ComputePipelineCompileInfo2
{
    return daxa::ComputePipelineCompileInfo2{
        .source = daxa::ShaderFile{"./src/rendering/pgi/pgi_update.hlsl"},
        .entry_point = "entry_update_probe",
        .language = daxa::ShaderLanguage::SLANG,
        .push_constant_size = s_cast<u32>(sizeof(PGIUpdateProbesPush)),
        .name = std::string{"entry_update_probe"},
    };
}

struct PGIUpdateProbesTask : PGIUpdateProbesH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    PGIState* pgi_state = {};
    void callback(daxa::TaskInterface ti)
    {
        if ((render_context->render_data.pgi_settings.enable_indirect_sparse != 0) || (render_context->render_data.pgi_settings.debug_probe_draw_mode != 0))
        {
            PGIUpdateProbesPush push = {};
            push.attach = ti.attachment_shader_blob;
            ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(pgi_update_probes_compute_compile_info3().name));
            ti.recorder.push_constant(push);
            push.attach = ti.attachment_shader_blob;
            if (render_context->render_data.pgi_settings.enable_indirect_sparse)
            {
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(AT.probe_indirections).ids[0],
                    .offset = offsetof(PGIIndirections, probe_update_dispatch),
                });
            }
            else
            {
                auto const x = render_context->render_data.pgi_settings.probe_count.x;
                auto const y = render_context->render_data.pgi_settings.probe_count.y;
                auto const z = render_context->render_data.pgi_settings.probe_count.z;
                auto const dispatch_x = round_up_div(x, PGI_UPDATE_WG_XY);
                auto const dispatch_y = round_up_div(y, PGI_UPDATE_WG_XY);
                auto const dispatch_z = round_up_div(z, PGI_UPDATE_WG_Z);
                ti.recorder.dispatch({dispatch_x,dispatch_y,dispatch_z});
            }
        }
    }
};

inline auto pgi_trace_probe_lighting_pipeline_compile_info() -> daxa::RayTracingPipelineCompileInfo
{
    return {
        .ray_gen_infos = {
            {
                .source = daxa::ShaderFile{"./src/rendering/pgi/pgi_trace_probe_lighting.hlsl"},
                .compile_options = {.entry_point = "entry_ray_gen", .language = daxa::ShaderLanguage::SLANG},
            },
        },
        .any_hit_infos = {
            {
                .source = daxa::ShaderFile{"./src/rendering/pgi/pgi_trace_probe_lighting.hlsl"},
                .compile_options = {.entry_point = "entry_any_hit", .language = daxa::ShaderLanguage::SLANG},
            }
        },
        .closest_hit_infos = {
            {
                .source = daxa::ShaderFile{"./src/rendering/pgi/pgi_trace_probe_lighting.hlsl"},
                .compile_options = {.entry_point = "entry_closest_hit", .language = daxa::ShaderLanguage::SLANG},
            },
        },
        .miss_hit_infos = {
            {
                .source = daxa::ShaderFile{"./src/rendering/pgi/pgi_trace_probe_lighting.hlsl"},
                .compile_options = {.entry_point = "entry_miss", .language = daxa::ShaderLanguage::SLANG},
            },
        },
        .shader_groups_infos = {
            // Gen Group
            daxa::RayTracingShaderGroupInfo{
                .type = daxa::ShaderGroup::GENERAL,
                .general_shader_index = 0,
            },
            // Miss group
            daxa::RayTracingShaderGroupInfo{
                .type = daxa::ShaderGroup::GENERAL,
                .general_shader_index = 3,
            },
            // Hit group
            daxa::RayTracingShaderGroupInfo{
                .type = daxa::ShaderGroup::TRIANGLES_HIT_GROUP,
                .closest_hit_shader_index = 2,
            },
            daxa::RayTracingShaderGroupInfo{
                .type = daxa::ShaderGroup::TRIANGLES_HIT_GROUP,
                .closest_hit_shader_index = 2,
                .any_hit_shader_index = 1,
            },
        },
        .max_ray_recursion_depth = 1,
        .push_constant_size = sizeof(PGITraceProbeLightingPush),
        .name = std::string{PGITraceProbeLightingH::NAME},
    };
}

// Traces one ray per probe texel.
// Results (color, depth) are written to the trace result texture.
struct PGITraceProbeRaysTask : PGITraceProbeLightingH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    PGIState* pgi_state = {};
    void callback(daxa::TaskInterface ti)
    {
        auto& pipeline = render_context->gpu_context->ray_tracing_pipelines.at(pgi_trace_probe_lighting_pipeline_compile_info().name);
        ti.recorder.set_pipeline(*pipeline.pipeline);

        PGITraceProbeLightingPush push = {};
        push.attach = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);

        if (render_context->render_data.pgi_settings.enable_indirect_sparse)
        {
            ti.recorder.trace_rays_indirect({
                .indirect_device_address = ti.device_address(AT.probe_indirections).value() + offsetof(PGIIndirections, probe_trace_dispatch),
                .shader_binding_table = pipeline.sbt,
            });
        }
        else
        {
            u32 const x = static_cast<u32>(render_context->render_data.pgi_settings.probe_count.x * render_context->render_data.pgi_settings.probe_trace_resolution);
            u32 const y = static_cast<u32>(render_context->render_data.pgi_settings.probe_count.y * render_context->render_data.pgi_settings.probe_trace_resolution);
            u32 const z = static_cast<u32>(render_context->render_data.pgi_settings.probe_count.z);
            ti.recorder.trace_rays({.width = x, .height = y, .depth = z, .shader_binding_table = pipeline.sbt});
        }
    }
};

inline auto pgi_pre_update_probes_compute_compile_info() -> daxa::ComputePipelineCompileInfo2
{
    return daxa::ComputePipelineCompileInfo2{
        .source = daxa::ShaderFile{"./src/rendering/pgi/pgi_update.hlsl"},
        .entry_point = "entry_pre_update_probes",
        .language = daxa::ShaderLanguage::SLANG,
        .push_constant_size = s_cast<u32>(sizeof(PGIPreUpdateProbesPush)),
        .name = std::string{"entry_pre_update_probes"},
    };
}

struct PGIPreUpdateProbesTask : PGIPreUpdateProbesH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    PGIState* pgi_state = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(pgi_pre_update_probes_compute_compile_info().name));
        auto const x = render_context->render_data.pgi_settings.probe_count.x;
        auto const y = render_context->render_data.pgi_settings.probe_count.y;
        auto const z = render_context->render_data.pgi_settings.probe_count.z;
        auto const dispatch_x = round_up_div(x, PGI_PRE_UPDATE_XYZ);
        auto const dispatch_y = round_up_div(y, PGI_PRE_UPDATE_XYZ);
        auto const dispatch_z = round_up_div(z, PGI_PRE_UPDATE_XYZ);
        PGIPreUpdateProbesPush push = {};
        push.attach = ti.attachment_shader_blob;
        push.workgroups_finished = reinterpret_cast<u32*>(ti.allocator->allocate_fill(0u).value().device_address);
        push.total_workgroups = dispatch_x * dispatch_y * dispatch_z;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({dispatch_x,dispatch_y,dispatch_z});
    }
};

inline auto pgi_create_trace_result_texture(daxa::TaskGraph& tg, PGISettings& settings, PGIState& state) -> daxa::TaskImageView
{
    return tg.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = { 
            static_cast<u32>(settings.probe_count.x * settings.probe_trace_resolution),
            static_cast<u32>(settings.probe_count.y * settings.probe_trace_resolution),
            1,
        },
        .array_layer_count = static_cast<u32>(settings.probe_count.z),
        .name = "pgi traced probe lighting and depth",
    });
}

inline auto pgi_create_probe_info_texture_prev_frame(daxa::TaskGraph& tg, PGISettings& settings, PGIState& state) -> daxa::TaskImageView
{
    return tg.create_transient_image({
        .dimensions = 2,
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = {
            static_cast<u32>(settings.probe_count.x),
            static_cast<u32>(settings.probe_count.y),
            1
        },
        .array_layer_count = static_cast<u32>(settings.probe_count.z),
        .name = "pgi probe info tex prev frame",
    });
}

inline auto pgi_create_probe_indirections(daxa::TaskGraph& tg, PGISettings& settings, PGIState& state) -> daxa::TaskBufferView
{
    return tg.create_transient_buffer({
        .size = static_cast<u32>(sizeof(PGIIndirections) + 2 * sizeof(daxa_u32) * settings.probe_count.x * settings.probe_count.y * settings.probe_count.z),
        .name = "pgi indirections",
    });
}