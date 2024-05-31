#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/aurora_shared.inl"

#define DISTRIBUTE_BEAM_ORIGINS_WG (16 * 16)
#define DEBUG_DRAW_BEAM_ORIGINS_WG (16 * 16)
#define BLUR_AURORA_IMAGE_WG (16 * 16)
#define MAX_BLUR_RADIUS 30

#define CHANNEL_R 0
#define CHANNEL_G 1
#define CHANNEL_B 2

DAXA_DECL_TASK_HEAD_BEGIN(DistributeBeamOriginsH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(AuroraGlobals), aurora_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(daxa_f32vec3), beam_paths)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(DebugDrawBeamOriginsH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(AuroraGlobals), aurora_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32vec3), beam_paths)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(DrawEmissionPointsH)
DAXA_TH_BUFFER_PTR(VERTEX_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(VERTEX_SHADER_READ, daxa_BufferPtr(AuroraGlobals), aurora_globals)
DAXA_TH_BUFFER_PTR(VERTEX_SHADER_READ, daxa_BufferPtr(daxa_f32vec3), beam_paths)
DAXA_TH_BUFFER_PTR(VERTEX_SHADER_READ, daxa_BufferPtr(daxa_f32), emission_luts)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(BlurAuroraImageH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(AuroraGlobals), aurora_globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, color_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, blured_image)
DAXA_DECL_TASK_HEAD_END

struct AuroraBlurPush
{
    DAXA_TH_BLOB(BlurAuroraImageH, uses)
    daxa_u32 color_channel;
};

#if defined(__cplusplus)
#include "aurora_state.hpp"
#include "../tasks/misc.hpp"
#include "../scene_renderer_context.hpp"

inline daxa::ComputePipelineCompileInfo aurora_distribute_beam_origins_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/aurora/distribute_beam_origins.hlsl"},
            .compile_options = {.language = daxa::ShaderLanguage::SLANG}},
        .push_constant_size = s_cast<u32>(sizeof(DistributeBeamOriginsH::AttachmentShaderBlob)),
        .name = std::string{DistributeBeamOriginsH::NAME},
    };
}

inline daxa::ComputePipelineCompileInfo aurora_debug_draw_beam_origins_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/aurora/debug_draw_beams.hlsl"},
            .compile_options = {.language = daxa::ShaderLanguage::SLANG}},
        .push_constant_size = s_cast<u32>(sizeof(DebugDrawBeamOriginsH::AttachmentShaderBlob)),
        .name = std::string{DebugDrawBeamOriginsH::NAME},
    };
}

inline daxa::RasterPipelineCompileInfo aurora_draw_emission_points_pipeline_compile_info()
{
    return {
        .vertex_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/aurora/draw_emission_points.hlsl"},
            .compile_options = {
                .entry_point = "vert_main",
                .language = daxa::ShaderLanguage::SLANG,
            },
        },
        .fragment_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/aurora/draw_emission_points.hlsl"},
            .compile_options = {
                .entry_point = "frag_main",
                .language = daxa::ShaderLanguage::SLANG,
            },
        },
        .color_attachments = std::vector{
            daxa::RenderAttachment{
                .format = daxa::Format::R32G32B32A32_SFLOAT,
                .blend = daxa::BlendInfo{
                    .src_color_blend_factor = daxa::BlendFactor::ONE,
                    .dst_color_blend_factor = daxa::BlendFactor::ONE,
                    .color_blend_op = daxa::BlendOp::ADD,
                },
            },
        },
        .raster = {
            .primitive_topology = daxa::PrimitiveTopology::POINT_LIST,
            .polygon_mode = daxa::PolygonMode::FILL,
            .face_culling = daxa::FaceCullFlagBits::NONE,
        },
        .push_constant_size = s_cast<u32>(sizeof(DrawEmissionPointsH::AttachmentShaderBlob)),
        .name = std::string{DrawEmissionPointsH::NAME},
    };
}


inline daxa::ComputePipelineCompileInfo aurora_blur_x_image_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/aurora/aurora_blur.hlsl"},
            .compile_options = {
                .language = daxa::ShaderLanguage::SLANG,
                .defines = {{"X_PASS", "1"}}}},
        .push_constant_size = s_cast<u32>(sizeof(AuroraBlurPush)),
        .name = std::string{BlurAuroraImageH::NAME} + "_X",
    };
}

inline daxa::ComputePipelineCompileInfo aurora_blur_y_image_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/aurora/aurora_blur.hlsl"},
            .compile_options = {
                .language = daxa::ShaderLanguage::SLANG,
                .defines = {{"Y_PASS", "1"}}}},
        .push_constant_size = s_cast<u32>(sizeof(AuroraBlurPush)),
        .name = std::string{BlurAuroraImageH::NAME} + "_Y",
    };
}

struct DistributeBeamOriginsTask : DistributeBeamOriginsH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    AuroraState * state = {};

    void callback(daxa::TaskInterface ti)
    {
        u32 const dispatch_size = round_up_div(state->cpu_globals.beam_count, DISTRIBUTE_BEAM_ORIGINS_WG);
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(aurora_distribute_beam_origins_pipeline_compile_info().name));
        DistributeBeamOriginsH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = dispatch_size});
    }
};

struct DebugDrawBeamOriginsTask : DebugDrawBeamOriginsH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    AuroraState * state = {};

    void callback(daxa::TaskInterface ti)
    {
        u32 const dispatch_size = round_up_div(state->cpu_globals.beam_count, DEBUG_DRAW_BEAM_ORIGINS_WG);
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(aurora_debug_draw_beam_origins_pipeline_compile_info().name));
        DebugDrawBeamOriginsH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = dispatch_size});
    }
};

struct DrawEmissionPointsTask : DrawEmissionPointsH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    AuroraState * state = {};

    void callback(daxa::TaskInterface ti)
    {
        u32 const emission_points_count = state->cpu_globals.beam_count * state->cpu_globals.beam_path_segment_count;

        auto const color_image_size = ti.device.info_image(ti.get(AT.color_image).ids[0]).value().size;
        auto render_recorder = std::move(ti.recorder).begin_renderpass({
            .color_attachments = {
                daxa::RenderAttachmentInfo{
                    .image_view = ti.get(AT.color_image).view_ids[0],
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = daxa::AttachmentLoadOp::CLEAR,
                    .clear_value = std::array<f32, 4>{0.0f, 0.0f, 0.0f, 0.0f},
                },
            },
            .render_area = {.width = color_image_size.x, .height = color_image_size.y},
        });

        render_recorder.set_pipeline(*render_context->gpuctx->raster_pipelines.at(aurora_draw_emission_points_pipeline_compile_info().name));
        DrawEmissionPointsH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        render_recorder.push_constant(push);
        render_recorder.draw({.vertex_count = emission_points_count});
        ti.recorder = std::move(render_recorder).end_renderpass();
    }
};

enum BlurAxis
{
    X_AXIS = 0,
    Y_AXIS = 1,
};

struct BlurAuroraImageTask : BlurAuroraImageH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    AuroraState * state = {};
    BlurAxis axis = {};
    AuroraBlurPush push = {};

    void callback(daxa::TaskInterface ti)
    {
        DBG_ASSERT_TRUE_M(state->cpu_globals.rgb_blur_kernels[CHANNEL_R].width < MAX_BLUR_RADIUS,
            "[BlurAurroraImageTask::callback()] R Blur radius outside allowed bounds");
        DBG_ASSERT_TRUE_M(state->cpu_globals.rgb_blur_kernels[CHANNEL_G].width < MAX_BLUR_RADIUS,
            "[BlurAurroraImageTask::callback()] G Blur radius outside allowed bounds");
        DBG_ASSERT_TRUE_M(state->cpu_globals.rgb_blur_kernels[CHANNEL_B].width < MAX_BLUR_RADIUS,
            "[BlurAurroraImageTask::callback()] B Blur radius outside allowed bounds");

        if(axis == X_AXIS)
        {
            u32 const dispatch_size_x = round_up_div(state->cpu_globals.aurora_image_resolution.x, DEBUG_DRAW_BEAM_ORIGINS_WG);
            u32 const dispatch_size_y = state->cpu_globals.aurora_image_resolution.y;
            ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(aurora_blur_x_image_pipeline_compile_info().name));
            assign_blob(push.uses, ti.attachment_shader_blob);
            ti.recorder.push_constant(push);
            ti.recorder.dispatch({.x = dispatch_size_x, .y = dispatch_size_y});
        }
        else 
        {
            u32 const dispatch_size_x = state->cpu_globals.aurora_image_resolution.x;
            u32 const dispatch_size_y = round_up_div(state->cpu_globals.aurora_image_resolution.y, DEBUG_DRAW_BEAM_ORIGINS_WG);
            ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(aurora_blur_y_image_pipeline_compile_info().name));
            assign_blob(push.uses, ti.attachment_shader_blob);
            ti.recorder.push_constant(push);
            ti.recorder.dispatch({.x = dispatch_size_x, .y = dispatch_size_y});
        }
    }
};

void record_aurora_task_graph(AuroraState * state, RenderContext * render_context)
{
    state->generate_aurora_task_graph = daxa::TaskGraph{{
        .device = render_context->gpuctx->device,
        .permutation_condition_count = 1,
        .name = "Generate aurora graph",
    }};


    auto & tg = state->generate_aurora_task_graph;
    tg.use_persistent_buffer(state->globals);
    tg.use_persistent_buffer(state->beam_paths);
    tg.use_persistent_buffer(state->emission_luts);
    tg.use_persistent_buffer(render_context->tgpu_render_data);
    tg.use_persistent_image(state->aurora_image);

    auto tmp_blur_image = tg.create_transient_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = {state->cpu_globals.aurora_image_resolution.x, state->cpu_globals.aurora_image_resolution.y, 1},
        .name = "aurora tmp blur image",
    });

    tg.conditional({.condition_index = 0,
        .when_true = {
            [&]()
            {
                tg.add_task(DistributeBeamOriginsTask{
                    .views = std::array{
                        daxa::attachment_view(DistributeBeamOriginsH::AT.aurora_globals, state->globals),
                        daxa::attachment_view(DistributeBeamOriginsH::AT.beam_paths, state->beam_paths),
                    },
                    .render_context = render_context,
                    .state = state,
                });
            }}

    });
    tg.add_task({
        .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, state->globals)},
        .task = [state](daxa::TaskInterface ti)
        {
            allocate_fill_copy(ti, state->cpu_globals, ti.get(state->globals));
        },
    });

    tg.add_task(DrawEmissionPointsTask{
        .views = std::array{
            daxa::attachment_view(DrawEmissionPointsTask::AT.globals, render_context->tgpu_render_data),
            daxa::attachment_view(DrawEmissionPointsTask::AT.aurora_globals, state->globals),
            daxa::attachment_view(DrawEmissionPointsTask::AT.beam_paths, state->beam_paths),
            daxa::attachment_view(DrawEmissionPointsTask::AT.color_image, state->aurora_image),
            daxa::attachment_view(DrawEmissionPointsTask::AT.emission_luts, state->emission_luts),
        },
        .render_context = render_context,
        .state = state,
    });

    auto tmp_blur_task = BlurAuroraImageTask{
        .views = std::array{
            daxa::attachment_view(BlurAuroraImageTask::AT.aurora_globals, state->globals),
            daxa::attachment_view(BlurAuroraImageTask::AT.color_image, state->aurora_image),
            daxa::attachment_view(BlurAuroraImageTask::AT.blured_image, tmp_blur_image),
        },
        .render_context = render_context,
        .state = state,
    };

    tmp_blur_task.push.color_channel = CHANNEL_R;
    tmp_blur_task.axis = BlurAxis::X_AXIS,
    tg.add_task(tmp_blur_task);
    tmp_blur_task.axis = BlurAxis::Y_AXIS;
    tg.add_task(tmp_blur_task);

    tmp_blur_task.push.color_channel = CHANNEL_G;
    tmp_blur_task.axis = BlurAxis::X_AXIS;
    tg.add_task(tmp_blur_task);
    tmp_blur_task.axis = BlurAxis::Y_AXIS;
    tg.add_task(tmp_blur_task);

    tmp_blur_task.push.color_channel = CHANNEL_B;
    tmp_blur_task.axis = BlurAxis::X_AXIS;
    tg.add_task(tmp_blur_task);
    tmp_blur_task.axis = BlurAxis::Y_AXIS;
    tg.add_task(tmp_blur_task);

    tg.submit({});
    tg.complete({});
}

struct TaskDebugDrawAuroraInfo
{
    AuroraState * aurora_state = {};
    RenderContext * render_context = {};
    daxa::TaskImageView color_image = {};
    daxa::TaskGraph * tg = {};
};

void draw_aurora_local_coord_system(TaskDebugDrawAuroraInfo const & info)
{
    auto const aurora_start = f32vec3(std::bit_cast<f32vec2>(info.aurora_state->cpu_globals.start), info.aurora_state->cpu_globals.height);
    auto const aurora_end = f32vec3(std::bit_cast<f32vec2>(info.aurora_state->cpu_globals.end), info.aurora_state->cpu_globals.height);
    auto const aurora_B = f32vec3(std::bit_cast<f32vec3>(info.aurora_state->cpu_globals.B));
    auto const aurora_local_forward = glm::normalize(aurora_end - aurora_start);
    auto const aurora_cross = glm::cross(aurora_local_forward, aurora_B);
    auto const aurora_forward_color = daxa_f32vec3(1.0, 0.0, 0.0);
    auto const aurora_B_color = daxa_f32vec3(0.0, 1.0, 0.0);
    auto const aurora_cross_color = daxa_f32vec3(0.0, 0.0, 1.0);

    info.render_context->gpuctx->shader_debug_context.cpu_debug_line_draws.push_back(
        ShaderDebugLineDraw{
            .vertices = {
                std::bit_cast<daxa_f32vec3>(aurora_start),
                std::bit_cast<daxa_f32vec3>(aurora_start + aurora_local_forward * 10.0f)},
            .colors = {aurora_forward_color, aurora_forward_color},
            .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE,
        });

    info.render_context->gpuctx->shader_debug_context.cpu_debug_line_draws.push_back(
        ShaderDebugLineDraw{
            .vertices = {
                std::bit_cast<daxa_f32vec3>(aurora_start),
                std::bit_cast<daxa_f32vec3>(aurora_start + aurora_B * 10.0f)},
            .colors = {aurora_B_color, aurora_B_color},
            .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE,
        });

    info.render_context->gpuctx->shader_debug_context.cpu_debug_line_draws.push_back(
        ShaderDebugLineDraw{
            .vertices = {
                std::bit_cast<daxa_f32vec3>(aurora_start),
                std::bit_cast<daxa_f32vec3>(aurora_start + aurora_cross * 10.0f)},
            .colors = {aurora_cross_color, aurora_cross_color},
            .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE,
        });
}
#endif //__cplusplus