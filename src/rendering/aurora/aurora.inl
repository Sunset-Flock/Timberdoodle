#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/aurora_shared.inl"

#define DISTRIBUTE_BEAM_ORIGINS_WG (16 * 16)
#define DEBUG_DRAW_BEAM_ORIGINS_WG (16 * 16)

DAXA_DECL_TASK_HEAD_BEGIN(DistributeBeamOriginsH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(AuroraGlobals), aurora_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(daxa_f32vec3), beam_paths)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(DebugDrawBeamOriginsH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(AuroraGlobals), aurora_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32vec3), beam_paths)
DAXA_DECL_TASK_HEAD_END

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

void record_aurora_task_graph(AuroraState * state, RenderContext * render_context)
{
    state->generate_aurora_task_graph = daxa::TaskGraph{{
        .device = render_context->gpuctx->device,
        .name = "Generate aurora graph",
    }};

    auto & tg = state->generate_aurora_task_graph;
    tg.use_persistent_buffer(state->globals);
    tg.use_persistent_buffer(state->beam_paths);

    tg.add_task({.attachments = {
                     daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, state->globals),
                 },
        .task = [state](daxa::TaskInterface ti)
        {
            allocate_fill_copy(ti, state->cpu_globals, ti.get(state->globals));
        }});

    tg.add_task(DistributeBeamOriginsTask{
        .views = std::array{
            daxa::attachment_view(DistributeBeamOriginsH::AT.aurora_globals, state->globals),
            daxa::attachment_view(DistributeBeamOriginsH::AT.beam_paths, state->beam_paths),
        },
        .render_context = render_context,
        .state = state,
    });

    tg.submit({});
    tg.complete({});
}

struct TaskDebugDrawAuroraInfo
{
    AuroraState * aurora_state = {};
    RenderContext * render_context = {};
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

void debug_draw_aurora(TaskDebugDrawAuroraInfo const & info)
{
    info.tg->add_task(DebugDrawBeamOriginsTask{
        .views = std::array{
            daxa::attachment_view(DebugDrawBeamOriginsH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(DebugDrawBeamOriginsH::AT.aurora_globals, info.aurora_state->globals),
            daxa::attachment_view(DebugDrawBeamOriginsH::AT.beam_paths, info.aurora_state->beam_paths),
        },
        .render_context = info.render_context,
        .state = info.aurora_state,
    });
}
#endif //__cplusplus