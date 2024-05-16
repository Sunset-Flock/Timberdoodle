#pragma once
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../gpu_context.hpp"
#include "../../shader_shared/aurora_shared.inl"
#include "../../shader_shared/shared.inl"

struct AuroraState
{
    // Persistent state
    daxa::TaskBuffer globals;
    daxa::TaskBuffer beam_paths;

    daxa::TaskGraph generate_aurora_task_graph;
    AuroraGlobals cpu_globals = {};

    void initialize_perisitent_state(GPUContext * context)
    {
        globals = daxa::TaskBuffer({
            .initial_buffers = {
                .buffers = std::array{
                    context->device.create_buffer({
                        .size = s_cast<daxa_u32>(sizeof(AuroraGlobals)),
                        .name = "aurora globals physical",
                    }) ,
                },
            },
            .name = "aurora globals",
        });

        beam_paths = daxa::TaskBuffer({
            .initial_buffers = {
                .buffers = std::array{
                    context->device.create_buffer({
                        .size = s_cast<daxa_u32>(sizeof(daxa_f32vec3) * cpu_globals.beam_path_segment_count * cpu_globals.beam_count),
                        .name = "aurora beam origins physical",
                    }) ,
                },
            },
            .name = "aurora beam origins",
        });

        // debug_draw_beam_origins_indirect = daxa::TaskBuffer({
        //     .initial_buffers = {
        //         .buffers = std::array{
        //             context->device.create_buffer({
        //                 .size = s_cast<daxa_u32>(sizeof(DispatchIndirectStruct)),
        //                 .name = "debug draw beam origins indirect physical",
        //             }) ,
        //         },
        //     },
        //     .name = "debug draw beam origins indirect",
        // });
    }

    void cleanup_persistent_state(GPUContext * context)
    {
        context->device.destroy_buffer(globals.get_state().buffers[0]);
        context->device.destroy_buffer(beam_paths.get_state().buffers[0]);
        // context->device.destroy_buffer(debug_draw_beam_origins_indirect.get_state().buffers[0]);
    }
};