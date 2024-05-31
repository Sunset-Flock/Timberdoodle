#pragma once
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../gpu_context.hpp"
#include "../../shader_shared/aurora_shared.inl"
#include "../../shader_shared/shared.inl"
#include "../tasks/misc.hpp"

struct AuroraState
{
    // 100 - 300 km entry per 5 km
    static constexpr std::array<f32vec3, 40> xyz_aurora_intensities = {
        f32vec3{0.00f, 0.30f, 0.17}, f32vec3{0.00f, 0.56f, 0.23},
        f32vec3{0.00f, 0.77f, 0.29}, f32vec3{0.00f, 0.81f, 0.36},
        f32vec3{0.00f, 0.87f, 0.40}, f32vec3{0.00f, 0.92f, 0.47},
        f32vec3{0.00f, 1.00f, 0.54}, f32vec3{0.00f, 0.94f, 0.56},
        f32vec3{0.00f, 0.90f, 0.53}, f32vec3{0.00f, 0.87f, 0.50},
        f32vec3{0.00f, 0.83f, 0.44}, f32vec3{0.00f, 0.79f, 0.40},
        f32vec3{0.00f, 0.73f, 0.37}, f32vec3{0.09f, 0.66f, 0.33},
        f32vec3{0.04f, 0.60f, 0.31}, f32vec3{0.05f, 0.52f, 0.29},
        f32vec3{0.07f, 0.47f, 0.27}, f32vec3{0.08f, 0.42f, 0.25},
        f32vec3{0.09f, 0.40f, 0.22}, f32vec3{0.09f, 0.36f, 0.20},
        f32vec3{0.09f, 0.33f, 0.17}, f32vec3{0.10f, 0.30f, 0.13},
        f32vec3{0.10f, 0.26f, 0.13}, f32vec3{0.10f, 0.20f, 0.12},
        f32vec3{0.10f, 0.13f, 0.09}, f32vec3{0.11f, 0.10f, 0.07},
        f32vec3{0.11f, 0.07f, 0.00}, f32vec3{0.13f, 0.04f, 0.00},
        f32vec3{0.14f, 0.02f, 0.00}, f32vec3{0.18f, 0.00f, 0.00},
        f32vec3{0.19f, 0.00f, 0.00}, f32vec3{0.20f, 0.00f, 0.00},
        f32vec3{0.20f, 0.00f, 0.00}, f32vec3{0.22f, 0.00f, 0.00},
        f32vec3{0.20f, 0.00f, 0.00}, f32vec3{0.20f, 0.00f, 0.00},
        f32vec3{0.19f, 0.00f, 0.00}, f32vec3{0.18f, 0.00f, 0.00},
        f32vec3{0.17f, 0.00f, 0.00}, f32vec3{0.13f, 0.00f, 0.00}};

    // 100km - 160km entry per kilometer
    static constexpr std::array<f32, 61> emission_intensities = {
        0.400f, 0.750f, 0.830f, 0.900f, 0.980f,
        1.000f, 1.000f, 0.990f, 0.970f, 0.830f,
        0.780f, 0.520f, 0.400f, 0.350f, 0.290f,
        0.250f, 0.240f, 0.230f, 0.220f, 0.210f,
        0.200f, 0.200f, 0.200f, 0.200f, 0.190f,
        0.180f, 0.185f, 0.170f, 0.150f, 0.140f,
        0.125f, 0.115f, 0.110f, 0.100f, 0.100f,
        0.100f, 0.100f, 0.095f, 0.090f, 0.090f,
        0.085f, 0.085f, 0.085f, 0.080f, 0.075f,
        0.070f, 0.070f, 0.065f, 0.060f, 0.050f,
        0.050f, 0.050f, 0.040f, 0.030f, 0.030f,
        0.025f, 0.025f, 0.020f, 0.018f, 0.013f,
        0.007f};

    daxa::TaskBuffer globals;
    daxa::TaskBuffer beam_paths;
    daxa::TaskBuffer emission_luts;

    daxa::TaskImage aurora_image;

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
                    }),
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
                    }),
                },
            },
            .name = "aurora beam origins",
        });

        aurora_image = daxa::TaskImage({
            .initial_images = {
                .images = std::array{
                    context->device.create_image({
                        .format = daxa::Format::R32G32B32A32_SFLOAT,
                        .size = {cpu_globals.aurora_image_resolution.x, cpu_globals.aurora_image_resolution.y, 1},
                        .usage =
                            daxa::ImageUsageFlagBits::SHADER_SAMPLED |
                            daxa::ImageUsageFlagBits::SHADER_STORAGE |
                            daxa::ImageUsageFlagBits::COLOR_ATTACHMENT |
                            daxa::ImageUsageFlagBits::TRANSFER_DST,
                        .name = "aurora color physical image",
                    }),
                },
            },
            .name = "aurora color image",
        });

        auto const colors_lut_size = sizeof(f32vec3) * xyz_aurora_intensities.size();
        auto const emission_lut_size = sizeof(f32) * emission_intensities.size();
        emission_luts = daxa::TaskBuffer({
            .initial_buffers = {
                .buffers = std::array{
                    context->device.create_buffer({
                        .size = s_cast<daxa_u32>(colors_lut_size + emission_lut_size),
                        .name = "aurora emission luts physical",
                    }),
                },
            },
            .name = "aurora emission luts",
        });

        auto upload_task_graph = daxa::TaskGraph({
            .device = context->device,
            .name = "aurora upload tg",
        });
        upload_task_graph.use_persistent_buffer(emission_luts);
        upload_task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, emission_luts)},
            .task = [&](daxa::TaskInterface ti)
            {
                std::array<f32, xyz_aurora_intensities.size() * 3 + emission_intensities.size()> cpu_staging = {};
                auto const xyz_intensities_byte_size = xyz_aurora_intensities.size() * sizeof(f32vec3);
                std::memcpy(cpu_staging.data(), xyz_aurora_intensities.data(), xyz_intensities_byte_size);
                std::memcpy(cpu_staging.data() + xyz_aurora_intensities.size() * 3, emission_intensities.data(), emission_intensities.size() * sizeof(f32));
                allocate_fill_copy(ti, cpu_staging, ti.get(emission_luts));
            }
        });
        cpu_globals.emission_colors = context->device.get_device_address(emission_luts.get_state().buffers[0]).value();
        cpu_globals.emission_intensities = cpu_globals.emission_colors + xyz_aurora_intensities.size() * sizeof(f32vec3);
        upload_task_graph.submit({});
        upload_task_graph.complete({});
        upload_task_graph.execute({});
    }

    void cleanup_persistent_state(GPUContext * context)
    {
        context->device.destroy_buffer(globals.get_state().buffers[0]);
        context->device.destroy_buffer(beam_paths.get_state().buffers[0]);
        context->device.destroy_buffer(emission_luts.get_state().buffers[0]);
        context->device.destroy_image(aurora_image.get_state().images[0]);
    }
};