#include "renderer.hpp"

#include "../shader_shared/scene.inl"
#include "../daxa_helper.hpp"
#include "../shader_shared/debug.inl"
#include "../shader_shared/readback.inl"

#include "rasterize_visbuffer/rasterize_visbuffer.hpp"

#include "virtual_shadow_maps/vsm.hpp"

#include "rtao/rtao.inl"
#include "path_trace/path_trace.inl"
#include "path_trace/kajiya/brdf_fg.inl"
#include "rtgi/rtgi.hpp"

#include "tasks/memset.inl"
#include "tasks/prefix_sum.inl"
#include "tasks/write_swapchain.inl"
#include "tasks/shade_opaque.inl"
#include "tasks/sky.inl"
#include "tasks/autoexposure.inl"
#include "tasks/shader_debug_draws.inl"
#include "tasks/decode_visbuffer_test.inl"
#include "tasks/gen_gbuffer.hpp"
#include "tasks/cull_lights.hpp"
#include "tasks/copy_depth.hpp"

#include <daxa/types.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <thread>
#include <variant>
#include <iostream>

struct FixedFunctionContainer
{
    std::array<u8, 128> mem = {};
    void (*callback)(u8 * mem, daxa::TaskInterface ti) = {};

    FixedFunctionContainer() = default;
    template <typename CallbackT>
    FixedFunctionContainer(CallbackT const & func)
    {
        std::memcpy(mem.data(), &func, sizeof(func));
        callback = [](u8 * mem, daxa::TaskInterface ti)
        {
            reinterpret_cast<CallbackT *>(mem)->operator()(ti);
        };
    }
};

inline auto create_task_buffer(GPUContext * gpu_context, auto size, auto task_buf_name, auto buf_name)
{
    FixedFunctionContainer fcont = {};
    fcont = {[gpu_context](daxa::TaskInterface ti) {
    }};

    return daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                gpu_context->device.create_buffer({
                    .size = static_cast<u32>(size),
                    .name = buf_name,
                }),
            },
        },
        .name = task_buf_name,
    }};
}

Renderer::Renderer(
    Window * window, GPUContext * gpu_context, Scene * scene, AssetProcessor * asset_manager, daxa::ImGuiRenderer * imgui_renderer, UIEngine * ui_engine)
    : render_context{std::make_unique<RenderContext>(gpu_context)}, window{window}, gpu_context{gpu_context}, scene{scene}, asset_manager{asset_manager}, imgui_renderer{imgui_renderer}, ui_engine{ui_engine}
{
    meshlet_instances = create_task_buffer(gpu_context, size_of_meshlet_instance_buffer(), "meshlet_instances", "meshlet_instances_a");
    visible_mesh_instances = create_task_buffer(gpu_context, sizeof(VisibleMeshesList), "visible_mesh_instances", "visible_mesh_instances");
    exposure_state = create_task_buffer(gpu_context, sizeof(AutoExposureState), "exposure state", "exposure_state");
    general_readback_buffer = gpu_context->device.create_buffer({
        .size = sizeof(ReadbackValues) * (MAX_GPU_FRAMES_IN_FLIGHT),
        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "general readback buffer",
    });
    visible_meshlet_instances = create_task_buffer(gpu_context, sizeof(u32) * (MAX_MESHLET_INSTANCES + 4), "visible_meshlet_instances", "visible_meshlet_instances");

    buffers = {
        meshlet_instances,
        visible_meshlet_instances,
        visible_mesh_instances,
        exposure_state,
    };

    swapchain_image = daxa::TaskImage{{.swapchain_image = true, .name = "swapchain_image"}};
    transmittance = daxa::TaskImage{{.name = "transmittance"}};
    multiscattering = daxa::TaskImage{{.name = "multiscattering"}};
    sky_ibl_cube = daxa::TaskImage{{.name = "sky ibl cube"}};
    depth_history = daxa::TaskImage{{.name = "depth_history"}};
    path_trace_history = daxa::TaskImage{{.name = "path_trace_history"}};
    normal_history = daxa::TaskImage{{.name = "normal_history"}};
    rtao_history = daxa::TaskImage{{.name = "rtao_history"}};
    rtgi_depth_history = daxa::TaskImage{{.name = "rtgi_depth_history"}};
    rtgi_samplecnt_history = daxa::TaskImage{{.name = "rtgi_samplecnt_history"}};
    rtgi_face_normal_history = daxa::TaskImage{{.name = "rtgi_face_normal_history"}};

    rtgi_full_samplecount_history = daxa::TaskImage{{.name = "rtgi_full_samplecount_history"}};
    rtgi_full_face_normal_history = daxa::TaskImage{{.name = "rtgi_full_face_normal_history"}};
    rtgi_full_color_history = daxa::TaskImage{{.name = "rtgi_full_color_history"}};
    rtgi_full_statistics_history = daxa::TaskImage{{.name = "rtgi_full_statistics_history"}};

    vsm_state.initialize_persitent_state(gpu_context);
    pgi_state.initialize(gpu_context->device);

    images = {
        transmittance,
        multiscattering,
        sky_ibl_cube,
        depth_history,
        path_trace_history,
        normal_history,
        rtao_history,
        rtgi_full_samplecount_history,
        rtgi_full_face_normal_history,
        rtgi_full_color_history,
        rtgi_full_statistics_history,
    };

    frame_buffer_images = {
        {
            daxa::ImageInfo{
                .format = daxa::Format::R32_SFLOAT,
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "depth_history",
            },
            depth_history,
        },
        {
            daxa::ImageInfo{
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "path_trace_history",
            },
            path_trace_history,
        },
        {
            daxa::ImageInfo{
                .format = daxa::Format::R32_UINT,
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .sharing_mode = daxa::SharingMode::CONCURRENT,
                .name = "normal_history",
            },
            normal_history,
        },
        {
            daxa::ImageInfo{
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .sharing_mode = daxa::SharingMode::CONCURRENT,
                .name = "rtao_history",
            },
            rtao_history,
        },
        {
            daxa::ImageInfo{
                .format = daxa::Format::R16_SFLOAT,
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .sharing_mode = daxa::SharingMode::CONCURRENT,
                .name = "rtgi_full_samplecount_history",
            },
            rtgi_full_samplecount_history,
        },
        {
            daxa::ImageInfo{
                .format = daxa::Format::R32_UINT,
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .sharing_mode = daxa::SharingMode::CONCURRENT,
                .name = "rtgi_full_face_normal_history",
            },
            rtgi_full_face_normal_history,
        },
        {
            daxa::ImageInfo{
                .format = daxa::Format::R32G32_UINT,
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .sharing_mode = daxa::SharingMode::CONCURRENT,
                .name = "rtgi_full_color_history",
            },
            rtgi_full_color_history,
        },
        {
            daxa::ImageInfo{
                .format = daxa::Format::R32_UINT,
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .sharing_mode = daxa::SharingMode::CONCURRENT,
                .name = "rtgi_full_statistics_history",
            },
            rtgi_full_statistics_history,
        },
    };

    recreate_framebuffer();
    recreate_sky_luts();
    main_task_graph = create_main_task_graph();
    debug_task_graph = create_debug_task_graph();
    sky_task_graph = create_sky_lut_task_graph();
}

Renderer::~Renderer()
{
    for (auto & tbuffer : buffers)
    {
        if (tbuffer.is_owning()) continue;
        for (auto buffer : tbuffer.get_state().buffers)
        {
            this->gpu_context->device.destroy_buffer(buffer);
        }
    }
    for (auto & timage : images)
    {
        for (auto image : timage.get_state().images)
        {
            this->gpu_context->device.destroy_image(image);
        }
    }
    if (!general_readback_buffer.is_empty())
    {
        this->gpu_context->device.destroy_buffer(general_readback_buffer);
    }
    for (auto const & [name, rt_pipe] : this->gpu_context->ray_tracing_pipelines)
    {
        if (!rt_pipe.sbt_buffer_id.is_empty())
        {
            this->gpu_context->device.destroy_buffer(rt_pipe.sbt_buffer_id);
        }
    }
    
    if (!rtgi_depth_history.get_state().images.empty() && !rtgi_depth_history.get_state().images[0].is_empty())
    {
        gpu_context->device.destroy_image(rtgi_depth_history.get_state().images[0]);
    }
    if (!rtgi_samplecnt_history.get_state().images.empty() && !rtgi_samplecnt_history.get_state().images[0].is_empty())
    {
        gpu_context->device.destroy_image(rtgi_samplecnt_history.get_state().images[0]);
    }
    if (!rtgi_face_normal_history.get_state().images.empty() && !rtgi_face_normal_history.get_state().images[0].is_empty())
    {
        gpu_context->device.destroy_image(rtgi_face_normal_history.get_state().images[0]);
    }
    if (!stbn2d.is_empty())
    {
        gpu_context->device.destroy_image(stbn2d);
    }
    if (!stbnCosDir.is_empty())
    {
        gpu_context->device.destroy_image(stbnCosDir);
    }
    pgi_state.cleanup(gpu_context->device);
    vsm_state.cleanup_persistent_state(gpu_context);
    this->gpu_context->device.wait_idle();
    this->gpu_context->device.collect_garbage();
}

// struct PipelineCompileInfos
// {
//     InlineArray<daxa::RasterPipelineCompileInfo, 64> rasters = {};
//     u32 const RPID_DRAW_VISBUFFER_0 = rasters.add(draw_visbuffer_mesh_shader_pipelines[0]);
//     u32 const RPID_DRAW_VISBUFFER_1 = rasters.add(draw_visbuffer_mesh_shader_pipelines[1]);
//     u32 const RPID_DRAW_VISBUFFER_2 = rasters.add(draw_visbuffer_mesh_shader_pipelines[2]);
//     u32 const RPID_DRAW_VISBUFFER_3 = rasters.add(draw_visbuffer_mesh_shader_pipelines[3]);
//     u32 const RPID_DRAW_VISBUFFER_4 = rasters.add(draw_visbuffer_mesh_shader_pipelines[4]);
//     u32 const RPID_DRAW_VISBUFFER_5 = rasters.add(draw_visbuffer_mesh_shader_pipelines[5]);
//     u32 const RPID_DRAW_VISBUFFER_6 = rasters.add(draw_visbuffer_mesh_shader_pipelines[6]);
//     u32 const RPID_DRAW_VISBUFFER_7 = rasters.add(draw_visbuffer_mesh_shader_pipelines[7]);
//     u32 const RPID_CULL_DRAW_PAGES_0 = rasters.add();
//     u32 const RPID_CULL_DRAW_PAGES_1 = rasters.add();
//     u32 const RPID_DRAW_SHADER_DEBUG_LINES = rasters.add();
//     u32 const RPID_DRAW_SHADER_DEBUG_CIRCLES = rasters.add();
//     u32 const RPID_DRAW_SHADER_DEBUG_RECTANGLES = rasters.add();
//     u32 const RPID_DRAW_SHADER_DEBUG_AABBS = rasters.add();
//     u32 const RPID_DRAW_SHADER_DEBUG_BOXES = rasters.add();
//     u32 const RPID_PGI_DRAW_DEBUG_PROBES = rasters.add();
//     u32 const RPID_COUNT = rasters.add();
// };

void Renderer::compile_pipelines()
{
    auto add_if_not_present = [&](auto & map, auto & list, auto compile_info)
    {
        if (!map.contains(compile_info.name))
        {
            list.push_back(compile_info);
        }
    };

    std::vector<daxa::RasterPipelineCompileInfo2> rasters = {
        {draw_visbuffer_mesh_shader_pipelines[0]},
        {draw_visbuffer_mesh_shader_pipelines[1]},
        {draw_visbuffer_mesh_shader_pipelines[2]},
        {draw_visbuffer_mesh_shader_pipelines[3]},
        {cull_and_draw_directional_pages_pipelines[0]},
        {cull_and_draw_directional_pages_pipelines[1]},
        {cull_and_draw_point_pages_pipelines[0]},
        {cull_and_draw_point_pages_pipelines[1]},
        {draw_shader_debug_lines_pipeline_compile_info()},
        {pgi_draw_debug_probes_compile_info()},
    };
    for (auto info : rasters)
    {
        auto compilation_result = this->gpu_context->pipeline_manager.add_raster_pipeline2(info);
        if (compilation_result.value()->is_valid())
        {
            std::cout << fmt::format("[Renderer::compile_pipelines()] SUCCESFULLY compiled pipeline {}", info.name) << std::endl;
        }
        else
        {
            std::cout << fmt::format("[Renderer::compile_pipelines()] FAILED to compile pipeline {} with message \n {}", info.name,
                             compilation_result.message())
                      << std::endl;
        }
        this->gpu_context->raster_pipelines[info.name] = compilation_result.value();
    }
    std::vector<daxa::ComputePipelineCompileInfo2> computes = {
        {rtgi_reproject_diffuse_compile_info()},
        {rtgi_pre_blur_flatten_compile_info()},
        {rtgi_pre_blur_prepare_compile_info()},
        {rtgi_pre_blur_apply_compile_info()},
        {rtgi_adaptive_blur_diffuse_compile_info()},
        {rtgi_upscale_diffuse_compile_info()},
        {rtgi_diffuse_temporal_stabilization_compile_info()},
        {gen_hiz_pipeline_compile_info2()},
        {pgi_update_probe_texels_pipeline_compile_info()},
        {pgi_update_probes_compile_info()},
        {pgi_pre_update_probes_compute_compile_info()},
        {pgi_eval_screen_irradiance_compute_compile_info()},
        {pgi_upscale_screen_irradiance_compute_compile_info()},
        {cull_meshlets_compute_pipeline_compile_info()},
        {IndirectMemsetBufferTask::pipeline_compile_info},
        {analyze_visbufer_pipeline_compile_info()},
        {write_swapchain_pipeline_compile_info2()},
        {write_swapchain_debug_pipeline_compile_info2()},
        {shade_opaque_pipeline_compile_info()},
        {expand_meshes_pipeline_compile_info()},
        {PrefixSumCommandWriteTask::pipeline_compile_info},
        {prefix_sum_upsweep_pipeline_compile_info()},
        {prefix_sum_downsweep_pipeline_compile_info()},
        {compute_transmittance_pipeline_compile_info()},
        {compute_multiscattering_pipeline_compile_info()},
        {compute_sky_pipeline_compile_info()},
        {sky_into_cubemap_pipeline_compile_info()},
        {gen_luminace_histogram_pipeline_compile_info()},
        {gen_luminace_average_pipeline_compile_info()},
        {vsm_free_wrapped_pages_pipeline_compile_info()},
        {vsm_invalidate_directional_pages_pipeline_compile_info()},
        {vsm_force_always_resident_pages_pipeline_compile_info()},
        {vsm_mark_required_pages_pipeline_compile_info()},
        {vsm_find_free_pages_pipeline_compile_info()},
        {vsm_allocate_pages_pipeline_compile_info()},
        {vsm_clear_pages_pipeline_compile_info()},
        {vsm_gen_dirty_bit_hiz_pipeline_compile_info()},
        {vsm_gen_point_dirty_bit_hiz_pipeline_compile_info()},
        {vsm_clear_dirty_bit_pipeline_compile_info()},
        {vsm_debug_virtual_page_table_pipeline_compile_info()},
        {vsm_debug_meta_memory_table_pipeline_compile_info()},
        {vsm_recreate_shadow_map_pipeline_compile_info()},
        {vsm_get_debug_statistics_pipeline_compile_info()},
        {decode_visbuffer_test_pipeline_info2()},
        {DrawVisbuffer_WriteCommandTask2::pipeline_compile_info},
        {rtao_denoiser_pipeline_info()},
        {gen_gbuffer_pipeline_compile_info()},
        {brdf_fg_compute_pipeline_info()},
        {cull_lights_compile_info()},
        {copy_depth_pipeline_compile_info()},
    };
    for (auto const & info : computes)
    {
        auto compilation_result = this->gpu_context->pipeline_manager.add_compute_pipeline2(info);
        if (compilation_result.value()->is_valid())
        {
            std::cout << fmt::format("[Renderer::compile_pipelines()] SUCCESFULLY compiled pipeline {}", info.name) << std::endl;
        }
        else
        {
            std::cout << fmt::format("[Renderer::compile_pipelines()] FAILED to compile pipeline {} with message \n {}", info.name,
                             compilation_result.message())
                      << std::endl;
        }
        this->gpu_context->compute_pipelines[info.name] = compilation_result.value();
    }

    std::vector<daxa::RayTracingPipelineCompileInfo2> ray_tracing = {
        {ray_trace_ao_rt_pipeline_info()},
        {pgi_trace_probe_lighting_pipeline_compile_info()},
        {reference_path_trace_rt_pipeline_info()},
        {rtgi_trace_diffuse_compile_info()},
    };
    for (auto const & info : ray_tracing)
    {
        auto compilation_result = this->gpu_context->pipeline_manager.add_ray_tracing_pipeline2(info);
        if (compilation_result.value()->is_valid())
        {
            std::cout << fmt::format("[Renderer::compile_pipelines()] SUCCESFULLY compiled pipeline {}", info.name) << std::endl;
        }
        else
        {
            std::cout << fmt::format("[Renderer::compile_pipelines()] FAILED to compile pipeline {} with message \n {}", info.name,
                             compilation_result.message())
                      << std::endl;
        }
        this->gpu_context->ray_tracing_pipelines[info.name].pipeline = compilation_result.value();
        auto sbt_info = gpu_context->ray_tracing_pipelines[info.name].pipeline->create_default_sbt();
        this->gpu_context->ray_tracing_pipelines[info.name].sbt = sbt_info.table;
        this->gpu_context->ray_tracing_pipelines[info.name].sbt_buffer_id = sbt_info.buffer;
    }

    while (!gpu_context->pipeline_manager.all_pipelines_valid())
    {
        auto const result = gpu_context->pipeline_manager.reload_all();
        if (daxa::holds_alternative<daxa::PipelineReloadError>(result))
        {
            std::cout << daxa::get<daxa::PipelineReloadError>(result).message << std::endl;
        }
        using namespace std::literals;
        std::this_thread::sleep_for(30ms);
    }
}

void Renderer::recreate_sky_luts()
{
    if (!transmittance.get_state().images.empty() && !transmittance.get_state().images[0].is_empty())
    {
        gpu_context->device.destroy_image(transmittance.get_state().images[0]);
    }
    if (!multiscattering.get_state().images.empty() && !multiscattering.get_state().images[0].is_empty())
    {
        gpu_context->device.destroy_image(multiscattering.get_state().images[0]);
    }
    if (!sky_ibl_cube.get_state().images.empty() && !sky_ibl_cube.get_state().images[0].is_empty())
    {
        gpu_context->device.destroy_image(sky_ibl_cube.get_state().images[0]);
    }
    transmittance.set_images({
        .images = std::array{
            gpu_context->device.create_image({
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {render_context->render_data.sky_settings.transmittance_dimensions.x, render_context->render_data.sky_settings.transmittance_dimensions.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
                .name = "transmittance look up table",
            }),
        },
    });

    multiscattering.set_images({
        .images = std::array{
            gpu_context->device.create_image({
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {render_context->render_data.sky_settings.multiscattering_dimensions.x, render_context->render_data.sky_settings.multiscattering_dimensions.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
                .name = "multiscattering look up table",
            }),
        },
    });

    sky_ibl_cube.set_images({
        .images = std::array{
            gpu_context->device.create_image({
                .flags = daxa::ImageCreateFlagBits::COMPATIBLE_CUBE,
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {IBL_CUBE_RES, IBL_CUBE_RES, 1},
                .array_layer_count = 6,
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
                .name = "ibl cube",
            }),
        },
    });
}

void Renderer::recreate_framebuffer()
{
    for (auto & [info, timg] : frame_buffer_images)
    {
        if (!timg.get_state().images.empty() && !timg.get_state().images[0].is_empty())
        {
            gpu_context->device.destroy_image(timg.get_state().images[0]);
        }
        auto new_info = info;
        new_info.size = {render_context->render_data.settings.render_target_size.x, render_context->render_data.settings.render_target_size.y, 1};
        timg.set_images({.images = std::array{this->gpu_context->device.create_image(new_info)}});
    }
    if (!rtgi_depth_history.get_state().images.empty() && !rtgi_depth_history.get_state().images[0].is_empty())
    {
        gpu_context->device.destroy_image(rtgi_depth_history.get_state().images[0]);
    }
    rtgi_depth_history.set_images({.images = std::array{this->gpu_context->device.create_image(rtgi_create_depth_history_image_info(render_context.get()))}});
    if (!rtgi_samplecnt_history.get_state().images.empty() && !rtgi_samplecnt_history.get_state().images[0].is_empty())
    {
        gpu_context->device.destroy_image(rtgi_samplecnt_history.get_state().images[0]);
    }
    rtgi_samplecnt_history.set_images({.images = std::array{this->gpu_context->device.create_image(rtgi_create_samplecnt_history_image_info(render_context.get()))}});
    if (!rtgi_face_normal_history.get_state().images.empty() && !rtgi_face_normal_history.get_state().images[0].is_empty())
    {
        gpu_context->device.destroy_image(rtgi_face_normal_history.get_state().images[0]);
    }
    rtgi_face_normal_history.set_images({.images = std::array{this->gpu_context->device.create_image(rtgi_create_face_normal_history_image_info(render_context.get()))}});
}

void Renderer::clear_select_buffers()
{
    using namespace daxa;
    TaskGraph tg{{
        .device = this->gpu_context->device,
        .additional_transient_image_usage_flags = daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::SHADER_STORAGE,
        .name = "clear task list",
    }};
    tg.use_persistent_buffer(meshlet_instances);
    tg.add_task(daxa::InlineTask::Transfer("clear meshlet instance buffers")
            .writes(meshlet_instances)
            .executes([=](daxa::TaskInterface ti)
                {
                    auto mesh_instances_address = ti.device_address(meshlet_instances.view()).value();
                    MeshletInstancesBufferHead mesh_instances_reset = make_meshlet_instance_buffer_head(mesh_instances_address);
                    allocate_fill_copy(ti, mesh_instances_reset, ti.get(meshlet_instances)); }));
    tg.use_persistent_image(rtao_history);
    tg.clear_image({rtao_history.view(), {}, daxa::QUEUE_MAIN, "clear rtao history"});
    tg.use_persistent_buffer(visible_meshlet_instances);
    tg.clear_buffer({.buffer = visible_meshlet_instances, .size = sizeof(u32), .clear_value = 0});
    
    // RTGI
    {
        tg.use_persistent_image(rtgi_depth_history);
        tg.use_persistent_image(rtgi_face_normal_history);
        tg.use_persistent_image(rtgi_samplecnt_history);
        tg.clear_image({rtgi_depth_history.view(), {}, daxa::QUEUE_MAIN, "clear rtgi_depth_history"});
        tg.clear_image({rtgi_face_normal_history.view(), {}, daxa::QUEUE_MAIN, "clear rtgi_face_normal_history"});
        tg.clear_image({rtgi_samplecnt_history.view(), {}, daxa::QUEUE_MAIN, "clear rtgi_samplecnt_history"});
    }
    // tg.use_persistent_buffer(exposure_state);
    // tg.clear_buffer({.buffer = exposure_state, .size = sizeof(ExposureState), .clear_value = 0});
    tg.use_persistent_image(path_trace_history);
    tg.clear_image({.view = path_trace_history, .name = "clear pt history"});
    tg.submit({});
    tg.complete({});
    tg.execute({});
}

void Renderer::window_resized()
{
    if (this->window->size.x == 0 || this->window->size.y == 0)
    {
        DEBUG_MSG("minimized");
        return;
    }
    this->gpu_context->swapchain.resize();
}

auto Renderer::create_sky_lut_task_graph() -> daxa::TaskGraph
{
    daxa::TaskGraph tg{{
        .device = gpu_context->device,
        .additional_transient_image_usage_flags = daxa::ImageUsageFlagBits::TRANSFER_SRC,
        .name = "Calculate sky luts task graph",
    }};
    // TODO:    Do not use globals here, make a new buffer.
    //          Globals should only be used within the main task graph.
    tg.use_persistent_buffer(render_context->tgpu_render_data);
    tg.use_persistent_image(transmittance);
    tg.use_persistent_image(multiscattering);

    tg.add_task(daxa::InlineTask::Transfer("update sky settings globals")
            .writes(render_context->tgpu_render_data)
            .executes(
                [=](daxa::TaskInterface ti)
                {
                    allocate_fill_copy(
                        ti,
                        render_context->render_data.sky_settings,
                        ti.get(render_context->tgpu_render_data),
                        offsetof(RenderGlobalData, sky_settings));
                }));

    tg.add_task(ComputeTransmittanceTask{
        .views = ComputeTransmittanceTask::Views{
            .globals = render_context->tgpu_render_data.view(),
            .transmittance = transmittance.view(),
        },
        .gpu_context = gpu_context,
    });

    tg.add_task(ComputeMultiscatteringTask{
        .views = ComputeMultiscatteringTask::Views{
            .globals = render_context->tgpu_render_data.view(),
            .transmittance = transmittance.view(),
            .multiscattering = multiscattering.view(),
        },
        .render_context = render_context.get(),
    });
    tg.submit({});
    tg.complete({});
    return tg;
}

auto Renderer::create_debug_task_graph() -> daxa::TaskGraph
{
    using namespace daxa;
    TaskGraph tg{{
        .device = this->gpu_context->device,
        .swapchain = this->gpu_context->swapchain,
        .reorder_tasks = true,
        .use_split_barriers = false,
        .staging_memory_pool_size = 2'097'152, // 2MiB.
        // Extra flags are required for tg debug inspector:
        .additional_transient_image_usage_flags = daxa::ImageUsageFlagBits::TRANSFER_SRC,
        .name = "Timberdoodle main TaskGraph",
    }};
    tg.use_persistent_image(swapchain_image);
    tg.clear_image({.view = swapchain_image.view()});
    tg.submit({});

    tg.add_task(daxa::InlineTask{"ImGui Draw"}
            .color_attachment.reads_writes(swapchain_image)
            .executes(
                [=, this](daxa::TaskInterface ti)
                {
                    ImGui::Render();
                    auto size = ti.info(swapchain_image.view()).value().size;
                    imgui_renderer->record_commands(
                        ImGui::GetDrawData(), ti.recorder, ti.id(swapchain_image.view()), size.x, size.y);
                }));

    tg.submit({});
    tg.present({});
    tg.complete({});
    return tg;
}

auto Renderer::create_main_task_graph() -> daxa::TaskGraph
{
    using namespace daxa;
    TaskGraph tg{{
        .device = this->gpu_context->device,
        .swapchain = this->gpu_context->swapchain,
        .reorder_tasks = this->render_context->render_data.settings.enable_task_reordering != 0,
        .optimize_transient_lifetimes = this->render_context->render_data.settings.optimize_transient_lifetimes != 0,
        .alias_transients = this->render_context->render_data.settings.enable_memory_aliasing != 0,
        .use_split_barriers = false,
        .staging_memory_pool_size = 2'097'152, // 2MiB.
        // Extra flags are required for tg debug inspector:
        .additional_transient_image_usage_flags = daxa::ImageUsageFlagBits::TRANSFER_SRC,
        .name = "Timberdoodle main TaskGraph",
    }};
    tg.use_persistent_image(swapchain_image);
    for (auto const & tbuffer : buffers)
    {
        tg.use_persistent_buffer(tbuffer);
    }
    for (auto const & timage : images)
    {
        tg.use_persistent_image(timage);
    }
    tg.use_persistent_buffer(scene->_gpu_entity_meta);
    tg.use_persistent_buffer(scene->_gpu_entity_transforms);
    tg.use_persistent_buffer(scene->_gpu_entity_combined_transforms);
    tg.use_persistent_buffer(scene->_gpu_entity_parents);
    tg.use_persistent_buffer(scene->_gpu_entity_mesh_groups);
    tg.use_persistent_buffer(scene->_gpu_mesh_manifest);
    tg.use_persistent_buffer(scene->_gpu_mesh_group_manifest);
    tg.use_persistent_buffer(scene->_gpu_material_manifest);
    tg.use_persistent_buffer(scene->_scene_as_indirections);
    tg.use_persistent_buffer(scene->mesh_instances_buffer);
    tg.use_persistent_buffer(scene->_gpu_point_lights);
    tg.use_persistent_buffer(scene->_gpu_spot_lights);
    tg.use_persistent_buffer(render_context->tgpu_render_data);
    tg.use_persistent_buffer(vsm_state.globals);
    tg.use_persistent_image(vsm_state.memory_block);
    tg.use_persistent_image(vsm_state.meta_memory_table);
    tg.use_persistent_image(vsm_state.page_table);
    tg.use_persistent_image(vsm_state.page_view_pos_row);
    tg.use_persistent_image(vsm_state.point_spot_page_tables);
    tg.use_persistent_image(gpu_context->shader_debug_context.vsm_debug_page_table);
    tg.use_persistent_image(gpu_context->shader_debug_context.vsm_debug_meta_memory_table);
    tg.use_persistent_image(gpu_context->shader_debug_context.vsm_recreated_shadowmap_memory_table);

    // TODO: Move into an if and create persistent state only if necessary.
    tg.use_persistent_image(pgi_state.probe_color);
    tg.use_persistent_image(pgi_state.probe_visibility);
    tg.use_persistent_image(pgi_state.probe_info);
    tg.use_persistent_image(pgi_state.cell_requests);

    daxa::TaskImageView clocks_image = daxa::NullTaskImage;

    if (render_context->render_data.settings.debug_draw_mode == DEBUG_DRAW_MODE_PGI_EVAL_CLOCKS ||
        render_context->render_data.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTAO_TRACE_CLOCKS)
    {
        clocks_image = tg.create_transient_image({
            .format = daxa::Format::R32_UINT,
            .size = {
                render_context->render_data.settings.render_target_size.x,
                render_context->render_data.settings.render_target_size.y,
                1,
            },
            .name = "clocks_image",
        });
    }

    auto debug_image = tg.create_transient_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = {
            render_context->render_data.settings.render_target_size.x,
            render_context->render_data.settings.render_target_size.y,
            1,
        },
        .name = "debug_image",
    });
    tg.clear_image({debug_image, std::array{0.0f, 0.0f, 0.0f, 0.0f}});

    tg.add_task(ReadbackTask{
        .views = ReadbackTask::Views{
            .globals = render_context->tgpu_render_data.view()},
        .shader_debug_context = &gpu_context->shader_debug_context,
    });
    tg.add_task(daxa::InlineTask::Transfer("update global buffers")
            .writes(render_context->tgpu_render_data)
            .executes(
                [=](daxa::TaskInterface ti)
                {
                    allocate_fill_copy(ti, render_context->render_data, ti.get(render_context->tgpu_render_data));
                    gpu_context->shader_debug_context.update_debug_buffer(ti.device, ti.recorder, *ti.allocator);
                    render_context->render_times.reset_timestamps_for_current_frame(ti.recorder);
                }));

    if (render_context->render_data.settings.enable_async_compute)
    {
        tg.submit({}); // Submit to allow for concurrent queue access to globals.
    }

    ///
    /// === Misc Tasks Begin ===
    ///

    auto sky = tg.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {render_context->render_data.sky_settings.sky_dimensions.x, render_context->render_data.sky_settings.sky_dimensions.y, 1},
        .name = "sky look up table",
    });
    auto luminance_histogram = tg.create_transient_buffer({sizeof(u32) * (LUM_HISTOGRAM_BIN_COUNT), "luminance_histogram"});

    auto misc_tasks_queue = render_context->render_data.settings.enable_async_compute ? daxa::QUEUE_COMPUTE_0 : daxa::QUEUE_MAIN;
    auto tlas_build_task_queue = render_context->render_data.settings.enable_async_compute ? daxa::QUEUE_COMPUTE_0 : daxa::QUEUE_MAIN;

    daxa::TaskImageView sky_ibl_view = sky_ibl_cube.view().layers(0, 6);
    tg.add_task(daxa::HeadTask<ComputeSkyH::Info>()
        .head_views({
            .globals = render_context->tgpu_render_data.view(),
            .transmittance = transmittance.view(),
            .multiscattering = multiscattering.view(),
            .sky = sky,
        })
        .uses_queue(misc_tasks_queue)
        .executes(compute_sky_task, render_context.get()));

    tg.add_task(daxa::HeadTask<SkyIntoCubemapH::Info>()
                .head_views({
                    .globals = render_context->tgpu_render_data.view(),
                    .transmittance = transmittance.view(),
                    .sky = sky,
                    .ibl_cube = sky_ibl_view,
                })
                .uses_queue(misc_tasks_queue)
                .executes(sky_into_cubemap_task, gpu_context));

    daxa::TaskImageView light_mask_volume = create_light_mask_volume(tg, *render_context);
    tg.add_task(daxa::HeadTask<CullLightsH::Info>()
            .head_views({
                .globals = render_context->tgpu_render_data.view(),
                .light_mask_volume = light_mask_volume,
            })
            .uses_queue(misc_tasks_queue)
            .executes(cull_lights_task, render_context.get()));

    auto scene_main_tlas = tg.create_transient_tlas({.size = 1u << 25u /* 32 Mib */, .name = "scene_main_tlas"});
    tg.add_task(daxa::Task::RayTracing("build scene tlas")
            .acceleration_structure_build.writes(scene_main_tlas)
            .uses_queue(tlas_build_task_queue)
            .executes(
                [=](daxa::TaskInterface ti)
                {
                    render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"MISC", "BUILD_TLAS">());
                    scene->build_tlas_from_mesh_instances(ti.recorder, ti.id(scene_main_tlas));
                    render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"MISC", "BUILD_TLAS">());
                }));

    ///
    /// === Misc Tasks End ===
    ///

    auto const visbuffer_ret = raster_visbuf::task_draw_visbuffer_all({tg, render_context, scene, meshlet_instances, visible_meshlet_instances, debug_image, depth_history});
    daxa::TaskImageView main_camera_visbuffer = visbuffer_ret.main_camera_visbuffer;
    daxa::TaskImageView main_camera_depth = visbuffer_ret.main_camera_depth;
    daxa::TaskImageView view_camera_visbuffer = visbuffer_ret.view_camera_visbuffer;
    daxa::TaskImageView view_camera_depth = visbuffer_ret.view_camera_depth;
    daxa::TaskImageView overdraw_image = visbuffer_ret.view_camera_overdraw;

    daxa::TaskImageView main_camera_face_normal_image = tg.create_transient_image({
        .format = GBUFFER_NORMAL_FORMAT,
        .size = {render_context->render_data.settings.render_target_size.x, render_context->render_data.settings.render_target_size.y, 1},
        .name = "main_camera_face_normal_image",
    });
    daxa::TaskImageView main_camera_detail_normal_image = tg.create_transient_image({
        .format = GBUFFER_NORMAL_FORMAT,
        .size = {render_context->render_data.settings.render_target_size.x, render_context->render_data.settings.render_target_size.y, 1},
        .name = "main_camera_detail_normal_image",
    });
    daxa::TaskImageView view_camera_face_normal_image = main_camera_face_normal_image;
    daxa::TaskImageView view_camera_detail_normal_image = main_camera_detail_normal_image;
    
    daxa::TaskImageView main_camera_half_res_face_normal_image = tg.create_transient_image({
        .format = GBUFFER_NORMAL_FORMAT,
        .size = {render_context->render_data.settings.render_target_size.x/2, render_context->render_data.settings.render_target_size.y/2, 1},
        .name = "main_camera_half_res_face_normal_image",
    });
    daxa::TaskImageView main_camera_half_res_depth_image = tg.create_transient_image({
        .format = daxa::Format::R32_SFLOAT,
        .size = {render_context->render_data.settings.render_target_size.x/2, render_context->render_data.settings.render_target_size.y/2, 1},
        .name = "main_camera_half_res_depth_image",
    });
    daxa::TaskImageView view_camera_half_res_face_normal_image = main_camera_half_res_face_normal_image;
    daxa::TaskImageView view_camera_half_res_depth_image = main_camera_half_res_depth_image;

    tg.add_task(GenGbufferTask{
        .views = GenGbufferTask::Views{
            .globals = render_context->tgpu_render_data.view(),
            .debug_image = debug_image,
            .vis_image = main_camera_visbuffer,
            .face_normal_image = main_camera_face_normal_image,
            .detail_normal_image = main_camera_detail_normal_image,
            .half_res_face_normal_image = main_camera_half_res_face_normal_image,
            .half_res_depth_image = main_camera_half_res_depth_image,
            .material_manifest = scene->_gpu_material_manifest.view(),
            .meshes = scene->_gpu_mesh_manifest.view(),
            .combined_transforms = scene->_gpu_entity_combined_transforms.view(),
            .instantiated_meshlets = meshlet_instances.view(),
        },
        .render_context = render_context.get(),
    });

    // Some following passes need either the main views camera OR the views cameras perspective.
    // The observer camera is not always appropriate to be used.
    // For example shade opaque needs view camera information while VSMs always need the main cameras perspective for generation.
    if (render_context->render_data.settings.draw_from_observer)
    {
        view_camera_face_normal_image = tg.create_transient_image({
            .format = GBUFFER_NORMAL_FORMAT,
            .size = {render_context->render_data.settings.render_target_size.x, render_context->render_data.settings.render_target_size.y, 1},
            .name = "view_camera_geo_normal_image",
        });
        view_camera_detail_normal_image = tg.create_transient_image({
            .format = GBUFFER_NORMAL_FORMAT,
            .size = {render_context->render_data.settings.render_target_size.x, render_context->render_data.settings.render_target_size.y, 1},
            .name = "view_camera_detail_normal_image",
        });
        view_camera_half_res_face_normal_image = tg.create_transient_image({
            .format = GBUFFER_NORMAL_FORMAT,
            .size = {render_context->render_data.settings.render_target_size.x/2, render_context->render_data.settings.render_target_size.y/2, 1},
            .name = "view_camera_half_res_face_normal_image",
        });
        view_camera_half_res_depth_image = tg.create_transient_image({
            .format = daxa::Format::R32_SFLOAT,
            .size = {render_context->render_data.settings.render_target_size.x/2, render_context->render_data.settings.render_target_size.y/2, 1},
            .name = "view_camera_half_res_depth_image",
        });
        tg.add_task(GenGbufferTask{
            .views = GenGbufferTask::Views{
                .globals = render_context->tgpu_render_data.view(),
                .debug_image = debug_image,
                .vis_image = view_camera_visbuffer,
                .face_normal_image = view_camera_face_normal_image,
                .detail_normal_image = view_camera_detail_normal_image,
                .half_res_face_normal_image = view_camera_half_res_face_normal_image,
                .half_res_depth_image = view_camera_half_res_depth_image,
                .material_manifest = scene->_gpu_material_manifest.view(),
                .meshes = scene->_gpu_mesh_manifest.view(),
                .combined_transforms = scene->_gpu_entity_combined_transforms.view(),
                .instantiated_meshlets = meshlet_instances.view(),
            },
            .render_context = render_context.get(),
        });
    }

    if (render_context->render_data.settings.enable_async_compute)
    {
        tg.submit({});
    }
    

    #if 1
    if (render_context->render_data.vsm_settings.enable)
    {
        vsm_state.initialize_transient_state(tg, render_context->render_data);
        task_draw_vsms(TaskDrawVSMsInfo{
            .scene = scene,
            .render_context = render_context.get(),
            .tg = &tg,
            .vsm_state = &vsm_state,
            .meshlet_instances = meshlet_instances,
            .mesh_instances = scene->mesh_instances_buffer,
            .meshes = scene->_gpu_mesh_manifest,
            .entity_combined_transforms = scene->_gpu_entity_combined_transforms,
            .material_manifest = scene->_gpu_material_manifest,
            .g_buffer_depth = main_camera_depth,
            .g_buffer_face_normal = main_camera_face_normal_image,
            .light_mask_volume = light_mask_volume,
        });
    }
    else
    #endif
    {
        vsm_state.zero_out_transient_state();
    }


    auto const vsm_page_table_view = vsm_state.page_table.view().layers(0, VSM_CLIP_LEVELS);
    auto const vsm_page_heigh_offsets_view = vsm_state.page_view_pos_row.view().layers(0, VSM_CLIP_LEVELS);
    auto const vsm_point_spot_page_table_view =
        vsm_state.point_spot_page_tables.view()
            .mips(0, s_cast<u32>(std::log2(VSM_POINT_SPOT_PAGE_TABLE_RESOLUTION)) + 1)
            .layers(0, (6 * MAX_POINT_LIGHTS) + MAX_SPOT_LIGHTS);

    auto color_image = tg.create_transient_image({
        .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
        .size = {
            render_context->render_data.settings.render_target_size.x,
            render_context->render_data.settings.render_target_size.y,
            1,
        },
        .name = "color_image",
    });

    daxa::TaskBufferView pgi_indirections = daxa::NullTaskBuffer;
    daxa::TaskImageView pgi_screen_irrdiance = daxa::NullTaskImage;
    daxa::TaskImageView pgi_irradiance = daxa::NullTaskImage;
    daxa::TaskImageView pgi_visibility = daxa::NullTaskImage;
    daxa::TaskImageView pgi_info = daxa::NullTaskImage;
    daxa::TaskImageView pgi_requests = daxa::NullTaskImage;

    if (render_context->render_data.pgi_settings.enabled)
    {
        auto ret = task_pgi_update({
            tg,
            render_context.get(),
            pgi_state,
            light_mask_volume,
            scene->mesh_instances_buffer,
            scene_main_tlas,
            transmittance,
            sky,
            vsm_state.globals,
            vsm_state.vsm_point_lights,
            vsm_state.vsm_spot_lights,
            vsm_state.memory_block,
            vsm_point_spot_page_table_view,
        });
        pgi_indirections = ret.pgi_indirections;
        pgi_irradiance = ret.pgi_irradiance;
        pgi_visibility = ret.pgi_visibility;
        pgi_info = ret.pgi_info;
        pgi_requests = ret.pgi_requests;

        if (!render_context->render_data.rtgi_settings.enabled)
        {
            pgi_screen_irrdiance = task_pgi_eval_screen_irradiance({
                tg,
                render_context.get(),
                pgi_state,
                main_camera_depth,
                main_camera_face_normal_image,
                main_camera_detail_normal_image,
                debug_image,
                clocks_image,
            });
        }
    }

    #if 0
    #endif

    daxa::TaskImageView ao_image = daxa::NullTaskImage;
    if (render_context->render_data.ao_settings.mode == AMBIENT_OCCLUSION_MODE_RTAO)
    {
        auto ao_raw_image_info = daxa::TaskTransientImageInfo{
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {
                render_context->render_data.settings.render_target_size.x,
                render_context->render_data.settings.render_target_size.y,
                1,
            },
            .name = "ao_raw_image",
        };
        auto ao_raw_image = tg.create_transient_image(ao_raw_image_info);
        auto ao_image_info = ao_raw_image_info;
        ao_image_info.name = "ao_image";
        ao_image = tg.create_transient_image(ao_image_info);
        tg.clear_image({ao_raw_image, std::array{0.0f, 0.0f, 0.0f, 0.0f}});
        tg.clear_image({ao_image, std::array{0.0f, 0.0f, 0.0f, 0.0f}});
        tg.add_task(RayTraceAmbientOcclusionTask{
            .views = RayTraceAmbientOcclusionTask::Views{
                .globals = render_context->tgpu_render_data.view(),
                .debug_image = debug_image,
                .clocks_image = clocks_image,
                .rtao_raw_image = ao_raw_image,
                .view_cam_depth = view_camera_depth,
                .view_cam_detail_normals = view_camera_detail_normal_image,
                .view_cam_visbuffer = view_camera_visbuffer,
                .meshlet_instances = meshlet_instances.view(),
                .mesh_instances = scene->mesh_instances_buffer.view(),
                .tlas = scene_main_tlas,
            },
            .gpu_context = gpu_context,
            .render_context = render_context.get(),
        });

        tg.add_task(RTAODeoinserTask{
            .views = RTAODeoinserTask::Views{
                .globals = render_context->tgpu_render_data.view(),
                .debug_image = daxa::NullTaskImage,
                .depth_history = depth_history.view(),
                .normal_history = normal_history.view(),
                .rtao_history = rtao_history.view(),
                .depth = view_camera_depth,
                .normals = view_camera_detail_normal_image,
                .rtao_raw = ao_raw_image,
                .rtao_image = ao_image,
            },
            .gpu_context = gpu_context,
            .render_context = render_context.get(),
        });
        tg.copy_image_to_image({.src = ao_image, .dst = rtao_history, .name = "save rtao history"});
    }
    #if 1

    // RTGI
    daxa::TaskImageView rtgi_per_pixel_diffuse = daxa::NullTaskImage;
    daxa::TaskImageView rtgi_debug_primary_trace = daxa::NullTaskImage;
    if (render_context->render_data.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTGI_DEBUG_PRIMARY_TRACE)
    {
        rtgi_debug_primary_trace = rtgi_create_trace_diffuse_image(tg, render_context.get(), "rtgi_debug_primary_trace");
        tg.clear_image({ .view = rtgi_debug_primary_trace, .clear_value = std::array{0.0f, 0.0f, 0.0f, 0.0f}, .name = "clear rtgi_debug_primary_trace"});
        tg.add_task(daxa::HeadTask<RtgiTraceDiffuseH::Info>()
            .head_views(RtgiTraceDiffuseH::Info::Views{
                .globals = render_context->tgpu_render_data.view(),
                .debug_image = debug_image,
                .clocks_image = clocks_image,
                .rtgi_diffuse_raw = rtgi_debug_primary_trace,
                .view_cam_half_res_depth = view_camera_half_res_depth_image,
                .view_cam_half_res_face_normals = view_camera_half_res_face_normal_image,
                .meshlet_instances = meshlet_instances.view(),
                .mesh_instances = scene->mesh_instances_buffer.view(),
                .sky = sky,
                .sky_transmittance = transmittance.view(),
                .light_mask_volume = light_mask_volume,
                .pgi_irradiance = pgi_irradiance,
                .pgi_visibility = pgi_visibility,
                .pgi_info = pgi_info,
                .pgi_requests = pgi_requests,
                .tlas = scene_main_tlas,
                .vsm_globals = vsm_state.globals.view(),
                .vsm_point_lights = vsm_state.vsm_point_lights,
                .vsm_spot_lights = vsm_state.vsm_spot_lights,
                .vsm_memory_block = vsm_state.memory_block.view(),
                .vsm_point_spot_page_table = vsm_point_spot_page_table_view,
            })
            .executes(rtgi_trace_diffuse_callback, render_context.get(), RtgiTraceDiffuseCallbackInfo{.debug_primary_trace = true}));
    }
    if (render_context->render_data.rtgi_settings.enabled)
    {
        tg.use_persistent_image(rtgi_depth_history);
        tg.use_persistent_image(rtgi_samplecnt_history);
        tg.use_persistent_image(rtgi_face_normal_history);

        auto rtgi_trace_diffuse_image = rtgi_create_diffuse_image(tg, render_context.get(), "rtgi_diffuse_raw_image");
        auto rtgi_trace_diffuse2_image = rtgi_create_diffuse2_image(tg, render_context.get(), "rtgi_diffuse2_raw_image");
        tg.add_task(daxa::HeadTask<RtgiTraceDiffuseH::Info>()
            .head_views(RtgiTraceDiffuseH::Info::Views{
                .globals = render_context->tgpu_render_data.view(),
                .debug_image = debug_image,
                .clocks_image = clocks_image,
                .rtgi_diffuse_raw = rtgi_trace_diffuse_image,
                .rtgi_diffuse2_raw = rtgi_trace_diffuse2_image,
                .view_cam_half_res_depth = view_camera_half_res_depth_image,
                .view_cam_half_res_face_normals = view_camera_half_res_face_normal_image,
                .meshlet_instances = meshlet_instances.view(),
                .mesh_instances = scene->mesh_instances_buffer.view(),
                .sky = sky,
                .sky_transmittance = transmittance.view(),
                .light_mask_volume = light_mask_volume,
                .pgi_irradiance = pgi_irradiance,
                .pgi_visibility = pgi_visibility,
                .pgi_info = pgi_info,
                .pgi_requests = pgi_requests,
                .tlas = scene_main_tlas,
                .vsm_globals = vsm_state.globals.view(),
                .vsm_point_lights = vsm_state.vsm_point_lights,
                .vsm_spot_lights = vsm_state.vsm_spot_lights,
                .vsm_memory_block = vsm_state.memory_block.view(),
                .vsm_point_spot_page_table = vsm_point_spot_page_table_view,
            })
            .executes(rtgi_trace_diffuse_callback, render_context.get(), RtgiTraceDiffuseCallbackInfo{.debug_primary_trace = false}));

        //auto rtgi_diffuse_reprojected_image = rtgi_create_diffuse_image(tg, render_context.get(), "rtgi_diffuse_reprojected_image");
        //auto rtgi_diffuse2_reprojected_image = rtgi_create_diffuse2_image(tg, render_context.get(), "rtgi_diffuse2_reprojected_image");
        auto rtgi_samplecnt_image = rtgi_create_samplecnt_image(tg, render_context.get(), "rtgi_samplecnt_image");
        tg.add_task(daxa::HeadTask<RtgiReprojectDiffuseH::Info>()
            .head_views(RtgiReprojectDiffuseH::Info::Views{
                .globals = render_context->tgpu_render_data.view(),
                .debug_image = debug_image,
                .clocks_image = clocks_image,
                .rtgi_depth_history = rtgi_depth_history.view(),
                .rtgi_samplecnt_history = rtgi_samplecnt_history.view(),
                .rtgi_face_normal_history = rtgi_face_normal_history.view(),
                .rtgi_samplecnt = rtgi_samplecnt_image,
                .view_cam_half_res_depth = view_camera_half_res_depth_image,
                .view_cam_half_res_face_normals = view_camera_half_res_face_normal_image,
            })
            .executes(rtgi_denoise_diffuse_reproject_callback, render_context.get()));

        
        // tg.add_task(daxa::HeadTask<RtgiReprojectFullH::Info>()
        //     .head_views(RtgiReprojectFullH::Info::Views{
        //         .globals = render_context->tgpu_render_data.view(),
        //         .debug_image = debug_image,
        //         .clocks_image = clocks_image,
        //         .rtgi_diffuse_history = rtgi_full_diffuse_history.view(),
        //         .rtgi_diffuse2_history = rtgi_full_diffuse2_history.view(),
        //         .rtgi_depth_history = depth_history.view(),
        //         .rtgi_samplecnt_history = rtgi_full_samplecount_history.view(),
        //         .rtgi_face_normal_history = rtgi_full_face_normal_history.view(),
        //         .rtgi_samplecnt = rtgi_full_samplecount_image,
        //         .rtgi_diffuse_reprojected = rtgi_full_diffuse_reprojected_image,
        //         .rtgi_diffuse2_reprojected = rtgi_full_diffuse2_reprojected_image,
        //         .view_cam_depth = view_camera_depth,
        //         .view_cam_face_normals = view_camera_face_normal_image,
        //     })
        //     .executes(rtgi_reproject_full_callback, render_context.get()));
        auto rtgi_flattened_diffuse_image = rtgi_create_diffuse_image(tg, render_context.get(), "rtgi_flattened_diffuse_image");
        auto rtgi_flattened_diffuse2_image = rtgi_create_diffuse2_image(tg, render_context.get(), "rtgi_flattened_diffuse2_image");
        tg.add_task(daxa::HeadTask<RtgiPreBlurFlattenH::Info>()
            .head_views({
                .globals = render_context->tgpu_render_data.view(),
                .debug_image = debug_image,
                .clocks_image = clocks_image,
                .rtgi_diffuse_raw = rtgi_trace_diffuse_image,
                .rtgi_diffuse2_raw = rtgi_trace_diffuse2_image,
                .view_cam_half_res_normals = view_camera_half_res_face_normal_image,
                .rtgi_flattened_diffuse = rtgi_flattened_diffuse_image,
                .rtgi_flattened_diffuse2 = rtgi_flattened_diffuse2_image,
                .view_cam_half_res_depth = view_camera_half_res_depth_image,
            })
            .executes(rtgi_pre_blur_flatten_callback, render_context.get()));
        
        auto rtgi_pre_blur_mips_image = rtgi_create_reconstructed_history_image(tg, render_context.get(), "rtgi_pre_blur_mips_image").mips(0,5);
        auto rtgi_pre_blur_mips2_image = rtgi_create_reconstructed_history_image(tg, render_context.get(), "rtgi_pre_blur_mips2_image").mips(0,5);
        tg.add_task(daxa::HeadTask<RtgiPreblurPrepareH::Info>()
            .head_views({
                .globals = render_context->tgpu_render_data.view(),
                .debug_image = debug_image,
                .clocks_image = clocks_image,
                .rtgi_diffuse_raw = rtgi_flattened_diffuse_image,
                .rtgi_diffuse2_raw = rtgi_flattened_diffuse2_image,
                .rtgi_samplecnt = rtgi_samplecnt_image,
                .view_cam_half_res_normals = view_camera_half_res_face_normal_image,
                .rtgi_reconstructed_diffuse_history = rtgi_pre_blur_mips_image,
                .rtgi_reconstructed_diffuse2_history = rtgi_pre_blur_mips2_image,
                .view_cam_half_res_depth = view_camera_half_res_depth_image,
            })
            .executes(rtgi_pre_blur_prepare_callback, render_context.get()));

        auto rtgi_pre_blurred_diffuse_image = rtgi_create_diffuse_image(tg, render_context.get(), "rtgi_pre_blurred_diffuse_image");
        auto rtgi_pre_blurred_diffuse2_image = rtgi_create_diffuse2_image(tg, render_context.get(), "rtgi_pre_blurred_diffuse2_image");
        tg.add_task(daxa::HeadTask<RtgiPreBlurApply::Info>()
            .head_views({
                .globals = render_context->tgpu_render_data.view(),
                .debug_image = debug_image,
                .clocks_image = clocks_image,
                .rtgi_reconstructed_diffuse_history = rtgi_pre_blur_mips_image,
                .rtgi_reconstructed_diffuse2_history = rtgi_pre_blur_mips2_image,
                .view_cam_half_res_depth = view_camera_half_res_depth_image,
                .view_cam_half_res_normals = view_camera_half_res_face_normal_image,
                .rtgi_samplecnt = rtgi_samplecnt_image,
                .rtgi_diffuse_filtered = rtgi_pre_blurred_diffuse_image,
                .rtgi_diffuse2_filtered = rtgi_pre_blurred_diffuse2_image,
            })
            .executes(rtgi_pre_blur_apply_callback, render_context.get()));

        daxa::TaskImageView rtgi_diffuse_filtered = rtgi_pre_blurred_diffuse_image.mips(0); 
        daxa::TaskImageView rtgi_diffuse2_filtered = rtgi_pre_blurred_diffuse2_image;
        if (render_context->render_data.rtgi_settings.spatial_filter_enabled)
        {
            auto rtgi_blurred0_diffuse_image = rtgi_create_diffuse_image(tg, render_context.get(), "rtgi_blurred0_diffuse_image");
            auto rtgi_blurred0_diffuse2_image = rtgi_create_diffuse2_image(tg, render_context.get(), "rtgi_blurred0_diffuse2_image");
            tg.add_task(daxa::HeadTask<RtgiAdaptiveBlurH::Info>("RtgiAdaptiveBlur1")
                .head_views({
                    .globals = render_context->tgpu_render_data.view(),
                    .debug_image = debug_image,
                    .clocks_image = clocks_image,
                    .rtgi_diffuse_before = rtgi_pre_blurred_diffuse_image,
                    .rtgi_diffuse2_before = rtgi_pre_blurred_diffuse2_image,
                    .rtgi_samplecnt = rtgi_samplecnt_image,
                    .view_cam_half_res_depth = view_camera_half_res_depth_image,
                    .view_cam_half_res_face_normals = view_camera_half_res_face_normal_image,
                    .rtgi_diffuse_blurred = rtgi_blurred0_diffuse_image,
                    .rtgi_diffuse2_blurred = rtgi_blurred0_diffuse2_image,
                })
                .executes(rtgi_adaptive_blur_diffuse_callback, render_context.get(), 0u));

            auto rtgi_blurred1_diffuse_image = rtgi_create_diffuse_image(tg, render_context.get(), "rtgi_blurred1_diffuse_image");
            auto rtgi_blurred1_diffuse2_image = rtgi_create_diffuse2_image(tg, render_context.get(), "rtgi_blurred1_diffuse2_image");
            tg.add_task(daxa::HeadTask<RtgiAdaptiveBlurH::Info>("RtgiAdaptiveBlur2")
                .head_views({
                    .globals = render_context->tgpu_render_data.view(),
                    .debug_image = debug_image,
                    .clocks_image = clocks_image,
                    .rtgi_diffuse_before = rtgi_blurred0_diffuse_image,
                    .rtgi_diffuse2_before = rtgi_blurred0_diffuse2_image,
                    .rtgi_samplecnt = rtgi_samplecnt_image,
                    .view_cam_half_res_depth = view_camera_half_res_depth_image,
                    .view_cam_half_res_face_normals = view_camera_half_res_face_normal_image,
                    .rtgi_diffuse_blurred = rtgi_blurred1_diffuse_image,
                    .rtgi_diffuse2_blurred = rtgi_blurred1_diffuse2_image,
                })
                .executes(rtgi_adaptive_blur_diffuse_callback, render_context.get(), 1u));
            rtgi_diffuse_filtered = rtgi_blurred1_diffuse_image;
            rtgi_diffuse2_filtered = rtgi_blurred1_diffuse2_image;
        }

        //auto rtgi_diffuse_stable_image = rtgi_create_diffuse_image(tg, render_context.get(), "rtgi_diffuse_stable_image");
        //auto rtgi_diffuse2_stable_image = rtgi_create_diffuse2_image(tg, render_context.get(), "rtgi_diffuse2_stable_image");
        //auto rtgi_diffuse_accumulated_image = rtgi_create_diffuse_image(tg, render_context.get(), "rtgi_diffuse_accumulated_image");
        //auto rtgi_diffuse2_accumulated_image = rtgi_create_diffuse2_image(tg, render_context.get(), "rtgi_diffuse2_accumulated_image");
        //tg.add_task(daxa::HeadTask<RtgiDiffuseTemporalStabilizationH::Info>()
        //    .head_views({
        //        .globals = render_context->tgpu_render_data.view(),
        //        .debug_image = debug_image,
        //        .clocks_image = clocks_image,
        //        .rtgi_diffuse_blurred = rtgi_diffuse_filtered,
        //        .rtgi_diffuse2_blurred = rtgi_diffuse2_filtered,
        //        .rtgi_diffuse_reprojected = rtgi_diffuse_reprojected_image,
        //        .rtgi_diffuse2_reprojected = rtgi_diffuse2_reprojected_image,
        //        .rtgi_samplecnt = rtgi_samplecnt_image,
        //        .rtgi_diffuse_stable = rtgi_diffuse_stable_image,
        //        .rtgi_diffuse2_stable = rtgi_diffuse2_stable_image,
        //        .rtgi_diffuse_accumulated = rtgi_diffuse_accumulated_image,
        //        .rtgi_diffuse2_accumulated = rtgi_diffuse2_accumulated_image,
        //        .view_cam_half_res_depth = view_camera_half_res_depth_image,
        //        .view_cam_half_res_face_normals = view_camera_half_res_face_normal_image,
        //    })
        //    .executes(rtgi_diffuse_temporal_stabilization_callback, render_context.get()));

            
        auto rtgi_full_diffuse_accumulated_image = rtgi_create_diffuse_image(tg, render_context.get(), "rtgi_full_diffuse_accumulated_image", 1);
        auto rtgi_full_diffuse2_accumulated_image = rtgi_create_diffuse2_image(tg, render_context.get(), "rtgi_full_diffuse2_accumulated_image", 1);
        auto rtgi_full_samplecount_image = rtgi_create_samplecnt_image(tg, render_context.get(), "rtgi_full_samplecount_image", 1);
        rtgi_per_pixel_diffuse = rtgi_create_upscaled_diffuse_image(tg, render_context.get(), "rtgi_per_pixel_diffuse");

        auto rtgi_accumualted_color_image = tg.create_transient_image({
            .format = daxa::Format::R32G32_UINT,
            .size = { 
                render_context->render_data.settings.render_target_size.x,
                render_context->render_data.settings.render_target_size.y,
                1,
            },
            .name = "rtgi_accumualted_color_image",
        });
        auto rtgi_accumulated_statistics_image = tg.create_transient_image({
            .format = daxa::Format::R32_UINT,
            .size = { 
                render_context->render_data.settings.render_target_size.x,
                render_context->render_data.settings.render_target_size.y,
                1,
            },
            .name = "rtgi_accumulated_statistics_image",
        });

        tg.add_task(daxa::HeadTask<RtgiUpscaleDiffuseH::Info>()
            .head_views(RtgiUpscaleDiffuseH::Info::Views{
                .globals = render_context->tgpu_render_data.view(),
                .debug_image = debug_image,
                .clocks_image = clocks_image,

                .rtgi_color_history_full_res = rtgi_full_color_history.view(),
                .rtgi_statistics_history_full_res = rtgi_full_statistics_history.view(),
                .rtgi_accumulated_color_full_res = rtgi_accumualted_color_image,
                .rtgi_accumulated_statistics_full_res = rtgi_accumulated_statistics_image,

                .rtgi_depth_history_full_res = depth_history,
                .rtgi_face_normal_history_full_res = rtgi_full_face_normal_history,
                .rtgi_samplecount_history_full_res = rtgi_full_samplecount_history,

                .rtgi_diffuse_full_res = rtgi_full_diffuse_accumulated_image,
                .rtgi_diffuse2_full_res = rtgi_full_diffuse2_accumulated_image,
                .rtgi_samplecount_full_res = rtgi_full_samplecount_image,

                .rtgi_diffuse_half_res = rtgi_diffuse_filtered,
                .rtgi_diffuse2_half_res = rtgi_diffuse2_filtered,

                .rtgi_samplecount_half_res = rtgi_samplecnt_image,
                .view_cam_half_res_depth = view_camera_half_res_depth_image,
                .view_cam_half_res_face_normals = view_camera_half_res_face_normal_image,
                .view_cam_depth = view_camera_depth,
                .view_cam_face_normals = view_camera_face_normal_image,
                .view_camera_detail_normal_image = view_camera_detail_normal_image,
                .rtgi_diffuse_resolved = rtgi_per_pixel_diffuse,
            })
            .executes(rtgi_upscale_diffuse_callback, render_context.get()));

        tg.copy_image_to_image({.src = view_camera_half_res_depth_image, .dst = rtgi_depth_history.view(), .name = "save rtgi depth history"});
        tg.copy_image_to_image({.src = rtgi_samplecnt_image, .dst = rtgi_samplecnt_history.view(), .name = "save rtgi samplecnt history"});
        tg.copy_image_to_image({.src = view_camera_half_res_face_normal_image, .dst = rtgi_face_normal_history.view(), .name = "save rtgi face normal history"});
        tg.copy_image_to_image({.src = rtgi_full_samplecount_image, .dst = rtgi_full_samplecount_history, .name = "save rtgi_full_samplecount_history"});
        tg.copy_image_to_image({.src = view_camera_face_normal_image, .dst = rtgi_full_face_normal_history, .name = "save rtgi_full_face_normal_history"});
        tg.copy_image_to_image({.src = rtgi_accumualted_color_image, .dst = rtgi_full_color_history, .name = "save rtgi_full_color_history"});
        tg.copy_image_to_image({.src = rtgi_accumulated_statistics_image, .dst = rtgi_full_statistics_history, .name = "save rtgi_full_statistics_history"});
    }

    #endif

    auto selected_mark_image = tg.create_transient_image({
        .format = daxa::Format::R8_UNORM,
        .size = {render_context->render_data.settings.render_target_size.x, render_context->render_data.settings.render_target_size.y, 1},
        .name = "selected mark image",
    });
    tg.clear_image({selected_mark_image, {}, daxa::QUEUE_MAIN, "clear selected mark image"});
    if (render_context->render_data.settings.enable_reference_path_trace)
    {
        // TODO: Precompute once, and save
        auto brdf_fg_lut = tg.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {64, 64, 1},
            .name = "brdf_fg_lut",
        });
        tg.add_task(BrdfFgTask{
            .views = BrdfFgTask::Views{
                .globals = render_context->tgpu_render_data.view(),
                .output_tex = brdf_fg_lut,
            },
            .render_context = render_context.get(),
        });

        tg.add_task(ReferencePathTraceTask{
            .views = ReferencePathTraceTask::Views{
                .globals = render_context->tgpu_render_data.view(),
                .debug_image = debug_image,
                .pt_image = color_image,
                .history_image = path_trace_history.view(),
                .vis_image = view_camera_visbuffer,
                .transmittance = transmittance.view(),
                .sky = sky,
                .sky_ibl = sky_ibl_view,
                .brdf_lut = brdf_fg_lut,
                .exposure = exposure_state.view(),
                .meshlet_instances = meshlet_instances.view(),
                .mesh_instances = scene->mesh_instances_buffer.view(),
                .tlas = scene_main_tlas,
            },
            .gpu_context = gpu_context,
            .render_context = render_context.get(),
        });
    }
    else
    {
        tg.add_task(ShadeOpaqueTask{
            .views = ShadeOpaqueTask::Views{
                .globals = render_context->tgpu_render_data.view(),
                .color_image = color_image,
                .selected_mark_image = selected_mark_image,
                .ao_image = ao_image,
                .vis_image = view_camera_visbuffer,
                .pgi_screen_irrdiance = pgi_screen_irrdiance,
                .depth = view_camera_depth,
                .debug_image = debug_image,
                .clocks_image = clocks_image,
                .vsm_overdraw_debug = vsm_state.overdraw_debug_image,
                .transmittance = transmittance.view(),
                .sky = sky,
                .sky_ibl = sky_ibl_view,
                .vsm_page_table = vsm_page_table_view,
                .vsm_page_view_pos_row = vsm_page_heigh_offsets_view,
                .vsm_memory_block = vsm_state.memory_block.view(),
                .overdraw_image = overdraw_image,
                .vsm_point_spot_page_table = vsm_point_spot_page_table_view,
                .material_manifest = scene->_gpu_material_manifest.view(),
                .instantiated_meshlets = meshlet_instances.view(),
                .meshes = scene->_gpu_mesh_manifest.view(),
                .combined_transforms = scene->_gpu_entity_combined_transforms.view(),
                .exposure = exposure_state.view(),
                .vsm_clip_projections = vsm_state.clip_projections,
                .vsm_globals = vsm_state.globals.view(),
                .vsm_point_lights = vsm_state.vsm_point_lights,
                .vsm_spot_lights = vsm_state.vsm_spot_lights,
                .vsm_wrapped_pages = vsm_state.free_wrapped_pages_info,
                .point_lights = scene->_gpu_point_lights.view(),
                .spot_lights = scene->_gpu_spot_lights.view(),
                .mesh_instances = scene->mesh_instances_buffer.view(),
                .light_mask_volume = light_mask_volume,
                .pgi_irradiance = pgi_irradiance,
                .pgi_visibility = pgi_visibility,
                .pgi_info = pgi_info,
                .pgi_requests = pgi_requests,
                .rtgi_per_pixel_diffuse = rtgi_per_pixel_diffuse,
                .rtgi_debug_primary_trace = rtgi_debug_primary_trace,
                .tlas = scene_main_tlas,
            },
            .render_context = render_context.get(),
        });

        if(render_context->render_data.vsm_settings.enable)
        {
            tg.clear_image({render_context->gpu_context->shader_debug_context.vsm_debug_page_table, std::array{0.0f, 0.0f, 0.0f, 0.0f}});
            tg.add_task(DebugVirtualPageTableTask{
                .views = DebugVirtualPageTableTask::Views{
                    .globals = render_context->tgpu_render_data.view(),
                    .vsm_globals = vsm_state.globals.view(),
                    .vsm_page_table = vsm_page_table_view,
                    .vsm_debug_page_table = render_context->gpu_context->shader_debug_context.vsm_debug_page_table.view(),
                },
                .render_context = render_context.get(),
            });

            tg.clear_image({render_context->gpu_context->shader_debug_context.vsm_recreated_shadowmap_memory_table, std::array{0.0f, 0.0f, 0.0f, 0.0f}});
            tg.add_task(RecreateShadowMapTask{
                .views = RecreateShadowMapTask::Views{
                    .globals = render_context->tgpu_render_data.view(),
                    .vsm_clip_projections = vsm_state.clip_projections,
                    .vsm_page_table = vsm_page_table_view,
                    .vsm_memory_block = vsm_state.memory_block.view(),
                    .vsm_overdraw_debug = vsm_state.overdraw_debug_image,
                    .vsm_recreated_shadow_map = render_context->gpu_context->shader_debug_context.vsm_recreated_shadowmap_memory_table.view(),
                },
                .render_context = render_context.get(),
            });

            tg.clear_image({render_context->gpu_context->shader_debug_context.vsm_debug_meta_memory_table, std::array{0.0f, 0.0f, 0.0f, 0.0f}});
            tg.add_task(DebugMetaMemoryTableTask{
                .views = DebugMetaMemoryTableTask::Views{
                    .globals = render_context->tgpu_render_data.view(),
                    .vsm_page_table = vsm_page_table_view,
                    .vsm_meta_memory_table = vsm_state.meta_memory_table.view(),
                    .vsm_debug_meta_memory_table = render_context->gpu_context->shader_debug_context.vsm_debug_meta_memory_table.view(),
                    .vsm_point_spot_page_table = vsm_point_spot_page_table_view,
                },
                .render_context = render_context.get(),
            });
        }
    }

    tg.clear_buffer({.buffer = luminance_histogram, .clear_value = 0});
    tg.add_task(GenLuminanceHistogramTask{
        .views = GenLuminanceHistogramTask::Views{
            .globals = render_context->tgpu_render_data.view(),
            .histogram = luminance_histogram,
            .exposure_state = exposure_state.view(),
            .color_image = color_image,
        },
        .render_context = render_context.get(),
    });
    tg.add_task(GenLuminanceAverageTask{
        .views = GenLuminanceAverageTask::Views{
            .globals = render_context->tgpu_render_data.view(),
            .histogram = luminance_histogram,
            .exposure_state = exposure_state.view(),
        },
        .render_context = render_context.get(),
    });
    daxa::TaskImageView debug_draw_depth = tg.create_transient_image({
        .format = daxa::Format::D32_SFLOAT,
        .size = {render_context->render_data.settings.render_target_size.x, render_context->render_data.settings.render_target_size.y, 1},
        .name = "debug depth",
    });
    tg.copy_image_to_image({view_camera_depth, debug_draw_depth, daxa::QUEUE_MAIN, "copy depth to debug depth"});
    if (render_context->render_data.pgi_settings.enabled)
    {
        tg.add_task(PGIDrawDebugProbesTask{
            .views = PGIDrawDebugProbesTask::Views{
                .globals = render_context->tgpu_render_data.view(),
                .exposure = exposure_state.view(),
                .probe_indirections = pgi_indirections,
                .color_image = color_image,
                .depth_image = debug_draw_depth,
                .probe_color = pgi_state.probe_color_view,
                .probe_visibility = pgi_state.probe_visibility_view,
                .probe_info = pgi_state.probe_info_view,
                .probe_requests = pgi_state.cell_requests_view,
                .tlas = scene_main_tlas,
            },
            .render_context = render_context.get(),
            .pgi_state = &pgi_state,
        });
    }
    tg.add_task(DebugDrawTask{
        .views = DebugDrawTask::Views{
            .globals = render_context->tgpu_render_data.view(),
            .color_image = color_image,
            .depth_image = debug_draw_depth,
        },
        .render_context = render_context.get(),
    });

    tg.add_task(CopyDepthTask{
        .views = CopyDepthTask::Views{
            .depth_src = main_camera_depth,
            .depth_dst_f32 = depth_history.view(),
        },
        .render_context = render_context.get(),
    });
    tg.copy_image_to_image({main_camera_detail_normal_image, normal_history, daxa::QUEUE_MAIN, "copy to normal history"});

    tg.add_task(WriteSwapchainTask{
        .views = WriteSwapchainTask::Views{
            .globals = render_context->tgpu_render_data.view(),
            .selected_mark_image = selected_mark_image,
            .debug_image = debug_image,
            .color_image = color_image,
            .swapchain = swapchain_image.view(),
        },
        .gpu_context = gpu_context,
    });
    
    #if 0
    tg.add_task(daxa::HeadTask<WriteSwapchainDebugH::Info>()
        .head_views({
            .globals = render_context->tgpu_render_data.view(),
            .debug_image = debug_image,
            .depth_image = view_camera_depth,
            .swapchain = swapchain_image.view(),
        })
        .executes(write_swapchain_debug_callback, render_context.get()));
    #endif


    tg.add_task(daxa::InlineTask{"ImGui Draw"}
            .color_attachment.reads_writes(swapchain_image)
            .executes(
                [=, this](daxa::TaskInterface ti)
                {
                    ImGui::Render();
                    auto size = ti.info(swapchain_image.view()).value().size;
                    imgui_renderer->record_commands(
                        ImGui::GetDrawData(), ti.recorder, ti.id(swapchain_image.view()), size.x, size.y);
                }));

    tg.submit({});
    tg.present({});
    tg.complete({});
    return tg;
}

auto Renderer::prepare_frame(
    u32 frame_index,
    CameraInfo const & camera_info,
    CameraInfo const & observer_camera_info,
    f32 const delta_time,
    u64 const total_elapsed_us) -> bool
{
    if (window->size.x == 0 || window->size.y == 0)
    {
        return false;
    }

    render_context->render_data.vsm_settings.point_light_count = s_cast<u32>(scene->_point_lights.size());
    render_context->render_data.vsm_settings.spot_light_count = s_cast<u32>(scene->_spot_lights.size());

    // Calculate frame relevant values.
    daxa_u32vec2 render_target_size = {static_cast<daxa_u32>(window->size.x), static_cast<daxa_u32>(window->size.y)};
    if (render_context->render_data.settings.anti_aliasing_mode == AA_MODE_SUPER_SAMPLE)
    {
        render_target_size.x *= 2;
        render_target_size.y *= 2;
    }

    // Update render_data.
    {
        u32 res_factor = 1;
        render_context->render_data.settings.window_size = {static_cast<u32>(window->size.x), static_cast<u32>(window->size.y)};
        if (render_context->render_data.settings.anti_aliasing_mode == AA_MODE_SUPER_SAMPLE)
        {
            res_factor = 2;
        }
        render_context->render_data.settings.render_target_size.x = render_target_size.x;
        render_context->render_data.settings.render_target_size.y = render_target_size.y;
        render_context->render_data.settings.render_target_size_inv = {
            1.0f / render_context->render_data.settings.render_target_size.x,
            1.0f / render_context->render_data.settings.render_target_size.y,
        };
        render_context->render_data.settings.next_lower_po2_render_target_size.x = find_next_lower_po2(render_target_size.x);
        render_context->render_data.settings.next_lower_po2_render_target_size.y = find_next_lower_po2(render_target_size.y);
        render_context->render_data.settings.next_lower_po2_render_target_size_inv = {
            1.0f / render_context->render_data.settings.next_lower_po2_render_target_size.x,
            1.0f / render_context->render_data.settings.next_lower_po2_render_target_size.y,
        };
        render_context->mesh_instance_counts = scene->cpu_mesh_instance_counts;

        /// THIS SHOULD BE DONE SOMEWHERE ELSE!
        {
            auto reloaded_result = gpu_context->pipeline_manager.reload_all();
            if (auto reload_err = daxa::get_if<daxa::PipelineReloadError>(&reloaded_result))
            {
                std::cout << "Failed to reload " << reload_err->message << '\n';
            }
            if (auto _ = daxa::get_if<daxa::PipelineReloadSuccess>(&reloaded_result))
            {
                std::cout << "Successfully reloaded!\n";
                for (auto [name, pipe] : gpu_context->ray_tracing_pipelines)
                {
                    this->gpu_context->device.destroy_buffer(pipe.sbt_buffer_id);
                    auto sbt_info = gpu_context->ray_tracing_pipelines[name].pipeline->create_default_sbt();
                    this->gpu_context->ray_tracing_pipelines[name].sbt = sbt_info.table;
                    this->gpu_context->ray_tracing_pipelines[name].sbt_buffer_id = sbt_info.buffer;
                }
            }
        }

        CameraInfo real_camera_info = camera_info;
        if (render_context->debug_frustum >= 0)
        {
            real_camera_info = vsm_state.point_lights_cpu.at(std::max(render_context->render_data.vsm_settings.force_point_light_idx, 0)).face_cameras[render_context->debug_frustum];
            real_camera_info.screen_size = camera_info.screen_size;
            real_camera_info.inv_screen_size = camera_info.inv_screen_size;
        }

        // Set Render Data.

        // Written by ui     render_context->render_data.hovered_entity_index
        // Written by ui     render_context->render_data.selected_entity_index

        render_context->render_data.main_camera_prev_frame = render_context->render_data.main_camera;
        render_context->render_data.main_camera = real_camera_info;
        render_context->render_data.view_camera_prev_frame = render_context->render_data.view_camera;
        render_context->render_data.view_camera = render_context->render_data.settings.draw_from_observer ? observer_camera_info : real_camera_info;
        render_context->render_data.frame_index = frame_index;
        render_context->render_data.frames_in_flight = MAX_GPU_FRAMES_IN_FLIGHT;
        render_context->render_data.delta_time = delta_time;
        render_context->render_data.total_elapsed_us = total_elapsed_us;

        render_context->render_data.cull_data = fill_cull_data(*render_context);

        pgi_resolve_settings(render_context->prev_pgi_settings, render_context->render_data);
    }

    ///
    /// SETTING RESOLVE END
    ///
    /// GRAPH RECORD START
    ///

    if (render_context->render_data.frame_index == 0)
    {
        pgi_state.recreate_and_clear(render_context->gpu_context->device, render_context->render_data.pgi_settings);
        clear_select_buffers();
    }

    bool const settings_changed = render_context->render_data.settings != render_context->prev_settings;
    bool const ao_settings_changed = render_context->render_data.ao_settings.mode != render_context->prev_ao_settings.mode;
    bool const rtgi_settings_changed = 
        render_context->render_data.rtgi_settings.enabled != render_context->prev_rtgi_settings.enabled || 
        render_context->render_data.rtgi_settings.spatial_filter_enabled != render_context->prev_rtgi_settings.spatial_filter_enabled;
    bool const light_settings_changed = lights_significant_settings_change(render_context->render_data.light_settings, render_context->prev_light_settings);
    bool const pgi_settings_changed = pgi_significant_settings_change(render_context->prev_pgi_settings, render_context->render_data.pgi_settings);
    bool const sky_settings_changed = render_context->render_data.sky_settings != render_context->prev_sky_settings;
    auto const sky_res_changed_flags = render_context->render_data.sky_settings.resolutions_changed(render_context->prev_sky_settings);
    bool const vsm_settings_changed =
        render_context->render_data.vsm_settings.enable != render_context->prev_vsm_settings.enable;
    if (pgi_settings_changed)
    {
        pgi_state.recreate_and_clear(render_context->gpu_context->device, render_context->render_data.pgi_settings);
    }
    if (settings_changed || sky_res_changed_flags.sky_changed || vsm_settings_changed || pgi_settings_changed || light_settings_changed || ao_settings_changed || rtgi_settings_changed)
    {
        this->gpu_context->device.wait_idle();
        this->gpu_context->device.collect_garbage();
        gpu_context->swapchain.set_present_mode(render_context->render_data.settings.enable_vsync ? daxa::PresentMode::FIFO : daxa::PresentMode::IMMEDIATE);
        main_task_graph = create_main_task_graph();
        recreate_framebuffer();
        clear_select_buffers();
    }
    {
        // std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        // main_task_graph = create_main_task_graph();
        // std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        // u32 duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        // std::cout << "tg compile took " << duration << "us" << std::endl;
    }
    daxa::DeviceAddress render_data_device_address =
        gpu_context->device.buffer_device_address(render_context->tgpu_render_data.get_state().buffers[0]).value();

    // Do General Readback
    {
        u32 const index = render_context->render_data.frame_index % MAX_GPU_FRAMES_IN_FLIGHT;
        render_context->render_data.readback = gpu_context->device.buffer_device_address(general_readback_buffer).value() + sizeof(ReadbackValues) * index;
        render_context->general_readback = render_context->gpu_context->device.buffer_host_address_as<ReadbackValues>(general_readback_buffer).value()[index];
        render_context->gpu_context->device.buffer_host_address_as<ReadbackValues>(general_readback_buffer).value()[index] = {}; // clear for next frame
    }

    if (sky_settings_changed)
    {
        // Potentially wastefull, ideally we want to only recreate the resource that changed the name
        if (sky_res_changed_flags.multiscattering_changed || sky_res_changed_flags.transmittance_changed)
        {
            recreate_sky_luts();
        }
        // Whenever the settings change we need to recalculate the transmittance and multiscattering look up textures
        auto const sky_settings_offset = offsetof(RenderGlobalData, sky_settings);
        auto const mie_density_offset = sky_settings_offset + offsetof(SkySettings, mie_density);
        render_context->render_data.sky_settings.mie_density_ptr = render_data_device_address + mie_density_offset;
        auto const rayleigh_density_offset = sky_settings_offset + offsetof(SkySettings, rayleigh_density);
        render_context->render_data.sky_settings.rayleigh_density_ptr = render_data_device_address + rayleigh_density_offset;
        auto const absoprtion_density_offset = sky_settings_offset + offsetof(SkySettings, absorption_density);
        render_context->render_data.sky_settings.absorption_density_ptr = render_data_device_address + absoprtion_density_offset;

        render_context->render_data.sky_settings = render_context->render_data.sky_settings;
        sky_task_graph.execute({});
    }
    bool sun_moved = std::bit_cast<f32vec3>(render_context->prev_sky_settings.sun_direction) ==
                     std::bit_cast<f32vec3>(render_context->render_data.sky_settings.sun_direction);
    render_context->render_data.vsm_settings.sun_moved = sun_moved ? 0u : 1u;
    render_context->prev_settings = render_context->render_data.settings;
    render_context->prev_pgi_settings = render_context->render_data.pgi_settings;
    render_context->prev_sky_settings = render_context->render_data.sky_settings;
    render_context->prev_vsm_settings = render_context->render_data.vsm_settings;
    render_context->prev_light_settings = render_context->render_data.light_settings;
    render_context->prev_ao_settings = render_context->render_data.ao_settings;
    render_context->prev_rtgi_settings = render_context->render_data.rtgi_settings;

    // Write GPUScene
    {
        auto & device = render_context->gpu_context->device;
        render_context->render_data.scene.meshes = device.device_address(scene->_gpu_mesh_manifest.get_state().buffers[0]).value();
        render_context->render_data.scene.mesh_lod_groups = device.device_address(scene->_gpu_mesh_lod_group_manifest.get_state().buffers[0]).value();
        render_context->render_data.scene.mesh_groups = device.device_address(scene->_gpu_mesh_group_manifest.get_state().buffers[0]).value();
        render_context->render_data.scene.entity_to_meshgroup = device.device_address(scene->_gpu_entity_mesh_groups.get_state().buffers[0]).value();
        render_context->render_data.scene.materials = device.device_address(scene->_gpu_material_manifest.get_state().buffers[0]).value();
        render_context->render_data.scene.material_count = scene->_material_manifest.size();
        render_context->render_data.scene.entity_transforms = device.device_address(scene->_gpu_entity_transforms.get_state().buffers[0]).value();
        render_context->render_data.scene.entity_combined_transforms = device.device_address(scene->_gpu_entity_combined_transforms.get_state().buffers[0]).value();
        render_context->render_data.scene.point_lights = device.device_address(scene->_gpu_point_lights.get_state().buffers[0]).value();
        render_context->render_data.scene.spot_lights = device.device_address(scene->_gpu_spot_lights.get_state().buffers[0]).value();
    }

    auto const vsm_projections_info = GetVSMProjectionsInfo{
        .camera_info = &render_context->render_data.main_camera,
        .sun_direction = std::bit_cast<f32vec3>(render_context->render_data.sky_settings.sun_direction),
        .clip_0_scale = render_context->render_data.vsm_settings.clip_0_frustum_scale,
        .clip_0_near = render_context->render_data.vsm_settings.fixed_near_far ? -1'000.0f : 0.01f,
        .clip_0_far = render_context->render_data.vsm_settings.fixed_near_far ? 1'000.0f : 10.0f,
        .clip_0_height_offset = 5.0f,
        .use_fixed_near_far = s_cast<bool>(render_context->render_data.vsm_settings.fixed_near_far),
        .debug_context = &gpu_context->shader_debug_context,
    };
    vsm_state.clip_projections_cpu = get_vsm_projections(vsm_projections_info);

    for (i32 clip = 0; clip < VSM_CLIP_LEVELS; clip++)
    {
        auto const clear_offset = std::bit_cast<i32vec2>(vsm_state.clip_projections_cpu.at(clip).page_offset) - vsm_state.last_frame_offsets.at(clip);
        vsm_state.free_wrapped_pages_info_cpu.at(clip).clear_offset = std::bit_cast<daxa_i32vec2>(clear_offset);

        vsm_state.last_frame_offsets.at(clip) = std::bit_cast<i32vec2>(vsm_state.clip_projections_cpu.at(clip).page_offset);
        vsm_state.clip_projections_cpu.at(clip).page_offset.x = vsm_state.clip_projections_cpu.at(clip).page_offset.x % VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION;
        vsm_state.clip_projections_cpu.at(clip).page_offset.y = vsm_state.clip_projections_cpu.at(clip).page_offset.y % VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION;
    }
    vsm_state.globals_cpu.clip_0_texel_world_size = (2.0f * render_context->render_data.vsm_settings.clip_0_frustum_scale) / VSM_DIRECTIONAL_TEXTURE_RESOLUTION;
    vsm_state.update_vsm_lights(scene->_point_lights, scene->_spot_lights);

    lights_resolve_settings(render_context->render_data);

    if (render_context->visualize_point_frustum)
    {
        debug_draw_point_frusti(DebugDrawPointFrusiInfo{
            .light = &vsm_state.point_lights_cpu.at(std::max(render_context->render_data.vsm_settings.force_point_light_idx, 0)),
            .state = &vsm_state,
            .debug_context = &gpu_context->shader_debug_context,
        });
    }

    if (render_context->visualize_spot_frustum)
    {
        debug_draw_spot_frustum(DebugDrawSpotFrustumInfo{
            .light = &vsm_state.spot_lights_cpu.at(std::max(render_context->render_data.vsm_settings.force_spot_light_idx, 0)),
            .state = &vsm_state,
            .debug_context = &gpu_context->shader_debug_context,
        });
    }

    debug_draw_clip_fusti(DebugDrawClipFrustiInfo{
        .proj_info = &vsm_projections_info,
        .clip_projections = &vsm_state.clip_projections_cpu,
        .draw_clip_frustum = &render_context->draw_clip_frustum,
        .debug_context = &gpu_context->shader_debug_context,
        .vsm_view_direction = -std::bit_cast<f32vec3>(render_context->render_data.sky_settings.sun_direction),
    });

    auto new_swapchain_image = gpu_context->swapchain.acquire_next_image();
    if (new_swapchain_image.is_empty())
    {
        return false;
    }
    swapchain_image.set_images({.images = std::array{new_swapchain_image}});

    render_context->render_times.readback_render_times(render_context->render_data.frame_index);

    // Draw Frustum Camera.
    gpu_context->shader_debug_context.aabb_draws.draw(ShaderDebugAABBDraw{
        .position = daxa_f32vec3(0, 0, 0.5),
        .size = daxa_f32vec3(2.01, 2.01, 0.999),
        .color = daxa_f32vec3(1, 0, 0),
        .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC_MAIN_CAMERA,
    });

    if (render_context->render_data.light_settings.debug_draw_point_influence)
    {
        u32 range[2] = {0, render_context->render_data.light_settings.point_light_count};
        if (render_context->render_data.light_settings.selected_debug_point_light != -1)
        {
            range[0] = render_context->render_data.light_settings.selected_debug_point_light;
            range[1] = range[0] + 1;
        }
        for (u32 i = range[0]; i < range[1]; ++i)
        {
            PointLight const & light = scene->_point_lights.at(i);

            gpu_context->shader_debug_context.sphere_draws.draw(ShaderDebugSphereDraw{
                .position = {
                    light.position[0],
                    light.position[1],
                    light.position[2],
                },
                .radius = light.cutoff,
                .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE,
                .color = std::bit_cast<daxa_f32vec3>(light.color),
            });
        }
    }

    if (render_context->render_data.light_settings.debug_draw_spot_influence)
    {
        u32 range[2] = {0, render_context->render_data.light_settings.spot_light_count};
        if (render_context->render_data.light_settings.selected_debug_spot_light != -1)
        {
            range[0] = render_context->render_data.light_settings.selected_debug_spot_light;
            range[1] = range[0] + 1;
        }
        for (u32 i = range[0]; i < range[1]; ++i)
        {
            SpotLight const & light = scene->_spot_lights.at(i);

            glm::mat4 transform4 = glm::mat4(
                glm::vec4(light.transform[0], 0.0f),
                glm::vec4(light.transform[1], 0.0f),
                glm::vec4(light.transform[2], 0.0f),
                glm::vec4(light.transform[3], 1.0f));
            f32vec3 const spot_direction = transform4 * f32vec4(0.0f, 0.0f, -1.0f, 0.0f);

            gpu_context->shader_debug_context.cone_draws.draw(ShaderDebugConeDraw{
                .position = {
                    light.transform[3][0],
                    light.transform[3][1],
                    light.transform[3][2],
                },
                .direction = {
                    spot_direction[0],
                    spot_direction[1],
                    spot_direction[2],
                },
                .size = light.cutoff,
                .angle = light.outer_cone_angle,
                .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE,
                .color = std::bit_cast<daxa_f32vec3>(light.color),
            });
        }
    }

    gpu_context->shader_debug_context.update(gpu_context->device, render_target_size, window->size, render_context->render_data.frame_index);

    return true;
}