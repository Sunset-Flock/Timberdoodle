#include "renderer.hpp"

#include "../shader_shared/scene.inl"
#include "../daxa_helper.hpp"
#include "../shader_shared/debug.inl"
#include "../shader_shared/readback.inl"

#include "rasterize_visbuffer/rasterize_visbuffer.hpp"

#include "virtual_shadow_maps/vsm.inl"
#include "pgi/pgi_update.inl"

#include "asteroids/draw_asteroids.hpp"

#include "ray_tracing/ray_tracing.inl"

#include "tasks/memset.inl"
#include "tasks/prefix_sum.inl"
#include "tasks/write_swapchain.inl"
#include "tasks/shade_opaque.inl"
#include "tasks/sky.inl"
#include "tasks/autoexposure.inl"
#include "tasks/shader_debug_draws.inl"
#include "tasks/decode_visbuffer_test.inl"
#include "tasks/gen_gbuffer.hpp"

#include <daxa/types.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <thread>
#include <variant>
#include <iostream>

inline auto create_task_buffer(GPUContext * gpu_context, auto size, auto task_buf_name, auto buf_name)
{
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
    zero_buffer = create_task_buffer(gpu_context, sizeof(u32), "zero_buffer", "zero_buffer");
    meshlet_instances = create_task_buffer(gpu_context, size_of_meshlet_instance_buffer(), "meshlet_instances", "meshlet_instances_a");
    meshlet_instances_last_frame = create_task_buffer(gpu_context, size_of_meshlet_instance_buffer(), "meshlet_instances_last_frame", "meshlet_instances_b");
    visible_mesh_instances = create_task_buffer(gpu_context, sizeof(VisibleMeshesList), "visible_mesh_instances", "visible_mesh_instances");
    luminance_average = create_task_buffer(gpu_context, sizeof(f32), "luminance average", "luminance_average");
    general_readback_buffer = gpu_context->device.create_buffer({
        .size = sizeof(ReadbackValues) * 4,
        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "general readback buffer",
    });
    visible_meshlet_instances = create_task_buffer(gpu_context, sizeof(u32) * (MAX_MESHLET_INSTANCES + 4), "visible_meshlet_instances", "visible_meshlet_instances");

    buffers = {
        zero_buffer,
        meshlet_instances,
        meshlet_instances_last_frame,
        visible_meshlet_instances,
        visible_mesh_instances,
        luminance_average};

    swapchain_image = daxa::TaskImage{{.swapchain_image = true, .name = "swapchain_image"}};
    transmittance = daxa::TaskImage{{.name = "transmittance"}};
    multiscattering = daxa::TaskImage{{.name = "multiscattering"}};
    sky_ibl_cube = daxa::TaskImage{{.name = "sky ibl cube"}};
    depth_vistory = daxa::TaskImage{{.name = "depth_history"}};
    f32_depth_vistory = daxa::TaskImage{{.name = "f32_depth_vistory"}};

    vsm_state.initialize_persitent_state(gpu_context);
    asteroid_state.initialize_persistent_state(gpu_context->device);
    pgi_state.initialize(gpu_context->device);

    images = {
        transmittance,
        multiscattering,
        sky_ibl_cube,
        depth_vistory,
        f32_depth_vistory,
    };

    frame_buffer_images = {
        {
            daxa::ImageInfo
            {
                .format = daxa::Format::D32_SFLOAT,
                .usage = daxa::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "depth_history",
            },
            depth_vistory,
        },
        {
            daxa::ImageInfo
            {
                .format = daxa::Format::R32_SFLOAT,
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "f32_depth_vistory",
            },
            f32_depth_vistory,
        },
    };

    recreate_framebuffer();
    recreate_sky_luts();
    main_task_graph = create_main_task_graph();
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
    pgi_state.cleanup(gpu_context->device);
    vsm_state.cleanup_persistent_state(gpu_context);
    this->gpu_context->device.wait_idle();
    this->gpu_context->device.collect_garbage();
}

void Renderer::compile_pipelines()
{
    auto add_if_not_present = [&](auto & map, auto & list, auto compile_info)
    {
        if (!map.contains(compile_info.name))
        {
            list.push_back(compile_info);
        }
    };

    std::vector<daxa::RasterPipelineCompileInfo> rasters = {
        {draw_visbuffer_mesh_shader_pipelines[0]},
        {draw_visbuffer_mesh_shader_pipelines[1]},
        {draw_visbuffer_mesh_shader_pipelines[2]},
        {draw_visbuffer_mesh_shader_pipelines[3]},
        {draw_visbuffer_mesh_shader_pipelines[4]},
        {draw_visbuffer_mesh_shader_pipelines[5]},
        {draw_visbuffer_mesh_shader_pipelines[6]},
        {draw_visbuffer_mesh_shader_pipelines[7]},
        {cull_and_draw_directional_pages_pipelines[0]},
        {cull_and_draw_directional_pages_pipelines[1]},
        {cull_and_draw_point_pages_pipelines[0]},
        {cull_and_draw_point_pages_pipelines[1]},
        {draw_shader_debug_lines_pipeline_compile_info()},
        {draw_shader_debug_circles_pipeline_compile_info()},
        {draw_shader_debug_rectangles_pipeline_compile_info()},
        {draw_shader_debug_aabb_pipeline_compile_info()},
        {draw_shader_debug_box_pipeline_compile_info()},
        {pgi_draw_debug_probes_compile_info()},
        {debug_draw_asteroids_compile_info()}
    };
    for (auto info : rasters)
    {
        auto compilation_result = this->gpu_context->pipeline_manager.add_raster_pipeline(info);
        if (compilation_result.value()->is_valid())
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] SUCCESFULLY compiled pipeline {}", info.name));
        }
        else
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] FAILED to compile pipeline {} with message \n {}", info.name,
                compilation_result.message()));
        }
        this->gpu_context->raster_pipelines[info.name] = compilation_result.value();
    }
    std::vector<daxa::ComputePipelineCompileInfo2> computes = {
        {pgi_update_probes_compute_compile_info()},
        {pgi_update_probes_compute_compile_info2()},
        {pgi_update_probes_compute_compile_info3()},
        {sfpm_allocate_ent_bitfield_lists()},
        {gen_hiz_pipeline_compile_info2()},
        {cull_meshlets_compute_pipeline_compile_info()},
        {draw_meshlets_compute_pipeline_compile_info()},
        {tido::upgrade_compute_pipeline_compile_info(alloc_entity_to_mesh_instances_offsets_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(set_entity_meshlets_visibility_bitmasks_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(prepopulate_meshlet_instances_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(IndirectMemsetBufferTask::pipeline_compile_info)},
        {tido::upgrade_compute_pipeline_compile_info(analyze_visbufer_pipeline_compile_info())},
        {write_swapchain_pipeline_compile_info2()},
        {tido::upgrade_compute_pipeline_compile_info(shade_opaque_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(expand_meshes_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(PrefixSumCommandWriteTask::pipeline_compile_info)},
        {tido::upgrade_compute_pipeline_compile_info(prefix_sum_upsweep_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(prefix_sum_downsweep_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(compute_transmittance_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(compute_multiscattering_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(compute_sky_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(sky_into_cubemap_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(gen_luminace_histogram_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(gen_luminace_average_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_free_wrapped_pages_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_invalidate_pages_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_mark_required_pages_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_find_free_pages_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_allocate_pages_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_clear_pages_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_gen_dirty_bit_hiz_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_gen_point_dirty_bit_hiz_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_clear_dirty_bit_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_debug_virtual_page_table_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_debug_meta_memory_table_pipeline_compile_info())},
        {decode_visbuffer_test_pipeline_info2()},
        {tido::upgrade_compute_pipeline_compile_info(SplitAtomicVisbufferTask::pipeline_compile_info)},
        {tido::upgrade_compute_pipeline_compile_info(DrawVisbuffer_WriteCommandTask2::pipeline_compile_info)},
        {tido::upgrade_compute_pipeline_compile_info(ray_trace_ao_compute_pipeline_info())},
        {debug_task_draw_display_image_pipeline_info()},
        {rtao_denoiser_pipeline_info()},
        {gen_gbuffer_pipeline_compile_info()},
    };
    for (auto const & info : computes)
    {
        auto compilation_result = this->gpu_context->pipeline_manager.add_compute_pipeline2(info);
        if (compilation_result.value()->is_valid())
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] SUCCESFULLY compiled pipeline {}", info.name));
        }
        else
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] FAILED to compile pipeline {} with message \n {}", info.name,
                compilation_result.message()));
        }
        this->gpu_context->compute_pipelines[info.name] = compilation_result.value();
    }

    std::vector<daxa::RayTracingPipelineCompileInfo> ray_tracing = {
        {ray_trace_ao_rt_pipeline_info()},
        {pgi_trace_probe_lighting_pipeline_compile_info()},
    };
    // for (auto const & info : ray_tracing)
    // {
    //     auto compilation_result = this->gpu_context->pipeline_manager.add_ray_tracing_pipeline(info);
    //     if (compilation_result.value()->is_valid())
    //     {
    //         DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] SUCCESFULLY compiled pipeline {}", info.name));
    //     }
    //     else
    //     {
    //         DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] FAILED to compile pipeline {} with message \n {}", info.name,
    //             compilation_result.message()));
    //     }
    //     this->gpu_context->ray_tracing_pipelines[info.name].pipeline = compilation_result.value();
    //     auto sbt_info = gpu_context->ray_tracing_pipelines[info.name].pipeline->create_default_sbt();
    //     this->gpu_context->ray_tracing_pipelines[info.name].sbt = sbt_info.table;
    //     this->gpu_context->ray_tracing_pipelines[info.name].sbt_buffer_id = sbt_info.buffer;
    // }

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
    this->pgi_state.recreate_resources(gpu_context->device, render_context->render_data.pgi_settings);
}

void Renderer::clear_select_buffers()
{
    using namespace daxa;
    TaskGraph tg{{
        .device = this->gpu_context->device,
        .swapchain = this->gpu_context->swapchain,
        .additional_transient_image_usage_flags = daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::SHADER_STORAGE,
        .pre_task_callback = [=, this](daxa::TaskInterface ti)
        { debug_task(ti, render_context->tg_debug, *render_context->gpu_context->compute_pipelines.at(std::string("debug_task_pipeline")), true); },
        .post_task_callback = [=, this](daxa::TaskInterface ti)
        { debug_task(ti, render_context->tg_debug, *render_context->gpu_context->compute_pipelines.at(std::string("debug_task_pipeline")), false); },
        .name = "clear task list",
    }};
    tg.use_persistent_buffer(meshlet_instances);
    tg.use_persistent_buffer(meshlet_instances_last_frame);
    tg.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, meshlet_instances),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, meshlet_instances_last_frame),
        },
        .task = [=](daxa::TaskInterface ti)
        {
            auto mesh_instances_address = ti.device.buffer_device_address(ti.get(meshlet_instances).ids[0]).value();
            MeshletInstancesBufferHead mesh_instances_reset = make_meshlet_instance_buffer_head(mesh_instances_address);
            allocate_fill_copy(ti, mesh_instances_reset, ti.get(meshlet_instances));
            auto mesh_instances_prev_address = ti.device.buffer_device_address(ti.get(meshlet_instances_last_frame).ids[0]).value();
            MeshletInstancesBufferHead mesh_instances_prev_reset = make_meshlet_instance_buffer_head(mesh_instances_prev_address);
            allocate_fill_copy(ti, mesh_instances_prev_reset, ti.get(meshlet_instances_last_frame));
        },
        .name = "clear meshlet instance buffers",
    });
    tg.use_persistent_buffer(visible_meshlet_instances);
    tg.clear_buffer({.buffer = visible_meshlet_instances, .size = sizeof(u32), .clear_value = 0});
    //tg.use_persistent_buffer(luminance_average);
    //tg.clear_buffer({.buffer = luminance_average, .size = sizeof(f32), .clear_value = 0});
    tg.use_persistent_image(pgi_state.probe_radiance);
    tg.use_persistent_image(pgi_state.probe_visibility);
    tg.use_persistent_image(pgi_state.probe_info);
    tg.clear_image({.view = pgi_state.probe_radiance_view, .name = "clear pgi radiance"});
    tg.clear_image({.view = pgi_state.probe_visibility_view, .name = "clear pgi visibility"});
    tg.clear_image({.view = pgi_state.probe_info_view, .name = "clear pgi info"});
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
        .pre_task_callback = [=, this](daxa::TaskInterface ti)
        { debug_task(ti, render_context->tg_debug, *render_context->gpu_context->compute_pipelines.at(std::string("debug_task_pipeline")), true); },
        .post_task_callback = [=, this](daxa::TaskInterface ti)
        { debug_task(ti, render_context->tg_debug, *render_context->gpu_context->compute_pipelines.at(std::string("debug_task_pipeline")), false); },
        .name = "Calculate sky luts task graph",
    }};
    // TODO:    Do not use globals here, make a new buffer.
    //          Globals should only be used within the main task graph.
    tg.use_persistent_buffer(render_context->tgpu_render_data);
    tg.use_persistent_image(transmittance);
    tg.use_persistent_image(multiscattering);

    tg.add_task({
        .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, render_context->tgpu_render_data)},
        .task = [&](daxa::TaskInterface ti)
        {
            allocate_fill_copy(
                ti,
                render_context->render_data.sky_settings,
                ti.get(render_context->tgpu_render_data),
                offsetof(RenderGlobalData, sky_settings));
            allocate_fill_copy(
                ti,
                render_context->render_data.sky_settings_ptr,
                ti.get(render_context->tgpu_render_data),
                offsetof(RenderGlobalData, sky_settings_ptr));
        },
        .name = "update sky settings globals",
    });

    tg.add_task(ComputeTransmittanceTask{
        .views = std::array{
            ComputeTransmittanceH::AT.globals | render_context->tgpu_render_data,
            ComputeTransmittanceH::AT.transmittance | transmittance,
        },
        .gpu_context = gpu_context,
    });

    tg.add_task(ComputeMultiscatteringTask{
        .views = std::array{
            ComputeMultiscatteringH::AT.globals | render_context->tgpu_render_data,
            ComputeMultiscatteringH::AT.transmittance | transmittance,
            ComputeMultiscatteringH::AT.multiscattering | multiscattering,
        },
        .render_context = render_context.get(),
    });
    tg.submit({});
    tg.complete({});
    return tg;
}

auto Renderer::create_main_task_graph() -> daxa::TaskGraph
{
    // Rasterize Visbuffer:
    // - reset/clear certain buffers and images (~130mics for 1440p RTX4080.... really really bad. Investigate)
    // - pre-populate meshlet instances from last frame (~25mics for 1440p RTX4080)
    //     - uses list of visible meshlets of last frame (visible_meshlet_instances) and meshlet instance list from last
    //     frame (meshlet_instances_last_frame)
    //     - checks if visible meshlet from last frame is also to be drawn this frame
    //     - builds bitfields (entity_meshlet_visibility_bitfield_offsets), that denote if a meshlet of an entity is
    //     drawn in the first pass.
    //     - bitfield rebuild into a little arena buffer every frame from scratch.
    // - draw first pass (~50-500mics for 1440p RTX4080)
    //     - draws meshlet instances, generated by pre-populate_instantiated_meshlets.
    //     - draws triangle id and depth. triangle id indexes into the meshlet instance list (that is freshly generated
    //     every frame).
    //     - effectively draws the meshlets that were visible last frame.
    // - build hiz depth map (~35mics for 1440p RTX4080)
    //      - hiz's mips are generated in a single pass
    //      - hiz size and mip sizes are a power of two. This is required for it to work in a single pass
    //      - lowest hiz mip is the next smaller power of two size relative to the render resolution. Example: 1440p -> 1024p.
    // - cull meshes (~8-20mics for 1440p RTX4080)
    //     - goes over the main pass draw lists, culls each mesh instance
    //     - culls meshes against occlusion (using hiz) and frustum
    //     - writes all meshlets of not culled meshes to po2expansion buffers for each pipeline draw (masked and opaque)
    // - cull and draw meshlets (~50-500mics for 1440p RTX4080)
    //     - ran for each main pass pipeline (masked, opaque)
    //     - use dispatch indirect commands within po2expansion buffer to know draw sizes
    //     - use po2expansion buffers to map task shader invocation to meshlet
    //     - task shaders cull meshlets against occlusion (using hiz) and frustum
    //     - non-culled meshlets are written to the meshlet instance list by the task shader threads
    //     - mesh shaders cull triangles against: occlusion (using hiz), frustum, importance and backface
    //     - like the first pass, draws triangle id and depth
    //     - IMPORTANT: uses entity meshlet bitfields to avoid drawing meshlets that were already drawn in the first pass.
    // - analyze visbuffer (~70-200mics for 1440p RTX4080)
    //     - reads final opaque visbuffer
    //     - generates list of visible meshlet instance indices
    //     - marks visible triangles of meshlet instances in bitfield.
    using namespace daxa;
    TaskGraph tg{{
        .device = this->gpu_context->device,
        .swapchain = this->gpu_context->swapchain,
        .reorder_tasks = true,
        .staging_memory_pool_size = 2'097'152, // 2MiB.
        // Extra flags are required for tg debug inspector:
        .additional_transient_image_usage_flags = daxa::ImageUsageFlagBits::TRANSFER_SRC,
        .pre_task_callback = [=, this](daxa::TaskInterface ti)
        { debug_task(ti, render_context->tg_debug, *render_context->gpu_context->compute_pipelines.at(std::string("debug_task_pipeline")), true); },
        .post_task_callback = [=, this](daxa::TaskInterface ti)
        { debug_task(ti, render_context->tg_debug, *render_context->gpu_context->compute_pipelines.at(std::string("debug_task_pipeline")), false); },
        .name = "Sandbox main TaskGraph",
    }};
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
    tg.use_persistent_buffer(render_context->tgpu_render_data);
    tg.use_persistent_buffer(vsm_state.globals);
    tg.use_persistent_image(vsm_state.memory_block);
    tg.use_persistent_image(vsm_state.meta_memory_table);
    tg.use_persistent_image(vsm_state.page_table);
    tg.use_persistent_image(vsm_state.page_view_pos_row);
    tg.use_persistent_image(vsm_state.point_page_tables);
    tg.use_persistent_image(gpu_context->shader_debug_context.vsm_debug_page_table);
    tg.use_persistent_image(gpu_context->shader_debug_context.vsm_debug_meta_memory_table);
    auto debug_lens_image = gpu_context->shader_debug_context.tdebug_lens_image;
    tg.use_persistent_image(debug_lens_image);
    tg.use_persistent_image(swapchain_image);
    // tg.use_persistent_tlas(scene->_scene_tlas);

    // TODO: Move into an if and create persistent state only if necessary.
    tg.use_persistent_image(pgi_state.probe_radiance);
    tg.use_persistent_image(pgi_state.probe_visibility);
    tg.use_persistent_image(pgi_state.probe_info);

    tg.clear_image({debug_lens_image, std::array{0.0f, 0.0f, 0.0f, 1.0f}});

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
        .views = std::array{daxa::attachment_view(ReadbackH::AT.globals, render_context->tgpu_render_data)},
        .shader_debug_context = &gpu_context->shader_debug_context,
    });
    tg.add_task({
        .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, render_context->tgpu_render_data)},
        .task = [&](daxa::TaskInterface ti)
        {
            allocate_fill_copy(ti, render_context->render_data, ti.get(render_context->tgpu_render_data));
            gpu_context->shader_debug_context.update_debug_buffer(ti.device, ti.recorder, *ti.allocator);
            render_context->render_times.reset_timestamps_for_current_frame(ti.recorder);
        },
        .name = "update global buffers",
    });

    auto depth_hist = render_context->render_data.settings.enable_atomic_visbuffer ? f32_depth_vistory : depth_vistory;
    auto const visbuffer_ret = raster_visbuf::task_draw_visbuffer_all({tg, render_context, scene, meshlet_instances, meshlet_instances_last_frame, visible_meshlet_instances, debug_image, depth_hist});
    daxa::TaskImageView main_camera_visbuffer = visbuffer_ret.main_camera_visbuffer;
    daxa::TaskImageView main_camera_depth = visbuffer_ret.main_camera_depth;
    daxa::TaskImageView view_camera_visbuffer = visbuffer_ret.view_camera_visbuffer;
    daxa::TaskImageView view_camera_depth = visbuffer_ret.view_camera_depth;
    daxa::TaskImageView overdraw_image = visbuffer_ret.view_camera_overdraw;

    daxa::TaskImageView main_camera_geo_normal_image = tg.create_transient_image({
        .format = GBUFFER_GEO_NORMAL_FORMAT,
        .size = { render_context->render_data.settings.render_target_size.x, render_context->render_data.settings.render_target_size.y, 1 },
        .name = "main_camera_geo_normal_image",
    }); 
    daxa::TaskImageView view_camera_geo_normal_image = main_camera_geo_normal_image;
    tg.add_task(GenGbufferTask{
        .views = std::array{
            GenGbufferTask::AT.globals | render_context->tgpu_render_data,
            GenGbufferTask::AT.debug_image | debug_image,
            GenGbufferTask::AT.vis_image | main_camera_visbuffer,
            GenGbufferTask::AT.geo_normal_image | main_camera_geo_normal_image,
            GenGbufferTask::AT.material_manifest | scene->_gpu_material_manifest,
            GenGbufferTask::AT.instantiated_meshlets | meshlet_instances,
            GenGbufferTask::AT.meshes | scene->_gpu_mesh_manifest,
            GenGbufferTask::AT.combined_transforms | scene->_gpu_entity_combined_transforms,
        },
        .render_context = render_context.get(),
    });

    // Some following passes need either the main views camera OR the views cameras perspective.
    // The observer camera is not always appropriate to be used.
    // For example shade opaque needs view camera information while VSMs always need the main cameras perspective for generation.
    if (render_context->render_data.settings.draw_from_observer)
    {
        view_camera_geo_normal_image = tg.create_transient_image({
            .format = GBUFFER_GEO_NORMAL_FORMAT,
            .size = { render_context->render_data.settings.render_target_size.x, render_context->render_data.settings.render_target_size.y, 1 },
            .name = "view_camera_geo_normal_image",
        }); 
        tg.add_task(GenGbufferTask{
            .views = std::array{
                GenGbufferTask::AT.globals | render_context->tgpu_render_data,
                GenGbufferTask::AT.debug_image | debug_image,
                GenGbufferTask::AT.vis_image | view_camera_visbuffer,
                GenGbufferTask::AT.geo_normal_image | view_camera_geo_normal_image,
                GenGbufferTask::AT.material_manifest | scene->_gpu_material_manifest,
                GenGbufferTask::AT.instantiated_meshlets | meshlet_instances,
                GenGbufferTask::AT.meshes | scene->_gpu_mesh_manifest,
                GenGbufferTask::AT.combined_transforms | scene->_gpu_entity_combined_transforms,
            },
            .render_context = render_context.get(),
        });
    }

    auto sky = tg.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {render_context->render_data.sky_settings.sky_dimensions.x, render_context->render_data.sky_settings.sky_dimensions.y, 1},
        .name = "sky look up table",
    });
    auto luminance_histogram = tg.create_transient_buffer({sizeof(u32) * (LUM_HISTOGRAM_BIN_COUNT), "luminance_histogram"});

    daxa::TaskImageView sky_ibl_view = sky_ibl_cube.view().view({.layer_count = 6});
    tg.add_task(ComputeSkyTask{
        .views = std::array{
            ComputeSkyH::AT.globals | render_context->tgpu_render_data,
            ComputeSkyH::AT.transmittance | transmittance,
            ComputeSkyH::AT.multiscattering | multiscattering,
            ComputeSkyH::AT.sky | sky,
        },
        .render_context = render_context.get(),
    });
    tg.add_task(SkyIntoCubemapTask{
        .views = std::array{
            SkyIntoCubemapH::AT.globals | render_context->tgpu_render_data,
            SkyIntoCubemapH::AT.transmittance | transmittance,
            SkyIntoCubemapH::AT.sky | sky,
            SkyIntoCubemapH::AT.ibl_cube | sky_ibl_view,
        },
        .gpu_context = gpu_context,
    });

    auto color_image = tg.create_transient_image({
        .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
        .size = {
            render_context->render_data.settings.render_target_size.x,
            render_context->render_data.settings.render_target_size.y,
            1,
        },
        .name = "color_image",
    });

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
            .g_buffer_geo_normal = main_camera_geo_normal_image,
        });
    }
    else
    {
        vsm_state.zero_out_transient_state(tg, render_context->render_data);
    }

    tg.submit({});

    daxa::TaskImageView ao_image = daxa::NullTaskImage;
#if 0
    if (render_context->render_data.settings.ao_mode == AO_MODE_RT)
    {
        auto ao_image_info = daxa::TaskTransientImageInfo{
            .format = daxa::Format::R16_SFLOAT,
            .size = {
                render_context->render_data.settings.render_target_size.x,
                render_context->render_data.settings.render_target_size.y,
                1,
            },
            .name = "ao_image_raw",
        };
        ao_image = tg.create_transient_image(ao_image_info);
        ao_image_info.name = "ao_image";
        auto ao_image_raw = tg.create_transient_image(ao_image_info);
        tg.clear_image({ao_image_raw, std::array{0.0f, 0.0f, 0.0f, 0.0f}});
        tg.clear_image({ao_image, std::array{0.0f, 0.0f, 0.0f, 0.0f}});
        tg.add_task(RayTraceAmbientOcclusionTask{
            .views = std::array{
                RayTraceAmbientOcclusionH::AT.globals | render_context->tgpu_render_data,
                RayTraceAmbientOcclusionH::AT.debug_image | debug_image,
                RayTraceAmbientOcclusionH::AT.debug_lens_image | debug_lens_image,
                RayTraceAmbientOcclusionH::AT.ao_image | ao_image_raw,
                RayTraceAmbientOcclusionH::AT.vis_image | view_camera_visbuffer,
                RayTraceAmbientOcclusionH::AT.sky | sky,
                RayTraceAmbientOcclusionH::AT.material_manifest | scene->_gpu_material_manifest,
                RayTraceAmbientOcclusionH::AT.instantiated_meshlets | meshlet_instances,
                RayTraceAmbientOcclusionH::AT.meshes | scene->_gpu_mesh_manifest,
                RayTraceAmbientOcclusionH::AT.mesh_groups | scene->_gpu_mesh_group_manifest,
                RayTraceAmbientOcclusionH::AT.combined_transforms | scene->_gpu_entity_combined_transforms,
                RayTraceAmbientOcclusionH::AT.tlas | scene->_scene_tlas,
                RayTraceAmbientOcclusionH::AT.entity_to_meshgroup | scene->_gpu_entity_mesh_groups,
                RayTraceAmbientOcclusionH::AT.mesh_instances | scene->mesh_instances_buffer,
            },
            .gpu_context = gpu_context,
            .render_context = render_context.get(),
        });
        tg.add_task(RTAODeoinserTask{
            .views = std::array{
                RTAODeoinserTask::AT.globals | render_context->tgpu_render_data,
                RTAODeoinserTask::AT.depth | view_camera_depth,
                RTAODeoinserTask::AT.src | ao_image_raw,
                RTAODeoinserTask::AT.dst | ao_image,
            },
            .gpu_context = gpu_context,
            .render_context = render_context.get(),
        });
    }
    if (render_context->render_data.pgi_settings.enabled)
    {
        daxa::TaskImageView pgi_trace_result = pgi_create_trace_result_texture(tg, render_context->render_data.pgi_settings, pgi_state);
        tg.add_task(PGITraceProbeRaysTask{
            .views = std::array{
                PGITraceProbeRaysTask::AT.globals | render_context->tgpu_render_data,
                PGITraceProbeRaysTask::AT.probe_radiance | pgi_state.probe_radiance_view,
                PGITraceProbeRaysTask::AT.probe_visibility | pgi_state.probe_visibility_view,
                PGITraceProbeRaysTask::AT.probe_info | pgi_state.probe_info_view,
                PGITraceProbeRaysTask::AT.tlas | scene->_scene_tlas,
                PGITraceProbeRaysTask::AT.sky_transmittance | transmittance,
                PGITraceProbeRaysTask::AT.sky | sky,
                PGITraceProbeRaysTask::AT.trace_result | pgi_trace_result,
                PGITraceProbeRaysTask::AT.mesh_instances | scene->mesh_instances_buffer,
            },
            .render_context = render_context.get(),
            .pgi_state = &this->pgi_state,
        });
        tg.add_task(PGIUpdateProbeTexelsTask{
            .views = std::array{
                PGIUpdateProbeTexelsTask::AT.globals | render_context->tgpu_render_data,
                PGIUpdateProbeTexelsTask::AT.probe_radiance | pgi_state.probe_radiance_view,
                PGIUpdateProbeTexelsTask::AT.probe_visibility | pgi_state.probe_visibility_view,
                PGIUpdateProbeTexelsTask::AT.probe_info | pgi_state.probe_info_view,
                PGIUpdateProbeTexelsTask::AT.trace_result | pgi_trace_result,
            },
            .render_context = render_context.get(),
            .pgi_state = &this->pgi_state,
        });
        tg.add_task(PGIUpdateProbesTask{
            .views = std::array{
                PGIUpdateProbesTask::AT.globals | render_context->tgpu_render_data,
                PGIUpdateProbesTask::AT.probe_radiance | pgi_state.probe_radiance_view,
                PGIUpdateProbesTask::AT.probe_visibility | pgi_state.probe_visibility_view,
                PGIUpdateProbesTask::AT.probe_info | pgi_state.probe_info_view,
                PGIUpdateProbesTask::AT.trace_result | pgi_trace_result,
            },
            .render_context = render_context.get(),
            .pgi_state = &this->pgi_state,
        });
    }
#endif
    auto const vsm_page_table_view = vsm_state.page_table.view().view({.base_array_layer = 0, .layer_count = VSM_CLIP_LEVELS});
    auto const vsm_page_heigh_offsets_view = vsm_state.page_view_pos_row.view().view({.base_array_layer = 0, .layer_count = VSM_CLIP_LEVELS});
    auto const vsm_point_page_table_view = vsm_state.point_page_tables.view().view({
        .base_mip_level = 0,
        .level_count = s_cast<u32>(std::log2(VSM_PAGE_TABLE_RESOLUTION)) + 1,
        .base_array_layer = 0,
        .layer_count = 6 * MAX_POINT_LIGHTS,
    });
    tg.add_task(ShadeOpaqueTask{
        .views = std::array{
            ShadeOpaqueH::AT.debug_lens_image | debug_lens_image,
            ShadeOpaqueH::AT.ao_image | ao_image,
            ShadeOpaqueH::AT.globals | render_context->tgpu_render_data,
            ShadeOpaqueH::AT.color_image | color_image,
            ShadeOpaqueH::AT.vis_image | view_camera_visbuffer,
            ShadeOpaqueH::AT.transmittance | transmittance,
            ShadeOpaqueH::AT.sky | sky,
            ShadeOpaqueH::AT.sky_ibl | sky_ibl_view,
            ShadeOpaqueH::AT.vsm_page_table | vsm_page_table_view,
            ShadeOpaqueH::AT.vsm_page_view_pos_row | vsm_page_heigh_offsets_view,
            ShadeOpaqueH::AT.vsm_point_page_table | vsm_point_page_table_view,
            ShadeOpaqueH::AT.material_manifest | scene->_gpu_material_manifest,
            ShadeOpaqueH::AT.instantiated_meshlets | meshlet_instances,
            ShadeOpaqueH::AT.meshes | scene->_gpu_mesh_manifest,
            ShadeOpaqueH::AT.combined_transforms | scene->_gpu_entity_combined_transforms,
            ShadeOpaqueH::AT.luminance_average | luminance_average,
            ShadeOpaqueH::AT.vsm_memory_block | vsm_state.memory_block,
            ShadeOpaqueH::AT.vsm_clip_projections | vsm_state.clip_projections,
            ShadeOpaqueH::AT.vsm_globals | vsm_state.globals,
            ShadeOpaqueH::AT.vsm_overdraw_debug | vsm_state.overdraw_debug_image,
            ShadeOpaqueH::AT.vsm_wrapped_pages | vsm_state.free_wrapped_pages_info,
            ShadeOpaqueH::AT.debug_image | debug_image,
            ShadeOpaqueH::AT.overdraw_image | overdraw_image,
            ShadeOpaqueH::AT.tlas | daxa::NullTaskTlas,
            ShadeOpaqueH::AT.point_lights | scene->_gpu_point_lights,
            ShadeOpaqueH::AT.vsm_point_lights | vsm_state.vsm_point_lights,
            ShadeOpaqueH::AT.mesh_instances | scene->mesh_instances_buffer,
            ShadeOpaqueH::AT.pgi_probe_radiance | pgi_state.probe_radiance_view,
            ShadeOpaqueH::AT.pgi_probe_visibility | pgi_state.probe_visibility_view,
            ShadeOpaqueH::AT.pgi_probe_info | pgi_state.probe_info_view,
        },
        .render_context = render_context.get(),
    });

    tg.clear_buffer({.buffer = luminance_histogram, .clear_value = 0});
    tg.add_task(GenLuminanceHistogramTask{
        .views = std::array{
            GenLuminanceHistogramH::AT.globals | render_context->tgpu_render_data,
            GenLuminanceHistogramH::AT.histogram | luminance_histogram,
            GenLuminanceHistogramH::AT.luminance_average | luminance_average,
            GenLuminanceHistogramH::AT.color_image | color_image,
        },
        .render_context = render_context.get(),
    });
    tg.add_task(GenLuminanceAverageTask{
        .views = std::array{
            GenLuminanceAverageH::AT.globals | render_context->tgpu_render_data,
            GenLuminanceAverageH::AT.histogram | luminance_histogram,
            GenLuminanceAverageH::AT.luminance_average | luminance_average,
        },
        .gpu_context = gpu_context,
    });
    daxa::TaskImageView debug_draw_depth = tg.create_transient_image({
        .format = daxa::Format::D32_SFLOAT,
        .size = { render_context->render_data.settings.render_target_size.x, render_context->render_data.settings.render_target_size.y, 1},
        .name = "debug depth",
    });
    tg.copy_image_to_image({view_camera_depth, debug_draw_depth, "copy depth to debug depth"});

    asteroid_state.initalize_transient_state(tg);
    task_draw_asteroids(TaskDrawAsteroidsInfo{
        .render_context = render_context.get(),
        .asteroids_state = &asteroid_state,
        .tg = &tg,
        .depth = main_camera_depth,
        .color = color_image,
    });
    if (render_context->render_data.pgi_settings.enabled && (render_context->render_data.pgi_settings.debug_probe_draw_mode != PGI_DEBUG_PROBE_DRAW_MODE_OFF))
    {
        tg.add_task(PGIDrawDebugProbesTask{
            .views = std::array{
                PGIDrawDebugProbesH::AT.globals | render_context->tgpu_render_data,
                PGIDrawDebugProbesH::AT.color_image | color_image,
                PGIDrawDebugProbesH::AT.depth_image | debug_draw_depth,
                PGIDrawDebugProbesH::AT.probe_radiance | pgi_state.probe_radiance_view,
                PGIDrawDebugProbesH::AT.probe_visibility | pgi_state.probe_visibility_view,
                PGIDrawDebugProbesH::AT.probe_info | pgi_state.probe_info_view,
                PGIDrawDebugProbesH::AT.tlas | scene->_scene_tlas,
            },
            .render_context = render_context.get(),
            .pgi_state = &pgi_state,
        });
    }
    tg.add_task(DebugDrawTask{
        .views = std::array{
            DebugDrawH::AT.globals | render_context->tgpu_render_data,
            DebugDrawH::AT.color_image | color_image,
            DebugDrawH::AT.depth_image | debug_draw_depth,
        },
        .render_context = render_context.get(),
    });
    tg.add_task(WriteSwapchainTask{
        .views = std::array{
            WriteSwapchainH::AT.globals | render_context->tgpu_render_data,
            WriteSwapchainH::AT.debug_image | debug_image,
            WriteSwapchainH::AT.swapchain | swapchain_image,
            WriteSwapchainH::AT.color_image | color_image,
        },
        .gpu_context = gpu_context,
    });

    tg.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskImageAccess::COLOR_ATTACHMENT, swapchain_image),
            daxa::inl_attachment(daxa::TaskImageAccess::FRAGMENT_SHADER_SAMPLED, debug_lens_image),
        },
        .task = [=, this](daxa::TaskInterface ti)
        {
            ImGui::Render();
            auto size = ti.device.image_info(ti.get(daxa::TaskImageAttachmentIndex(0)).ids[0]).value().size;
            imgui_renderer->record_commands(
                ImGui::GetDrawData(), ti.recorder, ti.get(daxa::TaskImageAttachmentIndex(0)).ids[0], size.x, size.y);
        },
        .name = "ImGui Draw",
    });

    tg.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, render_context->tgpu_render_data),
        },
        .task = [=, this](daxa::TaskInterface ti)
        {
            const u32 index = render_context->render_data.frame_index % 4;

            render_context->general_readback = ti.device.buffer_host_address_as<ReadbackValues>(general_readback_buffer).value()[index];
        },
        .name = "general readback",
    });

    tg.submit({});
    tg.present({});
    tg.complete({});
    return tg;
}

void Renderer::render_frame(
    CameraInfo const & camera_info,
    CameraInfo const & observer_camera_info,
    std::array<Asteroid, MAX_ASTEROID_COUNT> const & asteroids,
    f32 const delta_time)
{
    if (window->size.x == 0 || window->size.y == 0) { return; }

    // Calculate frame relevant values.
    u32 const flight_frame_index = gpu_context->swapchain.current_cpu_timeline_value() % (gpu_context->swapchain.info().max_allowed_frames_in_flight + 1);
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
                    auto sbt_info = gpu_context->ray_tracing_pipelines[name].pipeline->create_default_sbt();
                    this->gpu_context->ray_tracing_pipelines[name].sbt = sbt_info.table;
                    this->gpu_context->ray_tracing_pipelines[name].sbt_buffer_id = sbt_info.buffer;
                }
            }
        }

        vsm_state.update_vsm_lights(scene->_active_point_lights);
        CameraInfo real_camera_info = camera_info;
        if(render_context->debug_frustum >= 0) {
            real_camera_info = vsm_state.point_lights_cpu.at(0).face_cameras[render_context->debug_frustum];
            real_camera_info.screen_size = camera_info.screen_size;
            real_camera_info.inv_screen_size = camera_info.inv_screen_size;
        }

        // Set Render Data.
        render_context->render_data.camera_prev_frame = render_context->render_data.camera;
        render_context->render_data.observer_camera_prev_frame = render_context->render_data.observer_camera;
        render_context->render_data.camera = real_camera_info;
        render_context->render_data.observer_camera = observer_camera_info;
        render_context->render_data.frame_index = static_cast<u32>(gpu_context->swapchain.current_cpu_timeline_value());
        render_context->render_data.frames_in_flight = static_cast<u32>(gpu_context->swapchain.info().max_allowed_frames_in_flight);
        render_context->render_data.delta_time = delta_time;

        render_context->render_data.cull_data = fill_cull_data(*render_context);
    }

    // PGI Settings Resolve
    render_context->render_data.pgi_settings.probe_count.x = std::max(1, render_context->render_data.pgi_settings.probe_count.x);
    render_context->render_data.pgi_settings.probe_count.y = std::max(1, render_context->render_data.pgi_settings.probe_count.y);
    render_context->render_data.pgi_settings.probe_count.z = std::max(1, render_context->render_data.pgi_settings.probe_count.z);
    render_context->render_data.pgi_settings.probe_spacing = {
        static_cast<f32>(render_context->render_data.pgi_settings.probe_range.x) / static_cast<f32>(render_context->render_data.pgi_settings.probe_count.x),
        static_cast<f32>(render_context->render_data.pgi_settings.probe_range.y) / static_cast<f32>(render_context->render_data.pgi_settings.probe_count.y),
        static_cast<f32>(render_context->render_data.pgi_settings.probe_range.z) / static_cast<f32>(render_context->render_data.pgi_settings.probe_count.z),
    };    
    render_context->render_data.pgi_settings.probe_spacing_rcp = {
        1.0f / static_cast<f32>(render_context->render_data.pgi_settings.probe_spacing.x),
        1.0f / static_cast<f32>(render_context->render_data.pgi_settings.probe_spacing.y),
        1.0f / static_cast<f32>(render_context->render_data.pgi_settings.probe_spacing.z),
    };
    render_context->render_data.pgi_settings.max_visibility_distance = glm::length(glm::vec3{
        render_context->render_data.pgi_settings.probe_spacing.x,
        render_context->render_data.pgi_settings.probe_spacing.y,
        render_context->render_data.pgi_settings.probe_spacing.z,
    }) * 1.01f;

    bool const settings_changed = render_context->render_data.settings != render_context->prev_settings;
    bool const pgi_settings_changed = 
        render_context->render_data.pgi_settings.debug_probe_draw_mode != render_context->prev_pgi_settings.debug_probe_draw_mode ||
        render_context->render_data.pgi_settings.probe_count.x != render_context->prev_pgi_settings.probe_count.x ||
        render_context->render_data.pgi_settings.probe_count.y != render_context->prev_pgi_settings.probe_count.y ||
        render_context->render_data.pgi_settings.probe_count.z != render_context->prev_pgi_settings.probe_count.z ||
        render_context->render_data.pgi_settings.probe_radiance_resolution != render_context->prev_pgi_settings.probe_radiance_resolution ||
        render_context->render_data.pgi_settings.probe_trace_resolution != render_context->prev_pgi_settings.probe_trace_resolution ||
        render_context->render_data.pgi_settings.probe_visibility_resolution != render_context->prev_pgi_settings.probe_visibility_resolution ||
        render_context->render_data.pgi_settings.enabled != render_context->prev_pgi_settings.enabled;
    bool const sky_settings_changed = render_context->render_data.sky_settings != render_context->prev_sky_settings;
    auto const sky_res_changed_flags = render_context->render_data.sky_settings.resolutions_changed(render_context->prev_sky_settings);
    bool const vsm_settings_changed =
        render_context->render_data.vsm_settings.enable != render_context->prev_vsm_settings.enable;
    // Sky is transient of main task graph
    if (settings_changed || sky_res_changed_flags.sky_changed || vsm_settings_changed || pgi_settings_changed)
    {
        recreate_framebuffer();
        clear_select_buffers();
        main_task_graph = create_main_task_graph();
    }
    daxa::DeviceAddress render_data_device_address =
        gpu_context->device.buffer_device_address(render_context->tgpu_render_data.get_state().buffers[0]).value();

    render_context->render_data.readback =
        gpu_context->device.buffer_device_address(general_readback_buffer).value() +
        sizeof(ReadbackValues) * (render_context->render_data.frame_index % 4);
    if (sky_settings_changed)
    {
        // Potentially wastefull, ideally we want to only recreate the resource that changed the name
        if (sky_res_changed_flags.multiscattering_changed || sky_res_changed_flags.transmittance_changed)
        {
            recreate_sky_luts();
        }
        // Whenever the settings change we need to recalculate the transmittance and multiscattering look up textures
        auto const sky_settings_offset = offsetof(RenderGlobalData, sky_settings);
        render_context->render_data.sky_settings_ptr = render_data_device_address + sky_settings_offset;

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

    // Write GPUScene pointers
    {
        auto& device = render_context->gpu_context->device;
        render_context->render_data.scene.meshes = device.device_address(scene->_gpu_mesh_manifest.get_state().buffers[0]).value();
        render_context->render_data.scene.mesh_lod_groups = device.device_address(scene->_gpu_mesh_lod_group_manifest.get_state().buffers[0]).value();
        render_context->render_data.scene.mesh_groups = device.device_address(scene->_gpu_mesh_group_manifest.get_state().buffers[0]).value();
        render_context->render_data.scene.entity_to_meshgroup = device.device_address(scene->_gpu_entity_mesh_groups.get_state().buffers[0]).value();
        render_context->render_data.scene.materials = device.device_address(scene->_gpu_material_manifest.get_state().buffers[0]).value();
        render_context->render_data.scene.entity_transforms = device.device_address(scene->_gpu_entity_transforms.get_state().buffers[0]).value();
        render_context->render_data.scene.entity_combined_transforms = device.device_address(scene->_gpu_entity_combined_transforms.get_state().buffers[0]).value();
    }

    auto const vsm_projections_info = GetVSMProjectionsInfo{
        .camera_info = &render_context->render_data.camera,
        .sun_direction = std::bit_cast<f32vec3>(render_context->render_data.sky_settings.sun_direction),
        .clip_0_scale = render_context->render_data.vsm_settings.clip_0_frustum_scale,
        .clip_0_near = render_context->render_data.vsm_settings.fixed_near_far ? -1'000.0f : 0.01f,
        .clip_0_far = render_context->render_data.vsm_settings.fixed_near_far ? 1'000.0f : 10.0f,
        .clip_0_height_offset = 5.0f,
        .use_fixed_near_far = s_cast<bool>(render_context->render_data.vsm_settings.fixed_near_far),
        .debug_context = &gpu_context->shader_debug_context,
    };
    vsm_state.clip_projections_cpu = get_vsm_projections(vsm_projections_info);

    asteroid_state.update_cpu_data(asteroids);
    ShaderDebugAABBDraw aabb_draw;
    aabb_draw.position = daxa_f32vec3{0.0f, 0.0f, 0.0f};
    aabb_draw.size = daxa_f32vec3{2.0f * DOMAIN_BOUNDS * 0.001, 2.0f * DOMAIN_BOUNDS * 0.001, 2.0f * DOMAIN_BOUNDS * 0.001};
    aabb_draw.color = daxa_f32vec3{1.0f, 1.0f, 1.0f};
    aabb_draw.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
    render_context->gpu_context->shader_debug_context.aabb_draws.cpu_draws.push_back(aabb_draw);

    for (i32 clip = 0; clip < VSM_CLIP_LEVELS; clip++)
    {
        auto const clear_offset = std::bit_cast<i32vec2>(vsm_state.clip_projections_cpu.at(clip).page_offset) - vsm_state.last_frame_offsets.at(clip);
        vsm_state.free_wrapped_pages_info_cpu.at(clip).clear_offset = std::bit_cast<daxa_i32vec2>(clear_offset);

        vsm_state.last_frame_offsets.at(clip) = std::bit_cast<i32vec2>(vsm_state.clip_projections_cpu.at(clip).page_offset);
        vsm_state.clip_projections_cpu.at(clip).page_offset.x = vsm_state.clip_projections_cpu.at(clip).page_offset.x % VSM_PAGE_TABLE_RESOLUTION;
        vsm_state.clip_projections_cpu.at(clip).page_offset.y = vsm_state.clip_projections_cpu.at(clip).page_offset.y % VSM_PAGE_TABLE_RESOLUTION;
    }
    vsm_state.globals_cpu.clip_0_texel_world_size = (2.0f * render_context->render_data.vsm_settings.clip_0_frustum_scale) / VSM_TEXTURE_RESOLUTION;
    vsm_state.update_vsm_lights(scene->_active_point_lights);
    if(render_context->visualize_frustum){
        debug_draw_point_frusti(DebugDrawPointFrusiInfo{
            .light = &vsm_state.point_lights_cpu.at(std::max(render_context->render_data.vsm_settings.force_point_light_idx, 0)),
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
    if (new_swapchain_image.is_empty()) { return; }
    swapchain_image.set_images({.images = std::array{new_swapchain_image}});
    meshlet_instances.swap_buffers(meshlet_instances_last_frame);

    if (static_cast<daxa_u32>(gpu_context->swapchain.current_cpu_timeline_value()) == 0) { clear_select_buffers(); }

    render_context->render_times.readback_render_times(render_context->render_data.frame_index);

    // Draw Frustum Camera.
    gpu_context->shader_debug_context.aabb_draws.draw(ShaderDebugAABBDraw{
        .position = daxa_f32vec3(0, 0, 0.5),
        .size = daxa_f32vec3(2.01, 2.01, 0.999),
        .color = daxa_f32vec3(1, 0, 0),
        .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC,
    });

    gpu_context->shader_debug_context.update(gpu_context->device, render_target_size, window->size);

    u32 const fif_index = render_context->render_data.frame_index % (render_context->gpu_context->swapchain.info().max_allowed_frames_in_flight + 1);
    main_task_graph.execute({});
}