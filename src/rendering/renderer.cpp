#include "renderer.hpp"

#include "../shader_shared/scene.inl"
#include "../daxa_helper.hpp"
#include "../shader_shared/debug.inl"
#include "../shader_shared/readback.inl"

#include "rasterize_visbuffer/rasterize_visbuffer.hpp"

#include "virtual_shadow_maps/vsm.inl"

#include "ray_tracing/ray_tracing.inl"

#include "tasks/memset.inl"
#include "tasks/prefix_sum.inl"
#include "tasks/write_swapchain.inl"
#include "tasks/shade_opaque.inl"
#include "tasks/sky.inl"
#include "tasks/autoexposure.inl"
#include "tasks/shader_debug_draws.inl"
#include "tasks/decode_visbuffer_test.inl"

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
    general_readback_buffer = daxa::TaskBuffer{gpu_context->device, {sizeof(ReadbackValues) * 4, daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE, "general readback buffer"}};
    visible_meshlet_instances = create_task_buffer(gpu_context, sizeof(u32) * (MAX_MESHLET_INSTANCES + 4), "visible_meshlet_instances", "visible_meshlet_instances");

    buffers = {
        zero_buffer,
        meshlet_instances,
        meshlet_instances_last_frame,
        visible_meshlet_instances,
        visible_mesh_instances,
        luminance_average,
        general_readback_buffer};

    swapchain_image = daxa::TaskImage{{.swapchain_image = true, .name = "swapchain_image"}};
    transmittance = daxa::TaskImage{{.name = "transmittance"}};
    multiscattering = daxa::TaskImage{{.name = "multiscattering"}};
    sky_ibl_cube = daxa::TaskImage{{.name = "sky ibl cube"}};
    depth_vistory = daxa::TaskImage{{.name = "depth_history"}};
    f32_depth_vistory = daxa::TaskImage{{.name = "f32_depth_vistory"}};

    vsm_state.initialize_persitent_state(gpu_context);

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
        {cull_and_draw_pages_pipelines[0]},
        {cull_and_draw_pages_pipelines[1]},
        {draw_shader_debug_circles_pipeline_compile_info()},
        {draw_shader_debug_rectangles_pipeline_compile_info()},
        {draw_shader_debug_aabb_pipeline_compile_info()},
        {draw_shader_debug_box_pipeline_compile_info()},
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
        {sfpm_allocate_ent_bitfield_lists()},
        {gen_hiz_pipeline_compile_info2()},
        {cull_meshlets_compute_pipeline_compile_info()},
        {draw_meshlets_compute_pipeline_compile_info()},
        {tido::upgrade_compute_pipeline_compile_info(alloc_entity_to_mesh_instances_offsets_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(set_entity_meshlets_visibility_bitmasks_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(prepopulate_meshlet_instances_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(IndirectMemsetBufferTask::pipeline_compile_info)},
        {tido::upgrade_compute_pipeline_compile_info(analyze_visbufer_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(write_swapchain_pipeline_compile_info())},
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
        {tido::upgrade_compute_pipeline_compile_info(CullAndDrawPages_WriteCommandTask::pipeline_compile_info)},
        {tido::upgrade_compute_pipeline_compile_info(vsm_invalidate_pages_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_mark_required_pages_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_find_free_pages_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_allocate_pages_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_clear_pages_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_gen_dirty_bit_hiz_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_clear_dirty_bit_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_debug_virtual_page_table_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(vsm_debug_meta_memory_table_pipeline_compile_info())},
        {tido::upgrade_compute_pipeline_compile_info(decode_visbuffer_test_pipeline_info())},
        {tido::upgrade_compute_pipeline_compile_info(SplitAtomicVisbufferTask::pipeline_compile_info)},
        {tido::upgrade_compute_pipeline_compile_info(DrawVisbuffer_WriteCommandTask2::pipeline_compile_info)},
        {tido::upgrade_compute_pipeline_compile_info(ray_trace_ao_compute_pipeline_info())},
        {debug_task_draw_display_image_pipeline_info()},
        {rtao_denoiser_pipeline_info()},
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
    };
    for (auto const & info : ray_tracing)
    {
        auto compilation_result = this->gpu_context->pipeline_manager.add_ray_tracing_pipeline(info);
        if (compilation_result.value()->is_valid())
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] SUCCESFULLY compiled pipeline {}", info.name));
        }
        else
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] FAILED to compile pipeline {} with message \n {}", info.name,
                compilation_result.message()));
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
    tg.clear_buffer({.buffer = luminance_average, .size = sizeof(f32), .clear_value = 0});
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
    tg.use_persistent_tlas(scene->_scene_tlas);

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
    daxa::TaskImageView depth = visbuffer_ret.depth;
    daxa::TaskImageView visbuffer = visbuffer_ret.visbuffer;
    daxa::TaskImageView overdraw_image = visbuffer_ret.overdraw_image;

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
            .depth = depth,
        });
    }
    else
    {
        vsm_state.zero_out_transient_state(tg, render_context->render_data);
    }

    tg.submit({});

    auto color_image = tg.create_transient_image({
        .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
        .size = {
            render_context->render_data.settings.render_target_size.x,
            render_context->render_data.settings.render_target_size.y,
            1,
        },
        .name = "color_image",
    });
    daxa::TaskImageView ao_image = daxa::NullTaskImage;
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
                RayTraceAmbientOcclusionH::AT.vis_image | visbuffer,
                RayTraceAmbientOcclusionH::AT.sky | sky,
                RayTraceAmbientOcclusionH::AT.material_manifest | scene->_gpu_material_manifest,
                RayTraceAmbientOcclusionH::AT.instantiated_meshlets | meshlet_instances,
                RayTraceAmbientOcclusionH::AT.meshes | scene->_gpu_mesh_manifest,
                RayTraceAmbientOcclusionH::AT.mesh_groups | scene->_gpu_mesh_group_manifest,
                RayTraceAmbientOcclusionH::AT.combined_transforms | scene->_gpu_entity_combined_transforms,
                RayTraceAmbientOcclusionH::AT.geo_inst_indirections | scene->_scene_as_indirections,
                RayTraceAmbientOcclusionH::AT.tlas | scene->_scene_tlas,
            },
            .gpu_context = gpu_context,
            .render_context = render_context.get(),
        });
        tg.add_task(RTAODeoinserTask{
            .views = std::array{
                RTAODeoinserTask::AT.globals | render_context->tgpu_render_data,
                RTAODeoinserTask::AT.depth | depth,
                RTAODeoinserTask::AT.src | ao_image_raw,
                RTAODeoinserTask::AT.dst | ao_image,
            },
            .gpu_context = gpu_context,
            .render_context = render_context.get(),
        });
    }
    tg.add_task(DecodeVisbufferTestTask{
        .views = std::array{
            DecodeVisbufferTestH::AT.globals | render_context->tgpu_render_data,
            DecodeVisbufferTestH::AT.debug_image | debug_image,
            DecodeVisbufferTestH::AT.vis_image | visbuffer,
            DecodeVisbufferTestH::AT.material_manifest | scene->_gpu_material_manifest,
            DecodeVisbufferTestH::AT.instantiated_meshlets | meshlet_instances,
            DecodeVisbufferTestH::AT.meshes | scene->_gpu_mesh_manifest,
            DecodeVisbufferTestH::AT.combined_transforms | scene->_gpu_entity_combined_transforms,
        },
        .gpu_context = gpu_context,
    });
    auto const vsm_page_table_view = vsm_state.page_table.view().view({.base_array_layer = 0, .layer_count = VSM_CLIP_LEVELS});
    auto const vsm_page_heigh_offsets_view = vsm_state.page_view_pos_row.view().view({.base_array_layer = 0, .layer_count = VSM_CLIP_LEVELS});
    tg.add_task(ShadeOpaqueTask{
        .views = std::array{
            ShadeOpaqueH::AT.debug_lens_image | debug_lens_image,
            ShadeOpaqueH::AT.ao_image | ao_image,
            ShadeOpaqueH::AT.globals | render_context->tgpu_render_data,
            ShadeOpaqueH::AT.color_image | color_image,
            ShadeOpaqueH::AT.vis_image | visbuffer,
            ShadeOpaqueH::AT.transmittance | transmittance,
            ShadeOpaqueH::AT.sky | sky,
            ShadeOpaqueH::AT.sky_ibl | sky_ibl_view,
            ShadeOpaqueH::AT.vsm_page_table | vsm_page_table_view,
            ShadeOpaqueH::AT.vsm_page_view_pos_row | vsm_page_heigh_offsets_view,
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
            ShadeOpaqueH::AT.tlas | scene->_scene_tlas,
            ShadeOpaqueH::AT.point_lights | scene->_gpu_point_lights,
            ShadeOpaqueH::AT.vsm_point_lights | vsm_state.vsm_point_lights,
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
    tg.add_task(WriteSwapchainTask{
        .views = std::array{
            WriteSwapchainH::AT.globals | render_context->tgpu_render_data,
            WriteSwapchainH::AT.debug_image | debug_image,
            WriteSwapchainH::AT.swapchain | swapchain_image,
            WriteSwapchainH::AT.color_image | color_image,
        },
        .gpu_context = gpu_context,
    });

    tg.add_task(DebugDrawTask{
        .views = std::array{
            DebugDrawH::AT.globals | render_context->tgpu_render_data,
            DebugDrawH::AT.color_image | swapchain_image,
            DebugDrawH::AT.depth_image | depth,
        },
        .render_context = render_context.get(),
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
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, meshlet_instances),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, visible_mesh_instances),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, general_readback_buffer),
        },
        .task = [=, this](daxa::TaskInterface ti)
        {
            const u32 index = render_context->render_data.frame_index % 4;
#define READBACK_HELPER_MACRO(SRC_BUF, SRC_STRUCT, SRC_FIELD, DST_FIELD)                    \
    ti.recorder.copy_buffer_to_buffer({                                                     \
        .src_buffer = ti.get(SRC_BUF).ids[0],                                               \
        .dst_buffer = ti.get(general_readback_buffer).ids[0],                               \
        .src_offset = offsetof(SRC_STRUCT, SRC_FIELD),                                      \
        .dst_offset = offsetof(ReadbackValues, DST_FIELD) + sizeof(ReadbackValues) * index, \
        .size = sizeof(SRC_STRUCT::SRC_FIELD),                                              \
    })
            READBACK_HELPER_MACRO(meshlet_instances, MeshletInstancesBufferHead, prepass_draw_lists[0].pass_counts[0], first_pass_meshlet_count[0]);
            READBACK_HELPER_MACRO(meshlet_instances, MeshletInstancesBufferHead, prepass_draw_lists[0].pass_counts[1], second_pass_meshlet_count[0]);
            READBACK_HELPER_MACRO(meshlet_instances, MeshletInstancesBufferHead, prepass_draw_lists[1].pass_counts[0], first_pass_meshlet_count[1]);
            READBACK_HELPER_MACRO(meshlet_instances, MeshletInstancesBufferHead, prepass_draw_lists[1].pass_counts[1], second_pass_meshlet_count[1]);
            READBACK_HELPER_MACRO(visible_mesh_instances, VisibleMeshesList, count, visible_meshes);

            render_context->general_readback = ti.device.buffer_host_address_as<ReadbackValues>(ti.get(general_readback_buffer).ids[0]).value()[index];
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
            }
        }

        vsm_state.update_vsm_lights(scene->_active_point_lights);
        CameraInfo tmp = camera_info;
        if(render_context->debug_frustum >= 0) {
            tmp.position = scene->_active_point_lights.at(0).position;
            tmp.view = vsm_state.point_lights_cpu.at(0).view_matrices[render_context->debug_frustum];
            tmp.inv_view = vsm_state.point_lights_cpu.at(0).inverse_view_matrices[render_context->debug_frustum];
            tmp.proj = vsm_state.globals_cpu.point_light_projection_matrix;
            tmp.inv_proj = vsm_state.globals_cpu.inverse_point_light_projection_matrix;
            tmp.view_proj = tmp.proj * tmp.view;
            tmp.inv_view_proj = glm::inverse(tmp.view_proj);
        }
        // Set Render Data.
        render_context->render_data.camera_prev_frame = render_context->render_data.camera;
        render_context->render_data.observer_camera_prev_frame = render_context->render_data.observer_camera;
        render_context->render_data.camera = tmp;
        render_context->render_data.observer_camera = observer_camera_info;
        render_context->render_data.frame_index = static_cast<u32>(gpu_context->swapchain.current_cpu_timeline_value());
        render_context->render_data.frames_in_flight = static_cast<u32>(gpu_context->swapchain.info().max_allowed_frames_in_flight);
        render_context->render_data.delta_time = delta_time;

        render_context->render_data.cull_data = fill_cull_data(*render_context);
    }

    bool const settings_changed = render_context->render_data.settings != render_context->prev_settings;
    bool const sky_settings_changed = render_context->render_data.sky_settings != render_context->prev_sky_settings;
    auto const sky_res_changed_flags = render_context->render_data.sky_settings.resolutions_changed(render_context->prev_sky_settings);
    bool const vsm_settings_changed =
        render_context->render_data.vsm_settings.enable != render_context->prev_vsm_settings.enable;
    // Sky is transient of main task graph
    if (settings_changed || sky_res_changed_flags.sky_changed || vsm_settings_changed)
    {
        recreate_framebuffer();
        main_task_graph = create_main_task_graph();
    }
    daxa::DeviceAddress render_data_device_address =
        gpu_context->device.buffer_device_address(render_context->tgpu_render_data.get_state().buffers[0]).value();

    render_context->render_data.readback =
        gpu_context->device.buffer_device_address(general_readback_buffer.get_state().buffers[0]).value() +
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
    render_context->prev_sky_settings = render_context->render_data.sky_settings;
    render_context->prev_vsm_settings = render_context->render_data.vsm_settings;

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
            .light = &vsm_state.point_lights_cpu.at(0),
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
    gpu_context->shader_debug_context.cpu_debug_aabb_draws.push_back(ShaderDebugAABBDraw{
        .position = daxa_f32vec3(0, 0, 0.5),
        .size = daxa_f32vec3(2.01, 2.01, 0.999),
        .color = daxa_f32vec3(1, 0, 0),
        .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC,
    });

    gpu_context->shader_debug_context.update(gpu_context->device, render_target_size, window->size);

    u32 const fif_index = render_context->render_data.frame_index % (render_context->gpu_context->swapchain.info().max_allowed_frames_in_flight + 1);
    main_task_graph.execute({});
}