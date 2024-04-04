#include "renderer.hpp"

#include "../shader_shared/scene.inl"
#include "../shader_shared/debug.inl"

#include "rasterize_visbuffer/rasterize_visbuffer.hpp"

#include "virtual_shadow_maps/vsm.inl"

#include "tasks/memset.inl"
#include "tasks/dvmaa.inl"
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

inline auto create_task_buffer(GPUContext * context, auto size, auto task_buf_name, auto buf_name)
{
    return daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = static_cast<u32>(size),
                    .name = buf_name,
                }),
            },
        },
        .name = task_buf_name,
    }};
}

Renderer::Renderer(
    Window * window, GPUContext * context, Scene * scene, AssetProcessor * asset_manager, daxa::ImGuiRenderer * imgui_renderer)
    : render_context{std::make_unique<RenderContext>(context)}, window{window}, context{context}, scene{scene}, asset_manager{asset_manager}, imgui_renderer{imgui_renderer}
{
    zero_buffer = create_task_buffer(context, sizeof(u32), "zero_buffer", "zero_buffer");
    meshlet_instances = create_task_buffer(context, size_of_meshlet_instance_buffer(), "meshlet_instances", "meshlet_instances_a");
    meshlet_instances_last_frame = create_task_buffer(context, size_of_meshlet_instance_buffer(), "meshlet_instances_last_frame", "meshlet_instances_b");
    visible_meshlet_instances = create_task_buffer(context, sizeof(VisibleMeshletList), "visible_meshlet_instances", "visible_meshlet_instances");
    luminance_average = create_task_buffer(context, sizeof(f32), "luminance average", "luminance_average");

    buffers = {zero_buffer, meshlet_instances, meshlet_instances_last_frame, visible_meshlet_instances, luminance_average};

    swapchain_image = daxa::TaskImage{{.swapchain_image = true, .name = "swapchain_image"}};
    transmittance = daxa::TaskImage{{.name = "transmittance"}};
    multiscattering = daxa::TaskImage{{.name = "multiscattering"}};
    sky_ibl_cube = daxa::TaskImage{{.name = "sky ibl cube"}};

    vsm_state.initialize_persitent_state(context);

    images = {
        transmittance,
        multiscattering,
        sky_ibl_cube,
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
        for (auto buffer : tbuffer.get_state().buffers)
        {
            this->context->device.destroy_buffer(buffer);
        }
    }
    for (auto & timage : images)
    {
        for (auto image : timage.get_state().images)
        {
            this->context->device.destroy_image(image);
        }
    }
    vsm_state.cleanup_persistent_state(context);
    this->context->device.wait_idle();
    this->context->device.collect_garbage();
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
        {cull_and_draw_pages_pipelines[0]},
        {cull_and_draw_pages_pipelines[1]},
        {draw_shader_debug_circles_pipeline_compile_info()},
        {draw_shader_debug_rectangles_pipeline_compile_info()},
        {draw_shader_debug_aabb_pipeline_compile_info()},
        {draw_shader_debug_box_pipeline_compile_info()},
    };
    {
        add_if_not_present(this->context->raster_pipelines, rasters, slang_draw_visbuffer_pipelines[0]);
        add_if_not_present(this->context->raster_pipelines, rasters, slang_draw_visbuffer_pipelines[1]);
    }
    {
        add_if_not_present(this->context->raster_pipelines, rasters, slang_draw_visbuffer_mesh_shader_pipelines[0]);
        add_if_not_present(this->context->raster_pipelines, rasters, slang_draw_visbuffer_mesh_shader_pipelines[1]);
        add_if_not_present(this->context->raster_pipelines, rasters, slang_cull_meshlets_draw_visbuffer_pipelines[0]);
        add_if_not_present(this->context->raster_pipelines, rasters, slang_cull_meshlets_draw_visbuffer_pipelines[1]);
    }
    for (auto info : rasters)
    {
        auto compilation_result = this->context->pipeline_manager.add_raster_pipeline(info);
        if (compilation_result.value()->is_valid())
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] SUCCESFULLY compiled pipeline {}", info.name));
        }
        else
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] FAILED to compile pipeline {} with message \n {}", info.name,
                compilation_result.message()));
        }
        this->context->raster_pipelines[info.name] = compilation_result.value();
    }
    std::vector<daxa::ComputePipelineCompileInfo> computes = {
        {alloc_entity_to_mesh_instances_offsets_pipeline_compile_info()},
        {dvm_resolve_visbuffer_compile_info()},
        {set_entity_meshlets_visibility_bitmasks_pipeline_compile_info()},
        {AllocMeshletInstBitfieldsCommandWriteTask::pipeline_compile_info},
        {prepopulate_meshlet_instances_pipeline_compile_info()},
        {IndirectMemsetBufferTask::pipeline_compile_info},
        {analyze_visbufer_pipeline_compile_info()},
        {gen_hiz_pipeline_compile_info()},
        {write_swapchain_pipeline_compile_info()},
        {shade_opaque_pipeline_compile_info()},
        {cull_meshes_pipeline_compile_info()},
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
        {CullAndDrawPages_WriteCommandTask::pipeline_compile_info},
        {vsm_mark_required_pages_pipeline_compile_info()},
        {vsm_find_free_pages_pipeline_compile_info()},
        {vsm_allocate_pages_pipeline_compile_info()},
        {vsm_clear_pages_pipeline_compile_info()},
        {vsm_gen_dirty_bit_hiz_pipeline_compile_info()},
        {vsm_clear_dirty_bit_pipeline_compile_info()},
        {vsm_debug_virtual_page_table_pipeline_compile_info()},
        {vsm_debug_meta_memory_table_pipeline_compile_info()},
        {decode_visbuffer_test_pipeline_info()},
    };
    {
        add_if_not_present(this->context->compute_pipelines, computes, DrawVisbuffer_WriteCommandTask2::pipeline_compile_info);
    };
    {
        add_if_not_present(this->context->compute_pipelines, computes, DrawVisbuffer_WriteCommandTask2::pipeline_compile_info);
    }
    for (auto const & info : computes)
    {
        auto compilation_result = this->context->pipeline_manager.add_compute_pipeline(info);
        if (compilation_result.value()->is_valid())
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] SUCCESFULLY compiled pipeline {}", info.name));
        }
        else
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] FAILED to compile pipeline {} with message \n {}", info.name,
                compilation_result.message()));
        }
        this->context->compute_pipelines[info.name] = compilation_result.value();
    }

    while (!context->pipeline_manager.all_pipelines_valid())
    {
        auto const result = context->pipeline_manager.reload_all();
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
        context->device.destroy_image(transmittance.get_state().images[0]);
    }
    if (!multiscattering.get_state().images.empty() && !multiscattering.get_state().images[0].is_empty())
    {
        context->device.destroy_image(multiscattering.get_state().images[0]);
    }
    if (!sky_ibl_cube.get_state().images.empty() && !sky_ibl_cube.get_state().images[0].is_empty())
    {
        context->device.destroy_image(sky_ibl_cube.get_state().images[0]);
    }
    transmittance.set_images({
        .images = std::array{
            context->device.create_image({
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {render_context->render_data.sky_settings.transmittance_dimensions.x, render_context->render_data.sky_settings.transmittance_dimensions.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
                .name = "transmittance look up table",
            }),
        },
    });

    multiscattering.set_images({
        .images = std::array{
            context->device.create_image({
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {render_context->render_data.sky_settings.multiscattering_dimensions.x, render_context->render_data.sky_settings.multiscattering_dimensions.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
                .name = "multiscattering look up table",
            }),
        },
    });

    sky_ibl_cube.set_images({
        .images = std::array{
            context->device.create_image({
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
            context->device.destroy_image(timg.get_state().images[0]);
        }
        auto new_info = info;
        new_info.size = {this->window->get_width(), this->window->get_height(), 1};
        timg.set_images({.images = std::array{this->context->device.create_image(new_info)}});
    }
}

void Renderer::clear_select_buffers()
{
    using namespace daxa;
    TaskGraph list{{
        .device = this->context->device,
        .swapchain = this->context->swapchain,
        .name = "clear task list",
    }};
    list.use_persistent_buffer(meshlet_instances);
    list.use_persistent_buffer(meshlet_instances_last_frame);
    list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, meshlet_instances),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, meshlet_instances_last_frame),
        },
        .task = [=](daxa::TaskInterface ti)
        {
            auto mesh_instances_address = ti.device.get_device_address(ti.get(meshlet_instances).ids[0]).value();
            MeshletInstancesBufferHead mesh_instances_reset = make_meshlet_instance_buffer_head(mesh_instances_address);
            allocate_fill_copy(ti, mesh_instances_reset, ti.get(meshlet_instances));
            auto mesh_instances_prev_address = ti.device.get_device_address(ti.get(meshlet_instances_last_frame).ids[0]).value();
            MeshletInstancesBufferHead mesh_instances_prev_reset = make_meshlet_instance_buffer_head(mesh_instances_prev_address);
            allocate_fill_copy(ti, mesh_instances_prev_reset, ti.get(meshlet_instances_last_frame));
        },
        .name = "clear meshlet instance buffers",
    });
    list.use_persistent_buffer(visible_meshlet_instances);
    task_clear_buffer(list, visible_meshlet_instances, 0, sizeof(u32));
    task_clear_buffer(list, luminance_average, 0, sizeof(f32));
    list.submit({});
    list.complete({});
    list.execute({});
}

void Renderer::window_resized()
{
    if (this->window->size.x == 0 || this->window->size.y == 0)
    {
        DEBUG_MSG("minimized");
        return;
    }
    this->context->swapchain.resize();
    recreate_framebuffer();
}

auto Renderer::create_sky_lut_task_graph() -> daxa::TaskGraph
{
    daxa::TaskGraph task_graph{{
        .device = context->device,
        .name = "Calculate sky luts task graph",
    }};
    // TODO:    Do not use globals here, make a new buffer.
    //          Globals should only be used within the main task graph.
    task_graph.use_persistent_buffer(render_context->tgpu_render_data);
    task_graph.use_persistent_image(transmittance);
    task_graph.use_persistent_image(multiscattering);

    task_graph.add_task({
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

    task_graph.add_task(ComputeTransmittanceTask{
        .views = std::array{
            daxa::attachment_view(ComputeTransmittanceH::AT.globals, render_context->tgpu_render_data),
            daxa::attachment_view(ComputeTransmittanceH::AT.transmittance, transmittance),
        },
        .context = context,
    });

    task_graph.add_task(ComputeMultiscatteringTask{
        .views = std::array{
            daxa::attachment_view(ComputeMultiscatteringH::AT.globals, render_context->tgpu_render_data),
            daxa::attachment_view(ComputeMultiscatteringH::AT.transmittance, transmittance),
            daxa::attachment_view(ComputeMultiscatteringH::AT.multiscattering, multiscattering),
        },
        .render_context = render_context.get(),
    });
    task_graph.submit({});
    task_graph.complete({});
    return task_graph;
}

auto Renderer::create_main_task_graph() -> daxa::TaskGraph
{
    // Rasterize Visbuffer:
    // - reset/clear certain buffers
    // - prepopulate meshlet instances, these meshlet instances are drawn in the first pass.
    //     - uses list of visible meshlets of last frame (visible_meshlet_instances) and meshlet instance list from last
    //     frame (meshlet_instances_last_frame)
    //     - filteres meshlets when their entities/ meshes got invalidated.
    //     - builds bitfields (entity_meshlet_visibility_bitfield_offsets), that denote if a meshlet of an entity is
    //     drawn in the first pass.
    // - draw first pass
    //     - draws meshlet instances, generated by prepopulate_instantiated_meshlets.
    //     - draws trianlge id and depth. triangle id indexes into the meshlet instance list (that is freshly generated
    //     every frame), also stores triangle index within meshlet.
    //     - effectively draws the meshlets that were visible last frame as the first thing.
    // - build hiz depth map
    //     - lowest mip is half res of render target resolution, depth map at full res is not copied into the hiz.
    //     - single pass downsample dispatch. Each workgroup downsamples a 64x64 region, the very last workgroup to
    //     finish downsamples all the results of the previous workgroups.
    // - cull meshes
    //     - dispatch over all entities for all their meshes
    //     - cull against: hiz, frustum
    //     - builds argument lists for meshlet culling.
    //     - 32 meshlet cull argument lists, each beeing a bucket for arguments. An argument in each bucket represents
    //     2^bucket_index meshlets to be processed.
    // - cull and draw meshlets
    //     - 32 dispatches each going over one of the generated cull argument lists.
    //     - when mesh shaders are enabled, this is a single pipeline. Task shaders cull in this case.
    //     - when mesh shaders are disabled, a compute shader culls.
    //     - in either case, the task/compute cull shader fill the list of meshlet instances. This list is used to
    //     compactly reference meshlets via pixel id.
    //     - draws triangle id and depth
    //     - meshlet cull against: frustum, hiz
    //     - triangle cull (only on with mesh shaders) against: backface
    // - analyze visbuffer:
    //     - reads final opaque visbuffer
    //     - generates list of visible meshlets
    //     - marks visible triangles of meshlet instances in bitfield.
    //     - can optionally generate list of unique triangles.
    using namespace daxa;
    TaskGraph task_list{{
        .device = this->context->device,
        .swapchain = this->context->swapchain,
        .staging_memory_pool_size = 2'097'152, // 2MiB.
        .name = "Sandbox main TaskGraph",
    }};
    for (auto const & tbuffer : buffers)
    {
        task_list.use_persistent_buffer(tbuffer);
    }
    for (auto const & timage : images)
    {
        task_list.use_persistent_image(timage);
    }
    task_list.use_persistent_buffer(scene->_gpu_entity_meta);
    task_list.use_persistent_buffer(scene->_gpu_entity_transforms);
    task_list.use_persistent_buffer(scene->_gpu_entity_combined_transforms);
    task_list.use_persistent_buffer(scene->_gpu_entity_parents);
    task_list.use_persistent_buffer(scene->_gpu_entity_mesh_groups);
    task_list.use_persistent_buffer(scene->_gpu_mesh_manifest);
    task_list.use_persistent_buffer(scene->_gpu_mesh_group_manifest);
    task_list.use_persistent_buffer(scene->_gpu_material_manifest);
    task_list.use_persistent_buffer(render_context->tgpu_render_data);
    task_list.use_persistent_buffer(render_context->scene_draw.opaque_draw_list_buffer);
    task_list.use_persistent_buffer(vsm_state.globals);
    task_list.use_persistent_image(vsm_state.memory_block);
    task_list.use_persistent_image(vsm_state.meta_memory_table);
    task_list.use_persistent_image(vsm_state.page_table);
    task_list.use_persistent_image(vsm_state.page_height_offsets);
    task_list.use_persistent_image(context->shader_debug_context.vsm_debug_page_table);
    task_list.use_persistent_image(context->shader_debug_context.vsm_debug_meta_memory_table);
    auto debug_lens_image = context->shader_debug_context.tdebug_lens_image;
    task_list.use_persistent_image(debug_lens_image);
    task_list.use_persistent_image(swapchain_image);

    task_clear_image(task_list, debug_lens_image, std::array{0.0f, 0.0f, 0.0f, 1.0f});

    auto debug_image = task_list.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {
            render_context->render_data.settings.render_target_size.x,
            render_context->render_data.settings.render_target_size.y,
            1,
        },
        .name = "debug_image",
    });

    bool const dvmaa = render_context->render_data.settings.anti_aliasing_mode == AA_MODE_DVM;
    auto visbuffer = raster_visbuf::create_visbuffer(task_list, *render_context);
    daxa::TaskImageView depth = {};
    if (dvmaa)
    {
        depth = dvmaa::create_dvmaa_depth(task_list, *render_context);
    }
    else
    {
        depth = raster_visbuf::create_depth(task_list, *render_context);
    }
    daxa::TaskImageView dvmaa_visbuffer = daxa::NullTaskImage;
    daxa::TaskImageView dvmaa_depth = daxa::NullTaskImage;
    if (dvmaa)
    {
        dvmaa_visbuffer = dvmaa::create_dvmaa_ms_visbuffer(task_list, *render_context);
        dvmaa_depth = dvmaa::create_dvmaa_ms_depth(task_list, *render_context);
    }

    task_list.add_task(ReadbackTask{
        .views = std::array{daxa::attachment_view(ReadbackH::AT.globals, render_context->tgpu_render_data)},
        .shader_debug_context = &context->shader_debug_context,
    });
    task_list.add_task({
        .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, render_context->tgpu_render_data)},
        .task = [&](daxa::TaskInterface ti)
        {
            allocate_fill_copy(ti, render_context->render_data, ti.get(render_context->tgpu_render_data));
            context->shader_debug_context.update_debug_buffer(ti.device, ti.recorder, *ti.allocator);
        },
        .name = "update global buffers",
    });

    auto sky = task_list.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {render_context->render_data.sky_settings.sky_dimensions.x, render_context->render_data.sky_settings.sky_dimensions.y, 1},
        .name = "sky look up table",
    });
    auto luminance_histogram = task_list.create_transient_buffer({sizeof(u32) * (LUM_HISTOGRAM_BIN_COUNT), "luminance_histogram"});

    daxa::TaskImageView sky_ibl_view = sky_ibl_cube.view().view({.layer_count = 6});
    task_list.add_task(ComputeSkyTask{
        .views = std::array{
            daxa::attachment_view(ComputeSkyH::AT.globals, render_context->tgpu_render_data),
            daxa::attachment_view(ComputeSkyH::AT.transmittance, transmittance),
            daxa::attachment_view(ComputeSkyH::AT.multiscattering, multiscattering),
            daxa::attachment_view(ComputeSkyH::AT.sky, sky),
        },
        .render_context = render_context.get(),
    });
    task_list.add_task(SkyIntoCubemapTask{
        .views = std::array{
            daxa::attachment_view(SkyIntoCubemapH::AT.globals, render_context->tgpu_render_data),
            daxa::attachment_view(SkyIntoCubemapH::AT.transmittance, transmittance),
            daxa::attachment_view(SkyIntoCubemapH::AT.sky, sky),
            daxa::attachment_view(SkyIntoCubemapH::AT.ibl_cube, sky_ibl_view),
        },
        .context = context,
    });

    // Clear out counters for current meshlet instance lists.
    task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, meshlet_instances),
        },
        .task = [=](daxa::TaskInterface ti)
        {
            auto mesh_instances_address = ti.device.get_device_address(ti.get(meshlet_instances).ids[0]).value();
            MeshletInstancesBufferHead mesh_instances_reset = make_meshlet_instance_buffer_head(mesh_instances_address);
            allocate_fill_copy(ti, mesh_instances_reset, ti.get(meshlet_instances));
        },
        .name = "clear meshlet instance buffer",
    });

    daxa::TaskBufferView first_pass_meshlets_bitfield_offsets = {};
    daxa::TaskBufferView first_pass_meshlets_bitfield_arena = {};
    task_prepopulate_meshlet_instances({
        .render_context = render_context.get(),
        .task_graph = task_list,
        .meshes = scene->_gpu_mesh_manifest,
        .materials = scene->_gpu_material_manifest,
        .entity_mesh_groups = scene->_gpu_entity_mesh_groups,
        .mesh_group_manifest = scene->_gpu_mesh_group_manifest,
        .visible_meshlets_prev = visible_meshlet_instances,
        .meshlet_instances_last_frame = meshlet_instances_last_frame,
        .meshlet_instances = meshlet_instances,
        .first_pass_meshlets_bitfield_offsets = first_pass_meshlets_bitfield_offsets,
        .first_pass_meshlets_bitfield_arena = first_pass_meshlets_bitfield_arena,
    });

    task_draw_visbuffer({
        .render_context = render_context.get(),
        .task_graph = task_list,
        .pass = PASS0_DRAW_VISIBLE_LAST_FRAME,
        .meshlet_instances = meshlet_instances,
        .meshes = scene->_gpu_mesh_manifest,
        .material_manifest = scene->_gpu_material_manifest,
        .combined_transforms = scene->_gpu_entity_combined_transforms,
        .vis_image = visbuffer,
        .debug_image = debug_image,
        .depth_image = depth,
        .dvmaa_vis_image = dvmaa_visbuffer,
        .dvmaa_depth_image = dvmaa_depth,
    });

    daxa::TaskImageView hiz = {};
    task_gen_hiz_single_pass({render_context.get(), task_list, depth, render_context->tgpu_render_data, &hiz});

    daxa::TaskBufferView meshlets_cull_arg_buckets_buffers = {};
    tasks_cull_meshes({
        .render_context = render_context.get(),
        .task_list = task_list,
        .globals = render_context->tgpu_render_data,
        .opaque_draw_lists = render_context->scene_draw.opaque_draw_list_buffer,
        .meshes = scene->_gpu_mesh_manifest,
        .materials = scene->_gpu_material_manifest,
        .entity_meta = scene->_gpu_entity_meta,
        .entity_meshgroup_indices = scene->_gpu_entity_mesh_groups,
        .meshgroups = scene->_gpu_mesh_group_manifest,
        .entity_transforms = scene->_gpu_entity_transforms,
        .entity_combined_transforms = scene->_gpu_entity_combined_transforms,
        .hiz = hiz,
        .meshlets_cull_arg_buckets_buffers = meshlets_cull_arg_buckets_buffers,
    });

    task_cull_and_draw_visbuffer({
        .render_context = render_context.get(),
        .task_graph = task_list,
        .meshlets_cull_arg_buckets_buffers = meshlets_cull_arg_buckets_buffers,
        .entity_meta_data = scene->_gpu_entity_meta,
        .entity_meshgroups = scene->_gpu_entity_mesh_groups,
        .entity_combined_transforms = scene->_gpu_entity_combined_transforms,
        .mesh_groups = scene->_gpu_mesh_group_manifest,
        .meshes = scene->_gpu_mesh_manifest,
        .material_manifest = scene->_gpu_material_manifest,
        .first_pass_meshlets_bitfield_offsets = first_pass_meshlets_bitfield_offsets,
        .first_pass_meshlets_bitfield_arena = first_pass_meshlets_bitfield_arena,
        .hiz = hiz,
        .meshlet_instances = meshlet_instances,
        .vis_image = visbuffer,
        .debug_image = debug_image,
        .depth_image = depth,
        .dvmaa_vis_image = dvmaa_visbuffer,
        .dvmaa_depth_image = dvmaa_depth,
    });

    vsm_state.initialize_transient_state(task_list);
    task_clear_buffer(task_list, vsm_state.allocation_count, 0);
    task_clear_buffer(task_list, vsm_state.find_free_pages_header, 0);
    task_draw_vsms({
        .render_context = render_context.get(),
        .tg = &task_list,
        .vsm_state = &vsm_state,
        .meshlets_cull_arg_buckets_buffers = meshlets_cull_arg_buckets_buffers,
        .meshlet_instances = meshlet_instances,
        .meshes = scene->_gpu_mesh_manifest,
        .entity_combined_transforms = scene->_gpu_entity_combined_transforms,
        .material_manifest = scene->_gpu_material_manifest,
        .depth = depth,
    });

    auto visible_meshlets_bitfield = task_list.create_transient_buffer({
        sizeof(daxa_u32) * MAX_MESHLET_INSTANCES,
        "visible meshlets bitfield",
    });
    task_clear_buffer(task_list, visible_meshlets_bitfield, 0);
    task_clear_buffer(task_list, visible_meshlet_instances, 0);
    task_list.add_task(AnalyzeVisBufferTask2{
        .views = std::array{
            daxa::attachment_view(AnalyzeVisbuffer2H::AT.globals, render_context->tgpu_render_data),
            daxa::attachment_view(AnalyzeVisbuffer2H::AT.visbuffer, visbuffer),
            daxa::attachment_view(AnalyzeVisbuffer2H::AT.instantiated_meshlets, meshlet_instances),
            daxa::attachment_view(AnalyzeVisbuffer2H::AT.meshlet_visibility_bitfield, visible_meshlets_bitfield),
            daxa::attachment_view(AnalyzeVisbuffer2H::AT.visible_meshlets, visible_meshlet_instances),
        },
        .context = context,
    });

    if (render_context->render_data.settings.draw_from_observer)
    {
        task_draw_visbuffer({
            .render_context = render_context.get(),
            .task_graph = task_list,
            .pass = PASS4_OBSERVER_DRAW_ALL,
            .meshlet_instances = meshlet_instances,
            .meshes = scene->_gpu_mesh_manifest,
            .material_manifest = scene->_gpu_material_manifest,
            .combined_transforms = scene->_gpu_entity_combined_transforms,
            .vis_image = visbuffer,
            .debug_image = debug_image,
            .depth_image = depth,
            .dvmaa_vis_image = dvmaa_visbuffer,
            .dvmaa_depth_image = dvmaa_depth,
        });
    }
    task_list.submit({});

    auto color_image = task_list.create_transient_image({
        .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
        .size = {
            render_context->render_data.settings.render_target_size.x,
            render_context->render_data.settings.render_target_size.y,
            1,
        },
        .name = "color_image",
    });
    task_list.add_task(DecodeVisbufferTestTask{
        .views = std::array{
            daxa::attachment_view(DecodeVisbufferTestH::AT.globals, render_context->tgpu_render_data),
            daxa::attachment_view(DecodeVisbufferTestH::AT.debug_image, debug_image),
            daxa::attachment_view(DecodeVisbufferTestH::AT.vis_image, visbuffer),
            daxa::attachment_view(DecodeVisbufferTestH::AT.material_manifest, scene->_gpu_material_manifest),
            daxa::attachment_view(DecodeVisbufferTestH::AT.instantiated_meshlets, meshlet_instances),
            daxa::attachment_view(DecodeVisbufferTestH::AT.meshes, scene->_gpu_mesh_manifest),
            daxa::attachment_view(DecodeVisbufferTestH::AT.combined_transforms, scene->_gpu_entity_combined_transforms),
        },
        .context = context,
    });
    auto const vsm_page_table_view = vsm_state.page_table.view().view({.base_array_layer = 0, .layer_count = VSM_CLIP_LEVELS});
    auto const vsm_page_heigh_offsets_view = vsm_state.page_height_offsets.view().view({.base_array_layer = 0, .layer_count = VSM_CLIP_LEVELS});
    task_list.add_task(ShadeOpaqueTask{
        .views = std::array{
            daxa::attachment_view(ShadeOpaqueH::AT.debug_lens_image, debug_lens_image),
            daxa::attachment_view(ShadeOpaqueH::AT.globals, render_context->tgpu_render_data),
            daxa::attachment_view(ShadeOpaqueH::AT.color_image, color_image),
            daxa::attachment_view(ShadeOpaqueH::AT.vis_image, visbuffer),
            daxa::attachment_view(ShadeOpaqueH::AT.transmittance, transmittance),
            daxa::attachment_view(ShadeOpaqueH::AT.sky, sky),
            daxa::attachment_view(ShadeOpaqueH::AT.sky_ibl, sky_ibl_view),
            daxa::attachment_view(ShadeOpaqueH::AT.vsm_page_table, vsm_page_table_view),
            daxa::attachment_view(ShadeOpaqueH::AT.vsm_page_height_offsets, vsm_page_heigh_offsets_view),
            daxa::attachment_view(ShadeOpaqueH::AT.vsm_memory_block, vsm_state.memory_block),
            daxa::attachment_view(ShadeOpaqueH::AT.material_manifest, scene->_gpu_material_manifest),
            daxa::attachment_view(ShadeOpaqueH::AT.instantiated_meshlets, meshlet_instances),
            daxa::attachment_view(ShadeOpaqueH::AT.meshes, scene->_gpu_mesh_manifest),
            daxa::attachment_view(ShadeOpaqueH::AT.combined_transforms, scene->_gpu_entity_combined_transforms),
            daxa::attachment_view(ShadeOpaqueH::AT.luminance_average, luminance_average),
            daxa::attachment_view(ShadeOpaqueH::AT.vsm_clip_projections, vsm_state.clip_projections),
            daxa::attachment_view(ShadeOpaqueH::AT.vsm_globals, vsm_state.globals),
            daxa::attachment_view(ShadeOpaqueH::AT.debug_image, debug_image),
        },
        .context = context,
    });

    task_clear_buffer(task_list, luminance_histogram, 0);
    task_list.add_task(GenLuminanceHistogramTask{
        .views = std::array{
            daxa::attachment_view(GenLuminanceHistogramH::AT.globals, render_context->tgpu_render_data),
            daxa::attachment_view(GenLuminanceHistogramH::AT.histogram, luminance_histogram),
            daxa::attachment_view(GenLuminanceHistogramH::AT.luminance_average, luminance_average),
            daxa::attachment_view(GenLuminanceHistogramH::AT.color_image, color_image),
        },
        .render_context = render_context.get(),
    });
    task_list.add_task(GenLuminanceAverageTask{
        .views = std::array{
            daxa::attachment_view(GenLuminanceAverageH::AT.globals, render_context->tgpu_render_data),
            daxa::attachment_view(GenLuminanceAverageH::AT.histogram, luminance_histogram),
            daxa::attachment_view(GenLuminanceAverageH::AT.luminance_average, luminance_average),
        },
        .context = context,
    });
    task_list.add_task(WriteSwapchainTask{
        .views = std::array{
            daxa::attachment_view(WriteSwapchainH::AT.globals, render_context->tgpu_render_data),
            daxa::attachment_view(WriteSwapchainH::AT.swapchain, swapchain_image),
            daxa::attachment_view(WriteSwapchainH::AT.color_image, color_image),
        },
        .context = context,
    });

    task_list.add_task(DebugDrawTask{
        .views = std::array{
            daxa::attachment_view(DebugDrawH::AT.globals, render_context->tgpu_render_data),
            daxa::attachment_view(DebugDrawH::AT.color_image, swapchain_image),
            daxa::attachment_view(DebugDrawH::AT.depth_image, depth),
        },
        .rctx = render_context.get(),
    });
    task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskImageAccess::COLOR_ATTACHMENT, swapchain_image),
            daxa::inl_attachment(daxa::TaskImageAccess::FRAGMENT_SHADER_SAMPLED, debug_lens_image),
        },
        .task = [=, this](daxa::TaskInterface ti)
        {
            auto size = ti.device.info_image(ti.get(daxa::TaskImageAttachmentIndex(0)).ids[0]).value().size;
            imgui_renderer->record_commands(
                ImGui::GetDrawData(), ti.recorder, ti.get(daxa::TaskImageAttachmentIndex(0)).ids[0], size.x, size.y);
        },
        .name = "ImGui Draw",
    });

    task_list.submit({});
    task_list.present({});
    task_list.complete({});
    return task_list;
}

void Renderer::render_frame(
    CameraController const & camera_info,
    CameraController const & observer_camera_info,
    f32 const delta_time,
    SceneDraw scene_draw)
{
    if (window->size.x == 0 || window->size.y == 0) { return; }

    // Calculate frame relevant values.
    u32 const flight_frame_index = context->swapchain.current_cpu_timeline_value() % (context->swapchain.info().max_allowed_frames_in_flight + 1);
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
            1.0f / render_target_size.x,
            1.0f / render_target_size.y,
        };
        render_context->scene_draw = scene_draw;

        /// THIS SHOULD BE DONE SOMEWHERE ELSE!
        {
            auto reloaded_result = context->pipeline_manager.reload_all();
            if (auto reload_err = daxa::get_if<daxa::PipelineReloadError>(&reloaded_result))
            {
                std::cout << "Failed to reload " << reload_err->message << '\n';
            }
            if (auto _ = daxa::get_if<daxa::PipelineReloadSuccess>(&reloaded_result))
            {
                std::cout << "Successfully reloaded!\n";
            }
        }

        // Set Render Data.
        render_context->render_data.camera = camera_info.make_camera_info(render_context->render_data.settings);
        render_context->render_data.observer_camera = observer_camera_info.make_camera_info(render_context->render_data.settings);
        render_context->render_data.frame_index = static_cast<u32>(context->swapchain.current_cpu_timeline_value());
        render_context->render_data.delta_time = delta_time;
        render_context->render_data.test[0] = daxa_f32mat4x3{
            // rc = row column
            {11, 21, 31}, // col 1
            {12, 22, 32}, // col 2
            {13, 23, 33}, // col 3
            {14, 24, 34}, // col 4
        };
        render_context->render_data.test[1] = daxa_f32mat4x3{
            // rc = row column
            {11, 21, 31}, // col 1
            {12, 22, 32}, // col 2
            {13, 23, 33}, // col 3
            {14, 24, 34}, // col 4
        };
    }

    bool const settings_changed = render_context->render_data.settings != render_context->prev_settings;
    bool const sky_settings_changed = render_context->render_data.sky_settings != render_context->prev_sky_settings;
    auto const sky_res_changed_flags = render_context->render_data.sky_settings.resolutions_changed(render_context->prev_sky_settings);
    // Sky is transient of main task graph
    if (settings_changed || sky_res_changed_flags.sky_changed)
    {
        main_task_graph = create_main_task_graph();
    }
    daxa::DeviceAddress render_data_device_address =
        context->device.get_device_address(render_context->tgpu_render_data.get_state().buffers[0]).value();
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
    render_context->prev_settings = render_context->render_data.settings;
    render_context->prev_sky_settings = render_context->render_data.sky_settings;
    render_context->prev_vsm_settings = render_context->render_data.vsm_settings;

    vsm_state.clip_projections_cpu = get_vsm_projections(GetVSMProjectionsInfo{
        .camera_info = &camera_info,
        .sun_direction = std::bit_cast<f32vec3>(render_context->render_data.sky_settings.sun_direction),
        .clip_0_scale = render_context->render_data.vsm_settings.clip_0_frustum_scale,
        .clip_0_near = 0.1f,
        .clip_0_far = 10.0f,
        .clip_0_height_offset = 5.0f,
        .debug_context = &context->shader_debug_context,
    });

    for(i32 clip = 0; clip < VSM_CLIP_LEVELS; clip++)
    {
        const auto clear_offset = std::bit_cast<i32vec2>(vsm_state.clip_projections_cpu.at(clip).page_offset) - vsm_state.last_frame_offsets.at(clip);
        vsm_state.free_wrapped_pages_info_cpu.at(clip).clear_offset = std::bit_cast<daxa_i32vec2>(clear_offset);
    }
    for(i32 clip = 0; clip < VSM_CLIP_LEVELS; clip++)
    {
        vsm_state.last_frame_offsets.at(clip) = std::bit_cast<i32vec2>(vsm_state.clip_projections_cpu.at(clip).page_offset);
        vsm_state.clip_projections_cpu.at(clip).page_offset.x = vsm_state.clip_projections_cpu.at(clip).page_offset.x % VSM_PAGE_TABLE_RESOLUTION;
        vsm_state.clip_projections_cpu.at(clip).page_offset.y = vsm_state.clip_projections_cpu.at(clip).page_offset.y % VSM_PAGE_TABLE_RESOLUTION;
    }

    // clip_0.right - clip_0.left = 2.0f * clip_0_scale (HARDCODED FOR NOW TODO(msakmary) FIX)
    vsm_state.globals_cpu.clip_0_texel_world_size = (2.0f * render_context->render_data.vsm_settings.clip_0_frustum_scale) / VSM_TEXTURE_RESOLUTION;

    // debug_draw_clip_fusti(DebugDrawClipFrustiInfo{
    //     .clip_projections = std::span<const VSMClipProjection>(vsm_state.clip_projections_cpu.begin(), 1),
    //     .draw_individual_pages = false,
    //     .debug_context = &context->shader_debug_context,
    //     .vsm_view_direction = -std::bit_cast<f32vec3>(render_context->render_data.sky_settings.sun_direction),
    // });

    auto new_swapchain_image = context->swapchain.acquire_next_image();
    if (new_swapchain_image.is_empty()) { return; }
    swapchain_image.set_images({.images = std::array{new_swapchain_image}});
    meshlet_instances.swap_buffers(meshlet_instances_last_frame);

    if (static_cast<daxa_u32>(context->swapchain.current_cpu_timeline_value()) == 0) { clear_select_buffers(); }

    // Draw Frustum Camera.
    context->shader_debug_context.cpu_debug_aabb_draws.push_back(ShaderDebugAABBDraw{
        .position = daxa_f32vec3(0, 0, 0.5),
        .size = daxa_f32vec3(2.01, 2.01, 0.999),
        .color = daxa_f32vec3(1, 0, 0),
        .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC,
    });

    context->shader_debug_context.update(context->device, render_target_size, window->size);
    main_task_graph.execute({});
}