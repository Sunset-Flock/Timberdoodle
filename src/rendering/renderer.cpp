#include "renderer.hpp"

#include "../shader_shared/scene.inl"
#include "../shader_shared/debug.inl"

#include "rasterize_visbuffer/draw_visbuffer.inl"
#include "rasterize_visbuffer/cull_meshlets.inl"
#include "rasterize_visbuffer/cull_meshes.inl"
#include "rasterize_visbuffer/analyze_visbuffer.inl"
#include "rasterize_visbuffer/gen_hiz.inl"
#include "rasterize_visbuffer/prepopulate_meshlets.inl"

#include "tasks/prefix_sum.inl"
#include "tasks/write_swapchain.inl"
#include "tasks/shade_opaque.inl"
#include "tasks/sky.inl"
#include "tasks/autoexposure.inl"
#include "tasks/shader_debug_draws.inl"
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
    : window{window}, context{context}, scene{scene}, asset_manager{asset_manager}, imgui_renderer{imgui_renderer}
{
    zero_buffer = create_task_buffer(context, sizeof(u32), "zero_buffer", "zero_buffer");
    meshlet_instances = create_task_buffer(context, size_of_meshlet_instance_buffer(), "meshlet_instances", "meshlet_instances_a");
    meshlet_instances_last_frame = create_task_buffer(context, size_of_meshlet_instance_buffer(), "meshlet_instances_last_frame", "meshlet_instances_b");
    visible_meshlet_instances = create_task_buffer(context, sizeof(VisibleMeshletList), "visible_meshlet_instances", "visible_meshlet_instances");
    luminance_average = create_task_buffer(context, sizeof(f32), "luminance average", "luminance_average");

    buffers = {zero_buffer, meshlet_instances, meshlet_instances_last_frame, visible_meshlet_instances, luminance_average};

    swapchain_image = daxa::TaskImage{{.swapchain_image = true, .name = "swapchain_image"}};
    depth = daxa::TaskImage{{.name = "depth"}};
    visbuffer = daxa::TaskImage{{.name = "visbuffer"}};
    debug_image = daxa::TaskImage{{.name = "debug_image"}};
    color_image = daxa::TaskImage{{.name = "color_image"}};
    transmittance = daxa::TaskImage{{.name = "transmittance"}};
    multiscattering = daxa::TaskImage{{.name = "multiscattering"}};
    sky_ibl_cube = daxa::TaskImage{{.name = "sky ibl cube"}};

    images = {
        debug_image,
        visbuffer,
        depth,
        color_image,
        transmittance,
        multiscattering,
        sky_ibl_cube,
    };

    frame_buffer_images = {
        {
            {
                .format = daxa::Format::D32_SFLOAT,
                .usage = daxa::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = depth.info().name,
            },
            depth,
        },
        {
            {
                .format = daxa::Format::R32_UINT,
                .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT | daxa::ImageUsageFlagBits::TRANSFER_SRC |
                         daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = visbuffer.info().name,
            },
            visbuffer,
        },
        {
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT | daxa::ImageUsageFlagBits::TRANSFER_DST |
                         daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::SHADER_STORAGE |
                         daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = debug_image.info().name,
            },
            debug_image,
        },
        {
            {
                .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
                .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT | daxa::ImageUsageFlagBits::TRANSFER_DST |
                         daxa::ImageUsageFlagBits::TRANSFER_SRC |
                         daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = color_image.info().name,
            },
            color_image,
        },
    };

    recreate_framebuffer();
    recreate_sky_luts();
    update_settings();
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
    this->context->device.wait_idle();
    this->context->device.collect_garbage();
}

void Renderer::compile_pipelines()
{
    std::vector<std::tuple<std::string, daxa::RasterPipelineCompileInfo>> rasters = {
        {draw_visbuffer_no_mesh_shader_pipeline_opaque_compile_info().name, draw_visbuffer_no_mesh_shader_pipeline_opaque_compile_info()},
        {draw_visbuffer_no_mesh_shader_pipeline_discard_compile_info().name, draw_visbuffer_no_mesh_shader_pipeline_discard_compile_info()},
        {draw_shader_debug_circles_pipeline_compile_info().name, draw_shader_debug_circles_pipeline_compile_info()},
        {draw_shader_debug_rectangles_pipeline_compile_info().name, draw_shader_debug_rectangles_pipeline_compile_info()},
        {draw_shader_debug_aabb_pipeline_compile_info().name, draw_shader_debug_aabb_pipeline_compile_info()},
#if COMPILE_IN_MESH_SHADER
        {draw_visbuffer_mesh_shader_cull_and_draw_pipeline_compile_info().name, draw_visbuffer_mesh_shader_cull_and_draw_pipeline_compile_info()},
        {draw_visbuffer_mesh_shader_pipeline_compile_info().name, draw_visbuffer_mesh_shader_pipeline_compile_info()},
#endif
    };
    for (auto [name, info] : rasters)
    {
        auto compilation_result = this->context->pipeline_manager.add_raster_pipeline(info);
        if (compilation_result.value()->is_valid())
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] SUCCESFULLY compiled pipeline {}", name));
        }
        else
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] FAILED to compile pipeline {} with message \n {}", name,
                compilation_result.message()));
        }
        this->context->raster_pipelines[name] = compilation_result.value();
    }
    std::vector<std::tuple<std::string_view, daxa::ComputePipelineCompileInfo>> computes = {
        {AllocEntToMeshInstOffsetsOffsets{}.name(), alloc_entity_to_mesh_instances_offsets_pipeline_compile_info()},
        {WriteFirstPassMeshletBitfieldsTask{}.name(), set_entity_meshlets_visibility_bitmasks_pipeline_compile_info()},
        {PrepopulateMeshletInstancesCommandWriteTask{}.name(), prepopulate_meshlet_instances_command_write_pipeline_compile_info()},
        {PrepopulateMeshletInstancesTask{}.name(), prepopulate_meshlet_instances_pipeline_compile_info()},
        {IndirectMemsetBuffer{}.name(), indirect_memset_buffer_pipeline_compile_info()},
        {AnalyzeVisBufferTask2{}.name(), analyze_visbufer_pipeline_compile_info()},
        {GenHizTH{}.name(), gen_hiz_pipeline_compile_info()},
        {WriteSwapchainTask{}.name(), write_swapchain_pipeline_compile_info()},
        {ShadeOpaqueTask{}.name(), shade_opaque_pipeline_compile_info()},
        {DrawVisbuffer_WriteCommandTask{}.name(), draw_visbuffer_write_command_pipeline_compile_info()},
        {CullMeshesTask{}.name(), cull_meshes_pipeline_compile_info()},
        {PrefixSumCommandWriteTask{}.name(), prefix_sum_write_command_pipeline_compile_info()},
        {PrefixSumUpsweepTask{}.name(), prefix_sum_upsweep_pipeline_compile_info()},
        {PrefixSumDownsweepTask{}.name(), prefix_sum_downsweep_pipeline_compile_info()},
        {CullMeshletsTask{}.name(), cull_meshlets_pipeline_compile_info()},
        {ComputeTransmittance{}.name(), compute_transmittance_pipeline_compile_info()},
        {ComputeMultiscattering{}.name(), compute_multiscattering_pipeline_compile_info()},
        {ComputeSky{}.name(), compute_sky_pipeline_compile_info()},
        {SkyIntoCubemap{}.name(), sky_into_cubemap_pipeline_compile_info()},
        {GenLuminanceHistogram{}.name(), gen_luminace_histogram_pipeline_compile_info()},
        {GenLuminanceAverage{}.name(), gen_luminace_average_pipeline_compile_info()},
    };
    for (auto [name, info] : computes)
    {
        auto compilation_result = this->context->pipeline_manager.add_compute_pipeline(info);
        if (compilation_result.value()->is_valid())
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] SUCCESFULLY compiled pipeline {}", name));
        }
        else
        {
            DEBUG_MSG(fmt::format("[Renderer::compile_pipelines()] FAILED to compile pipeline {} with message \n {}", name,
                compilation_result.message()));
        }
        this->context->compute_pipelines[name] = compilation_result.value();
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
                .size = {context->sky_settings.transmittance_dimensions.x, context->sky_settings.transmittance_dimensions.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
                .name = "transmittance look up table",
            }),
        },
    });

    multiscattering.set_images({
        .images = std::array{
            context->device.create_image({
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {context->sky_settings.multiscattering_dimensions.x, context->sky_settings.multiscattering_dimensions.y, 1},
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
    task_graph.use_persistent_buffer(context->tshader_globals_buffer);
    task_graph.use_persistent_image(transmittance);
    task_graph.use_persistent_image(multiscattering);

    task_graph.add_task({
        .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, context->tshader_globals_buffer)},
        .task = [&](daxa::TaskInterface ti)
        {
            {
                auto const alloc = ti.allocator->allocate_fill(context->shader_globals.sky_settings).value();
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = ti.allocator->buffer(),
                    .dst_buffer = ti.get(context->tshader_globals_buffer).ids[0],
                    .src_offset = alloc.buffer_offset,
                    .dst_offset = offsetof(ShaderGlobals, sky_settings),
                    .size = alloc.size,
                });
            }
            {
                auto const alloc = ti.allocator->allocate_fill(context->shader_globals.sky_settings_ptr).value();
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = ti.allocator->buffer(),
                    .dst_buffer = ti.get(context->tshader_globals_buffer).ids[0],
                    .src_offset = alloc.buffer_offset,
                    .dst_offset = offsetof(ShaderGlobals, sky_settings_ptr),
                    .size = alloc.size,
                });
            }
        },
        .name = "update sky settings globals",
    });

    task_graph.add_task(ComputeTransmittanceTask{
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ComputeTransmittanceTask::globals, context->tshader_globals_buffer}},
            daxa::TaskViewVariant{std::pair{ComputeTransmittanceTask::transmittance, transmittance}},
        },
        .context = context,
    });

    task_graph.add_task(ComputeMultiscatteringTask{
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ComputeMultiscatteringTask::globals, context->tshader_globals_buffer}},
            daxa::TaskViewVariant{std::pair{ComputeMultiscatteringTask::transmittance, transmittance}},
            daxa::TaskViewVariant{std::pair{ComputeMultiscatteringTask::multiscattering, multiscattering}},
        },
        .context = context,
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
        .name = "Sandbox main TaskGraph",
    }};
    for (auto const & tbuffer : buffers)
    {
        task_list.use_persistent_buffer(tbuffer);
    }
    task_list.use_persistent_buffer(scene->_gpu_entity_meta);
    task_list.use_persistent_buffer(scene->_gpu_entity_transforms);
    task_list.use_persistent_buffer(scene->_gpu_entity_combined_transforms);
    task_list.use_persistent_buffer(scene->_gpu_entity_parents);
    task_list.use_persistent_buffer(scene->_gpu_entity_mesh_groups);
    task_list.use_persistent_buffer(scene->_gpu_mesh_manifest);
    task_list.use_persistent_buffer(scene->_gpu_mesh_group_manifest);
    task_list.use_persistent_buffer(scene->_gpu_material_manifest);
    task_list.use_persistent_buffer(context->tshader_globals_buffer);
    task_list.use_persistent_buffer(scene_context.opaque_draw_list_buffer);
    auto detector_image = context->shader_debug_context.tdetector_image;
    task_list.use_persistent_image(detector_image);
    for (auto const & timage : images)
    {
        task_list.use_persistent_image(timage);
    }
    task_list.use_persistent_image(swapchain_image);

    task_clear_image(task_list, detector_image, std::array{0.0f, 0.0f, 0.0f, 1.0f});

    task_list.add_task(ShaderDebugDrawContext::ReadbackTask{
        .views = std::array{daxa::attachment_view(ShaderDebugDrawContext::ReadbackTask::globals, context->tshader_globals_buffer)},
        .shader_debug_context = &context->shader_debug_context,
    });
    task_list.add_task({
        .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, context->tshader_globals_buffer)},
        .task = [&](daxa::TaskInterface ti)
        {
            auto const alloc = ti.allocator->allocate_fill(context->shader_globals).value();
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = ti.allocator->buffer(),
                .dst_buffer = ti.get(context->tshader_globals_buffer).ids[0],
                .src_offset = alloc.buffer_offset,
                .dst_offset = 0,
                .size = alloc.size,
            });
            context->shader_debug_context.update_debug_buffer(ti.device, ti.recorder, *ti.allocator);
        },
        .name = "update global buffers",
    });

    auto sky = task_list.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {context->sky_settings.sky_dimensions.x, context->sky_settings.sky_dimensions.y, 1},
        .name = "sky look up table",
    });
    auto luminance_histogram = task_list.create_transient_buffer({sizeof(u32) * (LUM_HISTOGRAM_BIN_COUNT), "luminance_histogram"});

    task_list.add_task(ComputeSkyTask{
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ComputeSkyTask::globals, context->tshader_globals_buffer}},
            daxa::TaskViewVariant{std::pair{ComputeSkyTask::transmittance, transmittance}},
            daxa::TaskViewVariant{std::pair{ComputeSkyTask::multiscattering, multiscattering}},
            daxa::TaskViewVariant{std::pair{ComputeSkyTask::sky, sky}},
        },
        .context = context,
    });
    task_list.add_task(SkyIntoCubemapTask{
        .views = std::array{
            daxa::TaskViewVariant{std::pair{SkyIntoCubemap::globals, context->tshader_globals_buffer}},
            daxa::TaskViewVariant{std::pair{SkyIntoCubemap::transmittance, transmittance}},
            daxa::TaskViewVariant{std::pair{SkyIntoCubemap::sky, sky}},
            daxa::TaskViewVariant{std::pair{SkyIntoCubemap::ibl_cube, sky_ibl_cube.view().view({.layer_count = 6})}},
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
    task_prepopulate_meshlet_instances(context, &scene_context, task_list,
        PrepopInfo{
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
    task_clear_buffer(task_list, visible_meshlet_instances, 0);
    task_draw_visbuffer({
        .context = context,
        .tg = task_list,
        .enable_mesh_shader = context->settings.enable_mesh_shader != 0,
        .pass = PASS0_DRAW_VISIBLE_LAST_FRAME,
        .meshlet_instances = meshlet_instances,
        .meshes = scene->_gpu_mesh_manifest,
        .material_manifest = scene->_gpu_material_manifest,
        .combined_transforms = scene->_gpu_entity_combined_transforms,
        .visible_meshlet_instances = visible_meshlet_instances,
        .vis_image = visbuffer,
        .debug_image = debug_image,
        .depth_image = depth,
    });
    auto hiz = task_gen_hiz_single_pass(context, task_list, depth, context->tshader_globals_buffer);
    auto [meshlets_cull_arg_buckets_buffer_opaque, meshlets_cull_arg_buckets_buffer_discard] = tasks_cull_meshes(
        TaskCullMeshesInfo{
            .context = context,
            .scene_context = &scene_context,
            .task_list = task_list,
            .globals = context->tshader_globals_buffer,
            .opaque_draw_lists = scene_context.opaque_draw_list_buffer,
            .meshes = scene->_gpu_mesh_manifest,
            .materials = scene->_gpu_material_manifest,
            .entity_meta = scene->_gpu_entity_meta,
            .entity_meshgroup_indices = scene->_gpu_entity_mesh_groups,
            .meshgroups = scene->_gpu_mesh_group_manifest,
            .entity_transforms = scene->_gpu_entity_transforms,
            .entity_combined_transforms = scene->_gpu_entity_combined_transforms,
            .hiz = hiz,
        });
    task_cull_and_draw_visbuffer({
        .context = context,
        .tg = task_list,
        .enable_mesh_shader = context->settings.enable_mesh_shader != 0,
        .meshlets_cull_arg_buckets_buffer_opaque = meshlets_cull_arg_buckets_buffer_opaque,
        .meshlets_cull_arg_buckets_buffer_discard = meshlets_cull_arg_buckets_buffer_discard,
        .entity_meta_data = scene->_gpu_entity_meta,
        .entity_meshgroups = scene->_gpu_entity_mesh_groups,
        .entity_combined_transforms = scene->_gpu_entity_combined_transforms,
        .mesh_groups = scene->_gpu_mesh_group_manifest,
        .meshes = scene->_gpu_mesh_manifest,
        .material_manifest = scene->_gpu_material_manifest,
        .first_pass_meshlets_bitfield_offsets = first_pass_meshlets_bitfield_offsets,
        .first_pass_meshlets_bitfield_arena = first_pass_meshlets_bitfield_arena,
        .hiz = hiz,
        .visible_meshlet_instances = visible_meshlet_instances,
        .meshlet_instances = meshlet_instances,
        .vis_image = visbuffer,
        .debug_image = debug_image,
        .depth_image = depth,
    });
    auto visible_meshlets_bitfield =
        task_list.create_transient_buffer({sizeof(daxa_u32) * MAX_MESHLET_INSTANCES, "visible meshlets bitfield"});
    task_clear_buffer(task_list, visible_meshlets_bitfield, 0);

    task_list.add_task(AnalyzeVisBufferTask2{
        .views = std::array{
            daxa::TaskViewVariant{std::pair{AnalyzeVisBufferTask2::globals, context->tshader_globals_buffer}},
            daxa::TaskViewVariant{std::pair{AnalyzeVisBufferTask2::visbuffer, visbuffer}},
            daxa::TaskViewVariant{std::pair{AnalyzeVisBufferTask2::instantiated_meshlets, meshlet_instances}},
            daxa::TaskViewVariant{std::pair{AnalyzeVisBufferTask2::meshlet_visibility_bitfield, visible_meshlets_bitfield}},
            daxa::TaskViewVariant{std::pair{AnalyzeVisBufferTask2::visible_meshlets, visible_meshlet_instances}},
        },
        .context = context,
    });

    if (context->settings.draw_from_observer)
    {
        task_draw_visbuffer({
            .context = context,
            .tg = task_list,
            .enable_mesh_shader = context->settings.enable_mesh_shader != 0,
            .pass = PASS4_OBSERVER_DRAW_ALL,
            .meshlet_instances = meshlet_instances,
            .meshes = scene->_gpu_mesh_manifest,
            .material_manifest = scene->_gpu_material_manifest,
            .combined_transforms = scene->_gpu_entity_combined_transforms,
            .visible_meshlet_instances = visible_meshlet_instances,
            .vis_image = visbuffer,
            .debug_image = debug_image,
            .depth_image = depth,
        });
    }
    task_list.submit({});
    task_list.add_task(ShadeOpaqueTask{
        .views = std::array{
            daxa::attachment_view(ShadeOpaqueTask::detector_image, detector_image),
            daxa::attachment_view(ShadeOpaqueTask::globals, context->tshader_globals_buffer),
            daxa::attachment_view(ShadeOpaqueTask::color_image, color_image),
            daxa::attachment_view(ShadeOpaqueTask::vis_image, visbuffer),
            daxa::attachment_view(ShadeOpaqueTask::transmittance, transmittance),
            daxa::attachment_view(ShadeOpaqueTask::sky, sky),
            daxa::attachment_view(ShadeOpaqueTask::sky_ibl, sky_ibl_cube),
            daxa::attachment_view(ShadeOpaqueTask::material_manifest, scene->_gpu_material_manifest),
            daxa::attachment_view(ShadeOpaqueTask::instantiated_meshlets, meshlet_instances),
            daxa::attachment_view(ShadeOpaqueTask::meshes, scene->_gpu_mesh_manifest),
            daxa::attachment_view(ShadeOpaqueTask::combined_transforms, scene->_gpu_entity_combined_transforms),
            daxa::attachment_view(ShadeOpaqueTask::luminance_average, luminance_average),
        },
        .context = context,
    });

    task_clear_buffer(task_list, luminance_histogram, 0);
    task_list.add_task(GenLuminanceHistogramTask{
        .views = std::array{
            daxa::attachment_view(GenLuminanceHistogramTask::globals, context->tshader_globals_buffer),
            daxa::attachment_view(GenLuminanceHistogramTask::histogram, luminance_histogram),
            daxa::attachment_view(GenLuminanceHistogramTask::luminance_average, luminance_average),
            daxa::attachment_view(GenLuminanceHistogramTask::color_image, color_image),
        },
        .context = context,
    });
    task_list.add_task(GenLuminanceAverageTask{
        .views = std::array{
            daxa::attachment_view(GenLuminanceAverageTask::globals, context->tshader_globals_buffer),
            daxa::attachment_view(GenLuminanceAverageTask::histogram, luminance_histogram),
            daxa::attachment_view(GenLuminanceAverageTask::luminance_average, luminance_average),
        },
        .context = context,
    });
    task_list.add_task(WriteSwapchainTask{
        .views = std::array{
            daxa::attachment_view(WriteSwapchainTask::globals, context->tshader_globals_buffer),
            daxa::attachment_view(WriteSwapchainTask::swapchain, swapchain_image),
            daxa::attachment_view(WriteSwapchainTask::color_image, color_image),
        },
        .context = context,
    });

    task_list.add_task(DebugDrawTask{
        .views = std::array{
            daxa::attachment_view(DebugDrawTask::globals, context->tshader_globals_buffer),
            daxa::attachment_view(DebugDrawTask::color_image, swapchain_image),
            daxa::attachment_view(DebugDrawTask::depth_image, depth),
        },
        .context = context,
    });
    task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskImageAccess::COLOR_ATTACHMENT, swapchain_image),
            daxa::inl_attachment(daxa::TaskImageAccess::FRAGMENT_SHADER_SAMPLED, detector_image),
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

void Renderer::update_settings()
{
    context->settings.render_target_size.x = window->size.x;
    context->settings.render_target_size.y = window->size.y;
    context->settings.render_target_size_inv = {
        1.0f / context->settings.render_target_size.x, 1.0f / context->settings.render_target_size.y};
}

void Renderer::render_frame(
    CameraInfo const & camera_info,
    CameraInfo const & observer_camera_info,
    f32 const delta_time,
    SceneRendererContext scene_context)
{
    this->scene_context = scene_context;
    if (window->size.x == 0 || window->size.y == 0) { return; }

    auto reloaded_result = context->pipeline_manager.reload_all();
    if (auto reload_err = daxa::get_if<daxa::PipelineReloadError>(&reloaded_result))
    {
        std::cout << "Failed to reload " << reload_err->message << '\n';
    }
    if (auto _ = daxa::get_if<daxa::PipelineReloadSuccess>(&reloaded_result))
    {
        std::cout << "Successfully reloaded!\n";
    }
    u32 const flight_frame_index = context->swapchain.current_cpu_timeline_value() % (context->swapchain.info().max_allowed_frames_in_flight + 1);
    daxa_u32vec2 render_target_size = {static_cast<daxa_u32>(window->size.x), static_cast<daxa_u32>(window->size.y)};

    update_settings();
    bool const settings_changed = context->settings != context->prev_settings;
    bool const sky_settings_changed = context->sky_settings != context->prev_sky_settings;
    auto const sky_res_changed_flags = context->sky_settings.resolutions_changed(context->prev_sky_settings);
    // Sky is transient of main task graph
    if (settings_changed || sky_res_changed_flags.sky_changed) { main_task_graph = create_main_task_graph(); }
    if (sky_settings_changed)
    {
        // Potentially wastefull, ideally we want to only recreate the resource that changed the name
        if (sky_res_changed_flags.multiscattering_changed || sky_res_changed_flags.transmittance_changed)
        {
            recreate_sky_luts();
        }
        // Whenever the settings change we need to recalculate the transmittance and multiscattering look up textures
        auto const sky_settings_offset = offsetof(ShaderGlobals, sky_settings);
        context->shader_globals.sky_settings_ptr = context->shader_globals_address + sky_settings_offset;

        auto const mie_density_offset = sky_settings_offset + offsetof(SkySettings, mie_density);
        context->sky_settings.mie_density_ptr = context->shader_globals_address + mie_density_offset;
        auto const rayleigh_density_offset = sky_settings_offset + offsetof(SkySettings, rayleigh_density);
        context->sky_settings.rayleigh_density_ptr = context->shader_globals_address + rayleigh_density_offset;
        auto const absoprtion_density_offset = sky_settings_offset + offsetof(SkySettings, absorption_density);
        context->sky_settings.absorption_density_ptr = context->shader_globals_address + absoprtion_density_offset;

        context->shader_globals.sky_settings = context->sky_settings;
        sky_task_graph.execute({});
    }
    context->prev_settings = context->settings;
    context->prev_sky_settings = context->sky_settings;

    // Set Shader Globals.
    context->shader_globals.camera = camera_info;
    context->shader_globals.observer_camera = observer_camera_info;
    context->shader_globals.settings = context->settings;
    context->shader_globals.frame_index = static_cast<u32>(context->swapchain.current_cpu_timeline_value());
    context->shader_globals.delta_time = delta_time;

    auto new_swapchain_image = context->swapchain.acquire_next_image();
    if (new_swapchain_image.is_empty()) { return; }
    swapchain_image.set_images({.images = std::array{new_swapchain_image}});
    meshlet_instances.swap_buffers(meshlet_instances_last_frame);

    if (static_cast<daxa_u32>(context->swapchain.current_cpu_timeline_value()) == 0) { clear_select_buffers(); }

    context->shader_debug_context.cpu_debug_aabb_draws.push_back(ShaderDebugAABBDraw{
        .position = daxa_f32vec3(0, 0, 0.5),
        .size = daxa_f32vec3(2.01, 2.01, 0.999),
        .color = daxa_f32vec3(1, 0, 0),
        .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC,
    });

    submit_info = {};
    auto const t_semas = std::array{std::pair{context->transient_mem.timeline_semaphore(), context->transient_mem.timeline_value()}};
    submit_info.signal_timeline_semaphores = t_semas;
    context->shader_debug_context.update(context->device, window->size.x, window->size.y);
    main_task_graph.execute({});
    context->device.collect_garbage();
}