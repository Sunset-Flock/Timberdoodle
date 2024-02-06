#include "renderer.hpp"

#include "../shader_shared/scene.inl"
#include "../shader_shared/debug.inl"

#include "rasterize_visbuffer/draw_visbuffer.inl"
#include "rasterize_visbuffer/cull_meshes.inl"
#include "rasterize_visbuffer/cull_meshlets.inl"
#include "rasterize_visbuffer/analyze_visbuffer.inl"
#include "rasterize_visbuffer/gen_hiz.inl"
#include "rasterize_visbuffer/prepopulate_inst_meshlets.inl"

#include "tasks/prefix_sum.inl"
#include "tasks/write_swapchain.inl"
#include "tasks/shader_debug_draws.inl"
#include <daxa/types.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <thread>
#include <variant>

inline auto create_task_buffer(GPUContext * context, auto size, auto task_buf_name, auto buf_name)
{
    return daxa::TaskBuffer{{
        .initial_buffers =
            {
                .buffers =
                    std::array{
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
    meshlet_instances = create_task_buffer(context, sizeof(MeshletInstances), "meshlet_instances", "meshlet_instances_a");
    meshlet_instances_last_frame = create_task_buffer(context, sizeof(MeshletInstances), "meshlet_instances_last_frame", "meshlet_instances_b");
    visible_meshlet_instances = create_task_buffer(context, sizeof(VisibleMeshletList), "visible_meshlet_instances", "visible_meshlet_instances");

    buffers = {zero_buffer, meshlet_instances, meshlet_instances_last_frame, visible_meshlet_instances};

    swapchain_image = daxa::TaskImage{{
        .swapchain_image = true,
        .name = "swapchain_image",
    }};
    depth = daxa::TaskImage{{
        .name = "depth",
    }};
    visbuffer = daxa::TaskImage{{
        .name = "visbuffer",
    }};
    debug_image = daxa::TaskImage{{
        .name = "debug_image",
    }};

    images = {
        debug_image,
        visbuffer,
        depth,
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
    };

    recreate_framebuffer();

    context->settings.enable_mesh_shader = 0;
    context->settings.draw_from_observer = 0;
    update_settings();
    main_task_graph = create_main_task_graph();
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
        {draw_visbuffer_no_mesh_shader_pipeline_compile_info().name, draw_visbuffer_no_mesh_shader_pipeline_compile_info()},
        {draw_shader_debug_circles_pipeline_compile_info().name, draw_shader_debug_circles_pipeline_compile_info()},
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
        {SetEntityMeshletVisibilityBitMasksTask{}.name(), set_entity_meshlets_visibility_bitmasks_pipeline_compile_info()},
        {PrepopulateInstantiatedMeshletsTask{}.name(), prepopulate_inst_meshlets_pipeline_compile_info()},
        {PrepopulateInstantiatedMeshletsCommandWriteTask{}.name(), prepopulate_instantiated_meshlets_command_write_pipeline_compile_info()},
        {AnalyzeVisBufferTask2{}.name(), analyze_visbufer_pipeline_compile_info()},
        {GenHizTH{}.name(), gen_hiz_pipeline_compile_info()},
        {WriteSwapchainTask{}.name(), write_swapchain_pipeline_compile_info()},
        {DrawVisbuffer_WriteCommandTask{}.name(), draw_visbuffer_write_command_pipeline_compile_info()},
        {CullMeshesCommandWriteTask{}.name(), cull_meshes_write_command_pipeline_compile_info()},
        {CullMeshesTask{}.name(), cull_meshes_pipeline_compile_info()},
        {PrefixSumCommandWriteTask{}.name(), prefix_sum_write_command_pipeline_compile_info()},
        {PrefixSumUpsweepTask{}.name(), prefix_sum_upsweep_pipeline_compile_info()},
        {PrefixSumDownsweepTask{}.name(), prefix_sum_downsweep_pipeline_compile_info()},
        {CullMeshletsTask{}.name(), cull_meshlets_pipeline_compile_info()},
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
    task_clear_buffer(list, meshlet_instances, 0, sizeof(u32));
    list.use_persistent_buffer(visible_meshlet_instances);
    task_clear_buffer(list, visible_meshlet_instances, 0, sizeof(u32));
    list.submit({});
    list.complete({});
    list.execute({});
}

void Renderer::window_resized()
{
    if (this->window->size.x == 0 || this->window->size.y == 0) { return; }
    this->context->swapchain.resize();
    recreate_framebuffer();
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
    task_list.use_persistent_buffer(context->shader_globals_task_buffer);
    for (auto const & timage : images)
    {
        task_list.use_persistent_image(timage);
    }
    task_list.use_persistent_image(swapchain_image);

    task_list.add_task({
        .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, context->shader_globals_task_buffer)},
        .task = [=](daxa::TaskInterface ti)
        {
            auto const alloc = ti.allocator->allocate_fill(context->shader_globals).value();
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = ti.allocator->buffer(),
                .dst_buffer = ti.get(context->shader_globals_task_buffer).ids[0],
                .src_offset = alloc.buffer_offset,
                .dst_offset = 0,
                .size = alloc.size,
            });
            context->debug_draw_info.update_debug_buffer(ti.device, ti.recorder, *ti.allocator);
        },
        .name = "update buffers",
    });

    auto entity_meshlet_visibility_bitfield_offsets = task_list.create_transient_buffer(
        {sizeof(EntityMeshletVisibilityBitfieldOffsets) * MAX_ENTITY_COUNT + sizeof(u32), "meshlet_visibility_bitfield_offsets"});
    auto entity_meshlet_visibility_bitfield_arena =
        task_list.create_transient_buffer({ENTITY_MESHLET_VISIBILITY_ARENA_SIZE, "meshlet_visibility_bitfield_arena"});
    task_prepopulate_instantiated_meshlets(context, task_list,
        PrepopInfo{
            .meshes = scene->_gpu_mesh_manifest,
            .visible_meshlets_prev = visible_meshlet_instances,
            .meshlet_instances_last_frame = meshlet_instances_last_frame,
            .meshlet_instances = meshlet_instances,
            .entity_meshlet_visibility_bitfield_offsets = entity_meshlet_visibility_bitfield_offsets,
            .entity_meshlet_visibility_bitfield_arena = entity_meshlet_visibility_bitfield_arena,
        });

    task_draw_visbuffer({
        .context = context,
        .tg = task_list,
        .enable_mesh_shader = context->settings.enable_mesh_shader != 0,
        .pass = DRAW_VISBUFFER_PASS_ONE,
        .meshlet_instances = meshlet_instances,
        .meshes = scene->_gpu_mesh_manifest,
        .combined_transforms = scene->_gpu_entity_combined_transforms,
        .vis_image = visbuffer,
        .debug_image = debug_image,
        .depth_image = depth,
    });
    auto hiz = task_gen_hiz_single_pass(context, task_list, depth, context->shader_globals_task_buffer);
    auto meshlet_cull_indirect_args = task_list.create_transient_buffer({
        .size = sizeof(MeshletCullIndirectArgTable) + sizeof(MeshletCullIndirectArg) * MAX_MESHLET_INSTANCES * 2,
        .name = "meshlet_cull_indirect_args",
    });
    auto cull_meshlets_commands = task_list.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct) * 32,
        .name = "CullMeshletsCommands",
    });
    CullMeshesTask cull_meshes_task = {
        .views = std::array{
            daxa::attachment_view(CullMeshesTask::globals, context->shader_globals_task_buffer),
            daxa::attachment_view(CullMeshesTask::command, daxa::TaskBufferView{}),
            daxa::attachment_view(CullMeshesTask::meshes, scene->_gpu_mesh_manifest),
            daxa::attachment_view(CullMeshesTask::entity_meta, scene->_gpu_entity_meta),
            daxa::attachment_view(CullMeshesTask::entity_meshgroup_indices, scene->_gpu_entity_mesh_groups),
            daxa::attachment_view(CullMeshesTask::meshgroups, scene->_gpu_mesh_group_manifest),
            daxa::attachment_view(CullMeshesTask::entity_transforms, scene->_gpu_entity_transforms),
            daxa::attachment_view(CullMeshesTask::entity_combined_transforms, scene->_gpu_entity_combined_transforms),
            daxa::attachment_view(CullMeshesTask::hiz, hiz),
            daxa::attachment_view(CullMeshesTask::meshlet_cull_indirect_args, meshlet_cull_indirect_args),
            daxa::attachment_view(CullMeshesTask::cull_meshlets_commands, cull_meshlets_commands),
        },
        .context = context,
    };
    tasks_cull_meshes(context, task_list, cull_meshes_task);
    task_cull_and_draw_visbuffer({
        .context = context,
        .tg = task_list,
        .enable_mesh_shader = context->settings.enable_mesh_shader != 0,
        .cull_meshlets_commands = cull_meshlets_commands,
        .meshlet_cull_indirect_args = meshlet_cull_indirect_args,
        .entity_meta_data = scene->_gpu_entity_meta,
        .entity_meshgroups = scene->_gpu_entity_mesh_groups,
        .entity_combined_transforms = scene->_gpu_entity_combined_transforms,
        .mesh_groups = scene->_gpu_mesh_group_manifest,
        .meshes = scene->_gpu_mesh_manifest,
        .entity_meshlet_visibility_bitfield_offsets = entity_meshlet_visibility_bitfield_offsets,
        .entity_meshlet_visibility_bitfield_arena = entity_meshlet_visibility_bitfield_arena,
        .hiz = hiz,
        .meshlet_instances = meshlet_instances,
        .vis_image = visbuffer,
        .debug_image = debug_image,
        .depth_image = depth,
    });
    auto visible_meshlets_bitfield =
        task_list.create_transient_buffer({sizeof(daxa_u32) * MAX_MESHLET_INSTANCES, "visible meshlets bitfield"});
    task_clear_buffer(task_list, visible_meshlet_instances, 0, 4);
    task_clear_buffer(task_list, visible_meshlets_bitfield, 0);
    task_clear_buffer(task_list, entity_meshlet_visibility_bitfield_arena, 0);

    task_list.add_task(AnalyzeVisBufferTask2{
        .views = std::array{
            daxa::TaskViewVariant{std::pair{AnalyzeVisBufferTask2::globals, context->shader_globals_task_buffer}},
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
            .pass = DRAW_VISBUFFER_PASS_OBSERVER,
            .meshlet_instances = meshlet_instances,
            .meshes = scene->_gpu_mesh_manifest,
            .combined_transforms = scene->_gpu_entity_combined_transforms,
            .vis_image = visbuffer,
            .debug_image = debug_image,
            .depth_image = depth,
        });
    }
    task_list.submit({});
    task_list.add_task(WriteSwapchainTask{
        .views = std::array{
            daxa::TaskViewVariant{std::pair{WriteSwapchainTask::globals, context->shader_globals_task_buffer}},
            daxa::TaskViewVariant{std::pair{WriteSwapchainTask::swapchain, swapchain_image}},
            daxa::TaskViewVariant{std::pair{WriteSwapchainTask::vis_image, visbuffer}},
            daxa::TaskViewVariant{std::pair{WriteSwapchainTask::debug_image, debug_image}},
            daxa::TaskViewVariant{std::pair{WriteSwapchainTask::material_manifest, scene->_gpu_material_manifest}},
            daxa::TaskViewVariant{std::pair{WriteSwapchainTask::instantiated_meshlets, meshlet_instances}},
        },
        .context = context,
    });

    task_list.add_task({
        .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COLOR_ATTACHMENT, swapchain_image)},
        .task =
            [=, this](daxa::TaskInterface ti)
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

void Renderer::render_frame(CameraInfo const & camera_info, CameraInfo const & observer_camera_info, f32 const delta_time)
{
    if (this->window->size.x == 0 || this->window->size.y == 0) { return; }
    auto reloaded_result = context->pipeline_manager.reload_all();
    if (auto reload_err = daxa::get_if<daxa::PipelineReloadError>(&reloaded_result))
    {
        std::cout << "Failed to reload " << reload_err->message << '\n';
    }
    if (auto _ = daxa::get_if<daxa::PipelineReloadSuccess>(&reloaded_result)) { std::cout << "Successfully reloaded!\n"; }
    u32 const flight_frame_index =
        context->swapchain.current_cpu_timeline_value() % (context->swapchain.info().max_allowed_frames_in_flight + 1);
    daxa_u32vec2 render_target_size = {static_cast<daxa_u32>(this->window->size.x), static_cast<daxa_u32>(this->window->size.y)};
    this->update_settings();
    this->context->shader_globals.settings = context->settings;
    bool const settings_changed = context->settings != context->prev_settings;
    if (settings_changed) { this->main_task_graph = create_main_task_graph(); }
    this->context->prev_settings = this->context->settings;

    // Set Shader Globals.
    this->context->shader_globals.camera = camera_info;
    this->context->shader_globals.observer_camera = observer_camera_info;
    this->context->shader_globals.settings = this->context->settings;
    this->context->shader_globals.frame_index = static_cast<u32>(context->swapchain.current_cpu_timeline_value());
    this->context->shader_globals.delta_time = delta_time;

    auto swapchain_image = context->swapchain.acquire_next_image();
    if (swapchain_image.is_empty()) { return; }
    this->swapchain_image.set_images({.images = std::array{swapchain_image}});
    meshlet_instances.swap_buffers(meshlet_instances_last_frame);

    if (static_cast<daxa_u32>(context->swapchain.current_cpu_timeline_value()) == 0) { clear_select_buffers(); }

    this->submit_info = {};
    auto const t_semas =
        std::array{std::pair{this->context->transient_mem.timeline_semaphore(), this->context->transient_mem.timeline_value()}};
    this->submit_info.signal_timeline_semaphores = t_semas;
    main_task_graph.execute({});
    context->prev_settings = context->settings;
    this->context->device.collect_garbage();
}