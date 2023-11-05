#include "renderer.hpp"

#include "../../shader_shared/scene.inl"

#include "rasterize_visbuffer/draw_visbuffer.inl"
#include "rasterize_visbuffer/cull_meshes.inl"
#include "rasterize_visbuffer/cull_meshlets.inl"
#include "rasterize_visbuffer/analyze_visbuffer.inl"
#include "rasterize_visbuffer/gen_hiz.inl"
#include "rasterize_visbuffer/prepopulate_inst_meshlets.inl"

#include "tasks/prefix_sum.inl"
#include "tasks/write_swapchain.inl"

inline auto create_task_buffer(GPUContext *context, auto size, auto task_buf_name, auto buf_name)
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

Renderer::Renderer(Window *window, GPUContext *context, Scene *scene, AssetProcessor *asset_manager)
    : window{window},
      context{context},
      scene{scene},
      asset_manager{asset_manager},
      imgui_renderer{{context->device, context->swapchain.get_format()}}
{
    zero_buffer = create_task_buffer(context, sizeof(u32), "zero_buffer", "zero_buffer");
    meshlet_instances = create_task_buffer(context, sizeof(MeshletInstances), "meshlet_instances", "meshlet_instances_a");
    meshlet_instances_last_frame = create_task_buffer(context, sizeof(MeshletInstances), "meshlet_instances_last_frame", "meshlet_instances_b");
    visible_meshlet_instances = create_task_buffer(context, sizeof(VisibleMeshletList), "visible_meshlet_instances", "visible_meshlet_instances");

    buffers = {
        zero_buffer,
        meshlet_instances,
        meshlet_instances_last_frame,
        visible_meshlet_instances};

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
                .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT |
                         daxa::ImageUsageFlagBits::TRANSFER_SRC |
                         daxa::ImageUsageFlagBits::SHADER_STORAGE |
                         daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = visbuffer.info().name,
            },
            visbuffer,
        },
        {
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT |
                         daxa::ImageUsageFlagBits::TRANSFER_DST |
                         daxa::ImageUsageFlagBits::TRANSFER_SRC |
                         daxa::ImageUsageFlagBits::SHADER_STORAGE |
                         daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = debug_image.info().name,
            },
            debug_image,
        },
    };

    recreate_framebuffer();

    compile_pipelines();

    context->settings.enable_mesh_shader = 0;
    context->settings.enable_observer = 0;
    update_settings();
    main_task_graph = create_main_task_graph();
}

Renderer::~Renderer()
{
    for (auto &tbuffer : buffers)
    {
        for (auto buffer : tbuffer.get_state().buffers)
        {
            this->context->device.destroy_buffer(buffer);
        }
    }
    for (auto &timage : images)
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
    std::vector<std::tuple<std::string_view, daxa::RasterPipelineCompileInfo>> rasters = {
        {DrawVisbufferTask::PIPELINE_COMPILE_INFO.name, DrawVisbufferTask::PIPELINE_COMPILE_INFO},
#if COMPILE_IN_MESH_SHADER
        {DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER_CULL_AND_DRAW.name, DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER_CULL_AND_DRAW},
        {DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER.name, DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER},
#endif
    };
    for (auto [name, info] : rasters)
    {
        auto compilation_result = this->context->pipeline_manager.add_raster_pipeline(info);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->raster_pipelines[name] = compilation_result.value();
    }
    std::vector<std::tuple<std::string_view, daxa::ComputePipelineCompileInfo>> computes = {
        {SetEntityMeshletVisibilityBitMasksTask::NAME, SetEntityMeshletVisibilityBitMasksTask::PIPELINE_COMPILE_INFO},
        {PrepopulateInstantiatedMeshletsTask::NAME, PrepopulateInstantiatedMeshletsTask::PIPELINE_COMPILE_INFO},
        {PrepopulateInstantiatedMeshletsCommandWriteTask::NAME, PrepopulateInstantiatedMeshletsCommandWriteTask::PIPELINE_COMPILE_INFO},
        {AnalyzeVisBufferTask2::NAME, AnalyzeVisBufferTask2::PIPELINE_COMPILE_INFO},
        {GEN_HIZ_PIPELINE_COMPILE_INFO.name, GEN_HIZ_PIPELINE_COMPILE_INFO},
        {WriteSwapchainTask::NAME, WriteSwapchainTask::PIPELINE_COMPILE_INFO},
        {DrawVisbufferWriteCommandTask::NAME, DrawVisbufferWriteCommandTask::PIPELINE_COMPILE_INFO},
        {CullMeshesCommandWriteTask::NAME, CullMeshesCommandWriteTask::PIPELINE_COMPILE_INFO},
        {CullMeshesTask::NAME, CullMeshesTask::PIPELINE_COMPILE_INFO},
        {PrefixSumCommandWriteTask::NAME, PrefixSumCommandWriteTask::PIPELINE_COMPILE_INFO},
        {PrefixSumUpsweepTask::NAME, PrefixSumUpsweepTask::PIPELINE_COMPILE_INFO},
        {PrefixSumDownsweepTask::NAME, PrefixSumDownsweepTask::PIPELINE_COMPILE_INFO},
        {CullMeshletsTask::NAME, CullMeshletsTask::PIPELINE_COMPILE_INFO},
    };
    for (auto [name, info] : computes)
    {
        auto compilation_result = this->context->pipeline_manager.add_compute_pipeline(info);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->compute_pipelines[name] = compilation_result.value();
    }
}

void Renderer::recreate_framebuffer()
{
    for (auto &[info, timg] : frame_buffer_images)
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
    if (this->window->size.x == 0 || this->window->size.y == 0)
    {
        return;
    }
    this->context->swapchain.resize();
    recreate_framebuffer();
}

auto Renderer::create_main_task_graph() -> daxa::TaskGraph
{
    // Rasterize Visbuffer:
    // - reset/clear certain buffers
    // - prepopulate meshlet instances, these meshlet instances are drawn in the first pass.
    //     - uses list of visible meshlets of last frame (visible_meshlet_instances) and meshlet instance list from last frame (meshlet_instances_last_frame)
    //     - filteres meshlets when their entities/ meshes got invalidated.
    //     - builds bitfields (entity_meshlet_visibility_bitfield_offsets), that denote if a meshlet of an entity is drawn in the first pass.
    // - draw first pass
    //     - draws meshlet instances, generated by prepopulate_instantiated_meshlets.
    //     - draws trianlge id and depth. triangle id indexes into the meshlet instance list (that is freshly generated every frame), also stores triangle index within meshlet.
    //     - effectively draws the meshlets that were visible last frame as the first thing.
    // - build hiz depth map
    //     - lowest mip is half res of render target resolution, depth map at full res is not copied into the hiz.
    //     - single pass downsample dispatch. Each workgroup downsamples a 64x64 region, the very last workgroup to finish downsamples all the results of the previous workgroups.
    // - cull meshes
    //     - dispatch over all entities for all their meshes
    //     - cull against: hiz, frustum
    //     - builds argument lists for meshlet culling.
    //     - 32 meshlet cull argument lists, each beeing a bucket for arguments. An argument in each bucket represents 2^bucket_index meshlets to be processed.
    // - cull and draw meshlets
    //     - 32 dispatches each going over one of the generated cull argument lists.
    //     - when mesh shaders are enabled, this is a single pipeline. Task shaders cull in this case.
    //     - when mesh shaders are disabled, a compute shader culls.
    //     - in either case, the task/compute cull shader fill the list of meshlet instances. This list is used to compactly reference meshlets via pixel id.
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
    for (auto const &tbuffer : buffers)
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
    for (auto const &timage : images)
    {
        task_list.use_persistent_image(timage);
    }
    task_list.use_persistent_image(swapchain_image);

    auto entity_meshlet_visibility_bitfield_offsets = task_list.create_transient_buffer({sizeof(EntityMeshletVisibilityBitfieldOffsets) * MAX_ENTITY_COUNT + sizeof(u32), "meshlet_visibility_bitfield_offsets"});
    auto entity_meshlet_visibility_bitfield_arena = task_list.create_transient_buffer({ENTITY_MESHLET_VISIBILITY_ARENA_SIZE, "meshlet_visibility_bitfield_arena"});
    task_prepopulate_instantiated_meshlets(
        context,
        task_list,
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
    auto hiz = task_gen_hiz_single_pass(context, task_list, depth);
    auto meshlet_cull_indirect_args = task_list.create_transient_buffer({
        .size = sizeof(MeshletCullIndirectArgTable) + sizeof(MeshletCullIndirectArg) * MAX_MESHLET_INSTANCES * 2,
        .name = "meshlet_cull_indirect_args",
    });
    auto cull_meshlets_commands = task_list.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct) * 32,
        .name = "CullMeshletsCommands",
    });
    tasks_cull_meshes(
        context,
        task_list,
        {
            .u_meshes = scene->_gpu_mesh_manifest,
            .u_entity_meta = scene->_gpu_entity_meta,
            .u_entity_meshgroup_indices = scene->_gpu_entity_mesh_groups,
            .u_meshgroups = scene->_gpu_mesh_group_manifest,
            .u_entity_transforms = scene->_gpu_entity_transforms,
            .u_entity_combined_transforms = scene->_gpu_entity_combined_transforms,
            .u_hiz = hiz,
            .u_meshlet_cull_indirect_args = meshlet_cull_indirect_args,
            .u_cull_meshlets_commands = cull_meshlets_commands,
        });
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
    auto visible_meshlets_bitfield = task_list.create_transient_buffer({sizeof(daxa_u32) * MAX_MESHLET_INSTANCES, "visible meshlets bitfield"});
    task_clear_buffer(task_list, visible_meshlet_instances, 0, 4);
    task_clear_buffer(task_list, visible_meshlets_bitfield, 0);
    task_clear_buffer(task_list, entity_meshlet_visibility_bitfield_arena, 0);
    task_list.add_task(AnalyzeVisBufferTask2{
        .uses = {
            .u_visbuffer = visbuffer,
            .u_instantiated_meshlets = meshlet_instances,
            .u_meshlet_visibility_bitfield = visible_meshlets_bitfield,
            .u_visible_meshlets = visible_meshlet_instances,
        },
        .context = context,
    });
    if (context->settings.enable_observer)
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
        .uses = {
            .swapchain = swapchain_image,
            .vis_image = visbuffer,
            .u_debug_image = debug_image,
            .u_instantiated_meshlets = meshlet_instances,
        },
        .context = context,
    });
    task_list.add_task({
        .uses = {
            ImageColorAttachment<>{swapchain_image},
        },
        .task = [=](daxa::TaskInterface ti)
        {
            auto & cmd_list = ti.get_recorder();
            auto size = ti.get_device().info_image(ti.uses[swapchain_image].image()).value().size;
            imgui_renderer.record_commands(ImGui::GetDrawData(), cmd_list, ti.uses[swapchain_image].image(), size.x, size.y);
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
    context->settings.render_target_size_inv = {1.0f / context->settings.render_target_size.x, 1.0f / context->settings.render_target_size.y};
}

void Renderer::render_frame(CameraInfo const &camera_info, CameraInfo const &observer_camera_info, f32 const delta_time)
{
    if (this->window->size.x == 0 || this->window->size.y == 0)
    {
        return;
    }
    auto reloaded_result = context->pipeline_manager.reload_all();
    if (auto reload_err = daxa::get_if<daxa::PipelineReloadError>(&reloaded_result))
    {
        std::cout << "Failed to reload " << reload_err->message << '\n';
    }
    if (auto _ = daxa::get_if<daxa::PipelineReloadSuccess>(&reloaded_result))
    {
        std::cout << "Successfully reloaded!\n";
    }
    u32 const flight_frame_index = context->swapchain.get_cpu_timeline_value() % (context->swapchain.info().max_allowed_frames_in_flight + 1);
    daxa_u32vec2 render_target_size = {static_cast<daxa_u32>(this->window->size.x), static_cast<daxa_u32>(this->window->size.y)};
    this->update_settings();
    this->context->shader_globals.globals.settings = context->settings;
    bool const settings_changed = context->settings != context->prev_settings;
    if (settings_changed)
    {
        this->main_task_graph = create_main_task_graph();
    }
    this->context->prev_settings = this->context->settings;

    // Set Shader Globals.
    this->context->shader_globals.globals.settings = this->context->settings;
    this->context->shader_globals.globals.frame_index = static_cast<u32>(context->swapchain.get_cpu_timeline_value());
    this->context->shader_globals.globals.delta_time = delta_time;
    this->context->shader_globals.globals.observer_camera_up = *reinterpret_cast<daxa_f32vec3 const *>(&observer_camera_info.up);
    this->context->shader_globals.globals.observer_camera_pos = *reinterpret_cast<daxa_f32vec3 const *>(&observer_camera_info.pos);
    this->context->shader_globals.globals.observer_camera_view = *reinterpret_cast<daxa_f32mat4x4 const *>(&observer_camera_info.view);
    this->context->shader_globals.globals.observer_camera_projection = *reinterpret_cast<daxa_f32mat4x4 const *>(&observer_camera_info.proj);
    this->context->shader_globals.globals.observer_camera_view_projection = *reinterpret_cast<daxa_f32mat4x4 const *>(&observer_camera_info.vp);
    this->context->shader_globals.globals.camera_up = *reinterpret_cast<daxa_f32vec3 const *>(&camera_info.up);
    this->context->shader_globals.globals.camera_pos = *reinterpret_cast<daxa_f32vec3 const *>(&camera_info.pos);
    this->context->shader_globals.globals.camera_view = *reinterpret_cast<daxa_f32mat4x4 const *>(&camera_info.view);
    this->context->shader_globals.globals.camera_projection = *reinterpret_cast<daxa_f32mat4x4 const *>(&camera_info.proj);
    this->context->shader_globals.globals.camera_view_projection = *reinterpret_cast<daxa_f32mat4x4 const *>(&camera_info.vp);
    this->context->shader_globals.globals.camera_near_plane_normal = *reinterpret_cast<daxa_f32vec3 const *>(&camera_info.camera_near_plane_normal);
    this->context->shader_globals.globals.camera_right_plane_normal = *reinterpret_cast<daxa_f32vec3 const *>(&camera_info.camera_right_plane_normal);
    this->context->shader_globals.globals.camera_left_plane_normal = *reinterpret_cast<daxa_f32vec3 const *>(&camera_info.camera_left_plane_normal);
    this->context->shader_globals.globals.camera_top_plane_normal = *reinterpret_cast<daxa_f32vec3 const *>(&camera_info.camera_top_plane_normal);
    this->context->shader_globals.globals.camera_bottom_plane_normal = *reinterpret_cast<daxa_f32vec3 const *>(&camera_info.camera_bottom_plane_normal);
    // Upload Shader Globals.
    context->device.get_host_address_as<ShaderGlobalsBlock>(context->shader_globals_buffer).value()[flight_frame_index] = context->shader_globals;
    context->shader_globals_ptr = context->device.get_device_address(context->shader_globals_buffer).value() + sizeof(ShaderGlobalsBlock) * flight_frame_index;
    context->shader_globals_set_info = {
        .slot = SHADER_GLOBALS_SLOT,
        .buffer = context->shader_globals_buffer,
        .size = sizeof(ShaderGlobalsBlock),
        .offset = sizeof(ShaderGlobalsBlock) * flight_frame_index,
    };

    auto swapchain_image = context->swapchain.acquire_next_image();
    if (swapchain_image.is_empty())
    {
        return;
    }
    this->swapchain_image.set_images({.images = std::array{swapchain_image}});
    meshlet_instances.swap_buffers(meshlet_instances_last_frame);

    if (static_cast<daxa_u32>(context->swapchain.get_cpu_timeline_value()) == 0)
    {
        clear_select_buffers();
    }

    this->submit_info = {};
    auto const t_semas = std::array{std::pair{this->context->transient_mem.timeline_semaphore(), this->context->transient_mem.timeline_value()}};
    this->submit_info.signal_timeline_semaphores = t_semas;
    main_task_graph.execute({});
    context->prev_settings = context->settings;
    this->context->device.collect_garbage();
}