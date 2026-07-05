#include "application.hpp"
#include "json_handler.hpp"
#include <fmt/core.h>
#include <fmt/format.h>

#include <intrin.h>
#include <png.h>
#include <chrono>
#include <ctime>
#include <filesystem>

#define DISABLE_EMISSIVE_BALLS 0

static void save_screenshot_png(std::filesystem::path const & path, u8 const * bgra_data, u32 width, u32 height)
{
    std::filesystem::create_directories(path.parent_path());

    FILE * fp = nullptr;
    fopen_s(&fp, path.string().c_str(), "wb");
    if (!fp) { fmt::print("[Screenshot] Failed to open file: {}\n", path.string()); return; }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { fclose(fp); return; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, nullptr); fclose(fp); return; }
    if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png, &info); fclose(fp); return; }

    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // swapchain is B8G8R8A8 — write as RGB swapping B and R, dropping A
    std::vector<u8> row(width * 3u);
    for (u32 y = 0; y < height; ++y)
    {
        u8 const * src = bgra_data + y * width * 4u;
        for (u32 x = 0; x < width; ++x)
        {
            row[x * 3 + 0] = src[x * 4 + 2]; // R ← B
            row[x * 3 + 1] = src[x * 4 + 1]; // G
            row[x * 3 + 2] = src[x * 4 + 0]; // B ← R
        }
        png_write_row(png, row.data());
    }

    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    fmt::print("[Screenshot] Saved to: {}\n", path.string());
}

struct ScreenshotWriteTask : Task
{
    std::filesystem::path path = {};
    std::vector<u8> pixels = {};
    u32 width = {};
    u32 height = {};
    void callback(u32, u32) override { save_screenshot_png(path, pixels.data(), width, height); }
};

static auto create_sphere_gpu_mesh(daxa::Device & device, u32 material_index) -> GPUMesh
{
    constexpr float SPHERE_PI = 3.14159265358979323846f;
    constexpr int STACKS = 8;
    constexpr int SLICES = 12;
    constexpr float R = 1.0f;

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<u32> indices;

    for (int s = 0; s <= STACKS; ++s)
    {
        float const phi = (float(s) / STACKS) * SPHERE_PI;
        for (int sl = 0; sl <= SLICES; ++sl)
        {
            float const theta = (float(sl) / SLICES) * 2.0f * SPHERE_PI;
            float const nx = std::sin(phi) * std::cos(theta);
            float const ny = std::cos(phi);
            float const nz = std::sin(phi) * std::sin(theta);
            positions.push_back({nx * R, ny * R, nz * R});
            normals.push_back({nx, ny, nz});
        }
    }

    for (int s = 0; s < STACKS; ++s)
    {
        for (int sl = 0; sl < SLICES; ++sl)
        {
            u32 const i0 = s * (SLICES + 1) + sl;
            u32 const i1 = i0 + 1;
            u32 const i2 = i0 + (SLICES + 1);
            u32 const i3 = i2 + 1;
            indices.push_back(i0); indices.push_back(i1); indices.push_back(i2);
            indices.push_back(i1); indices.push_back(i3); indices.push_back(i2);
        }
    }

    size_t const max_meshlets = meshopt_buildMeshletsBound(
        indices.size(), MAX_VERTICES_PER_MESHLET, MAX_TRIANGLES_PER_MESHLET);
    std::vector<meshopt_Meshlet> meshlets(max_meshlets);
    std::vector<u32> indirect_vertices(max_meshlets * MAX_VERTICES_PER_MESHLET);
    std::vector<u8> micro_indices_u8(max_meshlets * MAX_TRIANGLES_PER_MESHLET * 3);

    size_t const meshlet_count = meshopt_buildMeshlets(
        meshlets.data(), indirect_vertices.data(), micro_indices_u8.data(),
        indices.data(), indices.size(),
        &positions[0].x, positions.size(), sizeof(glm::vec3),
        MAX_VERTICES_PER_MESHLET, MAX_TRIANGLES_PER_MESHLET, 0.0f);

    auto const & last_ml = meshlets[meshlet_count - 1];
    indirect_vertices.resize(last_ml.vertex_offset + last_ml.vertex_count);
    size_t const micro_size = last_ml.triangle_offset + ((last_ml.triangle_count * 3u + 3u) & ~3u);
    micro_indices_u8.resize(micro_size);
    meshlets.resize(meshlet_count);

    std::vector<BoundingSphere> meshlet_bounds(meshlet_count);
    std::vector<AABB> meshlet_aabbs(meshlet_count);

    for (size_t mi = 0; mi < meshlet_count; ++mi)
    {
        auto const & ml = meshlets[mi];
        meshopt_Bounds const raw = meshopt_computeMeshletBounds(
            &indirect_vertices[ml.vertex_offset], &micro_indices_u8[ml.triangle_offset],
            ml.triangle_count, &positions[0].x, positions.size(), sizeof(glm::vec3));
        meshlet_bounds[mi] = {
            .center = daxa_f32vec3{raw.center[0], raw.center[1], raw.center[2]},
            .radius = raw.radius,
        };

        glm::vec3 ml_min = positions[indirect_vertices[ml.vertex_offset]];
        glm::vec3 ml_max = ml_min;
        for (u32 v = 1; v < ml.vertex_count; ++v)
        {
            glm::vec3 const p = positions[indirect_vertices[ml.vertex_offset + v]];
            ml_min = glm::min(ml_min, p);
            ml_max = glm::max(ml_max, p);
        }
        glm::vec3 const c = (ml_min + ml_max) * 0.5f;
        meshlet_aabbs[mi] = {
            .center = daxa_f32vec3{c.x, c.y, c.z},
            .size = daxa_f32vec3{ml_max.x - ml_min.x, ml_max.y - ml_min.y, ml_max.z - ml_min.z},
        };
    }

    AABB mesh_aabb;
    {
        glm::vec3 mesh_min = positions[0], mesh_max = positions[0];
        for (auto const & p : positions) { mesh_min = glm::min(mesh_min, p); mesh_max = glm::max(mesh_max, p); }
        glm::vec3 const c = (mesh_min + mesh_max) * 0.5f;
        mesh_aabb = {
            .center = daxa_f32vec3{c.x, c.y, c.z},
            .size = daxa_f32vec3{mesh_max.x - mesh_min.x, mesh_max.y - mesh_min.y, mesh_max.z - mesh_min.z},
        };
    }
    BoundingSphere const bsphere = { .center = mesh_aabb.center, .radius = R };

    u64 const total_size =
        sizeof(Meshlet) * meshlet_count +
        sizeof(BoundingSphere) * meshlet_count +
        sizeof(AABB) * meshlet_count +
        sizeof(u8) * micro_indices_u8.size() +
        sizeof(u32) * indirect_vertices.size() +
        sizeof(u32) * indices.size() +
        sizeof(glm::vec3) * positions.size() +
        sizeof(glm::vec3) * normals.size();

    GPUMesh mesh = {};
    mesh.material_index = material_index;
    mesh.meshlet_count = static_cast<u32>(meshlet_count);
    mesh.vertex_count = static_cast<u32>(positions.size());
    mesh.primitive_count = static_cast<u32>(indices.size() / 3);
    mesh.lod_error = 0.0f;
    mesh.aabb = mesh_aabb;
    mesh.bounding_sphere = bsphere;

    mesh.mesh_buffer = std::bit_cast<daxa_BufferId>(device.create_buffer({
        .size = static_cast<daxa::usize>(total_size),
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "procedural_sphere_mesh",
    }));

    daxa::BufferId const buf = std::bit_cast<daxa::BufferId>(mesh.mesh_buffer);
    daxa::DeviceAddress const bda = device.buffer_device_address(buf).value();
    std::byte * const ptr = device.buffer_host_address(buf).value();

    u64 off = 0;
    mesh.meshlets = bda + off;
    std::memcpy(ptr + off, meshlets.data(), sizeof(Meshlet) * meshlet_count);
    off += sizeof(Meshlet) * meshlet_count;

    mesh.meshlet_bounds = bda + off;
    std::memcpy(ptr + off, meshlet_bounds.data(), sizeof(BoundingSphere) * meshlet_count);
    off += sizeof(BoundingSphere) * meshlet_count;

    mesh.meshlet_aabbs = bda + off;
    std::memcpy(ptr + off, meshlet_aabbs.data(), sizeof(AABB) * meshlet_count);
    off += sizeof(AABB) * meshlet_count;

    mesh.micro_indices = bda + off;
    std::memcpy(ptr + off, micro_indices_u8.data(), sizeof(u8) * micro_indices_u8.size());
    off += sizeof(u8) * micro_indices_u8.size();

    mesh.indirect_vertices = bda + off;
    std::memcpy(ptr + off, indirect_vertices.data(), sizeof(u32) * indirect_vertices.size());
    off += sizeof(u32) * indirect_vertices.size();

    mesh.primitive_indices = bda + off;
    std::memcpy(ptr + off, indices.data(), sizeof(u32) * indices.size());
    off += sizeof(u32) * indices.size();

    mesh.vertex_uvs = 0;

    mesh.vertex_positions = bda + off;
    std::memcpy(ptr + off, positions.data(), sizeof(glm::vec3) * positions.size());
    off += sizeof(glm::vec3) * positions.size();

    mesh.vertex_normals = bda + off;
    std::memcpy(ptr + off, normals.data(), sizeof(glm::vec3) * normals.size());

    return mesh;
}

auto load_stbn2D(AssetProcessor & asset_processor) -> daxa::ImageId
{
    std::filesystem::path const STBN_BASE_PATH = "deps\\timberdoodle_assets\\STBN\\";
    std::filesystem::path const stbn_vec2_2Dx1D_128x128x64_base_path = STBN_BASE_PATH / "stbn_vec2_2Dx1D_128x128x64_0.png";

    AssetProcessor::NonmanifestLoadRet ret = asset_processor.load_nonmanifest_texture({
        stbn_vec2_2Dx1D_128x128x64_base_path,
        64,
        false
    });
    if (auto const * err = std::get_if<AssetProcessor::AssetLoadResultCode>(&ret))
    {
        DEBUG_MSG(fmt::format("[Renderer] ERROR failed to load Spatio Temporal Blue Noise (STBN) from path {}", stbn_vec2_2Dx1D_128x128x64_base_path.string()));
        return {};
    }

    return std::get<daxa::ImageId>(ret);
}

auto load_stbnCosDir(AssetProcessor & asset_processor) -> daxa::ImageId
{
    std::filesystem::path const STBN_BASE_PATH = "deps\\timberdoodle_assets\\STBN\\";
    std::filesystem::path const stbn_unitvec3_cosine_2Dx1D_128x128x64_base_path = STBN_BASE_PATH / "stbn_unitvec3_cosine_2Dx1D_128x128x64_0.png";

    AssetProcessor::NonmanifestLoadRet ret = asset_processor.load_nonmanifest_texture({
        stbn_unitvec3_cosine_2Dx1D_128x128x64_base_path,
        64,
        false
    });
    if (auto const * err = std::get_if<AssetProcessor::AssetLoadResultCode>(&ret))
    {
        DEBUG_MSG(fmt::format("[Renderer] ERROR failed to load Spatio Temporal Blue Noise (STBN) from path {}", stbn_unitvec3_cosine_2Dx1D_128x128x64_base_path.string()));
        return {};
    }

    return std::get<daxa::ImageId>(ret);
}

std::filesystem::path const DEFAULT_CLOUD_DATA_VDB_PATH = "deps\\timberdoodle_assets\\clouds\\cloud_data_fields.cloudbin";
std::filesystem::path const DEFAULT_CLOUD_DETAIL_NOISE_VDB_PATH = "deps\\timberdoodle_assets\\clouds\\cloud_detail_noise.cloudbin";

Application::Application()
{
    _threadpool = std::make_unique<ThreadPool>(6);
    _window = std::make_unique<Window>(1024, 1024, "Timberdoodle");
    _gpu_context = std::make_unique<GPUContext>(*_window);
    _scene = std::make_unique<Scene>(_gpu_context->device, _gpu_context.get());
    _asset_manager = std::make_unique<AssetProcessor>(_gpu_context->device);
    _ui_engine = std::make_unique<UIEngine>(*_window, *_asset_manager, _gpu_context.get());

    _renderer = std::make_unique<Renderer>(_window.get(), _gpu_context.get(), _scene.get(), _asset_manager.get(), &_ui_engine->imgui_renderer, _ui_engine.get());

    std::filesystem::path const DEFAULT_SKY_SETTINGS_PATH = "settings\\sky\\default.json";
    // std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\cam_path_sun_temple.json";
    // std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\cam_path_san_miguel.json";
    std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\cam_path_bistro.json";
    // std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\keypoints.json";
    // std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\exported_path.json";

    _renderer->stbn2d = load_stbn2D(*_asset_manager);
    _renderer->render_context->render_data.stbn2d = std::bit_cast<daxa_ImageViewId>(_renderer->stbn2d.default_view());
    _renderer->stbnCosDir = load_stbnCosDir(*_asset_manager);
    _renderer->render_context->render_data.stbnCosDir = std::bit_cast<daxa_ImageViewId>(_renderer->stbnCosDir.default_view());

    _renderer->render_context->render_data.sky_settings = load_sky_settings(DEFAULT_SKY_SETTINGS_PATH);
    app_state.cinematic_camera.update_keyframes(std::move(load_camera_animation(DEFAULT_CAMERA_ANIMATION_PATH)));

    auto const cloud_volume_index = _scene->add_cloud_volume(DEFAULT_CLOUD_DATA_VDB_PATH.string(), DEFAULT_CLOUD_DETAIL_NOISE_VDB_PATH.string(), _asset_manager.get(), _threadpool.get());
    const RenderEntityId cloud_entity_id = _scene->_render_entities.create_slot({
        .transform = glm::mat4x3(glm::translate(glm::scale(glm::identity<glm::mat4x4>(), f32vec3(512.0f, 512.0f, 64.0f) * 20.0f), f32vec3(-0.5f, -0.5f, 0.3f))),
        .cloud_volume_index = cloud_volume_index,
        .type = EntityType::CLOUD_VOLUME,
        .name = "Default cloud volume",
    });
    _scene->_dirty_render_entities.push_back(cloud_entity_id);

#if !DISABLE_EMISSIVE_BALLS
    {
        const u32 material_index = static_cast<u32>(_scene->_material_manifest.size());
        _scene->_material_manifest.push_back({
            .base_color = f32vec3{1.0f, 0.8f, 0.4f},
            .emissive_color = f32vec3{40.0f, 20.0f, 2.0f},
            .name = "Emissive Ball Material",
        });
        _scene->_new_material_manifest_entries += 1;

        const u32 lod_group_index = static_cast<u32>(_scene->_mesh_lod_group_manifest.size());
        const u32 mesh_group_index = static_cast<u32>(_scene->_mesh_group_manifest.size());
        const u32 lod_group_indices_offset = static_cast<u32>(_scene->_mesh_lod_group_manifest_indices.size());

        _scene->_mesh_lod_group_manifest.push_back({
            .mesh_group_manifest_index = mesh_group_index,
            .material_index = material_index,
            .name = "Emissive Ball Lod Group",
        });
        _scene->_new_mesh_lod_group_manifest_entries += 1;
        _scene->_mesh_lod_group_manifest_indices.push_back(lod_group_index);

        _scene->_mesh_group_manifest.push_back({
            .mesh_lod_group_manifest_indices_array_offset = lod_group_indices_offset,
            .mesh_lod_group_count = 1,
            .name = "Emissive Ball Mesh Group",
        });
        _scene->_new_mesh_group_manifest_entries += 1;

        AssetProcessor::MeshLodGroupUploadInfo upload_info = {};
        upload_info.lods[0] = create_sphere_gpu_mesh(_gpu_context->device, material_index);
        upload_info.lod_count = 1;
        upload_info.mesh_lod_manifest_index = lod_group_index;
        _pending_mesh_uploads.push_back(upload_info);

        const f32vec3 initial_pos = f32vec3{0.0f, 26.0f, 3.0f};
        app_state.emissive_ball = _scene->_render_entities.create_slot({
            .transform = glm::mat4x3(glm::translate(glm::identity<glm::mat4x4>(), initial_pos)),
            .mesh_group_manifest_index = mesh_group_index,
            .type = EntityType::MESHGROUP,
            .name = "Emissive Flying Ball",
        });
        _scene->_dirty_render_entities.push_back(app_state.emissive_ball);
    }

    {
        const u32 material_index2 = static_cast<u32>(_scene->_material_manifest.size());
        _scene->_material_manifest.push_back({
            .base_color = f32vec3{1.0f, 0.5f, 0.8f},
            .emissive_color = f32vec3{40.0f, 5.0f, 25.0f},
            .name = "Emissive Ball2 Material",
        });
        _scene->_new_material_manifest_entries += 1;

        const u32 lod_group_index2 = static_cast<u32>(_scene->_mesh_lod_group_manifest.size());
        const u32 mesh_group_index2 = static_cast<u32>(_scene->_mesh_group_manifest.size());
        const u32 lod_group_indices_offset2 = static_cast<u32>(_scene->_mesh_lod_group_manifest_indices.size());

        _scene->_mesh_lod_group_manifest.push_back({
            .mesh_group_manifest_index = mesh_group_index2,
            .material_index = material_index2,
            .name = "Emissive Ball2 Lod Group",
        });
        _scene->_new_mesh_lod_group_manifest_entries += 1;
        _scene->_mesh_lod_group_manifest_indices.push_back(lod_group_index2);

        _scene->_mesh_group_manifest.push_back({
            .mesh_lod_group_manifest_indices_array_offset = lod_group_indices_offset2,
            .mesh_lod_group_count = 1,
            .name = "Emissive Ball2 Mesh Group",
        });
        _scene->_new_mesh_group_manifest_entries += 1;

        AssetProcessor::MeshLodGroupUploadInfo upload_info2 = {};
        upload_info2.lods[0] = create_sphere_gpu_mesh(_gpu_context->device, material_index2);
        upload_info2.lod_count = 1;
        upload_info2.mesh_lod_manifest_index = lod_group_index2;
        _pending_mesh_uploads.push_back(upload_info2);

        const f32vec3 initial_pos2 = f32vec3{0.0f, 26.0f, 3.0f};
        app_state.emissive_ball2 = _scene->_render_entities.create_slot({
            .transform = glm::mat4x3(glm::translate(glm::identity<glm::mat4x4>(), initial_pos2)),
            .mesh_group_manifest_index = mesh_group_index2,
            .type = EntityType::MESHGROUP,
            .name = "Emissive Flying Ball 2",
        });
        _scene->_dirty_render_entities.push_back(app_state.emissive_ball2);
    }
#endif // !DISABLE_EMISSIVE_BALLS

    // compile_pipelines internally fans out per-pipeline work across the thread pool.
    _renderer->compile_pipelines(*_threadpool);

    app_state.last_time_point = app_state.startup_time_point = std::chrono::steady_clock::now();
    _renderer->render_context->render_times.enable_render_times = true;
}

using FpMicroSeconds = std::chrono::duration<float, std::chrono::microseconds::period>;

void Application::load_scene(std::filesystem::path const & path)
{
    if (!path.has_filename() || !path.has_parent_path())
    {
        return;
    }

    auto const result = _scene->load_manifest_from_gltf({
        .root_path = path.parent_path(),
        .asset_name = path.filename(),
        .thread_pool = _threadpool,
        .asset_processor = _asset_manager,
    });

    if (Scene::LoadManifestErrorCode const * err = std::get_if<Scene::LoadManifestErrorCode>(&result))
    {
        DEBUG_MSG(fmt::format("[WARN][Application::Application()] Loading \"{}\" Error: {}",
            path.string(), Scene::to_string(*err)));
    }
    // TODO(msakmary) HACKY - fix this
    // =========================================================================
    else
    {
        auto const r_id = std::get<RenderEntityId>(result);
        app_state.root_id = r_id;

        for (u32 entity_i = 0; entity_i < _scene->_render_entities.capacity(); ++entity_i)
        {
            RenderEntity const * r_ent = _scene->_render_entities.slot_by_index(entity_i);
            if (r_ent->name == "DYNAMIC_sphere")
            {
                app_state.dynamic_ball = _scene->_render_entities.id_from_index(entity_i);
            }
        }

        DEBUG_MSG(fmt::format("[INFO][Application::Application()] Loading \"{}\" Success", path.string()));
    }
    // =========================================================================
}

auto Application::run() -> i32
{
    while (app_state.keep_running)
    {
        auto new_time_point = std::chrono::steady_clock::now();
        app_state.delta_time = std::chrono::duration_cast<FpMicroSeconds>(new_time_point - app_state.last_time_point).count() / 1'000'000.0f;
        app_state.last_time_point = new_time_point;
        app_state.total_elapsed_us = s_cast<u64>(std::chrono::duration_cast<FpMicroSeconds>(new_time_point - app_state.startup_time_point).count());

        {
            auto start_time_taken_cpu_windowing = std::chrono::steady_clock::now();
            _window->update(app_state.delta_time);
            app_state.keep_running &= !static_cast<bool>(glfwWindowShouldClose(_window->glfw_handle));
            i32vec2 new_window_size;
            glfwGetWindowSize(this->_window->glfw_handle, &new_window_size.x, &new_window_size.y);
            if (this->_window->size.x != new_window_size.x || _window->size.y != new_window_size.y)
            {
                this->_window->size = new_window_size;
                _renderer->window_resized();
            }
            auto end_time_taken_cpu_windowing = std::chrono::steady_clock::now();
            app_state.time_taken_cpu_windowing = std::chrono::duration_cast<FpMicroSeconds>(end_time_taken_cpu_windowing - start_time_taken_cpu_windowing).count() / 1'000'000.0f;
        }
        if (_window->size.x != 0 && _window->size.y != 0)
        {
            {
                auto start_time_taken_cpu_application = std::chrono::steady_clock::now();
                update();
                auto end_time_taken_cpu_application = std::chrono::steady_clock::now();
                app_state.time_taken_cpu_application = std::chrono::duration_cast<FpMicroSeconds>(end_time_taken_cpu_application - start_time_taken_cpu_application).count() / 1'000'000.0f;
            }
            {
                auto start_time_taken_cpu_wait_for_gpu = std::chrono::steady_clock::now();
                _gpu_context->swapchain.wait_for_next_frame();
                auto end_time_taken_cpu_wait_for_gpu = std::chrono::steady_clock::now();
                app_state.time_taken_cpu_wait_for_gpu = std::chrono::duration_cast<FpMicroSeconds>(end_time_taken_cpu_wait_for_gpu - start_time_taken_cpu_wait_for_gpu).count() / 1'000'000.0f;
            }
            bool execute_frame = {};
            {
                auto start_time_taken_cpu_renderer_prepare = std::chrono::steady_clock::now();
                auto const camera_info = app_state.use_preset_camera ? 
                app_state.cinematic_camera.make_camera_info(_renderer->render_context->render_data.settings) :
                app_state.camera_controller.make_camera_info(_renderer->render_context->render_data.settings);
                execute_frame = _renderer->prepare_frame(
                    app_state.frame_index,
                    camera_info,
                    app_state.observer_camera_controller.make_camera_info(_renderer->render_context->render_data.settings),
                    app_state.delta_time,
                    app_state.total_elapsed_us);
                auto end_time_taken_cpu_renderer_prepare = std::chrono::steady_clock::now();
                app_state.time_taken_cpu_renderer_prepare = std::chrono::duration_cast<FpMicroSeconds>(end_time_taken_cpu_renderer_prepare - start_time_taken_cpu_renderer_prepare).count() / 1'000'000.0f;
            }
            bool const screenshot_write_in_progress =
                _renderer->screenshot_write_task &&
                _renderer->screenshot_write_task->not_finished > 0;
            app_state.screenshot_writing = screenshot_write_in_progress;
            _renderer->screenshot_pending = app_state.request_screenshot && !screenshot_write_in_progress;
            app_state.request_screenshot = false;

            if (execute_frame)
            {
                auto start_time_taken_cpu_renderer_record = std::chrono::steady_clock::now();
                _renderer->main_task_graph.execute({
                    .debug_ui = &_ui_engine->main_task_graph_debug_ui,
                 });
                auto end_time_taken_cpu_renderer_record = std::chrono::steady_clock::now();
                app_state.time_taken_cpu_renderer_record = std::chrono::duration_cast<FpMicroSeconds>(end_time_taken_cpu_renderer_record - start_time_taken_cpu_renderer_record).count() / 1'000'000.0f;
            }

            if (_renderer->screenshot_pending)
            {
                _gpu_context->device.wait_idle();
                auto const * data = _gpu_context->device.buffer_host_address_as<u8>(_renderer->screenshot_readback_buf).value();
                std::time_t const t = std::time(nullptr);
                std::tm tm_buf = {};
                localtime_s(&tm_buf, &t);
                char time_str[32];
                std::strftime(time_str, sizeof(time_str), "%Y%m%d_%H%M%S", &tm_buf);
                auto task = std::make_shared<ScreenshotWriteTask>();
                task->chunk_count = 1;
                task->not_finished = 1;
                auto const & cam = app_state.camera_controller;
                task->path = std::filesystem::path("screenshots") / fmt::format(
                    "screenshot_{}_pos{:.1f},{:.1f},{:.1f}_rot{:.2f},{:.2f}.png",
                    time_str,
                    cam.position.x, cam.position.y, cam.position.z,
                    cam.yaw, cam.pitch);
                task->pixels = std::vector<u8>(data, data + _renderer->screenshot_width * _renderer->screenshot_height * 4u);
                task->width = _renderer->screenshot_width;
                task->height = _renderer->screenshot_height;
                _renderer->screenshot_write_task = task;
                _threadpool->async_dispatch(task, TaskPriority::LOW);
                _renderer->screenshot_pending = false;
            }
        }
        _gpu_context->device.collect_garbage();
        ++app_state.frame_index;
    }
    return 0;
}

void Application::update()
{
    if (!app_state.desired_scene_path.empty())
    {
        fmt::print("Requested load: {}\n", app_state.desired_scene_path);
        load_scene(app_state.desired_scene_path);
        app_state.desired_scene_path.clear();
    }

    // TODO(msakmary) HACKY - fix this
    // ===== Saky's Ball =====
    {
        static f32 total_time = 0.0f;
        total_time += app_state.delta_time;

        // {
        //     RenderEntity * r_ent = _scene->_render_entities.slot(app_state.root_id);
        //     auto transform = mat_4x3_to_4x4(r_ent->transform);
        //     transform = glm::rotate(transform, glm::radians(90.0f), f32vec3(1.0f, 0.0f, 0.0f));

        //     r_ent->transform = transform;
        //     _scene->_dirty_render_entities.push_back(app_state.root_id);
        // }

        auto * dynamic_ball_ent = _scene->_render_entities.slot(app_state.dynamic_ball);
        // if (dynamic_ball_ent)
        if (false)
        {
            auto prev_transform = mat_4x3_to_4x4(dynamic_ball_ent->transform);

            auto new_position = f32vec4{
                std::sin(total_time) * 100.0f,
                std::cos(total_time) * 100.0f,
                prev_transform[3].z,
                1.0f};
            auto curr_transform = prev_transform;
            curr_transform[3] = new_position;

            dynamic_ball_ent->transform = curr_transform;
            _scene->_dirty_render_entities.push_back(app_state.dynamic_ball);
        }

#if !DISABLE_EMISSIVE_BALLS
        // Animate emissive flying balls — each on a local circle whose center itself orbits
        {
            const f32 local_radius  = 10.0f;
            const f32 local_speed   = 0.5f;
            const f32 orbit_radius  = 8.0f;
            const f32 orbit_speed   = 0.15f;
            const f32 center_y      = 0.0f;
            const f32 height        = 3.0f;

            auto ball_pos = [&](f32 orbit_phase, f32 local_phase) -> f32vec3
            {
                const f32 cx = std::cos(total_time * orbit_speed + orbit_phase) * orbit_radius;
                const f32 cy = center_y + std::sin(total_time * orbit_speed + orbit_phase) * orbit_radius;
                return f32vec3{
                    cx + std::cos(total_time * local_speed + local_phase) * local_radius,
                    cy + std::sin(total_time * local_speed + local_phase) * local_radius,
                    height,
                };
            };

            RenderEntity * ball_ent = _scene->_render_entities.slot(app_state.emissive_ball);
            if (ball_ent)
            {
                ball_ent->transform = glm::mat4x3(glm::translate(glm::identity<glm::mat4x4>(), ball_pos(0.0f, 0.0f)));
                _scene->_dirty_render_entities.push_back(app_state.emissive_ball);
            }

            RenderEntity * ball_ent2 = _scene->_render_entities.slot(app_state.emissive_ball2);
            if (ball_ent2)
            {
                ball_ent2->transform = glm::mat4x3(glm::translate(glm::identity<glm::mat4x4>(), ball_pos(3.14159265f, 3.14159265f)));
                _scene->_dirty_render_entities.push_back(app_state.emissive_ball2);
            }
        }
#endif // !DISABLE_EMISSIVE_BALLS

        if(app_state.decompose_bistro)
        {
            for (u32 entity_i = 0; entity_i < _scene->_render_entities.capacity(); ++entity_i)
            {
                RenderEntity * r_ent = _scene->_render_entities.slot_by_index(entity_i);
                if(r_ent->mesh_group_manifest_index.has_value())// && strstr(r_ent->name.c_str(), "StreetLight"))
                {
                    auto transform = mat_4x3_to_4x4(r_ent->transform);

                    transform = transform * glm::inverse(mat_4x3_to_4x4(_scene->_render_entities.slot(r_ent->parent.value())->combined_transform));
                    transform = glm::rotate(transform, glm::radians(sin(total_time * 0.00001f) * 50.0f), glm::normalize(glm::vec3(0.0, 1.0, 0.0)));
                    transform = transform * mat_4x3_to_4x4(_scene->_render_entities.slot(r_ent->parent.value())->combined_transform);

                    r_ent->transform = transform;
                    _scene->_dirty_render_entities.push_back(_scene->_render_entities.id_from_index(entity_i));
                }
            }
        }
    }
    // ===== Saky's Ball =====

    // ===== Process Render Entities, Generate Mesh Instances =====

    auto const scene_instances = _scene->process_entities(_renderer->render_context->render_data);
    _scene->current_frame_mesh_instances = scene_instances.mesh_instances;
    _scene->current_frame_cloud_volume_instances = scene_instances.cloud_volume_instances;

    // ===== Process Render Entities, Generate Mesh Instances =====

    // ===== Update GPU Scene Buffers =====

    _scene->write_gpu_mesh_instances_buffer(_scene->current_frame_mesh_instances);
    _scene->write_gpu_cloud_volume_instances_buffer(_scene->current_frame_cloud_volume_instances);

    usize cmd_list_count = 0ull;
    std::array<daxa::ExecutableCommandList, 16> cmd_lists = {};

    auto asset_data_upload_info = _asset_manager->collect_loaded_resources();
    for (auto & pending : _pending_mesh_uploads)
        asset_data_upload_info.uploaded_meshes.push_back(pending);
    _pending_mesh_uploads.clear();

    cmd_lists.at(cmd_list_count++) = _scene->record_gpu_manifest_update({
        .uploaded_meshes = asset_data_upload_info.uploaded_meshes,
        .uploaded_textures = asset_data_upload_info.uploaded_textures,
    });
    cmd_lists.at(cmd_list_count++) = _scene->create_mesh_acceleration_structures();
    _gpu_context->device.submit_commands({
        .command_lists = std::span{cmd_lists.data(), cmd_list_count},
    });

    // ===== Update GPU Scene Buffers =====

    // ===== Input Handling =====

    app_state.reset_observer = false;
    if (_window->size.x == 0 || _window->size.y == 0)
    {
        return;
    }
    _ui_engine->main_update(*_renderer->render_context, *_scene, app_state, *_threadpool);
    if (_renderer->main_task_graph.get() && _ui_engine->tg_debug_ui)
    {
        _ui_engine->tg_debug_ui = _ui_engine->main_task_graph_debug_ui.update(_renderer->main_task_graph);
    }
    if (app_state.use_preset_camera)
    {
        app_state.cinematic_camera.process_input(*_window, app_state.delta_time);
    }
    if (app_state.control_observer) {
        app_state.observer_camera_controller.process_input(*_window, app_state.delta_time);
    }
    else {
        app_state.camera_controller.process_input(*_window, app_state.delta_time);
    }

    if (!ImGui::GetIO().WantCaptureKeyboard)
    {
        if (_window->key_just_pressed(GLFW_KEY_H))
        {
            _renderer->render_context->render_data.settings.draw_from_observer = !_renderer->render_context->render_data.settings.draw_from_observer;
        }
        app_state.cinematic_camera.override_keyframe = 
            _window->key_just_pressed(GLFW_KEY_I) ?
            !app_state.cinematic_camera.override_keyframe :
            app_state.cinematic_camera.override_keyframe;
        if (_window->key_just_pressed(GLFW_KEY_J)) { app_state.control_observer = !app_state.control_observer; }
        if (_window->key_just_pressed(GLFW_KEY_K)) { app_state.reset_observer = true; }
        if (_window->key_just_pressed(GLFW_KEY_F1)) { app_state.request_screenshot = true; }
        if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->button_just_pressed(GLFW_MOUSE_BUTTON_1))
        {
            _renderer->gpu_context->shader_debug_context.detector_window_position = {
                _window->get_cursor_x(),
                _window->get_cursor_y(),
            };
        }
        if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->key_just_pressed(GLFW_KEY_LEFT))
        {
            _renderer->gpu_context->shader_debug_context.detector_window_position.x -= 1;
        }
        if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->key_just_pressed(GLFW_KEY_RIGHT))
        {
            _renderer->gpu_context->shader_debug_context.detector_window_position.x += 1;
        }
        if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->key_just_pressed(GLFW_KEY_UP))
        {
            _renderer->gpu_context->shader_debug_context.detector_window_position.y -= 1;
        }
        if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->key_just_pressed(GLFW_KEY_DOWN))
        {
            _renderer->gpu_context->shader_debug_context.detector_window_position.y += 1;
        }
    }

    if (app_state.reset_observer)
    {
        app_state.control_observer = false;
        _renderer->render_context->render_data.settings.draw_from_observer = static_cast<u32>(false);
        app_state.observer_camera_controller = app_state.camera_controller;
    }

    // ===== Input Handling =====
}


Application::~Application()
{
    _threadpool.reset();
    auto asset_data_upload_info = _asset_manager->collect_loaded_resources();
    auto manifest_update_commands = _scene->record_gpu_manifest_update({
        .uploaded_meshes = asset_data_upload_info.uploaded_meshes,
        .uploaded_textures = asset_data_upload_info.uploaded_textures,
    });
    auto cmd_lists = std::array{std::move(manifest_update_commands)};
    _gpu_context->device.submit_commands({.command_lists = cmd_lists});
    _gpu_context->device.wait_idle();
}