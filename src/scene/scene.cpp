#include "scene.hpp"

#include <fstream>
#include <array>
#include <algorithm>

#include <fastgltf/core.hpp>

#include <fmt/format.h>
#include <glm/gtx/quaternion.hpp>
#include <thread>
#include <chrono>
#include <ktx.h>
#include <random>
#include "../daxa_helper.hpp"

#include "../shader_shared/raytracing.inl"
#include "../shader_shared/scene.inl"
#include "../rendering/tasks/misc.hpp"

#include "mesh_lod.hpp"
#include "importers/gltf_importer.hpp"

Scene::Scene(daxa::Device device, GPUContext * gpu_context)
    : _device{std::move(device)}, gpu_context{gpu_context}

{
    /// TODO: THIS IS TEMPORARY! Make manifest and entity buffers growable!
    _gpu_entity_meta = tido::make_task_buffer(_device, sizeof(GPUEntityMetaData), "_gpu_entity_meta");
    _gpu_entity_parents = tido::make_task_buffer(_device, sizeof(RenderEntityId) * MAX_ENTITIES, "_gpu_entity_parents");
    _gpu_entity_transforms = tido::make_task_buffer(_device, sizeof(daxa_f32mat4x3) * MAX_ENTITIES, "_gpu_entity_transforms");
    _gpu_entity_combined_transforms = tido::make_task_buffer(_device, sizeof(daxa_f32mat4x3) * MAX_ENTITIES, "_gpu_entity_combined_transforms");
    _gpu_entity_mesh_groups = tido::make_task_buffer(_device, sizeof(u32) * MAX_ENTITIES, "_gpu_entity_mesh_groups");
    _gpu_mesh_manifest = tido::make_task_buffer(_device, sizeof(GPUMesh) * MAX_MESHES, "_gpu_mesh_manifest");
    _gpu_mesh_lod_group_manifest = tido::make_task_buffer(_device, sizeof(GPUMeshLodGroup) * MAX_MESH_LOD_GROUPS, "_gpu_mesh_lod_group_manifest");
    _gpu_mesh_group_manifest = tido::make_task_buffer(_device, sizeof(GPUMeshGroup) * MAX_MESH_LOD_GROUPS, "_gpu_mesh_group_manifest");
    _gpu_material_manifest = tido::make_task_buffer(_device, sizeof(GPUMaterial) * MAX_MATERIALS, "_gpu_material_manifest");
    _gpu_point_lights = tido::make_task_buffer(_device, sizeof(GPUPointLight) * MAX_POINT_LIGHTS, "_gpu_point_lights", daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE);
    _gpu_spot_lights = tido::make_task_buffer(_device, sizeof(GPUSpotLight) * MAX_SPOT_LIGHTS, "_gpu_spot_lights", daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE);
    _gpu_scratch_buffer = tido::make_task_buffer(_device, _gpu_scratch_buffer_size, "_gpu_scratch_buffer");
    _gpu_mesh_acceleration_structure_build_scratch_buffer = tido::make_task_buffer(_device, _gpu_mesh_acceleration_structure_build_scratch_buffer_size, "_gpu_mesh_acceleration_structure_build_scratch_buffer");
    _gpu_tlas_build_scratch_buffer = tido::make_task_buffer(_device, _gpu_tlas_build_scratch_buffer_size, "_gpu_tlas_build_scratch_buffer");
    mesh_instances_buffer = daxa::ExternalTaskBuffer{daxa::ExternalTaskBufferInfo{.name = "mesh_instances"}};
    cloud_volume_instances_buffer = daxa::ExternalTaskBuffer{daxa::ExternalTaskBufferInfo{.name = "cloud_volume_instances"}};
    _scene_as_indirections = tido::make_task_buffer(_device, _indirections_count, "_scene_as_indirections", daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE);
}

Scene::~Scene()
{
    if (!_scene_blas.is_empty()) { _device.destroy_blas(_scene_blas); }

    if (!_gpu_entity_meta.id().is_empty())
    {
        _device.destroy_buffer(_gpu_entity_meta.id());
        _gpu_entity_meta = {};
    }
    if (!_gpu_entity_parents.id().is_empty())
    {
        _device.destroy_buffer(_gpu_entity_parents.id());
        _gpu_entity_parents = {};
    }
    if (!_gpu_entity_transforms.id().is_empty())
    {
        _device.destroy_buffer(_gpu_entity_transforms.id());
        _gpu_entity_transforms = {};
    }
    if (!_gpu_entity_combined_transforms.id().is_empty())
    {
        _device.destroy_buffer(_gpu_entity_combined_transforms.id());
        _gpu_entity_combined_transforms = {};
    }
    if (!_gpu_entity_mesh_groups.id().is_empty())
    {
        _device.destroy_buffer(_gpu_entity_mesh_groups.id());
        _gpu_entity_mesh_groups = {};
    }
    if (!_gpu_mesh_manifest.id().is_empty())
    {
        _device.destroy_buffer(_gpu_mesh_manifest.id());
        _gpu_mesh_manifest = {};
    }
    if (!_gpu_mesh_lod_group_manifest.id().is_empty())
    {
        _device.destroy_buffer(_gpu_mesh_lod_group_manifest.id());
        _gpu_mesh_lod_group_manifest = {};
    }
    if (!_gpu_mesh_group_manifest.id().is_empty())
    {
        _device.destroy_buffer(_gpu_mesh_group_manifest.id());
        _gpu_mesh_group_manifest = {};
    }
    if (!_gpu_material_manifest.id().is_empty())
    {
        _device.destroy_buffer(_gpu_material_manifest.id());
        _gpu_material_manifest = {};
    }
    if (!_gpu_point_lights.id().is_empty())
    {
        _device.destroy_buffer(_gpu_point_lights.id());
        _gpu_point_lights = {};
    }
    if (!_gpu_spot_lights.id().is_empty())
    {
        _device.destroy_buffer(_gpu_spot_lights.id());
        _gpu_spot_lights = {};
    }
    if (!_gpu_scratch_buffer.id().is_empty())
    {
        _device.destroy_buffer(_gpu_scratch_buffer.id());
        _gpu_scratch_buffer = {};
    }
    if (!_gpu_mesh_acceleration_structure_build_scratch_buffer.id().is_empty())
    {
        _device.destroy_buffer(_gpu_mesh_acceleration_structure_build_scratch_buffer.id());
        _gpu_mesh_acceleration_structure_build_scratch_buffer = {};
    }
    if (!_gpu_tlas_build_scratch_buffer.id().is_empty())
    {
        _device.destroy_buffer(_gpu_tlas_build_scratch_buffer.id());
        _gpu_tlas_build_scratch_buffer = {};
    }
    if (!_scene_as_indirections.id().is_empty())
    {
        _device.destroy_buffer(_scene_as_indirections.id());
        _scene_as_indirections = {};
    }

    for (auto & mesh_group : _mesh_group_manifest)
    {
        if (!mesh_group.blas.is_empty())
        {
            _device.destroy_blas(mesh_group.blas);
        }
    }

    for (auto & mesh : _mesh_lod_group_manifest)
    {
        if (mesh.runtime.has_value())
        {
            for (daxa_u32 lod = 0; lod < mesh.runtime.value().lod_count; ++lod)
            {
                _device.destroy_buffer(std::bit_cast<daxa::BufferId>(mesh.runtime.value().lods[lod].mesh_buffer));
                if (!mesh.runtime.value().blas_lods[lod].is_empty())
                {
                    _device.destroy_blas(mesh.runtime.value().blas_lods[lod]);
                }
            }
        }
    }

    for (auto & texture : _material_texture_manifest)
    {
        if (texture.runtime_texture.has_value())
        {
            _device.destroy_image(std::bit_cast<daxa::ImageId>(texture.runtime_texture.value()));
        }
    }

    if (!mesh_instances_buffer.id().is_empty())
    {
        _device.destroy_buffer(mesh_instances_buffer.id());
    }

    if (!cloud_volume_instances_buffer.id().is_empty())
    {
        _device.destroy_buffer(cloud_volume_instances_buffer.id());
    }
}
static void start_async_loads_of_dirty_cloud_volumes(Scene & scene, AssetProcessor * asset_processor, ThreadPool * thread_pool);

auto Scene::load_manifest_from_gltf(LoadManifestInfo const & info) -> std::variant<RenderEntityId, LoadManifestErrorCode>
{
    // All glTF parsing + translation lives in the GltfImporter. It populates the scene exclusively
    // through the generic Scene::add_* builder API and owns the parsed fastgltf::Asset transiently.
    return GltfImporter{*this, info}.import();
}

/// --- Generic, format-agnostic scene builder API ---

auto Scene::add_texture(TextureManifestEntry texture) -> u32
{
    // add_texture stays cheap + non-blocking: it only records the entry and marks it for streaming.
    // The actual GPU upload happens asynchronously, driven by update_scene (which spawns the stream
    // task and later publishes the resident image). Entries without a cooked artifact (e.g. cloud
    // volumes, made resident elsewhere) are not queued.
    std::lock_guard<std::mutex> lock{*_manifest_mutex};
    u32 const index = s_cast<u32>(_material_texture_manifest.size());
    bool const needs_streaming = !texture.cooked_artifact.tido_path.empty();
    _material_texture_manifest.push_back(std::move(texture));
    if (needs_streaming)
    {
        _dirty_material_texture_manifest.mark(index);
    }
    return index;
}

void TextureStreamTask::callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index)
{
    // Read the cooked .tido off disk and upload it to the GPU (streamer). Runs on a worker thread.
    result = make_resident_image(device, artifact);
    finished.store(true, std::memory_order_release);
}

auto Scene::add_material(MaterialManifestEntry material) -> u32
{
    std::lock_guard<std::mutex> lock{*_manifest_mutex};
    u32 const index = s_cast<u32>(_material_manifest.size());
    // Wire up the texture -> material back-references. Dedup so a texture used in two roles by the
    // same material (e.g. diffuse + opacity sharing one image) is only recorded once.
    std::array<std::optional<MaterialManifestEntry::TextureInfo> const *, 4> const infos = {
        &material.diffuse_info, &material.opacity_mask_info, &material.normal_info, &material.roughness_metalness_info};
    std::array<u32, 4> seen = {};
    u32 seen_count = 0;
    for (auto const * info : infos)
    {
        if (!info->has_value()) { continue; }
        u32 const tex_index = info->value().tex_manifest_index;
        DBG_ASSERT_TRUE_M(tex_index < _material_texture_manifest.size(), "add_material: texture info references an invalid manifest index");
        if (std::find(seen.begin(), seen.begin() + seen_count, tex_index) != seen.begin() + seen_count) { continue; }
        seen[seen_count++] = tex_index;
        _material_texture_manifest.at(tex_index).material_manifest_indices.push_back({.material_manifest_index = index});
    }
    _material_manifest.push_back(std::move(material));
    _dirty_material_manifest.mark(index);
    DBG_ASSERT_TRUE_M(_material_manifest.size() <= MAX_MATERIALS, "EXCEEDED MAX_MATERIALS");
    return index;
}

auto Scene::add_mesh(MeshLodGroupManifestEntry mesh) -> u32
{
    // Mirrors add_texture: records the entry (cooked artifact already attached) and marks it for async
    // streaming. The GPU upload happens later, driven by update_scene. We also mark the GPU-resident mesh
    // manifest so its slot is initialized now (zeroed -> "not loaded yet") until the stream publishes the
    // real data. Entries without a cooked artifact are not queued for streaming.
    std::lock_guard<std::mutex> lock{*_manifest_mutex};
    u32 const index = s_cast<u32>(_mesh_lod_group_manifest.size());
    bool const needs_streaming = !mesh.cooked_artifact.tido_path.empty();
    _mesh_lod_group_manifest.push_back(std::move(mesh));
    _dirty_mesh_lod_group_manifest.mark(index);
    if (needs_streaming)
    {
        _dirty_mesh_lod_group_streaming.mark(index);
    }
    return index;
}

auto Scene::add_mesh_group(MeshGroupManifestEntry mesh_group, std::span<u32 const> mesh_manifest_indices) -> u32
{
    std::lock_guard<std::mutex> lock{*_manifest_mutex};
    u32 const group_index = s_cast<u32>(_mesh_group_manifest.size());
    // Allocate this group's contiguous slice of the shared indices array, record its meshes, and
    // back-link each mesh to this group (resolving the mesh<->group cycle on the scene side).
    mesh_group.mesh_lod_group_manifest_indices_array_offset = s_cast<u32>(_mesh_lod_group_manifest_indices.size());
    mesh_group.mesh_lod_group_count = s_cast<u32>(mesh_manifest_indices.size());
    for (u32 const mesh_manifest_index : mesh_manifest_indices)
    {
        _mesh_lod_group_manifest_indices.push_back(mesh_manifest_index);
        _mesh_lod_group_manifest.at(mesh_manifest_index).mesh_group_manifest_index = group_index;
    }
    _mesh_group_manifest.push_back(std::move(mesh_group));
    _dirty_mesh_group_manifest.mark(group_index);
    return group_index;
}

void MeshStreamTask::callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index)
{
    // Read the cooked .tido off disk and upload it to the GPU (streamer). Runs on a worker thread.
    result = make_resident_mesh(device, MakeResidentMeshInfo{
        .artifact = artifact,
        .mesh_lod_manifest_index = mesh_lod_manifest_index,
        .material_manifest_index = material_manifest_index,
        .name = name,
    });
    finished.store(true, std::memory_order_release);
}

auto Scene::add_point_light(PointLight light) -> u32
{
    std::lock_guard<std::mutex> lock{*_manifest_mutex};
    DBG_ASSERT_TRUE_M(_point_lights.size() < MAX_POINT_LIGHTS, "Maximum point light limit is currently hardcoded");
    u32 const index = s_cast<u32>(_point_lights.size());
    light.point_light_ptr = _device.buffer_device_address(_gpu_point_lights.id()).value() + index * sizeof(GPUPointLight);
    _point_lights.push_back(light);
    return index;
}

auto Scene::add_spot_light(SpotLight light) -> u32
{
    std::lock_guard<std::mutex> lock{*_manifest_mutex};
    DBG_ASSERT_TRUE_M(_spot_lights.size() < MAX_SPOT_LIGHTS, "Maximum spot light limit is currently hardcoded");
    u32 const index = s_cast<u32>(_spot_lights.size());
    light.spot_light_ptr = _device.buffer_device_address(_gpu_spot_lights.id()).value() + index * sizeof(GPUSpotLight);
    _spot_lights.push_back(light);
    return index;
}

auto Scene::add_entity(RenderEntity entity) -> RenderEntityId
{
    std::lock_guard<std::mutex> lock{*_manifest_mutex};
    RenderEntityId const id = _render_entities.create_slot(std::move(entity));
    _dirty_render_entities.push_back(id);
    return id;
}
auto Scene::add_cloud_volume(std::string const & cloud_volume_data_path, std::string const & detail_noise_path, AssetProcessor * asset_processor, ThreadPool * thread_pool) -> u32
{
    CloudVolume cpu_cloud_volume = {};
    cpu_cloud_volume.cloud_volume_data_path = cloud_volume_data_path;
    cpu_cloud_volume.detail_noise_path = detail_noise_path;

    // Preallocate manifest entries for all possible textures.
    // This potentially wastes some manifest entries (in case the cloud volume does not use separate sdf texture for example)
    // but I am limited by the way the texture manifest currently works (extremely dependent on gltf loading and not thread safe at all).
    // In the future this should be rewritten but for now this will work fine.
    cpu_cloud_volume.data_texture_manifest_index = s_cast<u32>(_material_texture_manifest.size());
    _material_texture_manifest.push_back(TextureManifestEntry{.name = fmt::format("{} cloud data", cloud_volume_data_path).c_str()});

    cpu_cloud_volume.sdf_texture_manifest_index = s_cast<u32>(_material_texture_manifest.size());
    _material_texture_manifest.push_back(TextureManifestEntry{.name = fmt::format("{} cloud sdf", cloud_volume_data_path).c_str()});

    cpu_cloud_volume.detail_noise_texture_manifest_index = s_cast<u32>(_material_texture_manifest.size());
    _material_texture_manifest.push_back(TextureManifestEntry{.name = fmt::format("{} cloud erosion noise", cloud_volume_data_path).c_str()});

    u32 const cloud_volume_manifest_index = s_cast<u32>(_cloud_volumes.size());
    _cloud_volumes.push_back(cpu_cloud_volume);

    _cloud_volumes_requesting_load.push_back(cloud_volume_manifest_index);
    start_async_loads_of_dirty_cloud_volumes(*this, asset_processor, thread_pool);

    return cloud_volume_manifest_index;
}

static void start_async_loads_of_dirty_cloud_volumes(Scene & scene, AssetProcessor * asset_processor, ThreadPool * thread_pool)
{
    struct LoadCloudVolumeTask : Task
    {
        struct TaskInfo
        {
            AssetProcessor * asset_processor = {};
            CloudVolume const * volume;
        };

        TaskInfo info = {};

        LoadCloudVolumeTask(TaskInfo const & info)
            : info{info}
        {
            chunk_count = 1;
        }

        virtual void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override{
            {AssetProcessor::LoadCloudVolumetricDataInfo load_data_info = {
                .volumetric_data_path = std::filesystem::path(info.volume -> cloud_volume_data_path),
                .cloud_data_texture_manifest_index = info.volume->data_texture_manifest_index,
                .cloud_sdf_texture_manifest_index = info.volume->sdf_texture_manifest_index,
            };

        auto const ret_status = info.asset_processor->load_cloud_volumetric_data(load_data_info);
        if (ret_status != AssetProcessor::AssetLoadResultCode::SUCCESS)
        {
            DEBUG_MSG(fmt::format("[ERROR] Failed to load cloud volume {} - error {}",
                info.volume->cloud_volume_data_path, AssetProcessor::to_string(ret_status)));
        }
        else
        {
            DEBUG_MSG(fmt::format("[SUCCESS] Successfuly loaded cloud volume {}", info.volume->cloud_volume_data_path));
        }
    }

    {
        AssetProcessor::LoadCloudVolumetricDataInfo load_data_info = {
            .volumetric_data_path = std::filesystem::path(info.volume->detail_noise_path),
            .cloud_data_texture_manifest_index = info.volume->detail_noise_texture_manifest_index,
            .cloud_sdf_texture_manifest_index = std::numeric_limits<u32>::max(), // Currently should be unused.
        };
        auto const ret_status = info.asset_processor->load_cloud_volumetric_data(load_data_info);
        if (ret_status != AssetProcessor::AssetLoadResultCode::SUCCESS)
        {
            DEBUG_MSG(fmt::format("[ERROR] Failed to load cloud volume {} - error {}",
                info.volume->detail_noise_path, AssetProcessor::to_string(ret_status)));
        }
        else
        {
            DEBUG_MSG(fmt::format("[SUCCESS] Successfuly loaded cloud volume {}", info.volume->detail_noise_path));
        }
    }
};
}
;

for (u32 cloud_volume_manifest_index : scene._cloud_volumes_requesting_load)
{
    // Launch loading of this cloud volume
    auto task = std::make_shared<LoadCloudVolumeTask>(LoadCloudVolumeTask::TaskInfo{
        .asset_processor = asset_processor,
        .volume = &scene._cloud_volumes.at(cloud_volume_manifest_index),
    });
    thread_pool->async_dispatch(task, TaskPriority::LOW);
}
scene._cloud_volumes_requesting_load.clear();
}

auto Scene::update_scene(UpdateSceneInfo const & info) -> daxa::ExecutableCommandList
{
    // --- Texture residency (async) ---
    // 1. Collect finished texture streams: publish each resident image as the texture's runtime, and
    //    re-mark the materials referencing it dirty so their GPUMaterial picks up the resolved id below.
    for (auto it = _inflight_texture_streams.begin(); it != _inflight_texture_streams.end();)
    {
        TextureStreamTask & task = **it;
        if (!task.finished.load(std::memory_order_acquire))
        {
            ++it;
            continue;
        }
        TextureManifestEntry & texture = _material_texture_manifest.at(task.texture_manifest_index);
        texture.runtime_texture = task.result;
        for (TextureManifestEntry::MaterialManifestIndex const & ref : texture.material_manifest_indices)
        {
            _dirty_material_manifest.mark(ref.material_manifest_index);
        }
        it = _inflight_texture_streams.erase(it);
    }
    // 2. Spawn a stream task for every newly dirtied texture. (No-op if there is no thread pool, e.g.
    //    at shutdown - those textures simply never become resident, which is fine.)
    if (info.thread_pool != nullptr)
    {
        for (u32 const texture_index : _dirty_material_texture_manifest.drain())
        {
            auto task = std::make_shared<TextureStreamTask>();
            task->chunk_count = 1;
            task->device = _device;
            task->artifact = _material_texture_manifest.at(texture_index).cooked_artifact;
            task->texture_manifest_index = texture_index;
            info.thread_pool->async_dispatch(task, TaskPriority::LOW);
            _inflight_texture_streams.push_back(std::move(task));
        }
    }

    // --- Mesh residency (async) --- (mirrors the texture residency above)
    // 1. Collect finished mesh streams: publish the uploaded GPUMesh array as the entry's runtime and
    //    mark the mesh-lod-group manifest dirty so the GPU sync below uploads the real data and does the
    //    BLAS-build + mesh-group completeness bookkeeping (the runtime.has_value() branch there).
    for (auto it = _inflight_mesh_streams.begin(); it != _inflight_mesh_streams.end();)
    {
        MeshStreamTask & task = **it;
        if (!task.finished.load(std::memory_order_acquire))
        {
            ++it;
            continue;
        }
        MeshLodGroupUploadInfo const & upload = task.result;
        _mesh_lod_group_manifest.at(upload.mesh_lod_manifest_index).runtime = MeshLodGroupManifestEntry::Runtime{
            .lods = upload.lods,
            .lod_count = upload.lod_count,
        };
        _dirty_mesh_lod_group_manifest.mark(upload.mesh_lod_manifest_index);
        it = _inflight_mesh_streams.erase(it);
    }
    // 2. Spawn a stream task for every newly requested mesh. (No-op without a thread pool, e.g. shutdown.)
    if (info.thread_pool != nullptr)
    {
        for (u32 const mesh_index : _dirty_mesh_lod_group_streaming.drain())
        {
            MeshLodGroupManifestEntry const & entry = _mesh_lod_group_manifest.at(mesh_index);
            auto task = std::make_shared<MeshStreamTask>();
            task->chunk_count = 1;
            task->device = _device;
            task->artifact = entry.cooked_artifact;
            task->mesh_lod_manifest_index = mesh_index;
            task->material_manifest_index = entry.material_index.value_or(INVALID_MANIFEST_INDEX);
            task->name = entry.name;
            info.thread_pool->async_dispatch(task, TaskPriority::LOW);
            _inflight_mesh_streams.push_back(std::move(task));
        }
    }

    auto recorder = _device.create_command_recorder({});
    /// TODO: Make buffers resize.

    // Calculate required staging buffer size:
    daxa::BufferId staging_buffer = {};
    usize staging_offset = 0;
    std::byte * host_ptr = {};
    if (_dirty_render_entities.size() > 0 || _modified_render_entities.size() > 0)
    {
        usize required_staging_size = 0;
        required_staging_size += sizeof(GPUEntityMetaData);                                                                   // _gpu_entity_meta
        required_staging_size += sizeof(daxa_f32mat4x3) * (_dirty_render_entities.size() + _modified_render_entities.size()); // _gpu_entity_transforms
        required_staging_size += sizeof(daxa_f32mat4x3) * (_dirty_render_entities.size() + _modified_render_entities.size()); // _gpu_entity_combined_transforms
        required_staging_size += sizeof(GPUMeshGroup) * (_dirty_render_entities.size() + _modified_render_entities.size());   // _gpu_entity_mesh_groups
        staging_buffer = _device.create_buffer({
            .size = required_staging_size,
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "entities update staging",
        });
        recorder.destroy_buffer_deferred(staging_buffer);
        host_ptr = _device.buffer_host_address(staging_buffer).value();
        *r_cast<GPUEntityMetaData *>(host_ptr) = {.entity_count = s_cast<u32>(_render_entities.size())};
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _gpu_entity_meta.id(),
            .src_offset = staging_offset,
            .size = sizeof(GPUEntityMetaData),
        });
        staging_offset += sizeof(GPUEntityMetaData);
    }

    /**
     * TODO:
     * - replace with compute shader
     * - write two arrays, one containing entity ids other containing update data
     * - write compute shader that reads both arrays, they then write the updates from staging to entity arrays
     */
    /// NOTE: Update dirty entities.
    auto update_entity = [&](i32 i, RenderEntity * entity, u32 entity_index) -> glm::mat4
    {
        usize offset = (staging_offset + (sizeof(glm::mat4x3) * 2 + sizeof(u32)) * i);
        glm::mat4 transform4 = glm::mat4(
            glm::vec4(entity->transform[0], 0.0f),
            glm::vec4(entity->transform[1], 0.0f),
            glm::vec4(entity->transform[2], 0.0f),
            glm::vec4(entity->transform[3], 1.0f));
        glm::mat4 combined_transform4 = transform4;
        glm::mat4 combined_parent_transform4 = glm::identity<glm::mat4>();
        std::optional<RenderEntityId> parent = entity->parent;
        while (parent.has_value())
        {
            glm::mat4x3 parent_transform4 = glm::mat4(
                glm::vec4(_render_entities.slot(parent.value())->transform[0], 0.0f),
                glm::vec4(_render_entities.slot(parent.value())->transform[1], 0.0f),
                glm::vec4(_render_entities.slot(parent.value())->transform[2], 0.0f),
                glm::vec4(_render_entities.slot(parent.value())->transform[3], 1.0f));
            combined_transform4 = parent_transform4 * combined_transform4;
            combined_parent_transform4 = parent_transform4 * combined_parent_transform4;
            parent = _render_entities.slot(parent.value())->parent;
        }
        entity->combined_transform = combined_transform4;
        u32 mesh_group_manifest_index = entity->mesh_group_manifest_index.value_or(INVALID_MANIFEST_INDEX);
        struct RenderEntityUpdateStagingMemoryView
        {
            glm::mat4x3 transform;
            glm::mat4x3 combined_transform;
            u32 mesh_group_manifest_index;
        };
        *r_cast<RenderEntityUpdateStagingMemoryView *>(host_ptr + offset) = {
            .transform = transform4,
            .combined_transform = combined_transform4,
            .mesh_group_manifest_index = mesh_group_manifest_index,
        };
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _gpu_entity_transforms.id(),
            .src_offset = offset + offsetof(RenderEntityUpdateStagingMemoryView, transform),
            .dst_offset = sizeof(glm::mat4x3) * entity_index,
            .size = sizeof(glm::mat4x3),
        });
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _gpu_entity_combined_transforms.id(),
            .src_offset = offset + offsetof(RenderEntityUpdateStagingMemoryView, combined_transform),
            .dst_offset = sizeof(glm::mat4x3) * entity_index,
            .size = sizeof(glm::mat4x3),
        });
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _gpu_entity_mesh_groups.id(),
            .src_offset = offset + offsetof(RenderEntityUpdateStagingMemoryView, mesh_group_manifest_index),
            .dst_offset = sizeof(u32) * entity_index,
            .size = sizeof(u32),
        });
        return combined_parent_transform4;
    };
    for (u32 i = 0; i < _dirty_render_entities.size(); ++i)
    {
        u32 entity_index = _dirty_render_entities[i].index;
        auto * entity = _render_entities.slot(_dirty_render_entities[i]);
        entity->dirty = true;
        update_entity(i, entity, entity_index);
    }

    _dirty_render_entities.clear();
    _modified_render_entities.clear();

    // Drain the per-manifest dirty-index lists: the indices that were added/updated since the last
    // sync. We re-upload exactly these entries rather than assuming a contiguous tail of new entries.
    std::vector<u32> const dirty_mesh_groups = _dirty_mesh_group_manifest.drain();
    std::vector<u32> const dirty_mesh_lod_groups = _dirty_mesh_lod_group_manifest.drain();
    std::vector<u32> const dirty_materials = _dirty_material_manifest.drain();

    // Add new mesh group manifest entries
    if (!dirty_mesh_groups.empty())
    {
        u32 const mesh_group_staging_buffer_size = sizeof(GPUMeshGroup) * s_cast<u32>(dirty_mesh_groups.size());
        daxa::BufferId mesh_group_staging_buffer = _device.create_buffer({
            .size = mesh_group_staging_buffer_size,
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "mesh group update staging buffer",
        });
        recorder.destroy_buffer_deferred(mesh_group_staging_buffer);
        GPUMeshGroup * staging_ptr = _device.buffer_host_address_as<GPUMeshGroup>(mesh_group_staging_buffer).value();
        for (u32 i = 0; i < s_cast<u32>(dirty_mesh_groups.size()); i++)
        {
            u32 const mesh_group_manifest_idx = dirty_mesh_groups[i];
            staging_ptr[i].mesh_lod_group_count = _mesh_group_manifest.at(mesh_group_manifest_idx).mesh_lod_group_count;
            recorder.copy_buffer_to_buffer({
                .src_buffer = mesh_group_staging_buffer,
                .dst_buffer = _gpu_mesh_group_manifest.id(),
                .src_offset = sizeof(GPUMeshGroup) * i,
                .dst_offset = sizeof(GPUMeshGroup) * mesh_group_manifest_idx,
                .size = sizeof(GPUMeshGroup),
            });
        }
    }

    // Sync each dirty mesh-lod-group's GPU state. A mesh is dirtied twice over its life: when it is
    // added (no runtime yet -> we upload zeroed GPUMesh/GPUMeshLodGroup slots) and when its cooked data
    // becomes resident (an async mesh stream finishes -> we upload the real data and do the load
    // bookkeeping). Dedup so a mesh added + loaded within the same frame is processed (load-counted) once.
    if (!dirty_mesh_lod_groups.empty())
    {
        std::vector<u32> unique_dirty_meshes = dirty_mesh_lod_groups;
        std::sort(unique_dirty_meshes.begin(), unique_dirty_meshes.end());
        unique_dirty_meshes.erase(std::unique(unique_dirty_meshes.begin(), unique_dirty_meshes.end()), unique_dirty_meshes.end());

        u32 const dirty_count = s_cast<u32>(unique_dirty_meshes.size());
        usize const meshes_staging_size = dirty_count * sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP;
        usize const mesh_lod_group_staging_size = dirty_count * sizeof(GPUMeshLodGroup);
        daxa::BufferId mesh_sync_staging_buffer = _device.create_buffer({
            .size = meshes_staging_size + mesh_lod_group_staging_size,
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "mesh + mesh lod group manifest update staging buffer",
        });
        recorder.destroy_buffer_deferred(mesh_sync_staging_buffer);
        GPUMesh * mesh_staging_ptr = _device.buffer_host_address_as<GPUMesh>(mesh_sync_staging_buffer).value();
        GPUMeshLodGroup * mesh_lod_group_staging_ptr = r_cast<GPUMeshLodGroup *>(_device.buffer_host_address(mesh_sync_staging_buffer).value() + meshes_staging_size);

        for (u32 i = 0; i < dirty_count; ++i)
        {
            u32 const mesh_lod_manifest_index = unique_dirty_meshes[i];
            MeshLodGroupManifestEntry & mesh_lod_group = _mesh_lod_group_manifest.at(mesh_lod_manifest_index);

            std::array<GPUMesh, MAX_MESHES_PER_LOD_GROUP> lods = {};
            u32 lod_count = 0;
            if (mesh_lod_group.runtime.has_value())
            {
                lods = mesh_lod_group.runtime.value().lods;
                lod_count = mesh_lod_group.runtime.value().lod_count;
                DAXA_DBG_ASSERT_TRUE_M(lods[0].material_index == mesh_lod_group.material_index.value_or(INVALID_MANIFEST_INDEX), "IMPOSSIBLE CASE! material index MUST MATCH!");

                // Queue every newly resident LOD for a BLAS build.
                for (u32 lod = 0; lod < lod_count; ++lod)
                {
                    _mesh_as_build_queue.push_back(mesh_lod_manifest_index * MAX_MESHES_PER_LOD_GROUP + lod);
                }
                // Bump the owning mesh group's loaded count; if every mesh in it is now resident, mark it complete.
                MeshGroupManifestEntry & mesh_group = _mesh_group_manifest.at(mesh_lod_group.mesh_group_manifest_index);
                mesh_group.loaded_mesh_lod_groups += 1;
                bool is_completely_loaded = true;
                u32 const range[] = {mesh_group.mesh_lod_group_manifest_indices_array_offset, mesh_group.mesh_lod_group_manifest_indices_array_offset + mesh_group.mesh_lod_group_count};
                for (u32 mesh_idx_array_idx = range[0]; mesh_idx_array_idx < range[1]; mesh_idx_array_idx++)
                {
                    if (!_mesh_lod_group_manifest.at(_mesh_lod_group_manifest_indices.at(mesh_idx_array_idx)).runtime.has_value())
                    {
                        is_completely_loaded = false;
                        break;
                    }
                }
                if (is_completely_loaded)
                {
                    _newly_completed_mesh_groups.push_back(mesh_lod_group.mesh_group_manifest_index);
                }
            }

            std::memcpy(mesh_staging_ptr + i * MAX_MESHES_PER_LOD_GROUP, lods.data(), sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP);
            mesh_lod_group_staging_ptr[i] = {.lod_count = lod_count};

            recorder.copy_buffer_to_buffer({
                .src_buffer = mesh_sync_staging_buffer,
                .dst_buffer = _gpu_mesh_manifest.id(),
                .src_offset = i * sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP,
                .dst_offset = mesh_lod_manifest_index * sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP,
                .size = sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP,
            });
            recorder.copy_buffer_to_buffer({
                .src_buffer = mesh_sync_staging_buffer,
                .dst_buffer = _gpu_mesh_lod_group_manifest.id(),
                .src_offset = meshes_staging_size + i * sizeof(GPUMeshLodGroup),
                .dst_offset = mesh_lod_manifest_index * sizeof(GPUMeshLodGroup),
                .size = sizeof(GPUMeshLodGroup),
            });
        }
    }

    // Sync each dirty material. We write the COMPLETE GPUMaterial, resolving its texture ids from the
    // texture manifest. Importers add a material only after its textures are made resident (add_texture),
    // so the runtime ids are already present here and a single write per material is enough - no separate
    // texture-propagation pass.
    if (!dirty_materials.empty())
    {
        u32 const dirty_material_count = s_cast<u32>(dirty_materials.size());
        daxa::BufferId material_staging_buffer = _device.create_buffer({
            .size = sizeof(GPUMaterial) * dirty_material_count,
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "material update staging buffer",
        });
        recorder.destroy_buffer_deferred(material_staging_buffer);
        GPUMaterial * staging_ptr = _device.buffer_host_address_as<GPUMaterial>(material_staging_buffer).value();

        auto resolve_texture_id = [&](std::optional<MaterialManifestEntry::TextureInfo> const & info) -> daxa::ImageId
        {
            if (!info.has_value()) { return {}; }
            return _material_texture_manifest.at(info.value().tex_manifest_index).runtime_texture.value_or(daxa::ImageId{});
        };

        // The normal map's BC5 encoding is deduced from its cooked texture format, not tracked through
        // the import: the shader needs to know whether to reconstruct Z from a two-channel normal map.
        auto normal_is_bc5_rg = [&](std::optional<MaterialManifestEntry::TextureInfo> const & info) -> bool
        {
            if (!info.has_value()) { return false; }
            return tido_format_is_bc5_rg(_material_texture_manifest.at(info.value().tex_manifest_index).cooked_artifact.descriptor.format);
        };

        for (u32 i = 0; i < dirty_material_count; ++i)
        {
            u32 const material_manifest_idx = dirty_materials[i];
            MaterialManifestEntry const & material = _material_manifest.at(material_manifest_idx);
            GPUMaterial gpu_material = {};
            gpu_material.diffuse_texture_id = resolve_texture_id(material.diffuse_info).default_view();
            gpu_material.opacity_texture_id = resolve_texture_id(material.opacity_mask_info).default_view();
            gpu_material.normal_texture_id = resolve_texture_id(material.normal_info).default_view();
            gpu_material.roughnes_metalness_id = resolve_texture_id(material.roughness_metalness_info).default_view();
            gpu_material.alpha_discard_enabled = material.alpha_discard_enabled;
            gpu_material.normal_compressed_bc5_rg = normal_is_bc5_rg(material.normal_info);
            gpu_material.base_color = std::bit_cast<daxa_f32vec3>(material.base_color);
            gpu_material.emissive_color = std::bit_cast<daxa_f32vec3>(material.emissive_color);
            gpu_material.double_sided_enabled = material.double_sided;
            gpu_material.blend_enabled = material.blend_enabled;
            staging_ptr[i] = gpu_material;
            recorder.copy_buffer_to_buffer({
                .src_buffer = material_staging_buffer,
                .dst_buffer = _gpu_material_manifest.id(),
                .src_offset = sizeof(GPUMaterial) * i,
                .dst_offset = sizeof(GPUMaterial) * material_manifest_idx,
                .size = sizeof(GPUMaterial),
            });
        }
    }

    // Make cloud-volume textures resident in the manifest. These still arrive through the AssetProcessor
    // upload queue (their load path is not yet ported); they are not referenced by any material, so we
    // only stash their runtime image id - no material update needed.
    for (AssetProcessor::LoadedTextureInfo const & texture_upload : info.uploaded_textures)
    {
        _material_texture_manifest.at(texture_upload.texture_manifest_index).runtime_texture = texture_upload.image;
    }

    /// TODO: Taskgraph this shit.
    recorder.pipeline_barrier({
        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_access = daxa::AccessConsts::READ_WRITE,
    });

    return recorder.complete_current_commands();
}

auto Scene::create_mesh_acceleration_structures() -> daxa::ExecutableCommandList
{
    u64 const scratch_buffer_offset_alignment =
        _device.properties().acceleration_structure_properties.value().min_acceleration_structure_scratch_offset_alignment;

    u64 current_scratch_buffer_offset = 0;
    auto const scratch_device_address = _device.buffer_device_address(_gpu_mesh_acceleration_structure_build_scratch_buffer.id()).value();
    std::vector<daxa::BlasTriangleGeometryInfo> build_geometries = {};
    // Reserve is nessecary to avoid memory resising.
    // We store pointers to the vector memory elsewhere, IT MUST NOT REALLOCATE!
    build_geometries.reserve(MAX_MESH_BLAS_BUILDS_PER_FRAME);
    std::vector<daxa::BlasBuildInfo> build_infos = {};
    while (!_mesh_as_build_queue.empty() && build_geometries.size() < MAX_MESH_BLAS_BUILDS_PER_FRAME)
    {
        auto const mesh_index = _mesh_as_build_queue.back();
        auto const lod = mesh_index % MAX_MESHES_PER_LOD_GROUP;
        auto const lod_group_index = mesh_index / MAX_MESHES_PER_LOD_GROUP;
        MeshLodGroupManifestEntry & mesh_lod_group = _mesh_lod_group_manifest.at(lod_group_index);

        bool is_alpha_discard = false;
        if (mesh_lod_group.material_index.has_value())
        {
            is_alpha_discard = _material_manifest.at(mesh_lod_group.material_index.value()).alpha_discard_enabled;
        }

        GPUMesh const & mesh = mesh_lod_group.runtime.value().lods[lod];

        // Must store geometries in vector as the memory address must persist for outside of the loop!
        build_geometries.push_back(daxa::BlasTriangleGeometryInfo{
            .vertex_data = mesh.vertex_positions,
            .max_vertex = mesh.vertex_count - 1,
            .index_data = mesh.primitive_indices,
            .count = static_cast<daxa_u32>(mesh.primitive_count),
            .flags = is_alpha_discard ? daxa::GeometryFlagBits::NONE : daxa::GeometryFlagBits::OPAQUE,
        });
        auto & geometry = build_geometries.back();
        daxa::BlasBuildInfo blas_build_info = daxa::BlasBuildInfo{
            .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE |
                     daxa::AccelerationStructureBuildFlagBits::ALLOW_DATA_ACCESS,
            .geometries = daxa::Span<daxa::BlasTriangleGeometryInfo const>(
                &geometry, 1ull),
        };

        auto const build_size_info = _device.blas_build_sizes(blas_build_info);
        DBG_ASSERT_TRUE_M(build_size_info.build_scratch_size < std::numeric_limits<u32>::max(), "Round up to multiple only handles 32bit values");
        DBG_ASSERT_TRUE_M(scratch_buffer_offset_alignment < std::numeric_limits<u32>::max(), "Round up to multiple only handles 32 bit values");
        u64 aligned_scratch_size = round_up_to_multiple(s_cast<u32>(build_size_info.build_scratch_size), s_cast<u32>(scratch_buffer_offset_alignment));
        DBG_ASSERT_TRUE_M(aligned_scratch_size < _gpu_mesh_acceleration_structure_build_scratch_buffer_size,
            "[ERROR][Scene::create_and_record_build_as()] Mesh group too big for the scratch buffer - increase scratch buffer size");

        bool const fits_scratch = (current_scratch_buffer_offset + aligned_scratch_size <= _gpu_mesh_acceleration_structure_build_scratch_buffer_size);
        if (!fits_scratch) { break; }

        blas_build_info.scratch_data = scratch_device_address + current_scratch_buffer_offset;
        current_scratch_buffer_offset += aligned_scratch_size;
        DBG_ASSERT_TRUE_M(build_size_info.acceleration_structure_size < std::numeric_limits<u32>::max(), "Round up to multiple only handles 32 bit values");
        auto const aligned_accel_structure_size = round_up_to_multiple(s_cast<u32>(build_size_info.acceleration_structure_size), 256u);
        auto blas = _device.create_blas({
            .size = aligned_accel_structure_size,
            .name = mesh_lod_group.name.empty() ? "mesh_lod_group blas" : mesh_lod_group.name.c_str(),
        });
        blas_build_info.dst_blas = blas;
        mesh_lod_group.runtime->blas_lods[lod] = blas;

        build_infos.push_back(std::move(blas_build_info));
        _mesh_as_build_queue.pop_back();
    }

    auto recorder = _device.create_command_recorder({daxa::QueueType::MAIN});
    if (!build_infos.empty())
    {
        DEBUG_MSG(fmt::format("[DEBUG][Scene::create_and_record_build_as()] Building {} blases this frame", build_infos.size()));
        recorder.build_acceleration_structures({.blas_build_infos = {build_infos.data(), build_infos.size()}});
    }
    recorder.pipeline_barrier({
        .src_access = daxa::AccessConsts::ACCELERATION_STRUCTURE_BUILD_WRITE,
        .dst_access = daxa::AccessConsts::ACCELERATION_STRUCTURE_BUILD_READ,
    });

    return recorder.complete_current_commands();
}

void Scene::build_tlas_from_mesh_instances(daxa::CommandRecorder & recorder, daxa::TlasId tlas)
{
    auto & mesh_instances = this->current_frame_mesh_instances;

    std::vector<daxa_BlasInstanceData> blas_instances = {};
    blas_instances.reserve(mesh_instances.mesh_instances.size());
    for (u32 mesh_inst_i = 0; mesh_inst_i < mesh_instances.mesh_instances.size(); ++mesh_inst_i)
    {
        MeshInstance const & mesh_instance = mesh_instances.mesh_instances[mesh_inst_i];
        auto const lod = mesh_instance.mesh_index % MAX_MESHES_PER_LOD_GROUP;
        auto const lod_group = mesh_instance.mesh_index / MAX_MESHES_PER_LOD_GROUP;

        if (!_mesh_lod_group_manifest[lod_group].runtime.has_value()) { continue; }
        if (_mesh_lod_group_manifest[lod_group].runtime.value().blas_lods[lod].is_empty()) { continue; }

        RenderEntity const * render_entity = _render_entities.slot_by_index(mesh_instance.entity_index);
        auto const & t = render_entity->combined_transform;
        blas_instances.push_back(daxa_BlasInstanceData{
            .transform = {
                {t[0][0], t[1][0], t[2][0], t[3][0]},
                {t[0][1], t[1][1], t[2][1], t[3][1]},
                {t[0][2], t[1][2], t[2][2], t[3][2]},
            },
            .instance_custom_index = mesh_inst_i,
            .mask = 0xFF,
            .instance_shader_binding_table_record_offset = ((mesh_instance.flags & MESH_INSTANCE_FLAG_MASKED) != 0) ? 1u : 0u,
            .flags = 0,
            .blas_device_address = _device.blas_device_address(_mesh_lod_group_manifest[lod_group].runtime.value().blas_lods[lod]).value(),
        });
    }

    daxa::BufferId blas_instances_buffer = {};
    if (!blas_instances.empty())
    {
        blas_instances_buffer = _device.create_buffer({.size = sizeof(daxa_BlasInstanceData) * blas_instances.size(),
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
            .name = "blas instances buffer"});
        recorder.destroy_buffer_deferred(blas_instances_buffer);

        std::memcpy(_device.buffer_host_address_as<daxa_BlasInstanceData>(blas_instances_buffer).value(),
            blas_instances.data(),
            blas_instances.size() * sizeof(daxa_BlasInstanceData));
    }
    else
    {
        return;
    }

    auto tlas_blas_instances_infos = std::array{daxa::TlasInstanceInfo{
        .data = blas_instances.empty() ? daxa::DeviceAddress{0ull} : _device.buffer_device_address(blas_instances_buffer).value(),
        .count = s_cast<u32>(blas_instances.size()),
        .is_data_array_of_pointers = false,
        .flags = daxa::GeometryFlagBits::NONE,
    }};

    auto tlas_build_info = daxa::TlasBuildInfo{
        .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE,
        .instances = tlas_blas_instances_infos,
    };

    daxa::AccelerationStructureBuildSizesInfo const tlas_build_sizes = _device.tlas_build_sizes(tlas_build_info);

    DAXA_DBG_ASSERT_TRUE_M(tlas_build_sizes.acceleration_structure_size <= _device.info(tlas).value().size, "Tlas Size Overflow");

    DBG_ASSERT_TRUE_M(tlas_build_sizes.build_scratch_size < _gpu_tlas_build_scratch_buffer_size,
        "[ERROR][Scene::build_tlas_from_mesh_instances] Tlas too big for scratch buffer - create bigger scratch buffer");

    daxa::DeviceAddress scratch_device_address = _device.buffer_device_address(_gpu_tlas_build_scratch_buffer.id()).value();
    tlas_build_info.dst_tlas = tlas;
    tlas_build_info.scratch_data = scratch_device_address;

    recorder.build_acceleration_structures({.tlas_build_infos = std::array{tlas_build_info}});
    recorder.pipeline_barrier({
        .src_access = daxa::AccessConsts::ACCELERATION_STRUCTURE_BUILD_READ_WRITE,
        .dst_access = daxa::AccessConsts::READ_WRITE,
    });
}

auto Scene::process_entities(RenderGlobalData & render_data) -> CPUSceneInstances
{
    CPUSceneInstances ret = {};

    auto * const gpu_point_lights_write_ptr = _device.buffer_host_address_as<GPUPointLight>(_gpu_point_lights.id()).value();
    auto * const gpu_spot_lights_write_ptr = _device.buffer_host_address_as<GPUSpotLight>(_gpu_spot_lights.id()).value();

    for (u32 entity_i = 0; entity_i < _render_entities.capacity(); ++entity_i)
    {
        RenderEntity * r_ent = _render_entities.slot_by_index(entity_i);
        bool const is_entity_dirty = r_ent->dirty;
        r_ent->dirty = false;

        if (r_ent != nullptr && r_ent->cloud_volume_index.has_value())
        {
            DBG_ASSERT_TRUE_M(r_ent->type == EntityType::CLOUD_VOLUME, "IMPOSSIBLE CASE! Only cloud volume entities can have cloud volume index");
            auto const & cloud_volume = _cloud_volumes.at(r_ent->cloud_volume_index.value());

            // ===================================== Calculate instance AABB =======================================
            {
                const f32vec3 cloud_bottom_left_corner = s_cast<f32vec3>((mat_4x3_to_4x4(r_ent->combined_transform) * f32vec4(0.0f, 0.0f, 0.0f, 1.0f)));
                const f32vec3 cloud_top_right_corner = s_cast<f32vec3>((mat_4x3_to_4x4(r_ent->combined_transform) * f32vec4(1.0f, 1.0f, 1.0f, 1.0f)));

                f32vec3 const instance_aabb_size = cloud_top_right_corner - cloud_bottom_left_corner;
                f32vec3 const instance_aabb_center = cloud_bottom_left_corner + (instance_aabb_size * f32vec3(0.5f));
                ret.cloud_volume_instances.instance_aabbs.push_back({
                    .center = std::bit_cast<daxa_f32vec3>(instance_aabb_center),
                    .size = std::bit_cast<daxa_f32vec3>(instance_aabb_size),
                });
            }

            // ===================================== Fill out cloud instance data =======================================
            {
                CloudVolumeInstance cloud_volume_instance = {};
                cloud_volume_instance.transform = std::bit_cast<daxa_f32mat4x3>(r_ent->combined_transform);
                cloud_volume_instance.albedo = 1.0f;
                cloud_volume_instance.density_scale = 0.1f;

                cloud_volume_instance.cloud_data_texture = _material_texture_manifest.at(cloud_volume.data_texture_manifest_index).runtime_texture.value_or(daxa::ImageId{}).default_view();
                cloud_volume_instance.cloud_sdf_texture = _material_texture_manifest.at(cloud_volume.sdf_texture_manifest_index).runtime_texture.value_or(daxa::ImageId{}).default_view();
                cloud_volume_instance.detail_noise_texture = _material_texture_manifest.at(cloud_volume.detail_noise_texture_manifest_index).runtime_texture.value_or(daxa::ImageId{}).default_view();

                cloud_volume_instance.texture_size = {0u, 0u, 0u};
                if(_material_texture_manifest.at(cloud_volume.data_texture_manifest_index).loaded())
                {
                    daxa::ImageId cloud_data_texture = _material_texture_manifest.at(cloud_volume.data_texture_manifest_index).runtime_texture.value();
                    daxa::ImageInfo const & cloud_data_texture_info = _device.image_info(cloud_data_texture).value();
                    cloud_volume_instance.texture_size = {cloud_data_texture_info.size.x, cloud_data_texture_info.size.y, cloud_data_texture_info.size.z};
                }
                ret.cloud_volume_instances.instances.push_back(cloud_volume_instance);
            }

        }

        if (r_ent != nullptr && r_ent->light_index.has_value())
        {
            if (r_ent->type == EntityType::POINT_LIGHT)
            {
                PointLight & point_light = _point_lights.at(r_ent->light_index.value());
                point_light.position = r_ent->combined_transform[3];

                gpu_point_lights_write_ptr[r_ent->light_index.value()] = GPUPointLight{
                    .position = std::bit_cast<daxa_f32vec3>(point_light.position),
                    .color = std::bit_cast<daxa_f32vec3>(point_light.color),
                    .intensity = point_light.intensity,
                    .cutoff = point_light.cutoff,
                };
            }
            else if (r_ent->type == EntityType::SPOT_LIGHT)
            {
                SpotLight & spot_light = _spot_lights.at(r_ent->light_index.value());
                spot_light.transform = r_ent->combined_transform;

                glm::mat4 transform4 = glm::mat4(
                    glm::vec4(spot_light.transform[0], 0.0f),
                    glm::vec4(spot_light.transform[1], 0.0f),
                    glm::vec4(spot_light.transform[2], 0.0f),
                    glm::vec4(spot_light.transform[3], 1.0f));
                f32vec3 const spot_direction = transform4 * f32vec4(0.0f, 0.0f, -1.0f, 0.0f);

                gpu_spot_lights_write_ptr[r_ent->light_index.value()] = GPUSpotLight{
                    .transform = std::bit_cast<daxa_f32mat4x3>(spot_light.transform),
                    .position = std::bit_cast<daxa_f32vec3>(spot_light.transform[3]),
                    .direction = std::bit_cast<daxa_f32vec3>(spot_direction),
                    .color = std::bit_cast<daxa_f32vec3>(spot_light.color),
                    .intensity = spot_light.intensity,
                    .cutoff = spot_light.cutoff,
                    .inner_cone_angle = spot_light.inner_cone_angle,
                    .outer_cone_angle = spot_light.outer_cone_angle,
                };
            }
        }

        if (r_ent != nullptr && r_ent->mesh_group_manifest_index.has_value())
        {
            usize mesh_group_index = r_ent->mesh_group_manifest_index.value();
            MeshGroupManifestEntry & mesh_group = _mesh_group_manifest.at(mesh_group_index);
            bool const is_mesh_group_loaded = (mesh_group.loaded_mesh_lod_groups == mesh_group.mesh_lod_group_count);
            bool const is_mesh_group_just_loaded = !mesh_group.fully_loaded_last_frame && is_mesh_group_loaded;
            mesh_group.fully_loaded_last_frame = is_mesh_group_loaded;

            // Process all fully loaded mesh groups
            if (is_mesh_group_loaded)
            {
                auto const mesh_lod_group_indices_meshgroup_offset = mesh_group.mesh_lod_group_manifest_indices_array_offset;
                for (u32 in_mesh_group_index = 0; in_mesh_group_index < mesh_group.mesh_lod_group_count; in_mesh_group_index++)
                {
                    u32 const mesh_lod_group_manifest_index = _mesh_lod_group_manifest_indices.at(mesh_lod_group_indices_meshgroup_offset + in_mesh_group_index);
                    auto const & mesh_lod_group = _mesh_lod_group_manifest.at(mesh_lod_group_manifest_index);
                    bool is_alpha_discard = false;
                    bool is_blend = false;
                    if (mesh_lod_group.material_index.has_value())
                    {
                        auto const & material = _material_manifest.at(mesh_lod_group.material_index.value());
                        is_alpha_discard = material.alpha_discard_enabled;
                        is_blend = material.blend_enabled;
                    }

                    // Put this mesh into appropriate drawlist for prepass
                    u32 const draw_list_type = is_alpha_discard ? PREPASS_DRAW_LIST_MASKED : PREPASS_DRAW_LIST_OPAQUE;

                    ret.mesh_instances.prepass_draw_lists[draw_list_type].push_back(static_cast<u32>(ret.mesh_instances.mesh_instances.size()));

                    // If the mesh loaded for the first time, it needs to invalidate VSM
                    if (is_mesh_group_just_loaded || is_entity_dirty)
                    {
                        ret.mesh_instances.vsm_invalidate_draw_list.push_back(static_cast<u32>(ret.mesh_instances.mesh_instances.size()));
                    }

                    u32 mesh_index = select_lod(render_data, mesh_lod_group, mesh_lod_group_manifest_index, r_ent);

                    // Because this mesh will be referenced by the prepass drawlist, we need also need it's appropriate mesh instance data
                    ret.mesh_instances.mesh_instances.push_back({
                        .entity_index = entity_i,
                        .mesh_index = mesh_index,
                        .in_mesh_group_index = in_mesh_group_index,
                        .mesh_group_index = static_cast<u32>(mesh_group_index),
                        .flags = s_cast<daxa_u32>(is_alpha_discard ? MESH_INSTANCE_FLAG_MASKED : MESH_INSTANCE_FLAG_OPAQUE),
                    });
                }
            }
        }
    }
    return ret;
}

void Scene::write_gpu_mesh_instances_buffer(CPUMeshInstances const & cpu_mesh_instances)
{
    // Calculate offsets into buffer and required size:
    usize offset = {};
    MeshInstancesBufferHead buffer_head = {};
    offset += sizeof(MeshInstancesBufferHead);
    buffer_head.instances = offset;
    buffer_head.count = static_cast<u32>(cpu_mesh_instances.mesh_instances.size());
    offset += sizeof(MeshInstance) * cpu_mesh_instances.mesh_instances.size();
    for (int i = 0; i < PREPASS_DRAW_LIST_TYPE_COUNT; ++i)
    {
        buffer_head.prepass_draw_lists[i].instances = offset;
        buffer_head.prepass_draw_lists[i].count = static_cast<u32>(cpu_mesh_instances.prepass_draw_lists[i].size());
        offset += sizeof(daxa_u32) * cpu_mesh_instances.prepass_draw_lists[i].size();
    }
    buffer_head.vsm_invalidate_draw_list.instances = offset;
    buffer_head.vsm_invalidate_draw_list.count = static_cast<u32>(cpu_mesh_instances.vsm_invalidate_draw_list.size());
    offset += sizeof(daxa_u32) * cpu_mesh_instances.vsm_invalidate_draw_list.size();
    usize const required_size = offset;

    // TODO: Allocate this into a ring buffer.
    // Allocate buffer
    if (!mesh_instances_buffer.id().is_empty())
    {
        _device.destroy_buffer(mesh_instances_buffer.id());
    }
    mesh_instances_buffer.set_buffer(_device.create_buffer({
        .size = required_size,
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "cpu_mesh_instances_buffer",
    }));
    daxa::DeviceAddress device_address = _device.buffer_device_address(mesh_instances_buffer.id()).value();
    std::byte * host_address = _device.buffer_host_address(mesh_instances_buffer.id()).value();

    // Write Buffer, add address on offsets and counts:
    cpu_mesh_instance_counts = {};
    usize const mesh_instances_size = sizeof(MeshInstance) * cpu_mesh_instances.mesh_instances.size();
    std::memcpy(host_address + buffer_head.instances, cpu_mesh_instances.mesh_instances.data(), mesh_instances_size);

    buffer_head.instances += device_address;
    cpu_mesh_instance_counts.mesh_instance_count = s_cast<u32>(cpu_mesh_instances.mesh_instances.size());
    for (int draw_list_type = 0; draw_list_type < PREPASS_DRAW_LIST_TYPE_COUNT; ++draw_list_type)
    {
        std::memcpy(
            host_address + buffer_head.prepass_draw_lists[draw_list_type].instances,
            cpu_mesh_instances.prepass_draw_lists[draw_list_type].data(),
            sizeof(u32) * cpu_mesh_instances.prepass_draw_lists[draw_list_type].size());
        buffer_head.prepass_draw_lists[draw_list_type].instances += device_address;
        cpu_mesh_instance_counts.prepass_instance_counts[draw_list_type] = s_cast<u32>(cpu_mesh_instances.prepass_draw_lists[draw_list_type].size());
    }
    std::memcpy(
        host_address + buffer_head.vsm_invalidate_draw_list.instances,
        cpu_mesh_instances.vsm_invalidate_draw_list.data(),
        sizeof(u32) * cpu_mesh_instances.vsm_invalidate_draw_list.size());
    buffer_head.vsm_invalidate_draw_list.instances += device_address;

    std::memcpy(host_address, &buffer_head, sizeof(MeshInstancesBufferHead));

    cpu_mesh_instance_counts.vsm_invalidate_instance_count = s_cast<u32>(cpu_mesh_instances.vsm_invalidate_draw_list.size());
}

void Scene::write_gpu_cloud_volume_instances_buffer(CPUCloudVolumeInstaces const & cpu_cloud_volume_instances)
{
    DBG_ASSERT_TRUE_M(cpu_cloud_volume_instances.instance_aabbs.size() == cpu_cloud_volume_instances.instances.size(),
                      "Each cloud volume instance must have a corresponding AABB");

    u32 const instance_count = s_cast<u32>(cpu_cloud_volume_instances.instances.size());

    // Calculate offsets into buffer and required size:
    // The offsets are relative to the start of the buffer.
    // Later they are added to the device address of the buffer to form the actual location.
    usize offset = {};
    CloudVolumeInstancesBufferHead buffer_head = {};
    offset += sizeof(CloudVolumeInstancesBufferHead);

    buffer_head.count = instance_count;

    buffer_head.instance_aabbs = offset;
    offset += sizeof(AABB) * instance_count;

    buffer_head.instances = offset;
    offset += sizeof(CloudVolumeInstance) * instance_count;
    usize const required_size = offset;

    // TODO: Allocate this into a ring buffer.
    // Allocate buffer
    if (!cloud_volume_instances_buffer.id().is_empty())
    {
        _device.destroy_buffer(cloud_volume_instances_buffer.id());
    }
    cloud_volume_instances_buffer.set_buffer(_device.create_buffer({
        .size = required_size,
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "cpu_cloud_volume_instances_buffer",
    }));
    daxa::DeviceAddress device_address = _device.buffer_device_address(cloud_volume_instances_buffer.id()).value();
    std::byte * host_address = _device.buffer_host_address(cloud_volume_instances_buffer.id()).value();

    // Write Buffer, add address on offsets and counts:
    usize const cloud_volume_aabbs_size = sizeof(AABB) * instance_count;
    std::memcpy(host_address + buffer_head.instance_aabbs, cpu_cloud_volume_instances.instance_aabbs.data(), cloud_volume_aabbs_size);

    usize const cloud_volume_instances_size = sizeof(CloudVolumeInstance) * instance_count;
    std::memcpy(host_address + buffer_head.instances, cpu_cloud_volume_instances.instances.data(), cloud_volume_instances_size);

    buffer_head.instance_aabbs += device_address;
    buffer_head.instances += device_address;
    std::memcpy(host_address, &buffer_head, sizeof(CloudVolumeInstancesBufferHead));
}

void Scene::clear(std::unique_ptr<ThreadPool> & thread_pool, std::unique_ptr<AssetProcessor> & asset_processor)
{
    // for (auto &task : scene_load_tasks)
    // {
    //     thread_pool->block_on(task);
    // }
    asset_processor->clear();

    // NOTE(grundlett): Destroy all GPU resources (from destructor)
    {
        if (!_gpu_mesh_group_indices_array_buffer.is_empty()) { _device.destroy_buffer(_gpu_mesh_group_indices_array_buffer); }
        if (!_scene_blas.is_empty()) { _device.destroy_blas(_scene_blas); }

        for (auto & mesh_group : _mesh_group_manifest)
        {
            if (!mesh_group.blas.is_empty())
            {
                _device.destroy_blas(mesh_group.blas);
            }
        }

        for (auto & mesh : _mesh_lod_group_manifest)
        {
            if (mesh.runtime.has_value())
            {
                for (daxa_u32 lod = 0; lod < mesh.runtime.value().lod_count; ++lod)
                {
                    _device.destroy_buffer(std::bit_cast<daxa::BufferId>(mesh.runtime.value().lods[lod].mesh_buffer));
                    if (!mesh.runtime.value().blas_lods[lod].is_empty())
                    {
                        _device.destroy_blas(mesh.runtime.value().blas_lods[lod]);
                    }
                }
            }
        }

        for (auto & texture : _material_texture_manifest)
        {
            if (texture.runtime_texture.has_value())
            {
                _device.destroy_image(std::bit_cast<daxa::ImageId>(texture.runtime_texture.value()));
            }
            // if (texture.secondary_runtime_texture.has_value())
            // {
            //     _device.destroy_image(std::bit_cast<daxa::ImageId>(texture.secondary_runtime_texture.value()));
            // }
        }

        if (!mesh_instances_buffer.id().is_empty())
        {
            _device.destroy_buffer(mesh_instances_buffer.id());
        }
        mesh_instances_buffer.set_buffer({});
    }

    // NOTE(grundlett): clear all state
    {
        _render_entities.clear();
        _dirty_render_entities.clear();
        _modified_render_entities.clear();
        _newly_completed_mesh_groups.clear();
        _mesh_as_build_queue.clear();

        _root_render_entities.clear();
        _material_texture_manifest.clear();
        _material_manifest.clear();
        _mesh_lod_group_manifest.clear();
        _mesh_lod_group_manifest_indices.clear();
        _mesh_group_manifest.clear();
        _point_lights.clear();
        _spot_lights.clear();

        // Discard any pending dirty indices; the manifests they referred to are gone.
        _dirty_material_manifest.drain();
        _dirty_mesh_lod_group_manifest.drain();
        _dirty_mesh_group_manifest.drain();
        _dirty_material_texture_manifest.drain();
        _dirty_mesh_lod_group_streaming.drain();
        // In-flight stream tasks reference manifest indices that are about to be invalid; drop them.
        _inflight_texture_streams.clear();
        _inflight_mesh_streams.clear();
    }
}
