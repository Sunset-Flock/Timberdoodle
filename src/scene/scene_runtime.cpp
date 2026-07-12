#include "scene_runtime.hpp"

#include <algorithm>

#include <fmt/format.h>

#include "importers/gltf_importer.hpp"

SceneRuntime::SceneRuntime(
    daxa::Device device,
    GPUContext * gpu_context,
    std::unique_ptr<ThreadPool> & thread_pool,
    std::unique_ptr<AssetProcessor> & asset_processor)
    : _scene{device, gpu_context},
      _thread_pool{thread_pool},
      _asset_processor{asset_processor},
      _device{std::move(device)}
{
}

SceneRuntime::~SceneRuntime() = default;

void SceneRuntime::request_import(std::filesystem::path const & path)
{
    if (!path.has_filename() || !path.has_parent_path())
    {
        return;
    }
    DBG_ASSERT_TRUE_M(_pending_scene_import == nullptr, "Only one scene import may be in flight at a time (poll guards this)");

    auto import_task = std::make_shared<GltfImportTask>(&_scene, Scene::LoadManifestInfo{
        .root_path = path.parent_path(),
        .asset_name = path.filename(),
        .thread_pool = _thread_pool,
        .asset_processor = _asset_processor,
    });
    _thread_pool->async_dispatch(import_task, TaskPriority::LOW);
    _pending_scene_import = std::move(import_task);
}

void SceneRuntime::poll(std::string & desired_scene_path)
{
    if (_pending_scene_import == nullptr)
    {
        // No import in flight - start one if a scene load is requested. While an import runs the
        // path is left untouched (a request made mid-import is not dropped, the newest one wins).
        if (!desired_scene_path.empty())
        {
            fmt::print("Requested load: {}\n", desired_scene_path);
            request_import(desired_scene_path);
            desired_scene_path.clear();
        }
        return;
    }

    GltfImportTask & import_task = *_pending_scene_import;
    if (!import_task.finished.load(std::memory_order_acquire))
    {
        return; // Still importing.
    }

    std::string const path_string = (import_task.info.root_path / import_task.info.asset_name).string();
    if (Scene::LoadManifestErrorCode const * err = std::get_if<Scene::LoadManifestErrorCode>(&import_task.result))
    {
        DEBUG_MSG(fmt::format("[WARN][SceneRuntime::poll()] Loading \"{}\" Error: {}", path_string, Scene::to_string(*err)));
    }
    else
    {
        apply_scene_metadata_batch(std::move(std::get<ImporterTaskResult::SceneMetadataBatch>(import_task.result)));
    }
    _pending_scene_import = nullptr;
}

void SceneRuntime::apply_scene_metadata_batch(ImporterTaskResult::SceneMetadataBatch batch)
{
    auto locked = _scene.lock();

    std::vector<u32> texture_local_to_global(batch.textures.size());
    for (u32 local_index = 0; local_index < s_cast<u32>(batch.textures.size()); ++local_index)
    {
        texture_local_to_global[local_index] = locked.add_texture(std::move(batch.textures[local_index]));
    }

    auto remap_texture_info = [&](std::optional<MaterialManifestEntry::TextureInfo> & info)
    {
        if (info.has_value())
        {
            info->tex_manifest_index = texture_local_to_global.at(info->tex_manifest_index);
        }
    };
    std::vector<u32> material_local_to_global(batch.materials.size());
    for (u32 local_index = 0; local_index < s_cast<u32>(batch.materials.size()); ++local_index)
    {
        MaterialManifestEntry material = std::move(batch.materials[local_index]);
        remap_texture_info(material.diffuse_info);
        remap_texture_info(material.opacity_mask_info);
        remap_texture_info(material.normal_info);
        remap_texture_info(material.roughness_metalness_info);
        material_local_to_global[local_index] = locked.add_material(std::move(material));
    }

    std::vector<u32> mesh_local_to_global(batch.mesh_lod_groups.size());
    for (u32 local_index = 0; local_index < s_cast<u32>(batch.mesh_lod_groups.size()); ++local_index)
    {
        MeshLodGroupManifestEntry mesh = std::move(batch.mesh_lod_groups[local_index]);
        if (mesh.material_index.has_value())
        {
            mesh.material_index = material_local_to_global.at(mesh.material_index.value());
        }
        mesh_local_to_global[local_index] = locked.add_mesh(std::move(mesh));
    }

    std::vector<u32> mesh_group_local_to_global(batch.mesh_groups.size());
    std::vector<u32> remapped_mesh_indices = {};
    for (u32 local_index = 0; local_index < s_cast<u32>(batch.mesh_groups.size()); ++local_index)
    {
        ImporterTaskResult::SceneMetadataBatch::MeshGroup const & mesh_group = batch.mesh_groups[local_index];
        remapped_mesh_indices.clear();
        remapped_mesh_indices.reserve(mesh_group.mesh_lod_group_indices.size());
        for (u32 const local_mesh_index : mesh_group.mesh_lod_group_indices)
        {
            remapped_mesh_indices.push_back(mesh_local_to_global.at(local_mesh_index));
        }
        mesh_group_local_to_global[local_index] = locked.add_mesh_group(remapped_mesh_indices, mesh_group.name);
    }

    std::vector<u32> point_light_local_to_global(batch.point_lights.size());
    for (u32 local_index = 0; local_index < s_cast<u32>(batch.point_lights.size()); ++local_index)
    {
        point_light_local_to_global[local_index] = locked.add_point_light(batch.point_lights[local_index]);
    }
    std::vector<u32> spot_light_local_to_global(batch.spot_lights.size());
    for (u32 local_index = 0; local_index < s_cast<u32>(batch.spot_lights.size()); ++local_index)
    {
        spot_light_local_to_global[local_index] = locked.add_spot_light(batch.spot_lights[local_index]);
    }

    // Entity ids must all exist before the tree's parent/child/sibling links (below) can reference them,
    // so every local entity gets an empty slot up front.
    std::vector<RenderEntityId> entity_local_to_global = {};
    entity_local_to_global.reserve(batch.entities.size());
    for (u32 local_index = 0; local_index < s_cast<u32>(batch.entities.size()); ++local_index)
    {
        entity_local_to_global.push_back(locked.add_entity({}));
    }
    for (u32 local_index = 0; local_index < s_cast<u32>(batch.entities.size()); ++local_index)
    {
        ImporterTaskResult::SceneMetadataBatch::Entity const & local_entity = batch.entities[local_index];
        RenderEntity entity = local_entity.entity;
        entity.parent = local_entity.parent_index.has_value()
                             ? std::optional{entity_local_to_global.at(local_entity.parent_index.value())}
                             : std::nullopt;
        entity.first_child = local_entity.first_child_index.has_value()
                                  ? std::optional{entity_local_to_global.at(local_entity.first_child_index.value())}
                                  : std::nullopt;
        entity.next_sibling = local_entity.next_sibling_index.has_value()
                                   ? std::optional{entity_local_to_global.at(local_entity.next_sibling_index.value())}
                                   : std::nullopt;
        if (entity.mesh_group_manifest_index.has_value())
        {
            entity.mesh_group_manifest_index = mesh_group_local_to_global.at(entity.mesh_group_manifest_index.value());
        }
        if (entity.light_index.has_value())
        {
            switch (entity.type)
            {
                case EntityType::POINT_LIGHT: entity.light_index = point_light_local_to_global.at(entity.light_index.value()); break;
                case EntityType::SPOT_LIGHT:  entity.light_index = spot_light_local_to_global.at(entity.light_index.value()); break;
                case EntityType::ROOT:
                case EntityType::TRANSFORM:
                case EntityType::CAMERA:
                case EntityType::MESHGROUP:
                case EntityType::CLOUD_VOLUME:
                case EntityType::UNKNOWN:
                    DBG_ASSERT_TRUE_M(false, "Entity has a light index but is not a light type");
                    break;
            }
        }
        locked.update_entity(entity_local_to_global.at(local_index), entity);
    }
    locked.add_root_entity(entity_local_to_global.at(batch.root_entity_index));
}

// Moves the accumulated indices out of a dirty-index vector and leaves it empty. Only ever called
// from update, which holds _manifest_mutex for its whole duration - the same lock the Locked
// add_* methods push indices under - so no lock is needed here.
static auto drain_dirty_indices(std::vector<u32> & indices) -> std::vector<u32>
{
    std::vector<u32> out = std::move(indices);
    indices.clear();
    return out;
}

auto SceneRuntime::update(UpdateInfo const & info) -> daxa::ExecutableCommandList
{
    // Touches manifests/entities throughout; importer worker threads may be appending concurrently.
    std::lock_guard<std::mutex> lock{*_scene._manifest_mutex};

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
        TextureManifestEntry & texture = _scene._texture_manifest.at(task.texture_manifest_index);
        texture.runtime_data.image = task.result;
        for (TextureManifestEntry::MaterialManifestIndex const & ref : texture.material_manifest_indices)
        {
            _scene._dirty_material_indices.push_back(ref.material_manifest_index);
        }
        it = _inflight_texture_streams.erase(it);
    }
    // 2. Spawn a stream task for every newly dirtied texture. (No-op if there is no thread pool, e.g.
    //    at shutdown - those textures simply never become resident, which is fine.)
    if (info.thread_pool != nullptr)
    {
        for (u32 const texture_index : drain_dirty_indices(_scene._dirty_texture_indices))
        {
            auto task = std::make_shared<TextureStreamTask>();
            task->chunk_count = 1;
            task->device = _device;
            task->artifact = _scene._texture_manifest.at(texture_index).streamer_data;
            task->texture_manifest_index = texture_index;
            info.thread_pool->async_dispatch(task, TaskPriority::LOW);
            _inflight_texture_streams.push_back(std::move(task));
        }
    }

    // --- Mesh residency (async) ---
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
        _scene._mesh_lod_group_manifest.at(upload.mesh_lod_manifest_index).runtime_data = MeshLodGroupManifestEntry::Runtime{
            .lods = upload.lods,
            .lod_count = upload.lod_count,
        };
        _scene._dirty_mesh_lod_group_indices.push_back(upload.mesh_lod_manifest_index);
        it = _inflight_mesh_streams.erase(it);
    }
    // 2. Spawn a stream task for every newly requested mesh. (No-op without a thread pool, e.g. shutdown.)
    if (info.thread_pool != nullptr)
    {
        for (u32 const mesh_index : drain_dirty_indices(_scene._dirty_mesh_lod_group_streaming_indices))
        {
            MeshLodGroupManifestEntry const & entry = _scene._mesh_lod_group_manifest.at(mesh_index);
            auto task = std::make_shared<MeshStreamTask>();
            task->chunk_count = 1;
            task->device = _device;
            task->artifact = entry.streamer_data;
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
    if (_scene._dirty_render_entities.size() > 0)
    {
        usize required_staging_size = 0;
        required_staging_size += sizeof(GPUEntityMetaData);                                       // _gpu_entity_meta
        required_staging_size += sizeof(daxa_f32mat4x3) * (_scene._dirty_render_entities.size()); // _gpu_entity_transforms
        required_staging_size += sizeof(daxa_f32mat4x3) * (_scene._dirty_render_entities.size()); // _gpu_entity_combined_transforms
        required_staging_size += sizeof(GPUMeshGroup) * (_scene._dirty_render_entities.size());   // _gpu_entity_mesh_groups
        staging_buffer = _device.create_buffer({
            .size = required_staging_size,
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "entities update staging",
        });
        recorder.destroy_buffer_deferred(staging_buffer);
        host_ptr = _device.buffer_host_address(staging_buffer).value();
        *r_cast<GPUEntityMetaData *>(host_ptr) = {.entity_count = s_cast<u32>(_scene._render_entities.size())};
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _scene._gpu_entity_meta.id(),
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
                glm::vec4(_scene._render_entities.slot(parent.value())->transform[0], 0.0f),
                glm::vec4(_scene._render_entities.slot(parent.value())->transform[1], 0.0f),
                glm::vec4(_scene._render_entities.slot(parent.value())->transform[2], 0.0f),
                glm::vec4(_scene._render_entities.slot(parent.value())->transform[3], 1.0f));
            combined_transform4 = parent_transform4 * combined_transform4;
            combined_parent_transform4 = parent_transform4 * combined_parent_transform4;
            parent = _scene._render_entities.slot(parent.value())->parent;
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
            .dst_buffer = _scene._gpu_entity_transforms.id(),
            .src_offset = offset + offsetof(RenderEntityUpdateStagingMemoryView, transform),
            .dst_offset = sizeof(glm::mat4x3) * entity_index,
            .size = sizeof(glm::mat4x3),
        });
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _scene._gpu_entity_combined_transforms.id(),
            .src_offset = offset + offsetof(RenderEntityUpdateStagingMemoryView, combined_transform),
            .dst_offset = sizeof(glm::mat4x3) * entity_index,
            .size = sizeof(glm::mat4x3),
        });
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _scene._gpu_entity_mesh_groups.id(),
            .src_offset = offset + offsetof(RenderEntityUpdateStagingMemoryView, mesh_group_manifest_index),
            .dst_offset = sizeof(u32) * entity_index,
            .size = sizeof(u32),
        });
        return combined_parent_transform4;
    };
    for (u32 i = 0; i < _scene._dirty_render_entities.size(); ++i)
    {
        u32 entity_index = _scene._dirty_render_entities[i].index;
        auto * entity = _scene._render_entities.slot(_scene._dirty_render_entities[i]);
        entity->dirty = true;
        update_entity(i, entity, entity_index);
    }

    _scene._dirty_render_entities.clear();

    // Drain the per-manifest dirty-index lists: the indices that were added/updated since the last
    // sync. We re-upload exactly these entries rather than assuming a contiguous tail of new entries.
    std::vector<u32> const dirty_mesh_groups = drain_dirty_indices(_scene._dirty_mesh_group_indices);
    std::vector<u32> const dirty_mesh_lod_groups = drain_dirty_indices(_scene._dirty_mesh_lod_group_indices);
    std::vector<u32> const dirty_materials = drain_dirty_indices(_scene._dirty_material_indices);

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
            staging_ptr[i].mesh_lod_group_count = _scene._mesh_group_manifest.at(mesh_group_manifest_idx).mesh_lod_group_count;
            recorder.copy_buffer_to_buffer({
                .src_buffer = mesh_group_staging_buffer,
                .dst_buffer = _scene._gpu_mesh_group_manifest.id(),
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
            MeshLodGroupManifestEntry & mesh_lod_group = _scene._mesh_lod_group_manifest.at(mesh_lod_manifest_index);

            std::array<GPUMesh, MAX_MESHES_PER_LOD_GROUP> lods = {};
            u32 lod_count = 0;
            if (mesh_lod_group.runtime_data.has_value())
            {
                lods = mesh_lod_group.runtime_data.value().lods;
                lod_count = mesh_lod_group.runtime_data.value().lod_count;
                DAXA_DBG_ASSERT_TRUE_M(lods[0].material_index == mesh_lod_group.material_index.value_or(INVALID_MANIFEST_INDEX), "IMPOSSIBLE CASE! material index MUST MATCH!");

                // Queue every newly resident LOD for a BLAS build.
                for (u32 lod = 0; lod < lod_count; ++lod)
                {
                    _mesh_as_build_queue.push_back(mesh_lod_manifest_index * MAX_MESHES_PER_LOD_GROUP + lod);
                }
                // Bump the owning mesh group's loaded count; if every mesh in it is now resident, mark it complete.
                // A mesh can become resident before it is assigned to any mesh group (add_mesh_group runs after
                // add_mesh), in which case there is nothing to bump yet - add_mesh_group counts it as loaded
                // once the group is created.
                if (mesh_lod_group.mesh_group_manifest_index.has_value())
                {
                    MeshGroupManifestEntry & mesh_group = _scene._mesh_group_manifest.at(mesh_lod_group.mesh_group_manifest_index.value());
                    mesh_group.loaded_mesh_lod_groups += 1;
                    bool is_completely_loaded = true;
                    u32 const range[] = {mesh_group.mesh_lod_group_manifest_indices_array_offset, mesh_group.mesh_lod_group_manifest_indices_array_offset + mesh_group.mesh_lod_group_count};
                    for (u32 mesh_idx_array_idx = range[0]; mesh_idx_array_idx < range[1]; mesh_idx_array_idx++)
                    {
                        if (!_scene._mesh_lod_group_manifest.at(_scene._mesh_lod_group_manifest_indices.at(mesh_idx_array_idx)).runtime_data.has_value())
                        {
                            is_completely_loaded = false;
                            break;
                        }
                    }
                    if (is_completely_loaded)
                    {
                        _scene._newly_completed_mesh_groups.push_back(mesh_lod_group.mesh_group_manifest_index.value());
                    }
                }
            }

            std::memcpy(mesh_staging_ptr + i * MAX_MESHES_PER_LOD_GROUP, lods.data(), sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP);
            mesh_lod_group_staging_ptr[i] = {.lod_count = lod_count};

            recorder.copy_buffer_to_buffer({
                .src_buffer = mesh_sync_staging_buffer,
                .dst_buffer = _scene._gpu_mesh_manifest.id(),
                .src_offset = i * sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP,
                .dst_offset = mesh_lod_manifest_index * sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP,
                .size = sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP,
            });
            recorder.copy_buffer_to_buffer({
                .src_buffer = mesh_sync_staging_buffer,
                .dst_buffer = _scene._gpu_mesh_lod_group_manifest.id(),
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
            return _scene._texture_manifest.at(info.value().tex_manifest_index).runtime_data.image.value_or(daxa::ImageId{});
        };

        // The normal map's BC5 encoding is deduced from its cooked texture format, not tracked through
        // the import: the shader needs to know whether to reconstruct Z from a two-channel normal map.
        auto normal_is_bc5_rg = [&](std::optional<MaterialManifestEntry::TextureInfo> const & info) -> bool
        {
            if (!info.has_value()) { return false; }
            return tido_format_is_bc5_rg(_scene._texture_manifest.at(info.value().tex_manifest_index).streamer_data.info.format);
        };

        for (u32 i = 0; i < dirty_material_count; ++i)
        {
            u32 const material_manifest_idx = dirty_materials[i];
            MaterialManifestEntry const & material = _scene._material_manifest.at(material_manifest_idx);
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
                .dst_buffer = _scene._gpu_material_manifest.id(),
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
        _scene._texture_manifest.at(texture_upload.texture_manifest_index).runtime_data.image = texture_upload.image;
    }

    /// TODO: Taskgraph this shit.
    recorder.pipeline_barrier({
        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_access = daxa::AccessConsts::READ_WRITE,
    });

    return recorder.complete_current_commands();
}

auto SceneRuntime::create_mesh_acceleration_structures() -> daxa::ExecutableCommandList
{
    // Reads the mesh/material manifests; importer worker threads may be appending concurrently.
    std::lock_guard<std::mutex> lock{*_scene._manifest_mutex};

    u64 const scratch_buffer_offset_alignment =
        _device.properties().acceleration_structure_properties.value().min_acceleration_structure_scratch_offset_alignment;

    u64 current_scratch_buffer_offset = 0;
    auto const scratch_device_address = _device.buffer_device_address(_scene._gpu_mesh_acceleration_structure_build_scratch_buffer.id()).value();
    std::vector<daxa::BlasTriangleGeometryInfo> build_geometries = {};
    // Reserve is nessecary to avoid memory resising.
    // We store pointers to the vector memory elsewhere, IT MUST NOT REALLOCATE!
    build_geometries.reserve(Scene::MAX_MESH_BLAS_BUILDS_PER_FRAME);
    std::vector<daxa::BlasBuildInfo> build_infos = {};
    while (!_mesh_as_build_queue.empty() && build_geometries.size() < Scene::MAX_MESH_BLAS_BUILDS_PER_FRAME)
    {
        auto const mesh_index = _mesh_as_build_queue.back();
        auto const lod = mesh_index % MAX_MESHES_PER_LOD_GROUP;
        auto const lod_group_index = mesh_index / MAX_MESHES_PER_LOD_GROUP;
        MeshLodGroupManifestEntry & mesh_lod_group = _scene._mesh_lod_group_manifest.at(lod_group_index);

        bool is_alpha_discard = false;
        if (mesh_lod_group.material_index.has_value())
        {
            is_alpha_discard = _scene._material_manifest.at(mesh_lod_group.material_index.value()).alpha_discard_enabled;
        }

        GPUMesh const & mesh = mesh_lod_group.runtime_data.value().lods[lod];

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
        DBG_ASSERT_TRUE_M(aligned_scratch_size < Scene::_gpu_mesh_acceleration_structure_build_scratch_buffer_size,
            "[ERROR][Scene::create_and_record_build_as()] Mesh group too big for the scratch buffer - increase scratch buffer size");

        bool const fits_scratch = (current_scratch_buffer_offset + aligned_scratch_size <= Scene::_gpu_mesh_acceleration_structure_build_scratch_buffer_size);
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
        mesh_lod_group.runtime_data->blas_lods[lod] = blas;

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
