#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "../timberdoodle.hpp"
#include "scene.hpp"
#include "importers/importer.hpp"
#include "importers/importer_task_result.hpp"
using namespace tido::types;

struct Importer;

/**
 * SceneRuntime is the center point of scene management. It owns a Scene (the passive manifest +
 * GPU-mirror container) and drives its whole import/stream lifecycle: it pushes ImporterTasks to the
 * Importer (an ImportScene per requested load, then one ImportAsset per manifest entry a metadata
 * batch produced), spawns and collects the async texture/mesh stream tasks, and each frame applies
 * the importer's results to the Scene's manifests and records the GPU manifest sync. It is the only
 * thing that mutates the Scene, always from the main thread.
 */
struct SceneRuntime
{
    SceneRuntime(
        daxa::Device device,
        GPUContext * gpu_context,
        std::unique_ptr<ThreadPool> & thread_pool,
        std::unique_ptr<AssetProcessor> & asset_processor,
        Importer * importer);
    ~SceneRuntime();

    auto scene() -> Scene & { return _scene; }
    auto scene() const -> Scene const & { return _scene; }
    auto scene_ptr() -> Scene * { return &_scene; }

    // Pushes an ImportScene task for `path`. 
    void request_import(std::filesystem::path const & path);

    // Once per frame: applies the importer's queued results.
    void poll();

    struct UpdateInfo
    {
        ThreadPool * thread_pool = {};
        // Only used for cloud volumes which still arrive via the AssetProcessor queue.
        // TODO(saky) Remove this once cloud volume path is included in the scene rewrite.
        std::span<const AssetProcessor::LoadedTextureInfo> uploaded_textures = {};
    };
    auto update(UpdateInfo const & info) -> daxa::ExecutableCommandList;

    auto create_mesh_acceleration_structures() -> daxa::ExecutableCommandList;

private:
    Scene _scene;
    // References to the Application-owned singletons; used to dispatch import/stream work.
    std::unique_ptr<ThreadPool> & _thread_pool;
    std::unique_ptr<AssetProcessor> & _asset_processor;
    Importer * _importer = {};
    daxa::Device _device = {};

    std::vector<std::shared_ptr<ImageStreamTask>> _inflight_image_streams = {};
    std::vector<std::shared_ptr<MeshStreamTask>> _inflight_mesh_streams = {};
    std::vector<u32> _mesh_as_build_queue = {};

    // Appends `batch`'s entries (metadata-only) and emits one ImportAsset task per texture/mesh entry,
    // each carrying the entry's freshly assigned global manifest index.
    void apply_scene_metadata_batch(Scene & scene, ImporterTaskResult::SceneMetadataBatch batch, std::vector<ImporterTask> & asset_tasks);
    // Fills in the cooked artifact `cooked_asset` targets (manifest_index is already global) and marks it dirty for streaming.
    void apply_cooked_asset(Scene & scene, ImporterTaskResult::CookedAsset cooked_asset);
};
