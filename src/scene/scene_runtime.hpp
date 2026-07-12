#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "../timberdoodle.hpp"
#include "scene.hpp"
#include "importers/importer_task_result.hpp"
using namespace tido::types;

struct GltfImportTask;

/**
 * SceneRuntime is the center point of scene management. It owns a Scene (the passive manifest +
 * GPU-mirror container) and drives its whole import/stream lifecycle: it kicks off + polls asset
 * imports, spawns and collects the async texture/mesh stream tasks, and each frame applies their
 * results to the Scene's manifests and records the GPU manifest sync. It is the only thing that
 * mutates the Scene from the main thread.
 */
struct SceneRuntime
{
    SceneRuntime(
        daxa::Device device,
        GPUContext * gpu_context,
        std::unique_ptr<ThreadPool> & thread_pool,
        std::unique_ptr<AssetProcessor> & asset_processor);
    ~SceneRuntime();

    auto scene() -> Scene & { return _scene; }
    auto scene() const -> Scene const & { return _scene; }
    auto scene_ptr() -> Scene * { return &_scene; }

    // Dispatches an async glTF import of `path`. Only one import may be in flight at a time (poll guards this).
    void request_import(std::filesystem::path const & path);
    // Once per frame: collects a finished import, and - if none is in flight - starts one for the newest
    // requested path (last-write-wins, a request made mid-import is not dropped).
    void poll(std::string & desired_scene_path);

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
    daxa::Device _device = {};

    // Import job currently in flight (nullptr if none). At most one at a time.
    std::shared_ptr<GltfImportTask> _pending_scene_import = {};

    // Texture stream jobs currently in flight.
    std::vector<std::shared_ptr<TextureStreamTask>> _inflight_texture_streams = {};
    // Mesh stream jobs currently in flight (spawned by update, collected once finished).
    std::vector<std::shared_ptr<MeshStreamTask>> _inflight_mesh_streams = {};
    // Mesh LODs (encoded mesh_lod_group * MAX_MESHES_PER_LOD_GROUP + lod) awaiting a BLAS build.
    // update queues newly resident LODs here; create_mesh_acceleration_structures drains it.
    std::vector<u32> _mesh_as_build_queue = {};

    void apply_scene_metadata_batch(ImporterTaskResult::SceneMetadataBatch batch);
};
