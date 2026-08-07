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
 * GPU-mirror container) and drives its residency lifecycle: it asks the Importer to import a source,
 * spawns and collects the async texture/mesh stream tasks, and each frame applies the importer's
 * results to the Scene's manifests and records the GPU manifest sync. It is the only thing that
 * mutates the Scene, always from the main thread. Production is the Importer's half: it queues the
 * cooks for the slots it publishes, so nothing here can cause one.
 */
struct SceneRuntime
{
    SceneRuntime(
        daxa::Device device,
        GPUContext * gpu_context,
        std::unique_ptr<ThreadPool> & thread_pool,
        Importer * importer);
    ~SceneRuntime();

    auto scene() -> Scene & { return _scene; }
    auto scene() const -> Scene const & { return _scene; }
    auto scene_ptr() -> Scene * { return &_scene; }

    // Pushes an ImportSource task for `path`, rejecting anything outside the assets root or of an extension
    // no source backend claims.
    void request_import(std::filesystem::path const & path);

    // Once per frame: applies the importer's queued results.
    void poll();

    struct UpdateInfo
    {
        ThreadPool * thread_pool = {};
    };
    auto update(UpdateInfo const & info) -> daxa::ExecutableCommandList;

    auto create_mesh_acceleration_structures() -> daxa::ExecutableCommandList;

private:
    Scene _scene;
    // References to the Application-owned singletons; used to dispatch import/stream work.
    std::unique_ptr<ThreadPool> & _thread_pool;
    Importer * _importer = {};
    daxa::Device _device = {};

    std::vector<std::shared_ptr<ImageStreamTask>> _inflight_image_streams = {};
    std::vector<std::shared_ptr<MeshStreamTask>> _inflight_mesh_streams = {};
    std::vector<u32> _mesh_as_build_queue = {};

    // Applies `batch`'s modifications - each element creates an entry or modifies the one it names -
    // returning where the created ones landed so the producer can name them later. The Scene itself keeps no
    // record of which source an entry came from.
    auto apply_scene_metadata_batch(Scene & scene, ImporterTaskResult::SceneMetadataBatch batch) -> ImporterTaskResult::AppliedBatch;
};
