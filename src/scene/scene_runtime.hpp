#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "../timberdoodle.hpp"
#include "scene.hpp"
using namespace tido::types;

/**
 * SceneRuntime drives residency lifecycle of the Scene.
 * It spawns and collects the async texture/mesh stream tasks and manages the synchronization of the GPU state associated with the scene's manifests.
 * Should always be called from the main thread.
 */
struct SceneRuntime
{
    SceneRuntime(
        daxa::Device device,
        GPUContext * gpu_context,
        std::unique_ptr<ThreadPool> & thread_pool);
    ~SceneRuntime();

    auto scene() -> Scene & { return _scene; }
    auto scene() const -> Scene const & { return _scene; }
    auto scene_ptr() -> Scene * { return &_scene; }

    // Null thread_pool flushes the pending GPU updates without spawning new streams, which is what shutdown
    // needs once the pool is gone.
    auto update(ThreadPool * thread_pool) -> daxa::ExecutableCommandList;

    auto create_mesh_acceleration_structures() -> daxa::ExecutableCommandList;

private:
    Scene _scene;
    // Reference to the Application-owned singleton; used to dispatch stream work.
    std::unique_ptr<ThreadPool> & _thread_pool;
    daxa::Device _device = {};

    std::vector<std::shared_ptr<ImageStreamTask>> _inflight_image_streams = {};
    std::vector<std::shared_ptr<MeshStreamTask>> _inflight_mesh_streams = {};
    std::vector<u32> _mesh_as_build_queue = {};
};
