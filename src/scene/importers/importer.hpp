#pragma once

#include <condition_variable>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "../../timberdoodle.hpp"
#include "../../multithreading/thread_pool.hpp"
#include "importer_task_result.hpp"
using namespace tido::types;

// Stable per-image-artifact source-identity key: a hash of the resolved source byte range (location) folded
// with the cook recipe (container format, channel mapping, target format). Shared by scene-parse (to dedup
// images into one manifest entry) and the image cook (as the artifact's cache key and file stem).
auto image_identity_key(ImageImporterData const & importer_data) -> u64;

// Stable per-mesh-artifact source-identity key: a hash of the resolved attribute-source byte ranges
// (location). Meshes have no cook recipe, so location alone keys the artifact. Shared by scene-parse (to
// dedup meshes into one manifest entry) and the mesh cook (as the artifact's cache key and file stem).
auto mesh_identity_key(MeshImporterData const & importer_data) -> u64;

// Stable per-VDB-image-artifact source-identity key: a hash of the whole-file source range (location) folded
// with the recipe (per-channel grid name / background / value range, and the target format). cache_path is
// routing only and excluded, so one .vdb cooked under different grid selections or formats gets distinct keys.
auto vdb_identity_key(VdbImporterData const & importer_data) -> u64;

struct ImporterTask
{
    struct ImportScene
    {
        std::filesystem::path path = {};
    };

    struct ImportImageAsset
    {
        ImageImporterData importer_data = {};
        u32 image_manifest_index = {};
    };

    struct ImportMeshAsset
    {
        MeshImporterData importer_data = {};
        u32 mesh_manifest_index = {};
    };

    struct ImportVdbAsset
    {
        VdbImporterData importer_data = {};
        u32 image_manifest_index = {};
    };

    std::variant<ImportScene, ImportImageAsset, ImportMeshAsset, ImportVdbAsset> data = {};
};

struct Importer;

// Consumes ImportScene tasks, dispatching one gltf scene-parse per source; other task kinds are left in place.
void dispatch_scene_parses(Importer & importer, std::vector<ImporterTask> & tasks);

struct Importer
{
    explicit Importer(ThreadPool * thread_pool);
    ~Importer();

    // Signals the orchestration thread to exit and joins it; still-queued tasks are dropped. Idempotent.
    // Must run before ~ThreadPool so nothing new is dispatched into a joining pool; the pool's own join
    // then completes the in-flight parse/cook/cache-write tasks while this Importer is still alive.
    void stop();

    // Pushes a group of tasks under one lock, so a single drain of the orchestration loop sees them
    // together (a source's asset tasks then batch into a single parse). Callable from any thread.
    void push_tasks(std::span<ImporterTask> tasks);
    // Moves out every result produced so far; SceneRuntime drains this once per frame.
    auto pop_results() -> std::vector<ImporterTaskResult>;

    // Worker-side API, called from the ThreadPool tasks the importers dispatch:
    void push_result(ImporterTaskResult result);
    // Wakes the orchestration loop so upkeep can act on finished work (cook chunk / cache write done).
    void notify();

    ThreadPool * thread_pool = {};

  private:
    std::mutex _queue_mutex = {};
    std::condition_variable _wake_signal = {};
    // All three guarded by _queue_mutex.
    std::vector<ImporterTask> _task_queue = {};
    bool _stop_requested = false;
    bool _upkeep_requested = false;

    std::mutex _result_queue_mutex = {};
    std::vector<ImporterTaskResult> _result_queue = {};

    std::thread _thread = {};
    void thread_main();
};