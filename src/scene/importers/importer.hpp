#pragma once

#include <condition_variable>
#include <mutex>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include "../../timberdoodle.hpp"
#include "../../multithreading/thread_pool.hpp"
#include "importer_task_result.hpp"
using namespace tido::types;

struct ImporterTask
{
    struct ImportSource
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

    std::variant<ImportSource, ImportImageAsset, ImportMeshAsset, ImportVdbAsset> data = {};
};

struct Importer;

/// --- Source backends ---
/// A backend turns one source file into a list of slots - { location, type, recipe } each - which reach the
/// scene as a SceneMetadataBatch. A whole-file source is not a second path through this, it is a one-slot
/// source, and nothing downstream (hashing, artifact keys, cook policies, streamer) learns which backend a
/// slot came from. The table is an open list: engine-authored materials and future engine-native asset kinds
/// are added to it rather than branched on at the call site.
struct SourceBackend
{
    // Lowercase, dot included. One backend may claim several.
    std::string_view extension = {};
    void (*dispatch)(Importer & importer, std::filesystem::path const & path) = {};
};

// Null when no backend claims the path's extension, which is what request_import rejects on.
auto find_source_backend(std::filesystem::path const & path) -> SourceBackend const *;

// Parses the source and emits its N slots - a slice location per image/mesh, recipes from the material bindings.
void dispatch_gltf_source(Importer & importer, std::filesystem::path const & path);
// Emits the source's single whole-file slot; the recipes stay hardcoded until the project document carries
// authored ones.
void dispatch_standalone_image_source(Importer & importer, std::filesystem::path const & path);

// What one image cooked out of a .vdb is made of: which grids to densify, how their channels map into the
// target and what to cook them to. A .vdb carries none of this, so a recipe is authored data - supplied by
// whoever asks for the import, from the project document once that exists.
struct VdbSlotRecipe
{
    // Appended to the source's stem to name the manifest entry.
    std::string name = {};
    std::vector<std::string> grid_names = {};
    std::vector<u8> channel_mapping = {};
    daxa::Format target_format = {};
};

// One image slot per recipe, all reading the same whole .vdb, plus the synthetic root every batch carries.
// Knows nothing about what the volumes are for; a caller wanting them grouped into something adds that to the
// batch itself, which is why this returns the batch instead of pushing it.
auto build_vdb_source_batch(std::filesystem::path const & path, std::span<VdbSlotRecipe const> recipes)
    -> ImporterTaskResult::SceneMetadataBatch;

// PROVISIONAL: takes any .vdb to be one cloud volume, hardcoding the three recipes, their grouping and the
// placement. It exists because none of that can be authored yet. Once a cloud material lets a cloud be an
// ordinary entity referencing three cooked volumes, this goes away and .vdb claims a plain recipe-driven
// import instead.
void dispatch_cloud_volume_source(Importer & importer, std::filesystem::path const & path);

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