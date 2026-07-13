#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "../tido_format/tido_cache.hpp"
#include "importer_task.hpp"

struct Importer;

/// --- glTF Importer ---
/// The ONLY place fastgltf lives. Consumes the glTF-provenance ImporterTasks on the importer thread and
/// dispatches all heavy work to the ThreadPool; never touches the Scene.
///
/// Flow:
///   - ImportScene: dispatches a parse task that translates the file into one SceneMetadataBatch
///     (manifest metadata only, importer_data provenance filled - no cooking, no cache involvement).
///     SceneRuntime applies the batch and pushes back one ImportTextureAsset/ImportMeshAsset per entry.
///   - ImportTextureAsset/ImportMeshAsset: grouped by source path into one batch per update() drain
///     (SceneRuntime pushes a whole source's tasks under one lock, so a drain sees the group together).
///     The batch task re-parses the source once - the second parse is an accepted cost of keeping the
///     two task kinds fully decoupled - then resolves each artifact via the shared cache (mtime fast
///     path) or a per-artifact cook chunk that keeps the parsed asset alive through the batch.
///   - An OPACITY texture task cooks its source's alpha channel independently of the DIFFUSE task
///     sharing the same source image; the shared source is deliberately not exploited.
///
/// All results go back through the owning Importer's result queue; SceneRuntime applies them.
struct GltfImporter
{
    explicit GltfImporter(Importer * importer);

    // Importer-thread only: consumes the glTF tasks out of `tasks` (leaving other importers' tasks in
    // place), then runs cache upkeep - dispatches a write task for every dirty source cache and evicts
    // idle source contexts.
    void update(std::vector<ImporterTask> & tasks);

    // Per-source shared state, alive from a source's first asset import until it is idle again. The
    // cache is threadsafe (cook chunks read/update it concurrently through the locked API below);
    // everything else is immutable after creation or atomic.
    struct SourceContext
    {
        std::filesystem::path source_path = {};
        std::filesystem::path cache_dir = {};       // per-import output folder for the .tido data files
        std::filesystem::path cache_file_path = {}; // the .tido_cache this context persists to

        // One per ImportAsset task handed to this source; each task's resolution (fast path, cook, or
        // failure) decrements exactly once. Zero + clean cache lets upkeep evict the context.
        std::atomic<u32> outstanding_asset_imports = {};
        // True while a cache-write task for this source is on the ThreadPool; blocks a second snapshot
        // so at most one write per source is in flight.
        std::atomic<bool> cache_write_in_flight = false;

        auto lookup_texture(u64 cache_key) -> std::optional<TidoTextureCookResult>;
        auto lookup_mesh(u64 cache_key) -> std::optional<TidoMeshCookResult>;
        // Upserts the artifact under its cache_key and marks the cache dirty (persisted by upkeep).
        void store_texture(TidoTextureCookResult artifact);
        void store_mesh(TidoMeshCookResult artifact);
        // Copies the cache out and clears the dirty flag; nullopt when nothing changed since last snapshot.
        auto snapshot_if_dirty() -> std::optional<TidoCache>;

      private:
        friend struct GltfImporter;
        std::mutex _cache_mutex = {};
        TidoCache _cache = {};     // guarded by _cache_mutex
        bool _cache_dirty = false; // guarded by _cache_mutex
    };

  private:
    Importer * _importer = {};
    // Importer-thread only: the live per-source contexts, keyed by the source path's string form.
    std::unordered_map<std::string, std::shared_ptr<SourceContext>> _source_contexts = {};

    // Loads the source's .tido_cache on first use; entries of a kind whose cook version no longer
    // matches are dropped at load so they can never fast-path (the recook overwrites them).
    auto find_or_create_source_context(std::filesystem::path const & source_path) -> std::shared_ptr<SourceContext>;
    void run_cache_upkeep();
};
