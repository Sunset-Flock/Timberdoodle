#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "../../timberdoodle.hpp"
#include "gltf_cache.hpp"
using namespace tido::types;

struct Importer;

/// --- Shared per-source cache registry ---
/// Generic per-source cook-cache machinery shared by every importer that persists a JSON sidecar cache -

// The cache is threadsafe (cook chunks read/update it concurrently through the locked API below);
struct SourceContext
{
    std::filesystem::path source_path = {};
    std::filesystem::path cache_dir = {};       // per-import output folder for the .tido_bin data files
    std::filesystem::path cache_file_path = {}; // the sidecar cache file this context persists to

    std::atomic<u32> outstanding_asset_imports = {};

    std::atomic<bool> cache_write_in_flight = false;

    auto lookup_texture(u64 cache_key) -> std::optional<TidoTextureCookResult>;
    auto lookup_mesh(u64 cache_key) -> std::optional<TidoMeshCookResult>;
    void store_texture(TidoTextureCookResult artifact);
    void store_mesh(TidoMeshCookResult artifact);

    // Copies the cache out and clears the dirty flag; nullopt when nothing changed since last snapshot.
    auto snapshot_if_dirty() -> std::optional<GltfCache>;

  private:
    friend struct SourceCacheRegistry;
    std::mutex _cache_mutex = {};
    GltfCache _cache = {};     // guarded by _cache_mutex
    bool _cache_dirty = false; // guarded by _cache_mutex
};

// Registry of one importer's live per-source contexts. The cache file location and per-kind cook versions
// are supplied by the caller at find_or_create time, so glTF and raw sources can share this machinery
// while persisting to their own sidecar files (.gltf_cache / .raw_cache) with independent cook versions.
struct SourceCacheRegistry
{
    explicit SourceCacheRegistry(Importer * importer);

    // Importer-thread only. Loads cache_file_path on first use for source_path; entries of a kind whose
    // cook version no longer matches texture_cook_version/mesh_cook_version are dropped at load so they
    // can never fast-path (the recook overwrites them). Returns the existing live context on a repeat
    // call for the same source.
    auto find_or_create(std::filesystem::path const & source_path, std::filesystem::path const & cache_dir,
        std::filesystem::path const & cache_file_path, u32 texture_cook_version, u32 mesh_cook_version) -> std::shared_ptr<SourceContext>;

    // Importer-thread only. Dispatches a write task for every dirty source cache and evicts idle source contexts.
    void run_upkeep();

  private:
    Importer * _importer = {};
    // Importer-thread only: the live per-source contexts, keyed by the source path's string form.
    std::unordered_map<std::string, std::shared_ptr<SourceContext>> _source_contexts = {};
};
