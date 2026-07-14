#include "source_cache.hpp"

#include <fstream>

#include <fmt/format.h>

#include "../../json_utils/gltf_cache.hpp" // read_gltf_cache + .gltf_cache record serializers (JSON, simdjson)
#include "../../multithreading/thread_pool.hpp"
#include "importer.hpp"

auto SourceContext::lookup_texture(u64 cache_key) -> std::optional<TidoTextureCookResult>
{
    std::lock_guard<std::mutex> lock{_cache_mutex};
    return _cache.lookup_texture(cache_key);
}

auto SourceContext::lookup_mesh(u64 cache_key) -> std::optional<TidoMeshCookResult>
{
    std::lock_guard<std::mutex> lock{_cache_mutex};
    return _cache.lookup_mesh(cache_key);
}

void SourceContext::store_texture(TidoTextureCookResult artifact)
{
    std::lock_guard<std::mutex> lock{_cache_mutex};
    _cache.textures[artifact.cache_key] = std::move(artifact);
    _cache_dirty = true;
}

void SourceContext::store_mesh(TidoMeshCookResult artifact)
{
    std::lock_guard<std::mutex> lock{_cache_mutex};
    _cache.meshes[artifact.cache_key] = std::move(artifact);
    _cache_dirty = true;
}

auto SourceContext::snapshot_if_dirty() -> std::optional<GltfCache>
{
    std::lock_guard<std::mutex> lock{_cache_mutex};
    if (!_cache_dirty) { return std::nullopt; }
    _cache_dirty = false;
    return _cache;
}

namespace
{
// Full-overwrite persist of one source's cache snapshot. Records are newline separated so simdjson's
// document-stream reader sees distinct top-level documents. Dispatched by upkeep with
// cache_write_in_flight already set; separate sources' writes overlap on the pool.
struct CacheWriteTask final : Task
{
    GltfCache const snapshot;
    std::shared_ptr<SourceContext> const context;
    Importer * const importer;

    CacheWriteTask(GltfCache snapshot, std::shared_ptr<SourceContext> context, Importer * importer)
        : snapshot{std::move(snapshot)}, context{std::move(context)}, importer{importer}
    {
        chunk_count = 1;
    }

    void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        std::error_code ec = {};
        std::filesystem::create_directories(context->cache_file_path.parent_path(), ec); // ignore "already exists"
        std::ofstream cache_stream{context->cache_file_path, std::ios::binary | std::ios::trunc};
        if (!cache_stream)
        {
            // A real I/O failure, not a programming error - the artifacts simply recook next import.
            DEBUG_MSG(fmt::format("[WARN][CacheWriteTask::callback] failed to open cache file for '{}'",
                context->source_path.string()));
        }
        else
        {
            auto write_record = [&](std::string const & record)
            {
                DBG_ASSERT_TRUE_M(!record.empty(), "CacheWriteTask: record serialization failed - must not write an empty record");
                cache_stream.write(record.data(), static_cast<std::streamsize>(record.size()));
                cache_stream.write("\n", 1);
            };
            // The file leads with the header record, followed by one record per cooked artifact.
            write_record(serialize_gltf_cache_header(snapshot.key));
            for (auto const & [cache_key, texture] : snapshot.textures)
            {
                write_record(serialize_gltf_cache_texture(texture));
            }
            for (auto const & [cache_key, mesh] : snapshot.meshes)
            {
                write_record(serialize_gltf_cache_mesh(mesh));
            }
        }
        context->cache_write_in_flight.store(false, std::memory_order_release);
        importer->notify(); // upkeep may write a re-dirtied cache or evict the now-idle context.
    }
};
} // namespace

SourceCacheRegistry::SourceCacheRegistry(Importer * importer)
    : _importer{importer}
{
}

auto SourceCacheRegistry::find_or_create(std::filesystem::path const & source_path, std::filesystem::path const & cache_dir,
    std::filesystem::path const & cache_file_path, u32 texture_cook_version, u32 mesh_cook_version) -> std::shared_ptr<SourceContext>
{
    std::string const map_key = source_path.string();
    if (auto const it = _source_contexts.find(map_key); it != _source_contexts.end())
    {
        return it->second;
    }

    auto context = std::make_shared<SourceContext>();
    context->source_path = source_path;
    // The cache file is LOCATED by the caller-supplied path; whether its entries are usable is decided
    // per kind by the cook version (a bumped version stales all of that kind), and then per artifact by
    // mtime/content hash in the asset batch. The source file merely changing (a new entity, reordered
    // nodes) does not invalidate anything on its own.
    std::string const asset_name = source_path.filename().string();
    GltfCacheKey const current_key = gltf_make_cache_key(source_path, texture_cook_version, mesh_cook_version);
    context->cache_dir = cache_dir;
    context->cache_file_path = cache_file_path;

    std::optional<GltfCache> loaded_cache = read_gltf_cache(context->cache_file_path);
    if (!loaded_cache.has_value())
    {
        DEBUG_MSG(fmt::format("[SourceCacheRegistry::find_or_create] '{}': no cache - cooking everything", asset_name));
    }
    else
    {
        bool const texture_cache_valid = loaded_cache->key.texture_cook_version == current_key.texture_cook_version;
        bool const mesh_cache_valid = loaded_cache->key.mesh_cook_version == current_key.mesh_cook_version;
        DEBUG_MSG(fmt::format("[SourceCacheRegistry::find_or_create] '{}': cache loaded ({} texture + {} mesh entries); textures {}, meshes {}",
            asset_name, loaded_cache->textures.size(), loaded_cache->meshes.size(),
            texture_cache_valid ? "valid" : "stale (cook version changed) - recooking",
            mesh_cache_valid ? "valid" : "stale (cook version changed) - recooking"));
        // Entries of a stale kind are dropped so they can never fast-path; the recook stores fresh ones.
        if (!texture_cache_valid) { loaded_cache->textures.clear(); }
        if (!mesh_cache_valid) { loaded_cache->meshes.clear(); }
        context->_cache = std::move(loaded_cache.value());
    }
    context->_cache.key = current_key;

    _source_contexts.emplace(map_key, context);
    return context;
}

void SourceCacheRegistry::run_upkeep()
{
    for (auto it = _source_contexts.begin(); it != _source_contexts.end();)
    {
        std::shared_ptr<SourceContext> const & context = it->second;
        // At most one write per source in flight: a snapshot is only taken once the previous write is
        // done, so a cache re-dirtied during a write is simply written on a later tick.
        if (!context->cache_write_in_flight.load(std::memory_order_acquire))
        {
            if (std::optional<GltfCache> snapshot = context->snapshot_if_dirty(); snapshot.has_value())
            {
                context->cache_write_in_flight.store(true, std::memory_order_release);
                // This tasks priority needs to be higher than the tasks that are running the cooks/imports themseves.
                // Otherwise the cooks/import tasks will starve this cache write and the cache will not be written after each artifact is cooked.
                _importer->thread_pool->async_dispatch(std::make_shared<CacheWriteTask>(std::move(snapshot.value()), context, _importer), TaskPriority::HIGH);
                ++it;
                continue;
            }
            // Idle: nothing outstanding, nothing dirty, no write in flight - the context can go. A later
            // task for the same source recreates it from the cache file on disk.
            if (context->outstanding_asset_imports.load(std::memory_order_acquire) == 0)
            {
                it = _source_contexts.erase(it);
                continue;
            }
        }
        ++it;
    }
}
