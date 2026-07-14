#pragma once

#include <vector>

#include "importer_task.hpp"
#include "source_cache.hpp"

struct Importer;

/// --- Raw Importer ---
/// The non-glTF importer: cooks ImportTextureAsset/ImportMeshAsset tasks whose provenance is
/// RawImporterData - plain PNG/KTX2 images and VDB volumes. Unlike GltfImporter there is no shared
/// scene-graph parse step: every task already carries everything its own cook needs, so items are
/// resolved and dispatched independently. Never touches the Scene; results go back through the owning
/// Importer's result queue.
///
/// Flow: groups a drain's raw asset tasks by source path (one SourceContext, persisting <stem>.raw_cache,
/// per source); per item a mtime fast path serves cache hits straight from the context, everything else
/// becomes a cook chunk on the ThreadPool. A `.vdb` source cooks through LoadVDBTask + process_volume; a
/// `.png`/`.ktx2` source is read and cooked through process_image. A raw ImportMeshAsset always fails -
/// no raw mesh cook exists yet.
struct RawImporter
{
    explicit RawImporter(Importer * importer);

    // Importer-thread only: consumes the raw tasks out of `tasks` (leaving other importers' tasks in
    // place), then runs cache upkeep - dispatches a write task for every dirty source cache and evicts
    // idle source contexts.
    void update(std::vector<ImporterTask> & tasks);

  private:
    Importer * _importer = {};
    // The per-source cache registry, persisting to .raw_cache; shared machinery with GltfImporter.
    SourceCacheRegistry _cache_registry;

    struct TextureItem
    {
        std::variant<TextureManifestEntry::RawImporterData::Image, TextureManifestEntry::RawImporterData::VdbVolume> recipe = {};
        TextureMaterialType type = {};
        u32 manifest_index = {};
    };
    void resolve_texture_items(std::shared_ptr<SourceContext> const & context, std::vector<TextureItem> const & items);
};
