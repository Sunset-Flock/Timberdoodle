#pragma once

#include <vector>

#include "importer_task.hpp"
#include "source_cache.hpp"

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

  private:
    Importer * _importer = {};
    // The per-source cache registry, persisting to .gltf_cache; shared machinery with the raw importer.
    SourceCacheRegistry _cache_registry;
};
