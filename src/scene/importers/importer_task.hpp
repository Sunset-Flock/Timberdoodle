#pragma once

#include <filesystem>
#include <variant>

#include "../../timberdoodle.hpp"
#include "../scene.hpp"
#include "../optimizers/image_optimizer.hpp"
using namespace tido::types;

/// --- Importer tasks ---
/// What SceneRuntime pushes to the Importer. ImportScene asks for a source file's scene metadata
/// (parsed into a SceneMetadataBatch); ImportTextureAsset/ImportMeshAsset ask for one manifest entry's
/// cooked artifact. An asset task already carries the global manifest index SceneRuntime assigned when
/// it appended the entry - the importer never resolves manifest indices itself. The provenance variant
/// (importer_data) selects which importer picks the task up.
struct ImporterTask
{
    struct ImportScene
    {
        std::filesystem::path path = {};
    };

    struct ImportTextureAsset
    {
        std::variant<TextureManifestEntry::GltfImporterData, TextureManifestEntry::RawImporterData> importer_data = {};
        // The usage the artifact is cooked for; an OPACITY task cooks its source's alpha channel alone,
        // independent of any DIFFUSE task sharing the same source image.
        TextureMaterialType type = {};
        u32 texture_manifest_index = {};
    };

    struct ImportMeshAsset
    {
        std::variant<MeshLodGroupManifestEntry::GltfImporterData, MeshLodGroupManifestEntry::RawImporterData> importer_data = {};
        u32 mesh_manifest_index = {};
    };

    std::variant<ImportScene, ImportTextureAsset, ImportMeshAsset> data = {};
};
