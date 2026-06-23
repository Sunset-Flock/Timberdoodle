#pragma once

#include <filesystem>
#include <memory>
#include <variant>
#include <vector>

#include <fastgltf/types.hpp>

#include "../scene.hpp"

/// --- glTF Importer ---
/// The ONLY place fastgltf lives. Parses a glTF/GLB file and translates it into the generic scene
/// via the format-agnostic Scene::add_* builder API. The parsed fastgltf::Asset is owned only for
/// the duration of the import (import() waits for all cook tasks before returning); it never resides
/// permanently in the Scene.
///
/// Flow:
///   1. collect_referenced_images - walk all materials, resolve each used texture to its IMAGE and
///      collect the set of images actually used (a glTF may contain unreferenced images, which we
///      skip) plus each image's type. We key on images, not textures: several textures can share one
///      image (differing only by sampler), and that image must be loaded only once.
///   2. load_images               - add + load/optimize every referenced image; wait until done.
///   3. translate_materials       - now that a material's images are loaded, add the material.
///   4. meshes / entities.
struct GltfImporter
{
    GltfImporter(Scene & scene, Scene::LoadManifestInfo const & info);

    auto import() -> std::variant<RenderEntityId, Scene::LoadManifestErrorCode>;

  private:
    Scene & scene;
    Scene::LoadManifestInfo const & info;

    std::filesystem::path file_path = {};
    // Owned by the importer for its whole lifetime. import() waits for the cook tasks (which borrow
    // it) before returning, so they can never outlive it — no shared ownership needed.
    fastgltf::Asset asset;
    // Mesh cook tasks dispatched during import; import() blocks on these before returning.
    std::vector<std::shared_ptr<Task>> mesh_cook_tasks = {};

    // Suffix for naming this import's root entity (file-agnostic running count, not a manifest offset).
    u32 import_index = {};

    // gltf-local index -> global manifest index, filled by the add_* return values as we translate.
    // This replaces capturing manifest base offsets: dependent entries are wired up through these
    // returned indices, so the importer never assumes a contiguous manifest layout.
    // image_manifest_indices is keyed by gltf IMAGE index (INVALID_MANIFEST_INDEX for unreferenced,
    // hence not added, images). Materials map their texture -> image -> manifest index.
    std::vector<u32> image_manifest_indices = {};
    std::vector<u32> material_manifest_indices = {};
    std::vector<u32> mesh_group_manifest_indices = {};
    // Per gltf image: the type it is used as (NONE == not referenced by any material -> skipped).
    std::vector<TextureMaterialType> image_types = {};
    // Per gltf image: whether the cook compressed it as BC5 (a normal-map property the material
    // needs). Filled in load_images from the cook result so the material entry is complete.
    std::vector<bool> image_compressed_bc5 = {};

    // Collected during mesh translation so the async cook can be dispatched without storing glTF
    // identity in the manifests (the cook still needs the asset-local indices).
    struct PendingMeshLoad
    {
        u32 mesh_manifest_index = {};
        u32 gltf_mesh_index = {};
        u32 gltf_primitive_index = {};
        u32 material_manifest_index = {};
    };
    std::vector<PendingMeshLoad> pending_mesh_loads = {};

    auto parse() -> std::optional<Scene::LoadManifestErrorCode>;
    void collect_referenced_images();
    void load_images();
    void translate_materials();
    void translate_meshes_and_mesh_groups();
    auto translate_entities() -> RenderEntityId;
    auto translate_light(fastgltf::Light const & light) -> u32;

    void dispatch_async_mesh_loads();

    auto gltf_texture_to_image_index(u32 gltf_texture_index) -> std::optional<u32>;
};
