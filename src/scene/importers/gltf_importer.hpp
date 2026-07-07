#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <fastgltf/types.hpp>

#include "../scene.hpp"
#include "../tido_format/tido_cache.hpp"

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
    // Owned by the importer for its whole lifetime. load_images / load_meshes wait for the cook tasks
    // (which borrow it) before returning, so they can never outlive it — no shared ownership needed.
    fastgltf::Asset asset;

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
    // Keyed [gltf mesh-group index][in-group primitive index] -> that mesh's manifest index (or
    // INVALID_MANIFEST_INDEX if its cook failed). Filled by load_meshes (a mesh is added only after it is
    // cooked); read by translate_mesh_groups to build each group over its already-added meshes. The mesh
    // analog of image_manifest_indices (nested because a gltf mesh-group owns a list of primitives).
    std::vector<std::vector<u32>> mesh_manifest_indices = {};
    // Per gltf image: the type it is used as (NONE == not referenced by any material -> skipped).
    std::vector<TextureMaterialType> image_types = {};
    // Every cooked texture / mesh artifact this import produced or read from the cache, recorded into the
    // shared .tido_cache when it is (re)written. Both lists are gathered single-threaded after the cook
    // tasks finish (each task stores its own result), so no locking is needed.
    std::vector<TidoTextureCookResult> cooked_texture_artifacts = {};
    std::vector<TidoMeshCookResult> cooked_mesh_artifacts = {};

    // The shared .tido_cache for this source file, loaded once by load_cache and reused by load_images +
    // load_meshes to serve hits. The per-kind validity flags are true only when the loaded cache's cook
    // version matches this importer's (textures and meshes are versioned independently); per-artifact
    // staleness (source mtime / content hash) is then checked entry-by-entry in load_images/load_meshes.
    std::optional<TidoCache> loaded_cache = {};
    bool texture_cache_valid = false;
    bool mesh_cache_valid = false;

    auto parse() -> std::optional<Scene::LoadManifestErrorCode>;
    void collect_referenced_images();
    // Loads the shared .tido_cache + sets the per-kind validity flags (before load_images / load_meshes use it).
    void load_cache();
    // load_images / load_meshes each return whether they dirtied the shared .tido_cache (cooked or
    // refreshed anything, or their cache kind was stale) so import() can skip rewriting the cache when
    // every artifact was a clean fast-path hit.
    auto load_images() -> bool;
    auto load_meshes() -> bool;
    // Writes the .tido_cache manifest (cook key + all cooked texture AND mesh artifacts). No-op unless
    // cache_dirty - a rewrite is only needed when a cook/refresh actually changed something.
    void write_cache_manifest(bool cache_dirty);
    void translate_materials();
    void translate_mesh_groups();
    auto translate_entities() -> RenderEntityId;
    auto translate_light(fastgltf::Light const & light) -> u32;

    auto gltf_texture_to_image_index(u32 gltf_texture_index) -> std::optional<u32>;
};
