#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <fastgltf/types.hpp>

#include "../scene.hpp"
#include "../tido_format/tido_cache.hpp"
#include "../../json_utils/tido_cache.hpp" // .tido_cache read + record serializers

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
    // Parallel to image_manifest_indices: gltf IMAGE index -> the manifest index of its split-off opacity
    // texture (see TC.4 / the DIFFUSE split in image_optimizer's process_image), or INVALID_MANIFEST_INDEX
    // when that image has no alpha (most images) or isn't a DIFFUSE image at all.
    std::vector<u32> opacity_manifest_indices = {};
    std::vector<u32> material_manifest_indices = {};
    std::vector<u32> mesh_group_manifest_indices = {};
    // Keyed [gltf mesh-group index][in-group primitive index] -> that mesh's manifest index (or
    // INVALID_MANIFEST_INDEX if its cook failed). Filled by load_meshes (a mesh is added only after it is
    // cooked); read by translate_mesh_groups to build each group over its already-added meshes. The mesh
    // analog of image_manifest_indices (nested because a gltf mesh-group owns a list of primitives).
    std::vector<std::vector<u32>> mesh_manifest_indices = {};
    // Per gltf image: the type it is used as (NONE == not referenced by any material -> skipped).
    std::vector<TextureMaterialType> image_types = {};

    // The shared .tido_cache for this source file, loaded once by load_cache and reused by load_images +
    // load_meshes to serve hits. The per-kind validity flags are true only when the loaded cache's cook
    // version matches this importer's (textures and meshes are versioned independently); per-artifact
    // staleness (source mtime / content hash) is then checked entry-by-entry in load_images/load_meshes.
    std::optional<TidoCache> loaded_cache = {};
    bool texture_cache_valid = false;
    bool mesh_cache_valid = false;

    // The per-import output directory (TIDO_ASSET_CACHE_DIR / "<asset stem>_<source hash>"), computed once
    // by load_cache. Every artifact this import produces - the .tido_cache and all .tido data files - is
    // written here, grouping one source's output in a single folder named after it.
    std::filesystem::path cache_output_dir = {};

    // The shared .tido_cache open for writing. import() decides whether a rewrite is needed (rewriting_cache =
    // !validate_cache()) and only then calls open_cache_writer; when validate_cache reports the loaded cache
    // fully usable, rewriting_cache stays false, open_cache_writer is never called, and cache_stream stays
    // closed. A rewrite truncates the file, writes the header, and load_images / load_meshes stream every
    // artifact's record into it as its cook drains - so a rewrite contains each key exactly once (no duplicates).
    // cache_write_mutex guards cache_stream: cook chunks append records to it concurrently.
    std::ofstream cache_stream = {};
    std::mutex cache_write_mutex = {};
    bool rewriting_cache = false;

    auto parse() -> std::optional<Scene::LoadManifestErrorCode>;
    void collect_referenced_images();
    // Loads the shared .tido_cache + sets the per-kind validity flags (before load_images / load_meshes use it).
    void load_cache();
    // Whether the loaded cache is fully usable as-is: it exists, its per-kind cook versions match, and every
    // referenced artifact has a cached entry whose source is unchanged (mtime match + .tido present). import()
    // calls this to decide rewriting_cache; reused verbatim if true, rewritten fresh (open_cache_writer) if not.
    auto validate_cache() -> bool;
    // Opens the shared .tido_cache for a fresh write: truncates it and writes the header record. Asserts
    // rewriting_cache is already true.
    void open_cache_writer();
    // Appends one serialized record (+ a separating newline) to the open cache_stream and flushes, under
    // cache_write_mutex so parallel cook chunks append safely. record must not be empty (asserted) - the
    // caller must not attempt to write a failed serialization. A no-op if the stream is not open (a real
    // open failure, already logged by open_cache_writer, not a programming error - the artifact recooks
    // next import rather than crashing this one).
    void write_cache_record(std::string const & record);
    // Cook + add every referenced image / mesh, streaming each artifact's record into the .tido_cache (when a
    // rewrite is underway) as its cook finishes.
    void load_images();
    void load_meshes();
    // Stable per-artifact source-identity key (also the .tido file stem), shared by the cache validation and
    // the load passes so both derive the same key.
    auto image_cache_key(u32 gltf_image_index) -> u64;
    // The split opacity artifact's own identity key (see opacity_manifest_indices): same source image as
    // image_cache_key but a distinct disambiguator, so it gets its own cache entry and .tido stem.
    auto image_opacity_cache_key(u32 gltf_image_index) -> u64;
    auto mesh_cache_key(u32 gltf_mesh_index, u32 gltf_primitive_index) -> u64;
    void translate_materials();
    void translate_mesh_groups();
    auto translate_entities() -> RenderEntityId;
    auto translate_light(fastgltf::Light const & light) -> u32;

    auto gltf_texture_to_image_index(u32 gltf_texture_index) -> std::optional<u32>;
};

/// --- Async import task ---
// A single-chunk task running a whole GltfImporter::import() on a worker thread.
// Application::load_scene dispatches one and stores it as ApplicationState::pending_scene_import;
// Application::poll_scene_import polls `finished` each frame and consumes `result` once it flips.
// `finished` uses release/acquire ordering so the polling thread's read of `result` is ordered
// after the worker's write.
struct GltfImportTask : Task
{
    Scene * scene = {};
    // Holds references (the thread-pool + asset-processor unique_ptrs). Safe to keep in this longer-
    // lived task: ThreadPool's destructor joins all workers - finishing any in-flight import - before
    // either referent is destroyed (Application's member order guarantees it).
    Scene::LoadManifestInfo info;
    std::variant<RenderEntityId, Scene::LoadManifestErrorCode> result = {};
    std::atomic<bool> finished = false;

    GltfImportTask(Scene * scene, Scene::LoadManifestInfo info)
        : scene{scene}, info{std::move(info)}
    {
        chunk_count = 1;
    }

    void callback(u32 chunk_index, u32 thread_index) override;
};
