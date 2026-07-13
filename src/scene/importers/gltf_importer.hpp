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
#include "importer_task_result.hpp"

/// --- glTF Importer ---
/// The ONLY place fastgltf lives. Parses a glTF/GLB file and translates it into the generic scene
/// via the format-agnostic Scene::add_* builder API. The parsed fastgltf::Asset is owned only for
/// the duration of the import; it never resides permanently in the Scene.
///
/// Flow:
///   1. collect_referenced_images - walk all materials, resolve each used texture to its IMAGE and
///      collect the set of images actually used (a glTF may contain unreferenced images, which we
///      skip) plus each image's type. We key on images, not textures: several textures can share one
///      image (differing only by sampler), and that image must be loaded only once.
///   2. add_texture_batch_entries / add_mesh_batch_entries - add every referenced image's/primitive's
///      metadata-only batch entry.
///   3. translate_materials / translate_mesh_groups / translate_entities - a material's/group's leaves
///      already have batch entries -> wire up the batch's own cross-references.
///   4. The batch is pushed to SceneRuntime, which appends it to the manifests (assigning global
///      indices) and, for every texture/mesh, calls request_texture_cook / request_mesh_cook with that
///      global index before calling begin_cooking() - see those methods. The importer never resolves a
///      cooked artifact's manifest index itself; SceneRuntime hands it over up front.
///
/// Dispatched as a Task by SceneRuntime::request_import and kept alive by shared_ptr - both by
/// SceneRuntime (until the whole group finishes) and by every cook chunk it dispatches (so the parsed
/// asset outlives them without re-parsing). See callback().
struct GltfImporter : Task, std::enable_shared_from_this<GltfImporter>
{
    GltfImporter(Scene * scene, Scene::LoadManifestInfo info, std::mutex * result_queue_mutex, std::vector<ImporterTaskResult> * result_queue);

    // Set once parsing + every cook chunk finish; gates new import requests (see SceneRuntime::poll).
    std::atomic<bool> group_finished = false;

    void callback(u32 chunk_index, u32 thread_index) override;

    // Called once SceneRuntime appends this image's texture entry, with its global index.
    void request_texture_cook(u32 gltf_image_index, u32 manifest_index);
    // Same as request_texture_cook, for the split-off opacity entry.
    void request_opacity_texture_cook(u32 gltf_image_index, u32 manifest_index);
    // Called once SceneRuntime appends this primitive's mesh entry, with its global index.
    void request_mesh_cook(u32 gltf_mesh_index, u32 gltf_primitive_index, u32 manifest_index);
    // Dispatches every requested cook; call once, after all requests for this batch are made.
    void begin_cooking();
    // Worker-thread entry point for begin_cooking(): resolves mtime cache hits, dispatches the rest.
    void dispatch_all_cooks();

    // Called by a finished cook chunk; the one that brings the outstanding count to zero marks the group done.
    void on_cook_chunk_finished();
    // Reports one cooked texture (+ optional split opacity) at its request_texture_cook'd manifest index.
    void push_cooked_texture(u32 gltf_image_index, TidoTextureCookResult const & artifact, std::optional<TidoTextureCookResult> const & opacity_artifact);
    // Reports one cooked mesh at its request_mesh_cook'd manifest index.
    void push_cooked_mesh(u32 gltf_mesh_index, u32 gltf_primitive_index, TidoMeshCookResult const & artifact);

  private:
    Scene * scene = {};
    Scene::LoadManifestInfo info;
    // Owned by SceneRuntime, which outlives every task this importer dispatches (Application's member order).
    std::mutex * result_queue_mutex = {};
    std::vector<ImporterTaskResult> * result_queue = {};

    std::filesystem::path file_path = {};
    fastgltf::Asset asset;
    u32 import_index = {};

    // gltf image index -> manifest index; batch-local until request_texture_cook makes it global.
    std::vector<u32> image_manifest_indices = {};
    // Parallel to image_manifest_indices, for the split-off opacity texture (see image_needs_opacity_split).
    std::vector<u32> opacity_manifest_indices = {};
    std::vector<u32> material_manifest_indices = {};
    std::vector<u32> mesh_group_manifest_indices = {};
    // [gltf mesh index][primitive index] -> manifest index; same batch-local-then-global lifecycle.
    std::vector<std::vector<u32>> mesh_manifest_indices = {};
    // Per gltf image: the type it is used as (NONE == not referenced by any material -> skipped).
    std::vector<TextureMaterialType> image_types = {};
    // Parallel to image_types: whether a Mask-mode material samples this image as an alpha cutoff.
    std::vector<bool> image_needs_opacity_split = {};

    // Every manifest entry's metadata, built purely from parsing; pushed to the result queue once complete.
    ImporterTaskResult::SceneMetadataBatch batch = {};

    // The shared .tido_cache for this source file, loaded once by load_cache and reused by
    // dispatch_texture_cooks/dispatch_mesh_cooks to serve hits. The per-kind validity flags are true only
    // when the loaded cache's cook version matches this importer's (textures and meshes are versioned
    // independently); per-artifact staleness (source mtime / content hash) is then checked entry-by-entry.
    std::optional<TidoCache> loaded_cache = {};
    bool texture_cache_valid = false;
    bool mesh_cache_valid = false;

    // The per-import output directory (TIDO_ASSET_CACHE_DIR / "<asset stem>_<source hash>"), computed once
    // by load_cache. Every artifact this import produces - the .tido_cache and all .tido data files - is
    // written here, grouping one source's output in a single folder named after it.
    std::filesystem::path cache_output_dir = {};

    // The shared .tido_cache open for writing. callback() decides whether a rewrite is needed
    // (rewriting_cache = !validate_cache()) and only then calls open_cache_writer; when validate_cache
    // reports the loaded cache fully usable, rewriting_cache stays false, open_cache_writer is never
    // called, and cache_stream stays closed. A rewrite truncates the file, writes the header, and
    // dispatch_all_cooks streams every artifact's record into it as its cook drains - so a rewrite
    // contains each key exactly once (no duplicates). cache_write_mutex guards cache_stream: cook chunks
    // append records to it concurrently, possibly outliving the parse phase itself.
    std::ofstream cache_stream = {};
    std::mutex cache_write_mutex = {};
    bool rewriting_cache = false;

    // Placeholder (1, held until dispatch_all_cooks finishes dispatching every cook chunk) + 1 per
    // dispatched chunk; whichever decrement brings this to zero sets group_finished.
    std::atomic<u32> outstanding_cook_chunks = 1;

    void push_cooked_asset(ImporterTaskResult::CookedAsset cooked_asset);

    auto parse() -> std::optional<Scene::LoadManifestErrorCode>;
    void collect_referenced_images();
    // Loads the shared .tido_cache + sets the per-kind validity flags (before dispatch_all_cooks uses it).
    void load_cache();
    // Whether the loaded cache is fully usable as-is: it exists, its per-kind cook versions match, and every
    // referenced artifact has a cached entry whose source is unchanged (mtime match + .tido present). callback()
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
    // Adds every referenced image's (+ split-off opacity's) metadata-only batch entry.
    void add_texture_batch_entries();
    void add_mesh_batch_entries();
    // Resolves mtime cache hits immediately (push_cooked_texture/push_cooked_mesh) and dispatches an async cook chunk for everything else.
    void dispatch_texture_cooks();
    void dispatch_mesh_cooks();
    // Stable per-artifact source-identity key (also the .tido file stem), shared by the cache validation and
    // the load passes so both derive the same key.
    auto image_cache_key(u32 gltf_image_index) -> u64;
    // The split opacity artifact's own identity key (see opacity_manifest_indices): same source image as
    // image_cache_key but a distinct disambiguator, so it gets its own cache entry and .tido stem.
    auto image_opacity_cache_key(u32 gltf_image_index) -> u64;
    auto mesh_cache_key(u32 gltf_mesh_index, u32 gltf_primitive_index) -> u64;
    void translate_materials();
    void translate_mesh_groups();
    // Returns the batch-local index (into batch.entities) of the imported subtree's synthetic root entity.
    auto translate_entities() -> u32;
    auto translate_light(fastgltf::Light const & light) -> u32;

    auto gltf_texture_to_image_index(u32 gltf_texture_index) -> std::optional<u32>;
};
