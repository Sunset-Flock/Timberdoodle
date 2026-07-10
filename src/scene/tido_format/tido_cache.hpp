#pragma once

#include <filesystem>
#include <optional>
#include <unordered_map>

#include "../../timberdoodle.hpp"
#include "tido_texture.hpp" // TidoTextureCookResult / descriptor / subresource entry
#include "tido_mesh.hpp"    // TidoMeshCookResult / descriptor / per-LOD entry
using namespace tido::types;

/// --- .tido_cache model ---
/// One .tido_cache per imported source file. It records the cook KEY (so a future import can tell
/// whether the cook is still valid) plus every cooked artifact the optimizer produced for that file -
/// textures AND meshes share one cache, keyed by the artifact's stable source-identity hash. On a cache
/// hit an importer reads this back and hands the stored artifacts straight to the scene, skipping the
/// optimizer entirely.
///
/// The on-disk form is a human-readable JSON document; the read/write pair lives in
/// json_utils/tido_cache.{hpp,cpp} (simdjson). This header defines only the in-memory model and the
/// identity/hashing helpers shared with the importer.

// FNV-1a 64-bit - the hash used for cache keys and source-path hashing.
auto tido_hash(std::string_view bytes) -> u64;

// The stable per-artifact source-identity key, shared by textures and meshes:
//   FNV-1a of "{source path}#{artifact name}#{disambiguator}".
// Used identically as BOTH the .tido_cache lookup key AND the .tido file-stem disambiguator, so it must be
// globally unique across all imported files (the path is folded in) and recomputable from the source at
// re-import. `disambiguator` separates artifacts that share a name within one file: the gltf image index
// for a texture, "{mesh}.{primitive}" for a mesh.
auto tido_source_identity_key(std::filesystem::path const & source_path, std::string const & name, std::string const & disambiguator) -> u64;

// Identifies a cooked source file for staleness checking. The cache file is LOCATED by source_hash; the
// per-kind cook versions gate whether its cached artifacts are still usable (bumping one recooks all of
// that kind). Per-artifact staleness (source mtime / content hash) lives on each entry, NOT here - the
// source file changing (e.g. a new entity) no longer invalidates artifacts whose bytes are unchanged.
struct TidoCacheKey
{
    u64 source_hash = {};          // hash of the source asset path (names/locates the .tido_cache file)
    u32 texture_cook_version = {}; // bump to invalidate every cached TEXTURE artifact this importer wrote
    u32 mesh_cook_version = {};    // bump to invalidate every cached MESH artifact this importer wrote
};

// Builds the cache key for a source asset: hashes its path and stamps the current per-kind cook versions.
auto tido_make_cache_key(std::filesystem::path const & source_path, u32 texture_cook_version, u32 mesh_cook_version) -> TidoCacheKey;

// The cache file name for a source asset: "<asset stem>_<source_hash hex>.tido_cache". Carries the asset
// name for readability; the appended source hash makes it unique and deterministic - the same stem scheme
// the .tido data files use (tido_stem).
auto tido_cache_file_name(std::string const & asset_name, u64 source_hash) -> std::string;

// The per-import output directory: "<TIDO_ASSET_CACHE_DIR>/<asset stem>_<source_hash hex>". Every artifact
// a source produces - its .tido_cache and all .tido data files - is written here, so one import's output is
// grouped in a single folder named after the source asset.
auto tido_cache_dir(std::string const & asset_name, u64 source_hash) -> std::filesystem::path;

// A loaded .tido_cache: the cook key + the cooked artifacts, each keyed by its stable source-identity
// hash (cache_key). Look artifacts up by the same key the writer stored, in the map matching the kind.
struct TidoCache
{
    TidoCacheKey key = {};
    std::unordered_map<u64, TidoTextureCookResult> textures = {}; // cache_key -> cooked texture metadata
    std::unordered_map<u64, TidoMeshCookResult> meshes = {};      // cache_key -> cooked mesh metadata

    auto lookup_texture(u64 cache_key) const -> std::optional<TidoTextureCookResult>;
    auto lookup_mesh(u64 cache_key) const -> std::optional<TidoMeshCookResult>;
};
