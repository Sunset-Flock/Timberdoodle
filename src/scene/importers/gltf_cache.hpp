#pragma once

#include <filesystem>
#include <optional>
#include <unordered_map>

#include "../../timberdoodle.hpp"
#include "../tido_format/tido_texture.hpp"
#include "../tido_format/tido_mesh.hpp"
using namespace tido::types;

/// --- .gltf_cache model ---
/// One .gltf_cache per imported source file. It records the cook KEY (so a future import can tell
/// whether the cook is still valid) plus every cooked artifact the optimizer produced for that file -
/// textures AND meshes share one cache, keyed by the artifact's stable source-identity hash. On a cache
/// hit an importer reads this back and hands the stored artifacts straight to the scene, skipping the
/// optimizer entirely.
///
/// The on-disk form is a human-readable JSON document; the read/write pair lives in
/// json_utils/gltf_cache.{hpp,cpp} (simdjson). This header defines only the in-memory model and the
/// identity/hashing helpers shared with the importer.

// FNV-1a 64-bit - the hash used for cache keys and source-path hashing.
auto tido_hash(std::string_view bytes) -> u64;

// The stable per-artifact source-identity key, shared by textures and meshes:
//   FNV-1a of "{source path}#{artifact name}#{disambiguator}".
// Used identically as BOTH the .gltf_cache lookup key AND the .tido_bin file-stem disambiguator, so it
// must be globally unique across all imported files (the path is folded in) and recomputable from the
// source at re-import. `disambiguator` separates artifacts that share a name within one file: the gltf
// image index for a texture, "{mesh}.{primitive}" for a mesh.
auto tido_source_identity_key(std::filesystem::path const & source_path, std::string const & name, std::string const & disambiguator) -> u64;

// Identifies a cooked source file for staleness checking. The cache file is LOCATED by its mirrored
// path under tido_asset_cache (see gltf_cache_dir/gltf_cache_file_name); source_hash here is carried for
// the record, not for locating the file. The per-kind cook versions gate whether the cache's artifacts
// are still usable (bumping one recooks all of that kind). Per-artifact staleness (source mtime /
// content hash) lives on each entry, NOT here - the source file changing (e.g. a new entity) no longer
// invalidates artifacts whose bytes are unchanged.
struct GltfCacheKey
{
    u64 source_hash = {};          // hash of the source asset path
    u32 texture_cook_version = {}; // bump to invalidate every cached TEXTURE artifact this importer wrote
    u32 mesh_cook_version = {};    // bump to invalidate every cached MESH artifact this importer wrote
};

// Builds the cache key for a source asset: hashes its path and stamps the current per-kind cook versions.
auto gltf_make_cache_key(std::filesystem::path const & source_path, u32 texture_cook_version, u32 mesh_cook_version) -> GltfCacheKey;

// The per-source cache file name: "<source stem>.gltf_cache" - e.g. "bistro.gltf" -> "bistro.gltf_cache".
// No hash suffix: gltf_cache_dir already mirrors the source's own directory, so the stem alone is unique
// within it.
auto gltf_cache_file_name(std::filesystem::path const & source_path) -> std::string;

// The per-source output directory: "<TIDO_ASSET_CACHE_DIR>/<source's directory relative to
// TIDO_ASSETS_ROOT>", e.g. "<assets>/bistro/bistro.gltf" -> "tido_asset_cache/bistro/". Every artifact a
// source produces - its .gltf_cache and all .tido_bin files - is written here, so one import's output is
// grouped in a single folder mirroring where the source lives under the assets root. `source_path` must
// already lie under TIDO_ASSETS_ROOT (validated at import request time - see tido_relative_to_assets_root).
auto gltf_cache_dir(std::filesystem::path const & source_path) -> std::filesystem::path;

// A loaded .gltf_cache: the cook key + the cooked artifacts, each keyed by its stable source-identity
// hash (cache_key). Look artifacts up by the same key the writer stored, in the map matching the kind.
struct GltfCache
{
    GltfCacheKey key = {};
    std::unordered_map<u64, TidoTextureCookResult> textures = {}; // cache_key -> cooked texture metadata
    std::unordered_map<u64, TidoMeshCookResult> meshes = {};      // cache_key -> cooked mesh metadata

    auto lookup_texture(u64 cache_key) const -> std::optional<TidoTextureCookResult>;
    auto lookup_mesh(u64 cache_key) const -> std::optional<TidoMeshCookResult>;
};
