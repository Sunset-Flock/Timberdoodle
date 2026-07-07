#pragma once

#include <filesystem>
#include <span>
#include <optional>
#include <unordered_map>
#include <vector>

#include "../../timberdoodle.hpp"
#include "tido_texture.hpp" // TidoTextureCookResult / descriptor / subresource entry
#include "tido_mesh.hpp"    // TidoMeshCookResult / descriptor / per-LOD entry
using namespace tido::types;

/// --- .tido_cache manifest (writer) ---
/// One .tido_cache per imported source file. It records the cook KEY (so a future import can tell
/// whether the cook is still valid) plus every cooked artifact the optimizer produced for that file —
/// textures AND meshes share one cache, distinguished only by which table the reader reconstructs. On
/// a cache hit (T5) an importer reads this back and hands the stored artifacts straight to the scene,
/// skipping the optimizer entirely.
///
/// The file begins with a fixed header, then two INDEXes (the "hashtables"): one {key, offset} pair
/// per texture and one per mesh, each pointing at that artifact's entry elsewhere in the file. A reader
/// loads just the indexes into in-memory maps and then seeks directly to the entry it wants by key — no
/// sequential scan. Texture and mesh keys are hashes of distinct source-identity strings, so a key is
/// looked up in exactly the table matching the artifact kind the caller wants.
///
/// File layout (little-endian, written field-by-field — no struct dumps for the scalar fields, so
/// packing/padding is irrelevant; the small fixed-layout math structs AABB/BoundingSphere are written
/// as raw POD since they are tightly packed float arrays):
///   magic "TIDC"
///   u32 texture_cook_version
///   u32 mesh_cook_version
///   u64 source_hash
///   u32 texture_count
///   u32 mesh_count
///   texture index: texture_count × { u64 key, u64 entry_offset }   (entry_offset = absolute file offset)
///   mesh index:    mesh_count    × { u64 key, u64 entry_offset }
///   texture entries: texture_count × {
///     TidoTextureDescriptor (6 × u32)
///     u32 subresource_count
///     subresource_count × { u64 offset, u32 byte_size }
///     u32 tido_path_byte_count
///     tido_path bytes (utf-8, no terminator)
///     i64 source_modified                 ← per-artifact staleness (max source mtime at cook time)
///     u64 content_hash                    ← per-artifact staleness (FNV-1a of raw source bytes)
///   }
///   mesh entries: mesh_count × {
///     TidoMeshDescriptor (u32 lod_count)
///     lod_count × TidoMeshLodDescriptor {
///       u64 blob_offset, u64 blob_byte_size,
///       AABB aabb, BoundingSphere bounding_sphere, f32 lod_error,
///       u32 vertex_count, u32 primitive_count, u32 meshlet_count,
///       u32 micro_indices_count, u32 indirect_vertices_count, u32 primitive_indices_count, u32 has_uv
///     }
///     u32 tido_path_byte_count
///     tido_path bytes (utf-8, no terminator)
///     i64 source_modified                 ← per-artifact staleness (max source mtime at cook time)
///     u64 content_hash                    ← per-artifact staleness (FNV-1a of raw source bytes)
///   }

inline constexpr std::array<char, 4> TIDO_CACHE_MAGIC = {'T', 'I', 'D', 'C'};

// FNV-1a 64-bit — the hash used for cache keys and source-path hashing.
auto tido_hash(std::string_view bytes) -> u64;

// The stable per-artifact source-identity key, shared by textures and meshes:
//   FNV-1a of "{source path}#{artifact name}#{disambiguator}".
// Used identically as BOTH the .tido_cache index key AND the .tido file-stem disambiguator, so it must be
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

// The cache file name for a source path (stem is the hashed path, so lookup is deterministic):
// "<source_hash hex>.tido_cache".
auto tido_cache_file_name(u64 source_hash) -> std::string;

// Writes a .tido_cache manifest (key + all cooked texture AND mesh artifacts). Returns false on IO
// failure.
auto write_tido_cache(std::filesystem::path const & cache_path, TidoCacheKey const & key,
    std::span<TidoTextureCookResult const> textures, std::span<TidoMeshCookResult const> meshes) -> bool;

// A loaded .tido_cache: the cook key + the in-memory indexes (key -> entry offset, one per artifact
// kind) + the raw file bytes the entries are parsed out of on demand. Look cooked artifacts up by the
// same source-identity key the writer stored, in the table matching the kind you want.
struct TidoCache
{
    TidoCacheKey key = {};
    std::unordered_map<u64, u64> texture_index = {}; // cache_key -> absolute byte offset of the texture entry
    std::unordered_map<u64, u64> mesh_index = {};    // cache_key -> absolute byte offset of the mesh entry
    std::vector<std::byte> data = {};                // whole file, entries read by offset

    // Reconstructs the cooked artifact for a source-identity key, or nullopt if absent / malformed.
    auto lookup_texture(u64 cache_key) const -> std::optional<TidoTextureCookResult>;
    auto lookup_mesh(u64 cache_key) const -> std::optional<TidoMeshCookResult>;
};

// Loads + parses a .tido_cache (header + indexes). Returns nullopt if the file is absent or malformed.
// Does NOT check staleness - the caller compares the returned `.key` against the current source key.
auto read_tido_cache(std::filesystem::path const & cache_path) -> std::optional<TidoCache>;
