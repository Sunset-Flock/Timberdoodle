#pragma once

#include <filesystem>
#include <optional>
#include <span>

#include "../scene/tido_format/tido_cache.hpp" // TidoCache / TidoCacheKey / cooked-artifact result types

/// --- .tido_cache JSON (de)serialization (simdjson) ---
/// The on-disk .tido_cache is a human-readable JSON document: a stream of per-record blocks - one
/// "header" record (cook key + counts) followed by one record per cooked artifact ("texture" / "mesh").
/// Each block is written pretty-printed (FracturedJson) and appended, so the file is a whitespace-
/// separated sequence of top-level JSON documents read back with simdjson's document-stream parser.
/// 64-bit fields are stored as strings (JSON numbers are doubles and would lose precision past 2^53).

// Writes the whole .tido_cache for a source file: the cook key + every cooked texture AND mesh artifact.
// Returns false on an IO failure.
auto write_tido_cache(std::filesystem::path const & cache_path, TidoCacheKey const & key,
    std::span<TidoTextureCookResult const> textures, std::span<TidoMeshCookResult const> meshes) -> bool;

// Loads + parses a .tido_cache into the in-memory model. Returns nullopt if the file is absent or has no
// valid header. Does NOT check staleness - the caller compares the returned `.key` against the current
// source key. A truncated trailing record (e.g. an interrupted cook) is tolerated: parsing stops at the
// first malformed record and keeps everything before it.
auto read_tido_cache(std::filesystem::path const & cache_path) -> std::optional<TidoCache>;
