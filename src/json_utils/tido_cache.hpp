#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "../scene/tido_format/tido_cache.hpp" // TidoCache / TidoCacheKey / cooked-artifact result types

/// --- .tido_cache JSON (de)serialization (simdjson) ---
/// The on-disk .tido_cache is a human-readable JSON document: a stream of per-record blocks - one
/// "header" record (the cook key) followed by one record per cooked artifact ("texture" / "mesh"). Each
/// block is pretty-printed (FracturedJson); the caller writes the blocks whitespace-separated so the file
/// is a sequence of top-level JSON documents read back with simdjson's document-stream parser. 64-bit
/// fields are stored as strings (JSON numbers are doubles and would lose precision past 2^53).
///
/// These functions only (de)serialize - they never touch the filesystem. simdjson provides file-reading
/// utilities (used by read_tido_cache) but no file-writing ones, so the writer serializes each record to a
/// string and the caller owns the actual disk write: it opens the cache once, writes the header, then
/// appends each artifact's record as its cook finishes (flushing per record leaves a valid partial cache
/// if an import is interrupted). A whole cache is written by one import, so a key never appears twice.

// Serialize one record to its pretty-printed (FracturedJson) JSON block. No trailing separator - the caller
// writes each block whitespace-separated. Returns an empty string on a serialization failure (never expected).
auto serialize_tido_cache_header(TidoCacheKey const & key) -> std::string;
auto serialize_tido_cache_texture(TidoTextureCookResult const & texture) -> std::string;
auto serialize_tido_cache_mesh(TidoMeshCookResult const & mesh) -> std::string;

// Loads + parses a .tido_cache into the in-memory model. Returns nullopt if the file is absent or has no
// valid header. Does NOT check staleness - the caller compares the returned `.key` against the current
// source key. A key appearing in two records means the file is corrupt (a correct writer emits each once);
// that asserts.
auto read_tido_cache(std::filesystem::path const & cache_path) -> std::optional<TidoCache>;
