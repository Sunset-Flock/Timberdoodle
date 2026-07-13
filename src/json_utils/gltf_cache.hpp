#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "../scene/importers/gltf_cache.hpp"

// Serialize one record to its pretty-printed (FracturedJson) JSON block. No trailing separator - the caller
// writes each block whitespace-separated. Returns an empty string on a serialization failure (never expected).
auto serialize_gltf_cache_header(GltfCacheKey const & key) -> std::string;
auto serialize_gltf_cache_texture(TidoTextureCookResult const & texture) -> std::string;
auto serialize_gltf_cache_mesh(TidoMeshCookResult const & mesh) -> std::string;

// Loads + parses a .gltf_cache into the in-memory model. Returns nullopt if the file is absent or has no
// valid header. Does NOT check staleness - the caller compares the returned `.key` against the current
// source key. A key appearing in two records means the file is corrupt (a correct writer emits each once);
// that asserts.
auto read_gltf_cache(std::filesystem::path const & cache_path) -> std::optional<GltfCache>;
