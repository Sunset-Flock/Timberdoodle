#pragma once

#include <span>
#include <string>
#include <vector>
#include <cstddef>

#include "../../timberdoodle.hpp"
using namespace tido::types;

/// --- .tido shared utilities ---
/// Small helpers shared by the .tido writers/readers (texture, mesh, cache): the canonical content
/// hash, asset-name -> file-stem sanitization, and in-memory byte-buffer appenders used to pre-build
/// blobs field-by-field (little-endian, no struct dumps).

// FNV-1a 64-bit over raw bytes. The one content/identity hash used everywhere in the .tido format
// (cooked-artifact file naming, cache keys, source-path hashing, per-artifact content hashing). Pass a
// running `seed` (a previous result) to hash several spans as one continuous stream - e.g. a mesh's
// separate vertex arrays; the default seed is the standard FNV-1a 64-bit offset basis.
auto tido_fnv1a(std::span<std::byte const> bytes, u64 seed = 0xcbf29ce484222325ull) -> u64;

// Turn an arbitrary asset name into a safe, extension-stripped file stem (asset names can be empty or
// contain characters that are not valid in a path). Returns "unnamed" if nothing usable remains.
auto tido_sanitize_stem(std::string const & name) -> std::string;

// The shared .tido file stem: "<sanitized name>_<16-hex identity key>". Both the texture and mesh writers
// use this so their cooked files are named identically in structure; the identity key (see
// tido_source_identity_key) disambiguates artifacts so two distinct ones never resolve to the same path.
auto tido_stem(std::string const & name, u64 identity_key) -> std::string;

// Append the raw bytes of a single POD to an in-memory byte buffer.
template <typename T>
void tido_append_pod(std::vector<std::byte> & buf, T const & value)
{
    auto const * bytes = r_cast<std::byte const *>(&value);
    buf.insert(buf.end(), bytes, bytes + sizeof(T));
}

// Append the raw bytes of a contiguous array (vector) to an in-memory byte buffer.
template <typename T>
void tido_append_array(std::vector<std::byte> & buf, std::vector<T> const & values)
{
    auto const * bytes = r_cast<std::byte const *>(values.data());
    buf.insert(buf.end(), bytes, bytes + values.size() * sizeof(T));
}

// Append raw bytes (ptr + size) to an in-memory byte buffer.
inline void tido_append_bytes(std::vector<std::byte> & buf, void const * data, usize size)
{
    auto const * bytes = r_cast<std::byte const *>(data);
    buf.insert(buf.end(), bytes, bytes + size);
}
