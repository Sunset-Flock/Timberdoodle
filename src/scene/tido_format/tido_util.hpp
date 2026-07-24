#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include <cstddef>
#include <type_traits>

#include "../../timberdoodle.hpp"
using namespace tido::types;

inline std::filesystem::path const TIDO_ASSETS_ROOT = "assets";

auto tido_relative_to_assets_root(std::filesystem::path const & path) -> std::optional<std::filesystem::path>;

auto tido_fnv1a(std::span<std::byte const> bytes, u64 seed = 0xcbf29ce484222325ull) -> u64;

// Turn an arbitrary asset name into a safe, extension-stripped file stem (asset names can be empty or
// contain characters that are not valid in a path). Returns "unnamed" if nothing usable remains.
auto tido_sanitize_stem(std::string const & name) -> std::string;

auto tido_stem(std::string const & name, u64 identity_key) -> std::string;

// Full path of an artifact's .tido_bin file: <cache_dir>/<stem>.tido_bin.
auto tido_artifact_path(std::filesystem::path const & cache_dir, std::string const & name, u64 identity_key) -> std::filesystem::path;

// Append the raw bytes of a single POD to an in-memory byte buffer.
template <typename T>
void tido_append_pod(std::vector<std::byte> & buf, T const & value)
{
    static_assert(std::is_trivially_copyable_v<T>, "tido_append_pod: T must be trivially copyable");
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
