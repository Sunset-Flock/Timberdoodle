#include "tido_util.hpp"

#include <algorithm>
#include <cctype>
#include <fmt/format.h>

auto tido_lowercase_extension(std::filesystem::path const & path) -> std::string
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](char const c) { return s_cast<char>(std::tolower(s_cast<unsigned char>(c))); });
    return extension;
}

auto tido_fnv1a(std::span<std::byte const> bytes, u64 seed) -> u64
{
    u64 hash = seed;
    for (std::byte const b : bytes)
    {
        hash ^= s_cast<u64>(s_cast<u8>(b));
        hash *= 0x100000001b3ull;
    }
    return hash;
}

auto tido_hash_file(std::filesystem::path const & path, std::optional<ByteSlice> slice, u64 seed) -> std::optional<u64>
{
    auto [open_result, reader] = FileReader::open(path);
    if (open_result != FileIoResult::SUCCESS) { return std::nullopt; }

    // A whole-file range can only learn its length here, matching read_file's handling of an absent slice.
    u64 const begin_byte_offset = slice.has_value() ? slice->byte_offset : 0;
    u64 remaining_byte_count = slice.has_value() ? slice->byte_length : reader.file_byte_size();
    if (reader.seek(begin_byte_offset) != FileIoResult::SUCCESS) { return std::nullopt; }

    // The bounded scratch is the whole point: a probe must not allocate its entire source to produce a hash.
    static constexpr u64 SCRATCH_BYTE_SIZE = 64 * 1024;
    std::vector<std::byte> scratch(s_cast<usize>(std::min(remaining_byte_count, SCRATCH_BYTE_SIZE)));
    u64 hash = seed;
    while (remaining_byte_count > 0)
    {
        u64 const chunk_byte_count = std::min(remaining_byte_count, SCRATCH_BYTE_SIZE);
        if (reader.read_into(scratch.data(), chunk_byte_count) != FileIoResult::SUCCESS) { return std::nullopt; }
        hash = tido_fnv1a(std::span<std::byte const>(scratch.data(), s_cast<usize>(chunk_byte_count)), hash);
        remaining_byte_count -= chunk_byte_count;
    }
    return hash;
}

auto tido_sanitize_stem(std::string const & name) -> std::string
{
    std::string out;
    out.reserve(name.size());
    for (char const c : name)
    {
        bool const ok = std::isalnum(s_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.';
        out.push_back(ok ? c : '_');
    }
    // Drop a trailing extension (e.g. ".png") so the stem is clean.
    auto const dot = out.find_last_of('.');
    if (dot != std::string::npos) { out.erase(dot); }
    if (out.empty()) { out = "unnamed"; }
    return out;
}

auto tido_artifact_path(std::filesystem::path const & store_dir, u64 artifact_key) -> std::filesystem::path
{
    return store_dir / fmt::format("{:02x}", artifact_key >> 56) / fmt::format("{:016x}.tido_bin", artifact_key);
}

auto tido_relative_to_assets_root(std::filesystem::path const & path) -> std::optional<std::filesystem::path>
{
    std::error_code ec = {};
    std::filesystem::path const canonical_root = std::filesystem::weakly_canonical(TIDO_ASSETS_ROOT, ec);
    if (ec) { return std::nullopt; }
    std::filesystem::path const canonical_path = std::filesystem::weakly_canonical(path, ec);
    if (ec) { return std::nullopt; }

    std::filesystem::path const relative = std::filesystem::relative(canonical_path, canonical_root, ec);
    if (ec || relative.empty() || relative.begin()->string() == "..") { return std::nullopt; }
    return relative;
}
