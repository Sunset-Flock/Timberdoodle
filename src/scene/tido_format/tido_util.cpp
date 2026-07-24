#include "tido_util.hpp"

#include <cctype>
#include <fmt/format.h>

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

auto tido_stem(std::string const & name, u64 identity_key) -> std::string
{
    return fmt::format("{}_{:016x}", tido_sanitize_stem(name), identity_key);
}

auto tido_artifact_path(std::filesystem::path const & cache_dir, std::string const & name, u64 identity_key) -> std::filesystem::path
{
    return cache_dir / (tido_stem(name, identity_key) + ".tido_bin");
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
