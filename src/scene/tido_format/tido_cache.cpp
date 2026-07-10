#include "tido_cache.hpp"

#include <span>

#include "tido_util.hpp"

auto tido_hash(std::string_view bytes) -> u64
{
    return tido_fnv1a(std::as_bytes(std::span{bytes.data(), bytes.size()}));
}

auto tido_source_identity_key(std::filesystem::path const & source_path, std::string const & name, std::string const & disambiguator) -> u64
{
    // generic_string() so the hash is stable across path separators / how the path was spelled.
    return tido_hash(fmt::format("{}#{}#{}", source_path.generic_string(), name, disambiguator));
}

auto tido_make_cache_key(std::filesystem::path const & source_path, u32 texture_cook_version, u32 mesh_cook_version) -> TidoCacheKey
{
    TidoCacheKey key = {};
    // generic_string() so the hash is stable across path separators / how the path was spelled.
    key.source_hash = tido_hash(source_path.generic_string());
    key.texture_cook_version = texture_cook_version;
    key.mesh_cook_version = mesh_cook_version;
    return key;
}

auto tido_cache_file_name(std::string const & asset_name, u64 source_hash) -> std::string
{
    return tido_stem(asset_name, source_hash) + ".tido_cache";
}

auto tido_cache_dir(std::string const & asset_name, u64 source_hash) -> std::filesystem::path
{
    return TIDO_ASSET_CACHE_DIR / tido_stem(asset_name, source_hash);
}

auto TidoCache::lookup_texture(u64 cache_key) const -> std::optional<TidoTextureCookResult>
{
    auto const it = textures.find(cache_key);
    if (it == textures.end()) { return std::nullopt; }
    return it->second;
}

auto TidoCache::lookup_mesh(u64 cache_key) const -> std::optional<TidoMeshCookResult>
{
    auto const it = meshes.find(cache_key);
    if (it == meshes.end()) { return std::nullopt; }
    return it->second;
}
