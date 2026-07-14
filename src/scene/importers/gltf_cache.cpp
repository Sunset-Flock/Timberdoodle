#include "gltf_cache.hpp"

#include <span>

#include "../tido_format/tido_util.hpp"

auto tido_hash(std::string_view bytes) -> u64
{
    return tido_fnv1a(std::as_bytes(std::span{bytes.data(), bytes.size()}));
}

auto tido_source_identity_key(std::filesystem::path const & source_path, std::string const & name, std::string const & disambiguator) -> u64
{
    // generic_string() so the hash is stable across path separators / how the path was spelled.
    return tido_hash(fmt::format("{}#{}#{}", source_path.generic_string(), name, disambiguator));
}

auto gltf_make_cache_key(std::filesystem::path const & source_path, u32 texture_cook_version, u32 mesh_cook_version) -> GltfCacheKey
{
    GltfCacheKey key = {};
    // generic_string() so the hash is stable across path separators / how the path was spelled.
    key.source_hash = tido_hash(source_path.generic_string());
    key.texture_cook_version = texture_cook_version;
    key.mesh_cook_version = mesh_cook_version;
    return key;
}

auto gltf_cache_file_name(std::filesystem::path const & source_path) -> std::string
{
    return source_path.stem().string() + ".gltf_cache";
}

auto raw_cache_file_name(std::filesystem::path const & source_path) -> std::string
{
    return source_path.stem().string() + ".raw_cache";
}

auto gltf_cache_dir(std::filesystem::path const & source_path) -> std::filesystem::path
{
    std::optional<std::filesystem::path> const relative = tido_relative_to_assets_root(source_path);
    DBG_ASSERT_TRUE_M(relative.has_value(), "gltf_cache_dir: source path must already be validated against the Tido Assets root");
    return TIDO_ASSET_CACHE_DIR / relative->parent_path();
}

auto GltfCache::lookup_texture(u64 cache_key) const -> std::optional<TidoTextureCookResult>
{
    auto const it = textures.find(cache_key);
    if (it == textures.end()) { return std::nullopt; }
    return it->second;
}

auto GltfCache::lookup_mesh(u64 cache_key) const -> std::optional<TidoMeshCookResult>
{
    auto const it = meshes.find(cache_key);
    if (it == meshes.end()) { return std::nullopt; }
    return it->second;
}
