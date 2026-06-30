#include "tido_cache.hpp"

#include <fstream>
#include <vector>
#include <utility>
#include <cstring>
#include <span>

#include "tido_util.hpp"

namespace
{
template <typename T>
void write_pod(std::ofstream & ofs, T const & value)
{
    ofs.write(r_cast<char const *>(&value), sizeof(T));
}

// Bounds-checked sequential reader over a byte buffer. Any read past the end sets `ok=false`, after
// which all reads are no-ops, so callers can do a batch of reads and check `ok` once at the end.
struct ByteReader
{
    std::span<std::byte const> data = {};
    usize cursor = {};
    bool ok = true;

    template <typename T>
    auto read_pod() -> T
    {
        T value = {};
        if (ok && cursor + sizeof(T) <= data.size())
        {
            std::memcpy(&value, data.data() + cursor, sizeof(T));
            cursor += sizeof(T);
        }
        else { ok = false; }
        return value;
    }
    auto read_string(usize size) -> std::string
    {
        std::string value = {};
        if (ok && cursor + size <= data.size())
        {
            value.assign(r_cast<char const *>(data.data() + cursor), size);
            cursor += size;
        }
        else { ok = false; }
        return value;
    }
};
} // namespace

auto tido_hash(std::string_view bytes) -> u64
{
    return tido_fnv1a(std::as_bytes(std::span{bytes.data(), bytes.size()}));
}

auto tido_make_cache_key(std::filesystem::path const & source_path, u32 importer_version) -> TidoCacheKey
{
    TidoCacheKey key = {};
    // generic_string() so the hash is stable across path separators / how the path was spelled.
    key.source_hash = tido_hash(source_path.generic_string());
    std::error_code ec = {};
    auto const modified = std::filesystem::last_write_time(source_path, ec);
    key.source_modified = ec ? 0 : modified.time_since_epoch().count();
    key.importer_version = importer_version;
    return key;
}

auto tido_cache_file_name(u64 source_hash) -> std::string
{
    return fmt::format("{:016x}.tido_cache", source_hash);
}

namespace
{
// Serializes one texture entry into the entries blob (see the layout comment in tido_cache.hpp).
void append_texture_entry(std::vector<std::byte> & entries, TidoTextureCookResult const & texture)
{
    TidoTextureDescriptor const & desc = texture.descriptor;
    tido_append_pod(entries, desc.format);
    tido_append_pod(entries, desc.width);
    tido_append_pod(entries, desc.height);
    tido_append_pod(entries, desc.depth);
    tido_append_pod(entries, desc.array_layers);
    tido_append_pod(entries, desc.mip_count);

    tido_append_pod(entries, s_cast<u32>(texture.subresources.size()));
    for (TidoSubresourceEntry const & sub : texture.subresources)
    {
        tido_append_pod(entries, sub.offset);
        tido_append_pod(entries, sub.byte_size);
    }

    std::string const path = texture.tido_path.generic_string();
    tido_append_pod(entries, s_cast<u32>(path.size()));
    tido_append_bytes(entries, path.data(), path.size());
}

// Serializes one mesh entry into the entries blob (see the layout comment in tido_cache.hpp). AABB /
// BoundingSphere are written as raw POD (tightly packed float arrays, no padding).
void append_mesh_entry(std::vector<std::byte> & entries, TidoMeshCookResult const & mesh)
{
    tido_append_pod(entries, mesh.descriptor.lod_count);
    for (u32 lod = 0; lod < mesh.descriptor.lod_count; ++lod)
    {
        TidoMeshLodDescriptor const & desc = mesh.lods[lod];
        tido_append_pod(entries, desc.blob_offset);
        tido_append_pod(entries, desc.blob_byte_size);
        tido_append_pod(entries, desc.aabb);
        tido_append_pod(entries, desc.bounding_sphere);
        tido_append_pod(entries, desc.lod_error);
        tido_append_pod(entries, desc.vertex_count);
        tido_append_pod(entries, desc.primitive_count);
        tido_append_pod(entries, desc.meshlet_count);
        tido_append_pod(entries, desc.micro_indices_count);
        tido_append_pod(entries, desc.indirect_vertices_count);
        tido_append_pod(entries, desc.primitive_indices_count);
        tido_append_pod(entries, desc.has_uv);
    }

    std::string const path = mesh.tido_path.generic_string();
    tido_append_pod(entries, s_cast<u32>(path.size()));
    tido_append_bytes(entries, path.data(), path.size());
}
} // namespace

auto write_tido_cache(std::filesystem::path const & cache_path, TidoCacheKey const & key,
    std::span<TidoTextureCookResult const> textures, std::span<TidoMeshCookResult const> meshes) -> bool
{
    std::error_code ec = {};
    std::filesystem::create_directories(cache_path.parent_path(), ec); // ignore "already exists"

    std::ofstream ofs{cache_path, std::ios::binary | std::ios::trunc};
    if (!ofs) { return false; }

    // Pre-build the entries blob so each entry's offset (relative to the blob start) is known. The
    // indexes then point at absolute file offsets once we know where the blob begins. Textures and
    // meshes share one contiguous blob (textures first, then meshes), each with its own index.
    std::vector<std::byte> entries = {};
    std::vector<std::pair<u64, u64>> texture_index = {}; // {cache_key, offset within entries blob}
    std::vector<std::pair<u64, u64>> mesh_index = {};
    texture_index.reserve(textures.size());
    mesh_index.reserve(meshes.size());
    for (TidoTextureCookResult const & texture : textures)
    {
        texture_index.emplace_back(texture.cache_key, s_cast<u64>(entries.size()));
        append_texture_entry(entries, texture);
    }
    for (TidoMeshCookResult const & mesh : meshes)
    {
        mesh_index.emplace_back(mesh.cache_key, s_cast<u64>(entries.size()));
        append_mesh_entry(entries, mesh);
    }

    // Header.
    ofs.write(TIDO_CACHE_MAGIC.data(), TIDO_CACHE_MAGIC.size());
    write_pod(ofs, key.importer_version);
    write_pod(ofs, key.source_hash);
    write_pod(ofs, key.source_modified);
    write_pod(ofs, s_cast<u32>(textures.size()));
    write_pod(ofs, s_cast<u32>(meshes.size()));

    // Indexes: {key, absolute entry offset}. The entries blob starts right after both indexes, so the
    // absolute offset of an entry is (current position + remaining index bytes) + its blob offset.
    u64 const index_entry_size = sizeof(u64) + sizeof(u64);
    u64 const index_byte_size = s_cast<u64>(texture_index.size() + mesh_index.size()) * index_entry_size;
    u64 const entries_base = s_cast<u64>(ofs.tellp()) + index_byte_size;
    for (auto const & [entry_key, blob_offset] : texture_index)
    {
        write_pod(ofs, entry_key);
        write_pod(ofs, entries_base + blob_offset);
    }
    for (auto const & [entry_key, blob_offset] : mesh_index)
    {
        write_pod(ofs, entry_key);
        write_pod(ofs, entries_base + blob_offset);
    }

    // Entries blob.
    ofs.write(r_cast<char const *>(entries.data()), s_cast<std::streamsize>(entries.size()));

    return ofs.good();
}

auto read_tido_cache(std::filesystem::path const & cache_path) -> std::optional<TidoCache>
{
    std::ifstream ifs{cache_path, std::ios::binary | std::ios::ate};
    if (!ifs) { return std::nullopt; }
    std::streamsize const size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<std::byte> data(s_cast<usize>(size));
    ifs.read(r_cast<char *>(data.data()), size);
    if (!ifs.good()) { return std::nullopt; }

    ByteReader reader{data};
    std::string const magic = reader.read_string(TIDO_CACHE_MAGIC.size());
    if (!reader.ok || magic != std::string(TIDO_CACHE_MAGIC.begin(), TIDO_CACHE_MAGIC.end()))
    {
        return std::nullopt; // not a .tido_cache
    }

    TidoCache cache = {};
    cache.key.importer_version = reader.read_pod<u32>();
    cache.key.source_hash = reader.read_pod<u64>();
    cache.key.source_modified = reader.read_pod<i64>();
    u32 const texture_count = reader.read_pod<u32>();
    u32 const mesh_count = reader.read_pod<u32>();
    cache.texture_index.reserve(texture_count);
    cache.mesh_index.reserve(mesh_count);
    for (u32 i = 0; i < texture_count; ++i)
    {
        u64 const entry_key = reader.read_pod<u64>();
        u64 const entry_offset = reader.read_pod<u64>();
        cache.texture_index.emplace(entry_key, entry_offset);
    }
    for (u32 i = 0; i < mesh_count; ++i)
    {
        u64 const entry_key = reader.read_pod<u64>();
        u64 const entry_offset = reader.read_pod<u64>();
        cache.mesh_index.emplace(entry_key, entry_offset);
    }
    if (!reader.ok) { return std::nullopt; } // truncated header / index

    cache.data = std::move(data);
    return cache;
}

auto TidoCache::lookup_texture(u64 cache_key) const -> std::optional<TidoTextureCookResult>
{
    auto const it = texture_index.find(cache_key);
    if (it == texture_index.end()) { return std::nullopt; }

    ByteReader reader{data, it->second};
    TidoTextureCookResult result = {};
    result.cache_key = cache_key;
    result.descriptor.format = reader.read_pod<u32>();
    result.descriptor.width = reader.read_pod<u32>();
    result.descriptor.height = reader.read_pod<u32>();
    result.descriptor.depth = reader.read_pod<u32>();
    result.descriptor.array_layers = reader.read_pod<u32>();
    result.descriptor.mip_count = reader.read_pod<u32>();

    u32 const subresource_count = reader.read_pod<u32>();
    result.subresources.resize(subresource_count);
    for (TidoSubresourceEntry & sub : result.subresources)
    {
        sub.offset = reader.read_pod<u64>();
        sub.byte_size = reader.read_pod<u32>();
    }

    u32 const path_size = reader.read_pod<u32>();
    result.tido_path = reader.read_string(path_size);

    if (!reader.ok) { return std::nullopt; } // corrupt / truncated entry
    return result;
}

auto TidoCache::lookup_mesh(u64 cache_key) const -> std::optional<TidoMeshCookResult>
{
    auto const it = mesh_index.find(cache_key);
    if (it == mesh_index.end()) { return std::nullopt; }

    ByteReader reader{data, it->second};
    TidoMeshCookResult result = {};
    result.cache_key = cache_key;
    result.descriptor.lod_count = reader.read_pod<u32>();
    if (result.descriptor.lod_count > result.lods.size()) { return std::nullopt; } // corrupt count
    for (u32 lod = 0; lod < result.descriptor.lod_count; ++lod)
    {
        TidoMeshLodDescriptor & desc = result.lods[lod];
        desc.blob_offset = reader.read_pod<u64>();
        desc.blob_byte_size = reader.read_pod<u64>();
        desc.aabb = reader.read_pod<AABB>();
        desc.bounding_sphere = reader.read_pod<BoundingSphere>();
        desc.lod_error = reader.read_pod<f32>();
        desc.vertex_count = reader.read_pod<u32>();
        desc.primitive_count = reader.read_pod<u32>();
        desc.meshlet_count = reader.read_pod<u32>();
        desc.micro_indices_count = reader.read_pod<u32>();
        desc.indirect_vertices_count = reader.read_pod<u32>();
        desc.primitive_indices_count = reader.read_pod<u32>();
        desc.has_uv = reader.read_pod<u32>();
    }

    u32 const path_size = reader.read_pod<u32>();
    result.tido_path = reader.read_string(path_size);

    if (!reader.ok) { return std::nullopt; } // corrupt / truncated entry
    return result;
}
