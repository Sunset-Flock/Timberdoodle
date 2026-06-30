#include "tido_cache.hpp"

#include <fstream>
#include <vector>
#include <utility>
#include <cstring>
#include <span>

namespace
{
template <typename T>
void write_pod(std::ofstream & ofs, T const & value)
{
    ofs.write(r_cast<char const *>(&value), sizeof(T));
}

// Append a POD / raw bytes to an in-memory byte buffer (used to pre-build the entries blob so each
// entry's offset is known before the index is written).
template <typename T>
void append_pod(std::vector<std::byte> & buf, T const & value)
{
    auto const * bytes = r_cast<std::byte const *>(&value);
    buf.insert(buf.end(), bytes, bytes + sizeof(T));
}
void append_bytes(std::vector<std::byte> & buf, void const * data, usize size)
{
    auto const * bytes = r_cast<std::byte const *>(data);
    buf.insert(buf.end(), bytes, bytes + size);
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
    u64 hash = 0xcbf29ce484222325ull;
    for (char const c : bytes)
    {
        hash ^= s_cast<u64>(s_cast<u8>(c));
        hash *= 0x100000001b3ull;
    }
    return hash;
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

auto write_texture_cache(std::filesystem::path const & cache_path, TidoCacheKey const & key,
    std::span<TidoTextureCookResult const> textures) -> bool
{
    std::error_code ec = {};
    std::filesystem::create_directories(cache_path.parent_path(), ec); // ignore "already exists"

    std::ofstream ofs{cache_path, std::ios::binary | std::ios::trunc};
    if (!ofs) { return false; }

    // Pre-build the entries blob so each entry's offset (relative to the blob start) is known. The
    // index then points at absolute file offsets once we know where the blob begins.
    std::vector<std::byte> entries = {};
    std::vector<std::pair<u64, u64>> index = {}; // {cache_key, offset within entries blob}
    index.reserve(textures.size());
    for (TidoTextureCookResult const & texture : textures)
    {
        index.emplace_back(texture.cache_key, s_cast<u64>(entries.size()));

        TidoTextureDescriptor const & desc = texture.descriptor;
        append_pod(entries, desc.format);
        append_pod(entries, desc.width);
        append_pod(entries, desc.height);
        append_pod(entries, desc.depth);
        append_pod(entries, desc.array_layers);
        append_pod(entries, desc.mip_count);

        append_pod(entries, s_cast<u32>(texture.subresources.size()));
        for (TidoSubresourceEntry const & sub : texture.subresources)
        {
            append_pod(entries, sub.offset);
            append_pod(entries, sub.byte_size);
        }

        std::string const path = texture.tido_path.generic_string();
        append_pod(entries, s_cast<u32>(path.size()));
        append_bytes(entries, path.data(), path.size());
    }

    // Header.
    ofs.write(TIDO_CACHE_MAGIC.data(), TIDO_CACHE_MAGIC.size());
    write_pod(ofs, key.importer_version);
    write_pod(ofs, key.source_hash);
    write_pod(ofs, key.source_modified);
    write_pod(ofs, s_cast<u32>(textures.size()));

    // Index: {key, absolute entry offset}. The entries blob starts right after the index, so the
    // absolute offset of an entry is (current position + remaining index bytes) + its blob offset.
    u64 const index_byte_size = s_cast<u64>(index.size()) * (sizeof(u64) + sizeof(u64));
    u64 const entries_base = s_cast<u64>(ofs.tellp()) + index_byte_size;
    for (auto const & [entry_key, blob_offset] : index)
    {
        write_pod(ofs, entry_key);
        write_pod(ofs, entries_base + blob_offset);
    }

    // Entries blob.
    ofs.write(r_cast<char const *>(entries.data()), s_cast<std::streamsize>(entries.size()));

    return ofs.good();
}

auto read_texture_cache(std::filesystem::path const & cache_path) -> std::optional<TidoTextureCache>
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

    TidoTextureCache cache = {};
    cache.key.importer_version = reader.read_pod<u32>();
    cache.key.source_hash = reader.read_pod<u64>();
    cache.key.source_modified = reader.read_pod<i64>();
    u32 const texture_count = reader.read_pod<u32>();
    cache.index.reserve(texture_count);
    for (u32 i = 0; i < texture_count; ++i)
    {
        u64 const entry_key = reader.read_pod<u64>();
        u64 const entry_offset = reader.read_pod<u64>();
        cache.index.emplace(entry_key, entry_offset);
    }
    if (!reader.ok) { return std::nullopt; } // truncated header / index

    cache.data = std::move(data);
    return cache;
}

auto TidoTextureCache::lookup(u64 cache_key) const -> std::optional<TidoTextureCookResult>
{
    auto const it = index.find(cache_key);
    if (it == index.end()) { return std::nullopt; }

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
