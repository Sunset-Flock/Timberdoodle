#include "tido_cache.hpp"

#include <fstream>
#include <charconv>
#include <string>
#include <string_view>

#include <simdjson.h>
#include <fmt/format.h>

// The .tido_cache is a stream of JSON records: a "header" record then one per cooked artifact. Each record
// type has a simdjson custom (de)serializer (tag_invoke, in namespace simdjson per the docs), so writing is
// `simdjson::to_json(record)` and reading is `document.get<TidoCacheRecord>()` - the field<->member mapping
// for each type lives in exactly one place. 64-bit fields are stored as strings (JSON numbers are doubles,
// exact only below 2^53; hashes and 100ns-tick mtimes exceed that): hashes hex, magnitudes decimal.
// Records are written and read strictly in field order, so no backward iteration is needed.

namespace
{
auto hex_u64(u64 value) -> std::string { return fmt::format("0x{:016x}", value); }
auto dec_u64(u64 value) -> std::string { return fmt::format("{}", value); }
auto dec_i64(i64 value) -> std::string { return fmt::format("{}", value); }

auto parse_hex_u64(std::string_view text) -> std::optional<u64>
{
    if (text.size() >= 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) { text.remove_prefix(2); }
    if (text.empty()) { return std::nullopt; }
    u64 value = 0;
    auto const result = std::from_chars(text.data(), text.data() + text.size(), value, 16);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) { return std::nullopt; }
    return value;
}

template <typename T>
auto parse_dec(std::string_view text) -> std::optional<T>
{
    T value = {};
    auto const result = std::from_chars(text.data(), text.data() + text.size(), value, 10);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) { return std::nullopt; }
    return value;
}

// The header record's payload: the cook key plus the artifact counts (counts are informational - the
// reader rebuilds the maps from the records themselves).
struct TidoCacheHeader
{
    TidoCacheKey key = {};
    u64 texture_count = {};
    u64 mesh_count = {};
};

// One parsed record, tagged by which "kind" it was so the reader can route it into the right map.
struct TidoCacheRecord
{
    enum struct Kind
    {
        HEADER,
        TEXTURE,
        MESH,
    };
    Kind kind = {};
    TidoCacheKey header_key = {};
    TidoTextureCookResult texture = {};
    TidoMeshCookResult mesh = {};
};

// --- read helpers: pull one string-encoded / numeric field from an object, in field order ---

auto read_hex(simdjson::ondemand::object & obj, char const * key, u64 & out) -> simdjson::error_code
{
    std::string_view text;
    SIMDJSON_TRY(obj[key].get_string().get(text));
    auto const parsed = parse_hex_u64(text);
    if (!parsed.has_value()) { return simdjson::INCORRECT_TYPE; }
    out = parsed.value();
    return simdjson::SUCCESS;
}

template <typename T>
auto read_dec(simdjson::ondemand::object & obj, char const * key, T & out) -> simdjson::error_code
{
    std::string_view text;
    SIMDJSON_TRY(obj[key].get_string().get(text));
    auto const parsed = parse_dec<T>(text);
    if (!parsed.has_value()) { return simdjson::INCORRECT_TYPE; }
    out = parsed.value();
    return simdjson::SUCCESS;
}

auto read_u32(simdjson::ondemand::object & obj, char const * key, u32 & out) -> simdjson::error_code
{
    u64 value = 0;
    SIMDJSON_TRY(obj[key].get_uint64().get(value));
    out = static_cast<u32>(value);
    return simdjson::SUCCESS;
}

auto read_path(simdjson::ondemand::object & obj, char const * key, std::filesystem::path & out) -> simdjson::error_code
{
    std::string_view text;
    SIMDJSON_TRY(obj[key].get_string().get(text));
    out = std::filesystem::path(text);
    return simdjson::SUCCESS;
}

// Read a fixed-length flat float array (aabb: 6, bounding_sphere: 4) in order.
auto read_floats(simdjson::ondemand::object & obj, char const * key, f32 * out, usize count) -> simdjson::error_code
{
    simdjson::ondemand::array array;
    SIMDJSON_TRY(obj[key].get_array().get(array));
    usize index = 0;
    for (auto element : array)
    {
        double value = 0.0;
        SIMDJSON_TRY(element.get_double().get(value));
        if (index >= count) { return simdjson::INCORRECT_TYPE; }
        out[index++] = static_cast<f32>(value);
    }
    return index == count ? simdjson::SUCCESS : simdjson::INCORRECT_TYPE;
}

// The three per-record readers. Each assumes the "kind" field has already been consumed and reads the
// remaining fields from `obj` in the same order the writer emitted them (no backward iteration).

auto read_header(simdjson::ondemand::object & obj, TidoCacheKey & key) -> simdjson::error_code
{
    SIMDJSON_TRY(read_u32(obj, "texture_cook_version", key.texture_cook_version));
    SIMDJSON_TRY(read_u32(obj, "mesh_cook_version", key.mesh_cook_version));
    SIMDJSON_TRY(read_hex(obj, "source_hash", key.source_hash));
    return simdjson::SUCCESS;
}

auto read_texture(simdjson::ondemand::object & obj, TidoTextureCookResult & out) -> simdjson::error_code
{
    SIMDJSON_TRY(read_hex(obj, "key", out.cache_key));
    SIMDJSON_TRY(read_u32(obj, "format", out.descriptor.format));
    SIMDJSON_TRY(read_u32(obj, "width", out.descriptor.width));
    SIMDJSON_TRY(read_u32(obj, "height", out.descriptor.height));
    SIMDJSON_TRY(read_u32(obj, "depth", out.descriptor.depth));
    SIMDJSON_TRY(read_u32(obj, "array_layers", out.descriptor.array_layers));
    SIMDJSON_TRY(read_u32(obj, "mip_count", out.descriptor.mip_count));

    simdjson::ondemand::array subresources;
    SIMDJSON_TRY(obj["subresources"].get_array().get(subresources));
    for (auto element : subresources)
    {
        simdjson::ondemand::object sub;
        SIMDJSON_TRY(element.get_object().get(sub));
        TidoSubresourceEntry entry = {};
        SIMDJSON_TRY(read_dec(sub, "offset", entry.offset));
        SIMDJSON_TRY(read_u32(sub, "byte_size", entry.byte_size));
        out.subresources.push_back(entry);
    }

    SIMDJSON_TRY(read_path(obj, "path", out.tido_path));
    SIMDJSON_TRY(read_dec(obj, "source_modified", out.source_modified));
    SIMDJSON_TRY(read_hex(obj, "content_hash", out.content_hash));
    return simdjson::SUCCESS;
}

auto read_mesh(simdjson::ondemand::object & obj, TidoMeshCookResult & out) -> simdjson::error_code
{
    SIMDJSON_TRY(read_hex(obj, "key", out.cache_key));

    simdjson::ondemand::array lods;
    SIMDJSON_TRY(obj["lods"].get_array().get(lods));
    u32 lod_count = 0;
    for (auto element : lods)
    {
        if (lod_count >= out.lods.size()) { return simdjson::INCORRECT_TYPE; } // more LODs than slots
        simdjson::ondemand::object lod;
        SIMDJSON_TRY(element.get_object().get(lod));
        TidoMeshLodDescriptor & desc = out.lods[lod_count];
        SIMDJSON_TRY(read_dec(lod, "blob_offset", desc.blob_offset));
        SIMDJSON_TRY(read_dec(lod, "blob_byte_size", desc.blob_byte_size));

        f32 aabb[6] = {};
        SIMDJSON_TRY(read_floats(lod, "aabb", aabb, 6));
        desc.aabb.center.x = aabb[0]; desc.aabb.center.y = aabb[1]; desc.aabb.center.z = aabb[2];
        desc.aabb.size.x = aabb[3]; desc.aabb.size.y = aabb[4]; desc.aabb.size.z = aabb[5];

        f32 sphere[4] = {};
        SIMDJSON_TRY(read_floats(lod, "bounding_sphere", sphere, 4));
        desc.bounding_sphere.center.x = sphere[0]; desc.bounding_sphere.center.y = sphere[1]; desc.bounding_sphere.center.z = sphere[2];
        desc.bounding_sphere.radius = sphere[3];

        double lod_error = 0.0;
        SIMDJSON_TRY(lod["lod_error"].get_double().get(lod_error));
        desc.lod_error = static_cast<f32>(lod_error);

        SIMDJSON_TRY(read_u32(lod, "vertex_count", desc.vertex_count));
        SIMDJSON_TRY(read_u32(lod, "primitive_count", desc.primitive_count));
        SIMDJSON_TRY(read_u32(lod, "meshlet_count", desc.meshlet_count));
        SIMDJSON_TRY(read_u32(lod, "micro_indices_count", desc.micro_indices_count));
        SIMDJSON_TRY(read_u32(lod, "indirect_vertices_count", desc.indirect_vertices_count));
        SIMDJSON_TRY(read_u32(lod, "primitive_indices_count", desc.primitive_indices_count));
        SIMDJSON_TRY(read_u32(lod, "has_uv", desc.has_uv));
        ++lod_count;
    }
    out.descriptor.lod_count = lod_count;

    SIMDJSON_TRY(read_path(obj, "path", out.tido_path));
    SIMDJSON_TRY(read_dec(obj, "source_modified", out.source_modified));
    SIMDJSON_TRY(read_hex(obj, "content_hash", out.content_hash));
    return simdjson::SUCCESS;
}
} // namespace

// Custom (de)serializers live in namespace simdjson so ADL finds them (the CPO tags serialize_tag /
// deserialize_tag are members of simdjson). Serializers are templated on the builder type so they match
// whichever implementation-specific string_builder simdjson::to_json instantiates.
namespace simdjson
{
error_code tag_invoke(deserialize_tag, auto & value, TidoCacheRecord & out)
{
    ondemand::object obj;
    SIMDJSON_TRY(value.get_object().get(obj));
    std::string_view kind;
    SIMDJSON_TRY(obj["kind"].get_string().get(kind));
    if (kind == "header")  { out.kind = TidoCacheRecord::Kind::HEADER;  return read_header(obj, out.header_key); }
    if (kind == "texture") { out.kind = TidoCacheRecord::Kind::TEXTURE; return read_texture(obj, out.texture); }
    if (kind == "mesh")    { out.kind = TidoCacheRecord::Kind::MESH;    return read_mesh(obj, out.mesh); }
    return INCORRECT_TYPE;
}

template <typename builder_type>
void tag_invoke(serialize_tag, builder_type & builder, TidoCacheHeader const & header)
{
    builder.start_object();
    builder.append_key_value("kind", std::string_view("header"));
    builder.append_comma();
    builder.append_key_value("texture_cook_version", static_cast<u64>(header.key.texture_cook_version));
    builder.append_comma();
    builder.append_key_value("mesh_cook_version", static_cast<u64>(header.key.mesh_cook_version));
    builder.append_comma();
    builder.append_key_value("source_hash", hex_u64(header.key.source_hash));
    builder.append_comma();
    builder.append_key_value("texture_count", header.texture_count);
    builder.append_comma();
    builder.append_key_value("mesh_count", header.mesh_count);
    builder.end_object();
}

template <typename builder_type>
void tag_invoke(serialize_tag, builder_type & builder, TidoTextureCookResult const & texture)
{
    TidoTextureDescriptor const & descriptor = texture.descriptor;
    builder.start_object();
    builder.append_key_value("kind", std::string_view("texture"));
    builder.append_comma();
    builder.append_key_value("key", hex_u64(texture.cache_key));
    builder.append_comma();
    builder.append_key_value("format", static_cast<u64>(descriptor.format));
    builder.append_comma();
    builder.append_key_value("width", static_cast<u64>(descriptor.width));
    builder.append_comma();
    builder.append_key_value("height", static_cast<u64>(descriptor.height));
    builder.append_comma();
    builder.append_key_value("depth", static_cast<u64>(descriptor.depth));
    builder.append_comma();
    builder.append_key_value("array_layers", static_cast<u64>(descriptor.array_layers));
    builder.append_comma();
    builder.append_key_value("mip_count", static_cast<u64>(descriptor.mip_count));
    builder.append_comma();
    builder.escape_and_append_with_quotes("subresources");
    builder.append_colon();
    builder.start_array();
    for (usize i = 0; i < texture.subresources.size(); ++i)
    {
        if (i != 0) { builder.append_comma(); }
        TidoSubresourceEntry const & subresource = texture.subresources[i];
        builder.start_object();
        builder.append_key_value("offset", dec_u64(subresource.offset));
        builder.append_comma();
        builder.append_key_value("byte_size", static_cast<u64>(subresource.byte_size));
        builder.end_object();
    }
    builder.end_array();
    builder.append_comma();
    builder.append_key_value("path", texture.tido_path.generic_string());
    builder.append_comma();
    builder.append_key_value("source_modified", dec_i64(texture.source_modified));
    builder.append_comma();
    builder.append_key_value("content_hash", hex_u64(texture.content_hash));
    builder.end_object();
}

template <typename builder_type>
void tag_invoke(serialize_tag, builder_type & builder, TidoMeshCookResult const & mesh)
{
    builder.start_object();
    builder.append_key_value("kind", std::string_view("mesh"));
    builder.append_comma();
    builder.append_key_value("key", hex_u64(mesh.cache_key));
    builder.append_comma();
    builder.escape_and_append_with_quotes("lods");
    builder.append_colon();
    builder.start_array();
    for (u32 lod = 0; lod < mesh.descriptor.lod_count; ++lod)
    {
        if (lod != 0) { builder.append_comma(); }
        TidoMeshLodDescriptor const & desc = mesh.lods[lod];
        builder.start_object();
        builder.append_key_value("blob_offset", dec_u64(desc.blob_offset));
        builder.append_comma();
        builder.append_key_value("blob_byte_size", dec_u64(desc.blob_byte_size));
        builder.append_comma();
        // aabb / bounding_sphere are flattened to plain float arrays so each LOD row stays low-complexity
        // (keeps FracturedJson's table alignment applicable to the LOD array).
        builder.escape_and_append_with_quotes("aabb");
        builder.append_colon();
        builder.start_array();
        builder.append(desc.aabb.center.x); builder.append_comma();
        builder.append(desc.aabb.center.y); builder.append_comma();
        builder.append(desc.aabb.center.z); builder.append_comma();
        builder.append(desc.aabb.size.x);   builder.append_comma();
        builder.append(desc.aabb.size.y);   builder.append_comma();
        builder.append(desc.aabb.size.z);
        builder.end_array();
        builder.append_comma();
        builder.escape_and_append_with_quotes("bounding_sphere");
        builder.append_colon();
        builder.start_array();
        builder.append(desc.bounding_sphere.center.x); builder.append_comma();
        builder.append(desc.bounding_sphere.center.y); builder.append_comma();
        builder.append(desc.bounding_sphere.center.z); builder.append_comma();
        builder.append(desc.bounding_sphere.radius);
        builder.end_array();
        builder.append_comma();
        builder.append_key_value("lod_error", desc.lod_error);
        builder.append_comma();
        builder.append_key_value("vertex_count", static_cast<u64>(desc.vertex_count));
        builder.append_comma();
        builder.append_key_value("primitive_count", static_cast<u64>(desc.primitive_count));
        builder.append_comma();
        builder.append_key_value("meshlet_count", static_cast<u64>(desc.meshlet_count));
        builder.append_comma();
        builder.append_key_value("micro_indices_count", static_cast<u64>(desc.micro_indices_count));
        builder.append_comma();
        builder.append_key_value("indirect_vertices_count", static_cast<u64>(desc.indirect_vertices_count));
        builder.append_comma();
        builder.append_key_value("primitive_indices_count", static_cast<u64>(desc.primitive_indices_count));
        builder.append_comma();
        builder.append_key_value("has_uv", static_cast<u64>(desc.has_uv));
        builder.end_object();
    }
    builder.end_array();
    builder.append_comma();
    builder.append_key_value("path", mesh.tido_path.generic_string());
    builder.append_comma();
    builder.append_key_value("source_modified", dec_i64(mesh.source_modified));
    builder.append_comma();
    builder.append_key_value("content_hash", hex_u64(mesh.content_hash));
    builder.end_object();
}
} // namespace simdjson

namespace
{
// Serialize one record to a pretty (FracturedJson) block and append it, blank-line separated. Returns
// false on a serialization or IO failure (via error codes, no exceptions).
template <typename Record>
auto write_record(std::ofstream & ofs, Record const & record) -> bool
{
    std::string compact;
    if (simdjson::to_json(record).get(compact)) { return false; }
    std::string const pretty = simdjson::fractured_json_string(compact);
    std::string_view const out = pretty.empty() ? std::string_view(compact) : std::string_view(pretty);
    ofs.write(out.data(), static_cast<std::streamsize>(out.size()));
    ofs.write("\n\n", 2);
    return static_cast<bool>(ofs);
}
} // namespace

auto write_tido_cache(std::filesystem::path const & cache_path, TidoCacheKey const & key,
    std::span<TidoTextureCookResult const> textures, std::span<TidoMeshCookResult const> meshes) -> bool
{
    std::error_code ec = {};
    std::filesystem::create_directories(cache_path.parent_path(), ec); // ignore "already exists"

    std::ofstream ofs{cache_path, std::ios::binary | std::ios::trunc};
    if (!ofs) { return false; }

    if (!write_record(ofs, TidoCacheHeader{key, textures.size(), meshes.size()})) { return false; }
    for (TidoTextureCookResult const & texture : textures)
    {
        if (!write_record(ofs, texture)) { return false; }
    }
    for (TidoMeshCookResult const & mesh : meshes)
    {
        if (!write_record(ofs, mesh)) { return false; }
    }
    return static_cast<bool>(ofs);
}

auto read_tido_cache(std::filesystem::path const & cache_path) -> std::optional<TidoCache>
{
    simdjson::padded_string json;
    if (simdjson::padded_string::load(cache_path.string()).get(json)) { return std::nullopt; } // absent
    if (json.size() == 0) { return std::nullopt; }

    simdjson::ondemand::parser parser;
    simdjson::ondemand::document_stream stream;
    if (parser.iterate_many(json).get(stream)) { return std::nullopt; }

    TidoCache cache = {};
    bool header_seen = false;
    for (auto document : stream)
    {
        TidoCacheRecord record;
        // We only ever append valid records, so any parse error means the file is corrupt: report and bail.
        if (auto const error = document.get<TidoCacheRecord>().get(record))
        {
            DEBUG_MSG(fmt::format("[read_tido_cache] corrupt .tido_cache '{}': {}",
                cache_path.string(), simdjson::error_message(error)));
            return std::nullopt;
        }
        switch (record.kind)
        {
            case TidoCacheRecord::Kind::HEADER:  cache.key = record.header_key; header_seen = true; break;
            case TidoCacheRecord::Kind::TEXTURE: cache.textures.emplace(record.texture.cache_key, std::move(record.texture)); break;
            case TidoCacheRecord::Kind::MESH:    cache.meshes.emplace(record.mesh.cache_key, std::move(record.mesh)); break;
        }
    }

    if (!header_seen) { return std::nullopt; } // no valid header - treat as no cache
    return cache;
}
