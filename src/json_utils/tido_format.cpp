#include "tido_format.hpp"

#include <charconv>
#include <string>
#include <string_view>

#include <simdjson.h>
#include <fmt/format.h>

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

auto read_i64(simdjson::ondemand::object & obj, char const * key, i64 & out) -> simdjson::error_code
{
    i64 value = 0;
    SIMDJSON_TRY(obj[key].get_int64().get(value));
    out = value;
    return simdjson::SUCCESS;
}

auto read_path(simdjson::ondemand::object & obj, char const * key, std::filesystem::path & out) -> simdjson::error_code
{
    std::string_view text;
    SIMDJSON_TRY(obj[key].get_string().get(text));
    out = std::filesystem::path(text);
    return simdjson::SUCCESS;
}

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

} // namespace

// Custom (de)serializers live in namespace simdjson so ADL finds them.
namespace simdjson
{
template <typename value_type>
auto tag_invoke(deserialize_tag, value_type & value, TidoMetadataHash & hash) -> error_code
{
    ondemand::object obj;
    SIMDJSON_TRY(value.get_object().get(obj));
    SIMDJSON_TRY(read_hex(obj, "cache_key", hash.cache_key));
    SIMDJSON_TRY(read_i64(obj, "source_mtime_at_bake", hash.source_mtime_at_bake));
    SIMDJSON_TRY(read_hex(obj, "content_hash", hash.content_hash));
    SIMDJSON_TRY(read_u32(obj, "version", hash.version));
    return SUCCESS;
}

template <typename value_type>
auto tag_invoke(deserialize_tag, value_type & value, TidoImageDescriptor & image) -> error_code
{
    ondemand::object obj;
    SIMDJSON_TRY(value.get_object().get(obj));

    u32 format_value = 0;
    SIMDJSON_TRY(read_u32(obj, "format", format_value));
    image.info.format = static_cast<daxa::Format>(format_value);

    SIMDJSON_TRY(read_u32(obj, "dimensions", image.info.dimensions));
    SIMDJSON_TRY(read_u32(obj, "width", image.info.size.x));
    SIMDJSON_TRY(read_u32(obj, "height", image.info.size.y));
    SIMDJSON_TRY(read_u32(obj, "depth", image.info.size.z));
    SIMDJSON_TRY(read_u32(obj, "mip_level_count", image.info.mip_level_count));
    SIMDJSON_TRY(read_u32(obj, "array_layer_count", image.info.array_layer_count));

    ondemand::array subresources;
    SIMDJSON_TRY(obj["subresources"].get_array().get(subresources));
    for (auto element : subresources)
    {
        ondemand::object sub;
        SIMDJSON_TRY(element.get_object().get(sub));
        TidoImageDescriptor::SubresourceEntry entry = {};
        SIMDJSON_TRY(read_dec(sub, "offset", entry.offset));
        SIMDJSON_TRY(read_u32(sub, "byte_size", entry.byte_size));
        image.subresources.push_back(entry);
    }

    return SUCCESS;
}

template <typename value_type>
auto tag_invoke(deserialize_tag, value_type & value, TidoMeshDescriptor & mesh) -> error_code
{
    ondemand::object obj;
    SIMDJSON_TRY(value.get_object().get(obj));

    ondemand::array lods;
    SIMDJSON_TRY(obj["lods"].get_array().get(lods));
    for (auto element : lods)
    {
        ondemand::object lod;
        SIMDJSON_TRY(element.get_object().get(lod));
        TidoMeshDescriptor::LodDescriptor desc = {};
        SIMDJSON_TRY(read_dec(lod, "offset", desc.offset));
        SIMDJSON_TRY(read_dec(lod, "byte_size", desc.byte_size));

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

        mesh.lods.push_back(desc);
    }
    return SUCCESS;
}

template <typename builder_type>
void tag_invoke(serialize_tag, builder_type & builder, TidoMetadataHash const & header)
{
    builder.start_object();
    builder.append_key_value("kind", std::string_view("tido_metadata_hash"));
    builder.append_comma();
    builder.append_key_value("cache_key", hex_u64(header.cache_key));
    builder.append_comma();
    builder.append_key_value("source_mtime_at_bake", static_cast<i64>(header.source_mtime_at_bake));
    builder.append_comma();
    builder.append_key_value("content_hash", hex_u64(header.content_hash));
    builder.append_comma();
    builder.append_key_value("version", static_cast<u32>(header.version));
    builder.end_object();
}

template <typename builder_type>
void tag_invoke(serialize_tag, builder_type & builder, TidoImageDescriptor const & image)
{
    builder.start_object();
    builder.append_key_value("kind", std::string_view("tido_image"));
    builder.append_comma();
    builder.append_key_value("format", static_cast<u64>(image.info.format));
    builder.append_comma();
    builder.append_key_value("dimensions", static_cast<u64>(image.info.dimensions));
    builder.append_comma();
    builder.append_key_value("width", static_cast<u64>(image.info.size.x));
    builder.append_comma();
    builder.append_key_value("height", static_cast<u64>(image.info.size.y));
    builder.append_comma();
    builder.append_key_value("depth", static_cast<u64>(image.info.size.z));
    builder.append_comma();
    builder.append_key_value("mip_level_count", static_cast<u64>(image.info.mip_level_count));
    builder.append_comma();
    builder.append_key_value("array_layer_count", static_cast<u64>(image.info.array_layer_count));
    builder.append_comma();
    builder.escape_and_append_with_quotes("subresources");
    builder.append_colon();
    builder.start_array();
    for (usize i = 0; i < image.subresources.size(); ++i)
    {
        if (i != 0) { builder.append_comma(); }
        TidoImageDescriptor::SubresourceEntry const & subresource = image.subresources[i];
        builder.start_object();
        builder.append_key_value("offset", dec_u64(subresource.offset));
        builder.append_comma();
        builder.append_key_value("byte_size", static_cast<u64>(subresource.byte_size));
        builder.end_object();
    }
    builder.end_array();
    builder.end_object();
}

template <typename builder_type>
void tag_invoke(serialize_tag, builder_type & builder, TidoMeshDescriptor const & mesh)
{
    builder.start_object();
    builder.append_key_value("kind", std::string_view("tido_mesh"));
    builder.append_comma();
    builder.escape_and_append_with_quotes("lods");
    builder.append_colon();
    builder.start_array();
    for (u32 lod = 0; lod < mesh.lods.size(); ++lod)
    {
        if (lod != 0) { builder.append_comma(); }
        TidoMeshDescriptor::LodDescriptor const & desc = mesh.lods[lod];
        builder.start_object();
        builder.append_key_value("offset", dec_u64(desc.offset));
        builder.append_comma();
        builder.append_key_value("byte_size", dec_u64(desc.byte_size));
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
    builder.end_object();
}
} // namespace simdjson

namespace
{
// Serialize one record to its pretty-printed (FracturedJson) JSON block. Returns an empty string on a
// serialization failure (via error codes, no exceptions) - never expected for these fixed-shape records.
template <typename Record>
auto serialize_record(Record const & record) -> std::string
{
    std::string compact;
    if (simdjson::to_json(record).get(compact)) { return {}; }
    std::string pretty = simdjson::fractured_json_string(compact);
    return pretty.empty() ? std::move(compact) : std::move(pretty);
}

// A .tido header stores its metadata objects back-to-back followed by the raw binary payload. Each object
// is pretty-printed (so it contains newlines) and the payload is not valid UTF-8, so the boundary can only
// be found structurally: advance cursor past leading whitespace and one brace/bracket-balanced value,
// skipping over string literals, and return that value's text.
auto next_json_object(std::string_view text, usize & cursor) -> std::optional<std::string_view>
{
    while (cursor < text.size() && (text[cursor] == ' ' || text[cursor] == '\n' || text[cursor] == '\r' || text[cursor] == '\t')) { ++cursor; }
    usize const start = cursor;
    if (start >= text.size() || (text[start] != '{' && text[start] != '[')) { return std::nullopt; }

    i32 depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (usize index = start; index < text.size(); ++index)
    {
        char const character = text[index];
        if (in_string)
        {
            if (escaped) { escaped = false; }
            else if (character == '\\') { escaped = true; }
            else if (character == '"') { in_string = false; }
            continue;
        }
        switch (character)
        {
            case '"': in_string = true; break;
            case '{': case '[': ++depth; break;
            case '}': case ']':
                --depth;
                if (depth == 0) { cursor = index + 1; return text.substr(start, cursor - start); }
                break;
            default: break;
        }
    }
    return std::nullopt;
}

template <typename Record>
auto parse_json_block(std::string_view block, char const * context) -> std::optional<Record>
{
    simdjson::padded_string json(block.data(), block.size());
    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    if (parser.iterate(json).get(doc)) { return std::nullopt; }
    Record record = {};
    if (auto const error = doc.get<Record>().get(record); error != simdjson::SUCCESS)
    {
        DEBUG_MSG(fmt::format("[{}] corrupt tido header: {}", context, simdjson::error_message(error)));
        return std::nullopt;
    }
    return record;
}
} // namespace


auto serialize_tido_metadata_hash(TidoMetadataHash const & hash) -> std::string
{
    return serialize_record(hash);
}

auto serialize_tido_image_descriptor(TidoImageDescriptor const & image) -> std::string
{
    return serialize_record(image);
}

auto serialize_tido_mesh_descriptor(TidoMeshDescriptor const & mesh) -> std::string
{
    return serialize_record(mesh);
}

auto read_tido_image_header_data(std::span<std::byte const> data) -> std::optional<std::pair<TidoMetadataHash, TidoImageDescriptor>>
{
    std::string_view const text(r_cast<char const *>(data.data()), data.size());
    usize cursor = 0;

    auto const hash_block = next_json_object(text, cursor);
    auto const image_block = next_json_object(text, cursor);
    if (!hash_block.has_value() || !image_block.has_value()) { return std::nullopt; }

    auto hash = parse_json_block<TidoMetadataHash>(hash_block.value(), "read_tido_image_header_data");
    auto image = parse_json_block<TidoImageDescriptor>(image_block.value(), "read_tido_image_header_data");
    if (!hash.has_value() || !image.has_value()) { return std::nullopt; }

    return std::make_optional(std::make_pair(std::move(hash.value()), std::move(image.value())));
}

auto read_tido_mesh_header_data(std::span<std::byte const> data) -> std::optional<std::pair<TidoMetadataHash, TidoMeshDescriptor>>
{
    std::string_view const text(r_cast<char const *>(data.data()), data.size());
    usize cursor = 0;

    auto const hash_block = next_json_object(text, cursor);
    auto const mesh_block = next_json_object(text, cursor);
    if (!hash_block.has_value() || !mesh_block.has_value()) { return std::nullopt; }

    auto hash = parse_json_block<TidoMetadataHash>(hash_block.value(), "read_tido_mesh_header_data");
    auto mesh = parse_json_block<TidoMeshDescriptor>(mesh_block.value(), "read_tido_mesh_header_data");
    if (!hash.has_value() || !mesh.has_value()) { return std::nullopt; }

    return std::make_optional(std::make_pair(std::move(hash.value()), std::move(mesh.value())));
}