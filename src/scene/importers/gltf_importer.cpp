#include "importer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <span>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fmt/format.h>
#include <glm/gtx/quaternion.hpp>

#include "../optimizers/image_processor.hpp"
#include "../optimizers/geometry_optimizer.hpp"
#include "../tido_format/tido_format.hpp"
#include "../tido_format/tido_util.hpp"

// Per-kind cook versions, stamped into every .gltf_cache this importer writes. Bump the relevant one
// whenever that pipeline's cook output or .tido_bin layout changes; on re-import a mismatching version
// marks all of that kind's cached artifacts stale and recooks them. Versioned independently so a
// texture-cook change does not needlessly recook meshes and vice versa.
static constexpr u32 GLTF_TEXTURE_COOK_VERSION = 1;
static constexpr u32 GLTF_MESH_COOK_VERSION = 1;

// =================== Mesh extraction: glTF accessors -> format-neutral RawMesh ====================
// The only place that reads glTF vertex/index accessors. Mirrors the texture part-1 (load_raw_image):
// glTF-specific extraction producing the generic input the optimizer (optimize_mesh) consumes. No
// cooking, no GPU work.

/// NOTE: Overload ElementTraits for glm vecs so fastgltf understands the types.
template <>
struct fastgltf::ElementTraits<glm::vec4> : fastgltf::ElementTraitsBase<float, fastgltf::AccessorType::Vec4>
{
};
template <>
struct fastgltf::ElementTraits<glm::vec3> : fastgltf::ElementTraitsBase<float, fastgltf::AccessorType::Vec3>
{
};
template <>
struct fastgltf::ElementTraits<glm::vec2> : fastgltf::ElementTraitsBase<float, fastgltf::AccessorType::Vec2>
{
};

namespace
{
static constexpr std::string_view VERT_ATTRIB_POSITION_NAME = "POSITION";
static constexpr std::string_view VERT_ATTRIB_TEXCOORD0_NAME = "TEXCOORD_0";
static constexpr std::string_view VERT_ATTRIB_NORMAL_NAME = "NORMAL";

template <typename ElemT, bool IS_INDEX_BUFFER>
static auto load_accessor_data_from_file(
    std::filesystem::path const & root_path,
    fastgltf::Asset const & gltf_asset,
    fastgltf::Accessor const & accesor)
    -> std::optional<std::vector<ElemT>>
{
    static_assert(!IS_INDEX_BUFFER || std::is_same_v<ElemT, u32>, "Index Buffer must be u32");
    fastgltf::BufferView const & gltf_buffer_view = gltf_asset.bufferViews.at(accesor.bufferViewIndex.value());
    fastgltf::Buffer const & gltf_buffer = gltf_asset.buffers.at(gltf_buffer_view.bufferIndex);
    if (!std::holds_alternative<fastgltf::sources::URI>(gltf_buffer.data))
    {
        return std::nullopt;
    }
    fastgltf::sources::URI uri = std::get<fastgltf::sources::URI>(gltf_buffer.data);

    /// NOTE: load the section of the file containing the buffer for the mesh index buffer.
    std::filesystem::path const full_buffer_path = root_path / uri.uri.fspath();
    std::ifstream ifs{full_buffer_path, std::ios::binary};
    if (!ifs)
    {
        return std::nullopt;
    }
    /// NOTE: Only load the relevant part of the file containing the view of the buffer we actually need.
    ifs.seekg(gltf_buffer_view.byteOffset + accesor.byteOffset + uri.fileByteOffset);
    std::vector<u16> raw = {};
    size_t const elem_byte_size = fastgltf::getElementByteSize(accesor.type, accesor.componentType);
    raw.resize((accesor.count * elem_byte_size) / 2);
    if (!ifs.read(r_cast<char *>(raw.data()), accesor.count * elem_byte_size))
    {
        return std::nullopt;
    }
    auto buffer_adapter = [&]([[maybe_unused]] fastgltf::Asset const & asset, [[maybe_unused]] u32 asset_index) -> fastgltf::span<std::byte const>
    {
        /// NOTE:   We only have a ptr to the loaded data to the accessors section of the buffer.
        ///         Fastgltf expects a ptr to the begin of the buffer VIEW, so we just subtract the offsets.
        ///         Fastgltf adds these on in the accessor tool, so in the end it gets the right ptr.
        return fastgltf::span<std::byte const>(reinterpret_cast<std::byte const *>(raw.data()) - accesor.byteOffset, accesor.count * elem_byte_size);
    };

    std::vector<ElemT> ret(accesor.count);
    if constexpr (IS_INDEX_BUFFER)
    {
        /// NOTE: Transform the loaded file section into a 32 bit index buffer.
        if (accesor.componentType == fastgltf::ComponentType::UnsignedShort)
        {
            std::vector<u16> u16_index_buffer(accesor.count);
            fastgltf::copyFromAccessor<u16>(gltf_asset, accesor, u16_index_buffer.data(), buffer_adapter);
            for (size_t i = 0; i < u16_index_buffer.size(); ++i)
            {
                ret[i] = s_cast<u32>(u16_index_buffer[i]);
            }
        }
        else
        {
            fastgltf::copyFromAccessor<u32>(gltf_asset, accesor, ret.data(), buffer_adapter);
        }
    }
    else
    {
        fastgltf::copyFromAccessor<ElemT>(gltf_asset, accesor, ret.data(), buffer_adapter);
    }
    return ret;
}

// Extract one primitive's index/position/normal/uv streams into the format-neutral RawMesh. Returns
// nullopt (with a logged reason) if the glTF primitive is missing or has invalid required attributes.
static auto extract_raw_mesh(
    fastgltf::Asset const & asset,
    std::filesystem::path const & asset_path,
    u32 gltf_mesh_index,
    u32 gltf_primitive_index) -> std::optional<RawMesh>
{
    std::filesystem::path const root_path = std::filesystem::path{asset_path}.remove_filename();
    fastgltf::Mesh const & gltf_mesh = asset.meshes[gltf_mesh_index];
    fastgltf::Primitive const & gltf_prim = gltf_mesh.primitives[gltf_primitive_index];

    // Indices (required).
    if (!gltf_prim.indicesAccessor.has_value())
    {
        DEBUG_MSG("[GltfImporter::extract_raw_mesh] missing index buffer");
        return std::nullopt;
    }
    fastgltf::Accessor const & index_accessor = asset.accessors.at(gltf_prim.indicesAccessor.value());
    bool const index_accessor_valid =
        (index_accessor.componentType == fastgltf::ComponentType::UnsignedInt ||
            index_accessor.componentType == fastgltf::ComponentType::UnsignedShort) &&
        index_accessor.type == fastgltf::AccessorType::Scalar &&
        index_accessor.bufferViewIndex.has_value();
    if (!index_accessor_valid)
    {
        DEBUG_MSG("[GltfImporter::extract_raw_mesh] faulty index buffer accessor");
        return std::nullopt;
    }
    auto indices = load_accessor_data_from_file<u32, true>(root_path, asset, index_accessor);
    if (!indices.has_value())
    {
        return std::nullopt;
    }

    // Vertex positions (required).
    auto pos_iter = gltf_prim.findAttribute(VERT_ATTRIB_POSITION_NAME);
    if (pos_iter == gltf_prim.attributes.end())
    {
        DEBUG_MSG("[GltfImporter::extract_raw_mesh] missing vertex positions");
        return std::nullopt;
    }
    fastgltf::Accessor const & pos_accessor = asset.accessors.at(pos_iter->accessorIndex);
    if (pos_accessor.componentType != fastgltf::ComponentType::Float || pos_accessor.type != fastgltf::AccessorType::Vec3)
    {
        DEBUG_MSG("[GltfImporter::extract_raw_mesh] faulty vertex positions");
        return std::nullopt;
    }
    auto positions = load_accessor_data_from_file<glm::vec3, false>(root_path, asset, pos_accessor);
    if (!positions.has_value())
    {
        return std::nullopt;
    }

    // Vertex UVs (optional).
    auto uv_iter = gltf_prim.findAttribute(VERT_ATTRIB_TEXCOORD0_NAME);
    bool const has_uv = uv_iter != gltf_prim.attributes.end();
    std::vector<glm::vec2> uvs = {};
    if (has_uv)
    {
        fastgltf::Accessor const & uv_accessor = asset.accessors.at(uv_iter->accessorIndex);
        if (uv_accessor.componentType != fastgltf::ComponentType::Float || uv_accessor.type != fastgltf::AccessorType::Vec2)
        {
            DEBUG_MSG("[GltfImporter::extract_raw_mesh] faulty vertex texcoord0");
            return std::nullopt;
        }
        auto uvs_opt = load_accessor_data_from_file<glm::vec2, false>(root_path, asset, uv_accessor);
        if (!uvs_opt.has_value())
        {
            return std::nullopt;
        }
        uvs = std::move(uvs_opt.value());
        DBG_ASSERT_TRUE_M(uvs.size() == positions.value().size(), "[GltfImporter::extract_raw_mesh] Mismatched position and uv count");
    }

    // Vertex normals (required).
    auto normal_iter = gltf_prim.findAttribute(VERT_ATTRIB_NORMAL_NAME);
    if (normal_iter == gltf_prim.attributes.end())
    {
        DEBUG_MSG("[GltfImporter::extract_raw_mesh] missing vertex normals");
        return std::nullopt;
    }
    fastgltf::Accessor const & normal_accessor = asset.accessors.at(normal_iter->accessorIndex);
    if (normal_accessor.componentType != fastgltf::ComponentType::Float || normal_accessor.type != fastgltf::AccessorType::Vec3)
    {
        DEBUG_MSG("[GltfImporter::extract_raw_mesh] faulty vertex normals");
        return std::nullopt;
    }
    auto normals = load_accessor_data_from_file<glm::vec3, false>(root_path, asset, normal_accessor);
    if (!normals.has_value())
    {
        return std::nullopt;
    }
    DBG_ASSERT_TRUE_M(normals.value().size() == positions.value().size(), "[GltfImporter::extract_raw_mesh] Mismatched position and normal count");

    return RawMesh{
        .indices = std::move(indices.value()),
        .positions = std::move(positions.value()),
        .normals = std::move(normals.value()),
        .uvs = std::move(uvs),
    };
}

// Last-write-time of a single file in filesystem-clock ticks; nullopt if it can't be stat'd (missing /
// unresolved). A nullopt means "unknown", so callers must NOT fast-path a cache hit on it. Read cheaply -
// no file contents. The per-artifact staleness timestamp is built from this (any touched source bumps it).
static auto file_mtime(std::filesystem::path const & path) -> std::optional<i64>
{
    std::error_code ec = {};
    auto const write_time = std::filesystem::last_write_time(path, ec);
    if (ec) { return std::nullopt; }
    return write_time.time_since_epoch().count();
}

// Max file_mtime over `paths` (a mesh's geometry can span several buffer files); nullopt if none stat'd.
static auto max_source_mtime(std::span<std::filesystem::path const> paths) -> std::optional<i64>
{
    std::optional<i64> newest = std::nullopt;
    for (auto const & path : paths)
    {
        if (auto const mtime = file_mtime(path)) { newest = newest.has_value() ? std::max(newest.value(), mtime.value()) : mtime.value(); }
    }
    return newest;
}

// FNV-1a over a primitive's extracted (raw, unoptimized) geometry - all four arrays hashed as one stream.
// This is the authoritative change detector: identical bytes => identical cook, so the .tido_bin is reused.
static auto raw_mesh_content_hash(RawMesh const & raw) -> u64
{
    u64 hash = tido_fnv1a(std::as_bytes(std::span{raw.indices}));
    hash = tido_fnv1a(std::as_bytes(std::span{raw.positions}), hash);
    hash = tido_fnv1a(std::as_bytes(std::span{raw.normals}), hash);
    hash = tido_fnv1a(std::as_bytes(std::span{raw.uvs}), hash);
    return hash;
}

// The distinct external buffer files a primitive's accessors (index / position / uv / normal) read from -
// the mesh's staleness is the max mtime over these. Mirrors extract_raw_mesh's accessor set so the mtime
// tracks exactly the files the cook consumes. Empty if a source is embedded/unsupported (mtime unknown).
static auto mesh_source_paths(fastgltf::Asset const & asset, std::filesystem::path const & asset_path, u32 gltf_mesh_index, u32 gltf_primitive_index) -> std::vector<std::filesystem::path>
{
    std::filesystem::path const root_path = std::filesystem::path{asset_path}.remove_filename();
    std::vector<std::filesystem::path> paths = {};
    auto add_accessor_source = [&](fastgltf::Accessor const & accessor)
    {
        if (!accessor.bufferViewIndex.has_value()) { return; }
        fastgltf::BufferView const & view = asset.bufferViews.at(accessor.bufferViewIndex.value());
        fastgltf::Buffer const & buffer = asset.buffers.at(view.bufferIndex);
        if (auto const * uri = std::get_if<fastgltf::sources::URI>(&buffer.data))
        {
            std::filesystem::path p = root_path / uri->uri.fspath();
            if (std::find(paths.begin(), paths.end(), p) == paths.end()) { paths.push_back(std::move(p)); }
        }
    };
    fastgltf::Mesh const & mesh = asset.meshes.at(gltf_mesh_index);
    fastgltf::Primitive const & prim = mesh.primitives.at(gltf_primitive_index);
    if (prim.indicesAccessor.has_value()) { add_accessor_source(asset.accessors.at(prim.indicesAccessor.value())); }
    for (std::string_view const attr : {VERT_ATTRIB_POSITION_NAME, VERT_ATTRIB_TEXCOORD0_NAME, VERT_ATTRIB_NORMAL_NAME})
    {
        auto const it = prim.findAttribute(attr);
        if (it != prim.attributes.end()) { add_accessor_source(asset.accessors.at(it->accessorIndex)); }
    }
    return paths;
}

// Byte range an accessor's tightly-packed data occupies in its external buffer file. nullopt if the buffer
// is embedded/unsupported or the accessor has no buffer view. Asserts the source is tightly packed - the
// generic cook reads a contiguous range and reinterprets it as a tight array, so interleaved sources aren't
// supported.
static auto locate_accessor_range(fastgltf::Asset const & asset, std::filesystem::path const & root_path, fastgltf::Accessor const & accessor) -> std::optional<FileByteRange>
{
    if (!accessor.bufferViewIndex.has_value()) { return std::nullopt; }
    fastgltf::BufferView const & view = asset.bufferViews.at(accessor.bufferViewIndex.value());
    fastgltf::Buffer const & buffer = asset.buffers.at(view.bufferIndex);
    if (!std::holds_alternative<fastgltf::sources::URI>(buffer.data)) { return std::nullopt; }
    fastgltf::sources::URI const & uri = std::get<fastgltf::sources::URI>(buffer.data);

    u64 const element_byte_size = fastgltf::getElementByteSize(accessor.type, accessor.componentType);
    DBG_ASSERT_TRUE_M(!view.byteStride.has_value() || view.byteStride.value() == element_byte_size, "Mesh accessor source is not tightly packed");
    return FileByteRange{
        .file = root_path / uri.uri.fspath(),
        .byte_offset = view.byteOffset + accessor.byteOffset + uri.fileByteOffset,
        .byte_length = accessor.count * element_byte_size,
    };
}

// Resolve where one primitive's tightly-packed vertex/index streams live without reading them. Mirrors
// extract_raw_mesh's accessor validation (F32 vec3 positions/normals, F32 vec2 uvs, U16|U32 scalar indices).
// nullopt if a required stream is missing/invalid or a source is embedded/unsupported. cache_path is left
// defaulted (routing-only, set by the caller).
static auto resolve_mesh_source(fastgltf::Asset const & asset, std::filesystem::path const & asset_path, u32 gltf_mesh_index, u32 gltf_primitive_index) -> std::optional<MeshImporterData>
{
    std::filesystem::path const root_path = std::filesystem::path{asset_path}.remove_filename();
    fastgltf::Mesh const & gltf_mesh = asset.meshes.at(gltf_mesh_index);
    fastgltf::Primitive const & gltf_prim = gltf_mesh.primitives.at(gltf_primitive_index);

    // Indices (required): U16 or U32 scalar.
    if (!gltf_prim.indicesAccessor.has_value()) { return std::nullopt; }
    fastgltf::Accessor const & index_accessor = asset.accessors.at(gltf_prim.indicesAccessor.value());
    if (index_accessor.type != fastgltf::AccessorType::Scalar) { return std::nullopt; }
    ComponentType index_component_type = {};
    if (index_accessor.componentType == fastgltf::ComponentType::UnsignedShort) { index_component_type = ComponentType::U16; }
    else if (index_accessor.componentType == fastgltf::ComponentType::UnsignedInt) { index_component_type = ComponentType::U32; }
    else { return std::nullopt; }
    auto const index_range = locate_accessor_range(asset, root_path, index_accessor);
    if (!index_range.has_value()) { return std::nullopt; }

    // Positions (required): F32 vec3.
    auto const pos_iter = gltf_prim.findAttribute(VERT_ATTRIB_POSITION_NAME);
    if (pos_iter == gltf_prim.attributes.end()) { return std::nullopt; }
    fastgltf::Accessor const & pos_accessor = asset.accessors.at(pos_iter->accessorIndex);
    if (pos_accessor.componentType != fastgltf::ComponentType::Float || pos_accessor.type != fastgltf::AccessorType::Vec3) { return std::nullopt; }
    auto const pos_range = locate_accessor_range(asset, root_path, pos_accessor);
    if (!pos_range.has_value()) { return std::nullopt; }

    // Normals (required): F32 vec3.
    auto const normal_iter = gltf_prim.findAttribute(VERT_ATTRIB_NORMAL_NAME);
    if (normal_iter == gltf_prim.attributes.end()) { return std::nullopt; }
    fastgltf::Accessor const & normal_accessor = asset.accessors.at(normal_iter->accessorIndex);
    if (normal_accessor.componentType != fastgltf::ComponentType::Float || normal_accessor.type != fastgltf::AccessorType::Vec3) { return std::nullopt; }
    auto const normal_range = locate_accessor_range(asset, root_path, normal_accessor);
    if (!normal_range.has_value()) { return std::nullopt; }
    DBG_ASSERT_TRUE_M(normal_accessor.count == pos_accessor.count, "Mismatched position and normal count");

    // UVs (optional): F32 vec2.
    std::optional<MeshAttribSource> uvs = {};
    auto const uv_iter = gltf_prim.findAttribute(VERT_ATTRIB_TEXCOORD0_NAME);
    if (uv_iter != gltf_prim.attributes.end())
    {
        fastgltf::Accessor const & uv_accessor = asset.accessors.at(uv_iter->accessorIndex);
        if (uv_accessor.componentType != fastgltf::ComponentType::Float || uv_accessor.type != fastgltf::AccessorType::Vec2) { return std::nullopt; }
        auto const uv_range = locate_accessor_range(asset, root_path, uv_accessor);
        if (!uv_range.has_value()) { return std::nullopt; }
        DBG_ASSERT_TRUE_M(uv_accessor.count == pos_accessor.count, "Mismatched position and uv count");
        uvs = MeshAttribSource{.range = uv_range.value(), .component_type = ComponentType::F32};
    }

    return MeshImporterData{
        .indices = MeshAttribSource{.range = index_range.value(), .component_type = index_component_type},
        .positions = MeshAttribSource{.range = pos_range.value(), .component_type = ComponentType::F32},
        .normals = MeshAttribSource{.range = normal_range.value(), .component_type = ComponentType::F32},
        .uvs = uvs,
        .vertex_count = s_cast<u32>(pos_accessor.count),
        .index_count = s_cast<u32>(index_accessor.count),
    };
}
} // namespace

// =================== Texture loading, part 1: load + decode the raw image data ===================
// Reads an image's source bytes out of the glTF (URI / embedded buffer view) and decodes them into
// the format-neutral RawImageData the optimizer understands. This is the only place that knows about
// the glTF image sources and the PNG/KTX2 container formats. No GPU work, no BC compression.
namespace
{
struct ImageFromRawInfo
{
    std::vector<std::byte> raw_data;
    std::filesystem::path image_path;
    fastgltf::MimeType mime_type;
};
using RawDataRet = std::variant<std::monostate, bool, ImageFromRawInfo>; // bool == failure

static auto raw_image_data_from_path(std::filesystem::path const & image_path) -> RawDataRet
{
    std::ifstream ifs{image_path, std::ios::binary};
    if (!ifs)
    {
        return false;
    }
    ifs.seekg(0, ifs.end);
    i64 const filesize = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    std::vector<std::byte> raw(filesize);
    if (!ifs.read(r_cast<char *>(raw.data()), filesize))
    {
        return false;
    }
    return ImageFromRawInfo{.raw_data = std::move(raw), .image_path = image_path, .mime_type = {}};
}

static auto raw_image_data_from_URI(fastgltf::sources::URI const & uri, std::filesystem::path const & scene_dir_path) -> RawDataRet
{
    if (!uri.uri.isLocalPath() || uri.fileByteOffset != 0)
    {
        return false;
    }
    std::filesystem::path const full_image_path = scene_dir_path / uri.uri.fspath();
    RawDataRet raw_image_data_ret = raw_image_data_from_path(full_image_path);
    if (!std::holds_alternative<ImageFromRawInfo>(raw_image_data_ret))
    {
        return raw_image_data_ret;
    }
    ImageFromRawInfo & raw_data = std::get<ImageFromRawInfo>(raw_image_data_ret);
    raw_data.mime_type = uri.mimeType;
    if (uri.uri.string().ends_with(".ktx2"))
    {
        raw_data.mime_type = fastgltf::MimeType::KTX2;
    }
    return raw_data;
}

static auto raw_image_data_from_buffer_view(fastgltf::sources::BufferView const & buffer_view, fastgltf::Asset const & asset, std::filesystem::path const & scene_dir_path) -> RawDataRet
{
    fastgltf::BufferView const & gltf_buffer_view = asset.bufferViews.at(buffer_view.bufferViewIndex);
    fastgltf::Buffer const & gltf_buffer = asset.buffers.at(gltf_buffer_view.bufferIndex);
    if (!std::holds_alternative<fastgltf::sources::URI>(gltf_buffer.data))
    {
        return false;
    }
    fastgltf::sources::URI uri = std::get<fastgltf::sources::URI>(gltf_buffer.data);
    std::filesystem::path const full_buffer_path = scene_dir_path / uri.uri.fspath();
    std::ifstream ifs{full_buffer_path, std::ios::binary};
    if (!ifs)
    {
        return false;
    }
    ifs.seekg(gltf_buffer_view.byteOffset + uri.fileByteOffset);
    std::vector<std::byte> raw = {};
    raw.resize(gltf_buffer_view.byteLength);
    if (!ifs.read(r_cast<char *>(raw.data()), gltf_buffer_view.byteLength))
    {
        return false;
    }
    return ImageFromRawInfo{.raw_data = std::move(raw), .image_path = full_buffer_path, .mime_type = buffer_view.mimeType};
}

// Part 1 entry point: read an image's source bytes out of the glTF and tag their format, ready to be
// handed to the optimizer. Does NOT decode/transcode - that is the optimizer's job.
static auto load_raw_image(fastgltf::Asset const & asset, u32 gltf_image_index, std::filesystem::path const & asset_path) -> std::optional<RawImage>
{
    fastgltf::Image const & fgltf_image = asset.images.at(gltf_image_index);
    std::filesystem::path const scene_dir_path = std::filesystem::path(asset_path).remove_filename();

    RawDataRet ret = {};
    if (auto const * uri = std::get_if<fastgltf::sources::URI>(&fgltf_image.data))
    {
        ret = raw_image_data_from_URI(*uri, scene_dir_path);
    }
    else if (auto const * buffer_view = std::get_if<fastgltf::sources::BufferView>(&fgltf_image.data))
    {
        ret = raw_image_data_from_buffer_view(*buffer_view, asset, scene_dir_path);
    }
    else
    {
        return std::nullopt;
    }
    if (!std::holds_alternative<ImageFromRawInfo>(ret))
    {
        return std::nullopt;
    }
    ImageFromRawInfo & raw_image_data = std::get<ImageFromRawInfo>(ret);

    ImageFileFormat format = {};
    if (raw_image_data.mime_type == fastgltf::MimeType::KTX2)
    {
        format = ImageFileFormat::KTX2;
    }
    else if (raw_image_data.mime_type == fastgltf::MimeType::PNG)
    {
        format = ImageFileFormat::PNG;
    }
    else
    {
        return std::nullopt; // Unsupported source format.
    }

    return RawImage{
        .data = std::move(raw_image_data.raw_data),
        .format = format,
        .name = raw_image_data.image_path.filename().string(),
    };
}

// The external source file an image reads from: the URI image file, or the .bin behind a buffer-view
// image. Empty if the source is embedded/unsupported (mtime unknown). The image's staleness keys off this
// single file (unlike a mesh, whose geometry can span several buffers - see mesh_source_paths).
static auto image_source_path(fastgltf::Asset const & asset, u32 gltf_image_index, std::filesystem::path const & asset_path) -> std::filesystem::path
{
    fastgltf::Image const & image = asset.images.at(gltf_image_index);
    std::filesystem::path const scene_dir_path = std::filesystem::path(asset_path).remove_filename();
    if (auto const * uri = std::get_if<fastgltf::sources::URI>(&image.data))
    {
        if (uri->uri.isLocalPath()) { return {scene_dir_path / uri->uri.fspath()}; }
    }
    else if (auto const * buffer_view = std::get_if<fastgltf::sources::BufferView>(&image.data))
    {
        fastgltf::BufferView const & view = asset.bufferViews.at(buffer_view->bufferViewIndex);
        fastgltf::Buffer const & buffer = asset.buffers.at(view.bufferIndex);
        if (auto const * uri = std::get_if<fastgltf::sources::URI>(&buffer.data))
        {
            return {scene_dir_path / uri->uri.fspath()};
        }
    }
    return {};
}

static auto mime_type_to_image_format(fastgltf::MimeType mime_type) -> std::optional<ImageFileFormat>
{
    if (mime_type == fastgltf::MimeType::KTX2) { return ImageFileFormat::KTX2; }
    if (mime_type == fastgltf::MimeType::PNG) { return ImageFileFormat::PNG; }
    return std::nullopt; // Unsupported source format.
}

struct ImageSourceLocate
{
    FileByteRange range = {};
    ImageFileFormat format = {};
};

// Resolve where an image's encoded bytes live without reading them: the whole URI image file, or a
// bufferView slice (even into a .glb's embedded buffer). nullopt if the source is embedded/unsupported.
static auto resolve_image_source(fastgltf::Asset const & asset, u32 gltf_image_index, std::filesystem::path const & asset_path) -> std::optional<ImageSourceLocate>
{
    fastgltf::Image const & image = asset.images.at(gltf_image_index);
    std::filesystem::path const scene_dir_path = std::filesystem::path(asset_path).remove_filename();

    if (auto const * uri = std::get_if<fastgltf::sources::URI>(&image.data))
    {
        if (!uri->uri.isLocalPath() || uri->fileByteOffset != 0)
        {
            return std::nullopt;
        }
        std::filesystem::path const full_image_path = scene_dir_path / uri->uri.fspath();
        std::error_code size_error = {};
        u64 const file_byte_length = std::filesystem::file_size(full_image_path, size_error);
        if (size_error)
        {
            return std::nullopt;
        }
        fastgltf::MimeType mime_type = uri->mimeType;
        // The URI mime is often unset; the .ktx2 extension is authoritative for basisu textures.
        if (uri->uri.string().ends_with(".ktx2"))
        {
            mime_type = fastgltf::MimeType::KTX2;
        }
        auto const format = mime_type_to_image_format(mime_type);
        if (!format.has_value())
        {
            return std::nullopt;
        }
        return ImageSourceLocate{
            .range = FileByteRange{.file = full_image_path, .byte_offset = 0, .byte_length = file_byte_length},
            .format = format.value(),
        };
    }
    else if (auto const * buffer_view = std::get_if<fastgltf::sources::BufferView>(&image.data))
    {
        fastgltf::BufferView const & gltf_buffer_view = asset.bufferViews.at(buffer_view->bufferViewIndex);
        fastgltf::Buffer const & gltf_buffer = asset.buffers.at(gltf_buffer_view.bufferIndex);
        if (!std::holds_alternative<fastgltf::sources::URI>(gltf_buffer.data))
        {
            return std::nullopt;
        }
        fastgltf::sources::URI const & uri = std::get<fastgltf::sources::URI>(gltf_buffer.data);
        std::filesystem::path const full_buffer_path = scene_dir_path / uri.uri.fspath();
        auto const format = mime_type_to_image_format(buffer_view->mimeType);
        if (!format.has_value())
        {
            return std::nullopt;
        }
        return ImageSourceLocate{
            .range = FileByteRange{
                .file = full_buffer_path,
                .byte_offset = gltf_buffer_view.byteOffset + uri.fileByteOffset,
                .byte_length = gltf_buffer_view.byteLength,
            },
            .format = format.value(),
        };
    }
    return std::nullopt;
}
} // namespace

// ============================ Shared parse + per-artifact identity keys ===========================
namespace
{
// Parses a .gltf/.glb into a fastgltf::Asset. Both the scene-parse task and the asset-import batch call
// this - the two task kinds are fully decoupled, so an asset batch re-parses its source rather than
// sharing the scene parse's asset.
static auto parse_gltf_file(std::filesystem::path const & file_path) -> std::variant<Scene::LoadManifestErrorCode, fastgltf::Asset>
{
    fastgltf::Parser parser{
        fastgltf::Extensions::KHR_texture_basisu |
        fastgltf::Extensions::KHR_lights_punctual};

    constexpr auto gltf_options =
        fastgltf::Options::DontRequireValidAssetMember |
        fastgltf::Options::AllowDouble;

    auto data_opt = fastgltf::GltfDataBuffer::FromPath(file_path);
    if (data_opt.error() != fastgltf::Error::None)
    {
        return Scene::LoadManifestErrorCode::FILE_NOT_FOUND;
    }
    fastgltf::GltfDataBuffer data = std::move(data_opt.get());
    auto const type = fastgltf::determineGltfFileType(data);

    switch (type)
    {
        case fastgltf::GltfType::glTF:
        {
            fastgltf::Expected<fastgltf::Asset> result = parser.loadGltf(data, file_path.parent_path(), gltf_options);
            if (result.error() != fastgltf::Error::None)
            {
                return Scene::LoadManifestErrorCode::COULD_NOT_LOAD_ASSET;
            }
            return std::move(result.get());
        }
        case fastgltf::GltfType::GLB:
        {
            fastgltf::Expected<fastgltf::Asset> result = parser.loadGltfBinary(data, file_path.parent_path(), gltf_options);
            if (result.error() != fastgltf::Error::None)
            {
                return Scene::LoadManifestErrorCode::COULD_NOT_LOAD_ASSET;
            }
            return std::move(result.get());
        }
        default:
            return Scene::LoadManifestErrorCode::INVALID_GLTF_FILE_TYPE;
    }
}

// Stable per-image-artifact source-identity key.
static auto image_identity_key(ImageImporterData const & importer_data) -> u64
{
    std::vector<std::byte> importer_data_as_bytes;
    // Location: the resolved source byte range (path + offset + length); cache_path is routing-only and excluded.
    std::string const source_path_string = importer_data.source_bytes.file.generic_string();
    importer_data_as_bytes.insert(importer_data_as_bytes.end(), source_path_string.begin(), source_path_string.end());
    importer_data_as_bytes.insert(importer_data_as_bytes.end(),
        r_cast<std::byte const *>(&importer_data.source_bytes.byte_offset),
        r_cast<std::byte const *>(&importer_data.source_bytes.byte_offset) + sizeof(importer_data.source_bytes.byte_offset));
    importer_data_as_bytes.insert(importer_data_as_bytes.end(),
        r_cast<std::byte const *>(&importer_data.source_bytes.byte_length),
        r_cast<std::byte const *>(&importer_data.source_bytes.byte_length) + sizeof(importer_data.source_bytes.byte_length));
    // Recipe: container format joins the channel mapping and target format so one source blob used under
    // different recipes gets distinct keys.
    importer_data_as_bytes.insert(importer_data_as_bytes.end(),
        r_cast<std::byte const *>(&importer_data.container_format),
        r_cast<std::byte const *>(&importer_data.container_format) + sizeof(importer_data.container_format));
    for(auto const & mapped_channel : importer_data.channel_mapping)
    {
        importer_data_as_bytes.push_back(static_cast<std::byte>(mapped_channel));
    }
    importer_data_as_bytes.insert(importer_data_as_bytes.end(),
        r_cast<std::byte const *>(&importer_data.target_format),
        r_cast<std::byte const *>(&importer_data.target_format) + sizeof(importer_data.target_format));

    return tido_fnv1a(importer_data_as_bytes, 0);
}

// Stable per-primitive source-identity key.
static auto mesh_cache_key(fastgltf::Asset const & asset, std::filesystem::path const & file_path, u32 gltf_mesh_index, u32 gltf_primitive_index) -> u64
{
    return tido_source_identity_key(file_path, asset.meshes[gltf_mesh_index].name.c_str(),
        fmt::format("{}.{}", gltf_mesh_index, gltf_primitive_index));
}
} // namespace

// ====================== ImportScene: parse -> SceneMetadataBatch (no cooking) =====================
namespace
{

// Parses one glTF/GLB file on a ThreadPool worker and translates it into a single SceneMetadataBatch
// result - manifest metadata only, with importer_data provenance filled and streamer/runtime data empty.
// Never touches the Scene or the cache; SceneRuntime applies the batch on the main thread and pushes
// back the ImportAsset tasks that drive the actual cooking.
//
// Flow:
//   1. collect_referenced_images - walk all materials, resolve each used texture to its IMAGE and
//      collect the set of images actually used (a glTF may contain unreferenced images, which we
//      skip) plus each image's type. We key on images, not textures: several textures can share one
//      image (differing only by sampler), and that image must be loaded only once.
//   2. add_texture_batch_entries / add_mesh_batch_entries - add every referenced image's/primitive's
//      metadata-only batch entry.
//   3. translate_materials / translate_mesh_groups / translate_entities - a material's/group's leaves
//      already have batch entries -> wire up the batch's own cross-references.

struct SceneParseTask final : Task
{
    SceneParseTask(std::filesystem::path file_path, Importer * importer)
        : file_path{std::move(file_path)}, importer{importer}
    {
        chunk_count = 1;
    }

    void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override;

  private:
    std::filesystem::path file_path = {};
    Importer * importer = {};

    fastgltf::Asset asset;

    ImporterTaskResult::SceneMetadataBatch batch = {};

    void translate_materials();
    void translate_mesh_groups();
    auto translate_entities() -> u32;
    auto translate_light(fastgltf::Light const & light) -> u32;
};

void SceneParseTask::callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index)
{
    auto parse_result = parse_gltf_file(file_path);
    if (auto const * error = std::get_if<Scene::LoadManifestErrorCode>(&parse_result))
    {
        DEBUG_MSG(fmt::format("[WARN][SceneParseTask::callback] Loading \"{}\" Error: {}", file_path.string(), Scene::to_string(*error)));
        importer->push_result(ImporterTaskResult{.data = ImporterTaskResult::Error{
            .kind = ImporterTaskResult::Error::TaskKind::IMPORT_SCENE,
            .source = file_path,
            .message = std::string{Scene::to_string(*error)},
        }});
        return;
    }
    asset = std::move(std::get<fastgltf::Asset>(parse_result));

    translate_materials();
    translate_mesh_groups();
    batch.root_entity_index = translate_entities();

    importer->push_result(ImporterTaskResult{.data = std::move(batch)});
}

void SceneParseTask::translate_materials()
{
    // image_source_key -> image_manifest_index
    std::unordered_map<u64, u32> image_manifest_map = {};

    auto gltf_texture_to_image_index = [&](u32 const gltf_texture_index) -> std::optional<u32>
    {
        auto const & texture = asset.textures.at(gltf_texture_index);
        if (texture.basisuImageIndex.has_value())
        {
            return s_cast<u32>(texture.basisuImageIndex.value());
        }
        else if (texture.imageIndex.has_value())
        {
            return s_cast<u32>(texture.imageIndex.value());
        }
        DBG_ASSERT_TRUE_M(false, "Texture type does not have image index nor basisu image index - we do not support dds or webp textures currently");
        return std::nullopt;
    };

    enum struct GLTFTextureMaterialType
    {
        NONE,
        DIFFUSE,
        OPACITY,
        NORMAL,
        ROUGHNESS_METALNESS,
    };

    auto default_image_import_info = [&](GLTFTextureMaterialType const texture_type, u32 const image_index) -> ImageImporterData
    {
        auto const source = resolve_image_source(asset, image_index, file_path);
        DBG_ASSERT_TRUE_M(source.has_value(), "Unsupported or unresolvable image source");
        FileByteRange const source_bytes = source.has_value() ? source->range : FileByteRange{};
        ImageFileFormat const container_format = source.has_value() ? source->format : ImageFileFormat{};

        switch(texture_type)
        {
            case GLTFTextureMaterialType::DIFFUSE:
                return ImageImporterData{ .cache_path = file_path, .source_bytes = source_bytes, .container_format = container_format, .channel_mapping = {0, 1, 2}, .target_format = daxa::Format::BC7_SRGB_BLOCK, };
            case GLTFTextureMaterialType::OPACITY:
                return ImageImporterData{ .cache_path = file_path, .source_bytes = source_bytes, .container_format = container_format, .channel_mapping = {3},  .target_format = daxa::Format::BC4_UNORM_BLOCK, };
            case GLTFTextureMaterialType::NORMAL:
                return ImageImporterData{ .cache_path = file_path, .source_bytes = source_bytes, .container_format = container_format, .channel_mapping = {0, 1, 2},  .target_format = daxa::Format::BC5_UNORM_BLOCK, };
            case GLTFTextureMaterialType::ROUGHNESS_METALNESS:
                return ImageImporterData{ .cache_path = file_path, .source_bytes = source_bytes, .container_format = container_format, .channel_mapping = {0, 1, 2, 3}, .target_format = daxa::Format::BC7_UNORM_BLOCK, };
            default:
                DBG_ASSERT_TRUE_M(false, "Unhandled texture type in default_image_import_info");
                return {};
        }
    };

    auto resolve_image_info = [&](ImporterTaskResult::SceneMetadataBatch::Image const & image_data, u32 const sampler_index) -> std::optional<MaterialManifestEntry::ImageInfo>
    {
        u64 const identity_key = image_identity_key(image_data.importer_data);
        auto const [iterator, inserted] = image_manifest_map.try_emplace(identity_key, s_cast<u32>(batch.images.size()));
        u32 const manifest_index = iterator->second;
        if (inserted)
        {
            batch.images.push_back(image_data);
        }

        // Sanity check: the same image under the same recipe must always resolve to the same manifest entry.
        DBG_ASSERT_TRUE_M(image_identity_key(batch.images.at(manifest_index).importer_data) == identity_key, "Image manifest entry mismatch");

        return MaterialManifestEntry::ImageInfo{.image_manifest_index = manifest_index, .sampler_index = sampler_index};
    };

    for (u32 material_index = 0; material_index < s_cast<u32>(asset.materials.size()); material_index++)
    {
        auto const & material = asset.materials.at(material_index);
        std::optional<MaterialManifestEntry::ImageInfo> diffuse_texture_info = {};
        std::optional<MaterialManifestEntry::ImageInfo> opacity_texture_info = {};
        std::optional<MaterialManifestEntry::ImageInfo> normal_texture_info = {};
        std::optional<MaterialManifestEntry::ImageInfo> roughness_metalness_info = {};
        if (material.pbrData.baseColorTexture.has_value())
        {
            auto const gltf_image_index = gltf_texture_to_image_index(s_cast<u32>(material.pbrData.baseColorTexture.value().textureIndex));
            auto const gltf_texture_name = asset.images.at(gltf_image_index.value()).name;
            diffuse_texture_info = resolve_image_info({.importer_data = default_image_import_info(GLTFTextureMaterialType::DIFFUSE, gltf_image_index.value())}, 0);
        }
        if(material.alphaMode == fastgltf::AlphaMode::Mask && material.pbrData.baseColorTexture.has_value())
        {
            auto const gltf_image_index = gltf_texture_to_image_index(s_cast<u32>(material.pbrData.baseColorTexture.value().textureIndex));
            auto const gltf_texture_name = asset.images.at(gltf_image_index.value()).name;
            opacity_texture_info = resolve_image_info({.importer_data = default_image_import_info(GLTFTextureMaterialType::OPACITY, gltf_image_index.value())}, 0);
        }
        if (material.normalTexture.has_value())
        {
            auto const gltf_image_index = gltf_texture_to_image_index(s_cast<u32>(material.normalTexture.value().textureIndex));
            auto const gltf_texture_name = asset.images.at(gltf_image_index.value()).name;
            normal_texture_info = resolve_image_info({.importer_data = default_image_import_info(GLTFTextureMaterialType::NORMAL, gltf_image_index.value())}, 0);
        }
        if (material.pbrData.metallicRoughnessTexture.has_value())
        {
            auto const gltf_image_index = gltf_texture_to_image_index(s_cast<u32>(material.pbrData.metallicRoughnessTexture.value().textureIndex));
            auto const gltf_texture_name = asset.images.at(gltf_image_index.value()).name;
            roughness_metalness_info = resolve_image_info({.importer_data = default_image_import_info(GLTFTextureMaterialType::ROUGHNESS_METALNESS, gltf_image_index.value())}, 0);
        }

        bool const alpha_discard_enabled = material.alphaMode == fastgltf::AlphaMode::Mask && opacity_texture_info.has_value();
        DBG_ASSERT_TRUE_M(!alpha_discard_enabled || opacity_texture_info.has_value(), "Alpha discard enabled but no opacity texture info");

        batch.materials.push_back(ImporterTaskResult::SceneMetadataBatch::Material{
            .diffuse_info = diffuse_texture_info,
            .opacity_mask_info = opacity_texture_info,
            .normal_info = normal_texture_info,
            .roughness_metalness_info = roughness_metalness_info,
            .alpha_discard_enabled = alpha_discard_enabled,
            .double_sided = material.doubleSided,
            .blend_enabled = material.alphaMode == fastgltf::AlphaMode::Blend,
            .base_color = f32vec3(material.pbrData.baseColorFactor[0], material.pbrData.baseColorFactor[1], material.pbrData.baseColorFactor[2]),
            .emissive_color = f32vec3(material.emissiveFactor[0] * material.emissiveStrength, material.emissiveFactor[1] * material.emissiveStrength, material.emissiveFactor[2] * material.emissiveStrength),
            .name = material.name.c_str(),
        });
    }
}

void SceneParseTask::translate_mesh_groups()
{
    /// NOTE: fastgltf::Mesh is a MeshGroup, fastgltf::Primitive is a Mesh (MeshLodGroup).
    batch.mesh_groups.reserve(asset.meshes.size());
    for (u32 mesh_group_index = 0; mesh_group_index < s_cast<u32>(asset.meshes.size()); ++mesh_group_index)
    {
        auto const & gltf_mesh = asset.meshes.at(mesh_group_index);
        auto & mesh_group = batch.mesh_groups.emplace_back( std::vector<u32>{}, gltf_mesh.name.c_str());
        mesh_group.mesh_lod_group_indices.reserve(gltf_mesh.primitives.size());

        for (u32 primitive_index = 0; primitive_index < s_cast<u32>(gltf_mesh.primitives.size()); ++primitive_index)
        {
            auto const & gltf_primitive = gltf_mesh.primitives.at(primitive_index);

            u32 const mesh_manifest_index = s_cast<u32>(batch.mesh_lod_groups.size());
            auto mesh_importer_data_opt = resolve_mesh_source(asset, file_path, mesh_group_index, primitive_index);
            DBG_ASSERT_TRUE_M(mesh_importer_data_opt.has_value(), "Unresolvable or unsupported mesh primitive source");
            MeshImporterData mesh_importer_data = mesh_importer_data_opt.value_or(MeshImporterData{});
            mesh_importer_data.cache_path = file_path;
            batch.mesh_lod_groups.push_back(ImporterTaskResult::SceneMetadataBatch::MeshLodGroup{
                .material_index = std::optional<u32>(gltf_primitive.materialIndex.value_or(std::nullopt)),
                .name = gltf_mesh.name.c_str(),
                .importer_data = std::move(mesh_importer_data),
            });
            mesh_group.mesh_lod_group_indices.push_back(mesh_manifest_index);
        }
    }
}

auto SceneParseTask::translate_light(fastgltf::Light const & light) -> u32
{
    f32 const LUMENS_PER_WATT = 683.0f;
    // Defines the minimum energy of a light before cutoff.
    // TODO(msakmary) hook this up to UI?
    f32 const E_min = 1.0f;

    switch (light.type)
    {
        case fastgltf::LightType::Point:
        {
            ImporterTaskResult::SceneMetadataBatch::PointLight cpu_point_light = {};
            cpu_point_light.position = f32vec3{0.0f, 0.0f, 0.0f}; // Filled/updated later when processing scene graph
            cpu_point_light.color = f32vec3{light.color.x(), light.color.y(), light.color.z()};
            // Converting candella to watt - blender (https://projects.blender.org/blender/blender-addons/issues/91035).
            cpu_point_light.intensity = (light.intensity * 4.0f * glm::pi<f32>()) / LUMENS_PER_WATT;
            // When the cutoff is not specified attempt to calculate one based on a minimum energy.
            cpu_point_light.cutoff = light.range.value_or(std::sqrt(light.intensity / E_min));
            u32 const index = s_cast<u32>(batch.point_lights.size());
            batch.point_lights.push_back(cpu_point_light);
            return index;
        }
        case fastgltf::LightType::Spot:
        {
            ImporterTaskResult::SceneMetadataBatch::SpotLight cpu_spot_light = {};
            cpu_spot_light.transform = {}; // Filled/updated later when processing scene graph
            cpu_spot_light.color = f32vec3{light.color.x(), light.color.y(), light.color.z()};
            // Converting candella to watt - https://google.github.io/filament/Filament.md.html#lighting
            cpu_spot_light.intensity = (light.intensity * glm::pi<f32>()) / LUMENS_PER_WATT;
            cpu_spot_light.inner_cone_angle = light.innerConeAngle.value();
            cpu_spot_light.outer_cone_angle = light.outerConeAngle.value();
            DBG_ASSERT_TRUE_M(light.range.has_value(), "Currently no auto deduce of range from intensity for spot lights");
            cpu_spot_light.cutoff = light.range.value();
            u32 const index = s_cast<u32>(batch.spot_lights.size());
            batch.spot_lights.push_back(cpu_spot_light);
            return index;
        }
        case fastgltf::LightType::Directional:
        {
            // TODO(msakmary) add handling of directional lights.
            DBG_ASSERT_TRUE_M(false, "TODO(msakmary) implement directional lights");
            return s_cast<u32>(-1);
        }
        default:
            DBG_ASSERT_TRUE_M(false, "Unhandled fastgltf::LightType");
            return s_cast<u32>(-1);
    }
}

auto SceneParseTask::translate_entities() -> u32
{
    /// NOTE: fastgltf::Node is Entity
    DBG_ASSERT_TRUE_M(asset.nodes.size() != 0, "[ERROR][SceneParseTask::translate_entities()] Empty node array - what to do now?");

    u32 const node_count = s_cast<u32>(asset.nodes.size());
    // The imported subtree's root entity, parenting every parentless node entity (wired below), sits one
    // past the node entities in this batch's local index space (node index == local index).
    u32 const root_entity_index = node_count;

    std::vector<ImporterTaskResult::SceneMetadataBatch::Entity> node_entities(node_count + 1);

    for (u32 node_index = 0; node_index < node_count; node_index++)
    {
        // TODO: For now store transform as a matrix - later should be changed to something else (TRS: translation, rotor, scale).
        auto fastgltf_to_glm_mat4x3_transform = [](std::variant<fastgltf::TRS, fastgltf::math::fmat4x4> const & trans) -> glm::mat4x3
        {
            glm::mat4x3 ret_trans;
            if (auto const * trs = std::get_if<fastgltf::TRS>(&trans))
            {
                auto const scale = glm::scale(glm::identity<glm::mat4x4>(), glm::vec3(trs->scale[0], trs->scale[1], trs->scale[2]));
                auto const rotation = glm::toMat4(glm::quat(trs->rotation[3], trs->rotation[0], trs->rotation[1], trs->rotation[2]));
                auto const translation = glm::translate(glm::identity<glm::mat4x4>(), glm::vec3(trs->translation[0], trs->translation[1], trs->translation[2]));
                auto const rotated_scaled = rotation * scale;
                auto const translated_rotated_scaled = translation * rotated_scaled;
                /// NOTE: As the last row is always (0,0,0,1) we dont store it.
                ret_trans = glm::mat4x3(translated_rotated_scaled);
            }
            else if (auto const * mat_trs = std::get_if<fastgltf::math::fmat4x4>(&trans))
            {
                // Gltf and glm matrices are column major.
                ret_trans = glm::mat4x3(*reinterpret_cast<glm::mat4x4 const *>(mat_trs->data()));
            }
            return ret_trans;
        };

        fastgltf::Node const & node = asset.nodes[node_index];
        ImporterTaskResult::SceneMetadataBatch::Entity & r_ent = node_entities[node_index];
        r_ent.mesh_group_manifest_index = std::optional<u32>(node.meshIndex.value_or(std::nullopt));
        r_ent.transform = fastgltf_to_glm_mat4x3_transform(node.transform);
        r_ent.name = node.name.c_str();

        r_ent.light_index = std::optional<u32>(std::nullopt);

        DBG_ASSERT_TRUE_M(
            s_cast<u32>(node.lightIndex.has_value()) +
                    s_cast<u32>(node.meshIndex.has_value()) +
                    s_cast<u32>(node.cameraIndex.has_value()) <=
                1u,
            "Node can only be of one type");

        if (node.lightIndex.has_value())
        {
            fastgltf::Light const & light = asset.lights.at(node.lightIndex.value());
            r_ent.light_index = translate_light(light);
            r_ent.type = light.type == fastgltf::LightType::Point ? EntityType::POINT_LIGHT : EntityType::SPOT_LIGHT;
        }
        else if (node.meshIndex.has_value())
        {
            r_ent.type = EntityType::MESHGROUP;
        }
        else if (node.cameraIndex.has_value())
        {
            r_ent.type = EntityType::CAMERA;
        }
        else if (!node.children.empty())
        {
            r_ent.type = EntityType::TRANSFORM;
        }

        if (!node.children.empty())
        {
            node_entities[node_index].first_child_index = s_cast<u32>(node.children[0]);
        }

        for (u32 curr_child_vec_idx = 0; curr_child_vec_idx < node.children.size(); curr_child_vec_idx++)
        {
            u32 const curr_child_node_idx = s_cast<u32>(node.children[curr_child_vec_idx]);
            node_entities[curr_child_node_idx].parent_index = node_index;
            bool const has_next_sibling = curr_child_vec_idx < (node.children.size() - 1ull);
            if (has_next_sibling)
            {
                node_entities[curr_child_node_idx].next_sibling_index = s_cast<u32>(node.children[curr_child_vec_idx + 1]);
            }
        }
    }

    /// NOTE: Find all root render entities (aka render entities that have no parent) and store them as
    //        Child root entites under scene root node
    ImporterTaskResult::SceneMetadataBatch::Entity & root_r_ent = node_entities[root_entity_index];
    // Named after the source file only; SceneRuntime appends the running import count when it applies
    // the batch, keeping repeat imports distinguishable.
    root_r_ent = ImporterTaskResult::SceneMetadataBatch::Entity{
        .transform = glm::mat4x3(glm::identity<glm::mat4x3>()),
        .type = EntityType::ROOT,
        .name = file_path.filename().replace_extension("").string(),
    };

    std::optional<u32> root_r_ent_prev_child_node_index = {};
    for (u32 node_index = 0; node_index < node_count; node_index++)
    {
        if (!node_entities[node_index].parent_index.has_value())
        {
            node_entities[node_index].parent_index = root_entity_index;
            if (!root_r_ent_prev_child_node_index.has_value()) // First child
            {
                node_entities[root_entity_index].first_child_index = node_index;
            }
            else // We have other root children already
            {
                node_entities[root_r_ent_prev_child_node_index.value()].next_sibling_index = node_index;
            }
            root_r_ent_prev_child_node_index = node_index;
        }
    }

    batch.entities = std::move(node_entities);
    return root_entity_index;
}
} // namespace

// =================== ImportAsset: per-source batch, cook chunks, cache writes ====================
namespace
{
// Cooks a batch's cache-missed texture artifacts, one chunk per artifact.
struct TextureCookTask final : Task
{
    struct Item
    {
        u32 gltf_image_index = {};
        u64 cache_key = {};
        u32 manifest_index = {};
        i64 current_mtime = {}; // current max source mtime, stamped onto the (re)cooked or refreshed artifact
        // Pre-seeded cache entry that failed the mtime fast path; reused if its content hash still matches.
        std::optional<TidoTextureCookResult> cached = {};
    };

    // Immutable for the run of the task: set once at construction (before dispatch) and only ever read
    // from callback, which may run concurrently across chunks - nothing here is mutated after dispatch.
    // keep_alive holds the batch task owning the parsed asset for as long as any chunk is still dispatched.
    fastgltf::Asset const * const asset;
    std::filesystem::path const asset_path;
    std::filesystem::path const cache_dir; // per-import output folder for the .tido_bin data files
    std::vector<Item> const items;
    std::shared_ptr<SourceContext> const context;
    Importer * const importer;
    std::shared_ptr<Task> const keep_alive;

    TextureCookTask(fastgltf::Asset const * asset, std::filesystem::path asset_path, std::vector<Item> items,
        std::shared_ptr<SourceContext> context, Importer * importer, std::shared_ptr<Task> keep_alive)
        : asset{asset}, asset_path{std::move(asset_path)}, cache_dir{context->cache_dir},
          items{std::move(items)}, context{std::move(context)}, importer{importer}, keep_alive{std::move(keep_alive)}
    {
        chunk_count = s_cast<u32>(this->items.size());
    }

    void run_cook(u32 chunk_index)
    {
        Item const & item = items.at(chunk_index);

        // Part 1: read the raw image bytes (+ tag their source format).
        auto raw = load_raw_image(*asset, item.gltf_image_index, asset_path);
        if (!raw.has_value())
        {
            DEBUG_MSG(fmt::format("[ERROR] Failed to load image index {} name {}", item.gltf_image_index, asset->images.at(item.gltf_image_index).name));
            return;
        }

        // Verify whether the content of the image actually changed using a content hash over the raw data.
        u64 const content_hash = tido_fnv1a(std::as_bytes(std::span{raw.value().data}));
        if (item.cached.has_value() && item.cached->content_hash == content_hash && std::filesystem::exists(item.cached->streamer_data.bin_source))
        {
            // Bytes unchanged (only the mtime moved): reuse the cached .tido_bin, just refresh its stored mtime.
            TidoTextureCookResult artifact = item.cached.value();
            artifact.source_modified = item.current_mtime;
            context->store_texture(artifact);
            importer->push_result(ImporterTaskResult{.data = ImporterTaskResult::CookedAsset{
                .streamer_data = item.cached->streamer_data,
                .manifest_index = item.manifest_index,
            }});
            return;
        }

        // Part 2: process the raw bytes into GPU-ready cooked CPU memory (decode/transcode/compress).
        OptimizeTextureInfo const optimize_info = {};
        auto processed_ret = process_image(raw.value(), optimize_info);
        if (auto const * error = std::get_if<ImageOptimizeError>(&processed_ret))
        {
            if (*error == ImageOptimizeError::SOURCE_HAS_NO_ALPHA)
            {
                DEBUG_MSG(fmt::format("[WARN] Image '{}' is Mask-sampled by a material but has no alpha channel - its opacity manifest entry will never become resident", asset->images.at(item.gltf_image_index).name));
            }
            else
            {
                DEBUG_MSG(fmt::format("[ERROR] Failed to process image index {} name {}", item.gltf_image_index, asset->images.at(item.gltf_image_index).name));
            }
            return;
        }
        ProcessedImage const & processed = std::get<ProcessedImage>(processed_ret);

        // Part 3: write the cooked image out as a .tido_bin artifact. The cache key (recipe-tag
        // disambiguated) is hashed into the stem, so artifacts sharing a source image never collide.
        std::string const artifact_name = raw.value().name;
        auto tido_result = write_texture_tido(processed, cache_dir, artifact_name, item.cache_key);
        if (!tido_result.has_value())
        {
            DEBUG_MSG(fmt::format("[WARN][write_texture_tido] failed to write .tido_bin for image '{}'", artifact_name));
            return;
        }

        DEBUG_MSG(fmt::format("[write_texture_tido] cooked '{}' -> {} ({}x{}, {} mips) -> '{}'",
            artifact_name, s_cast<u32>(processed.image_info.format), processed.image_info.size.x,
            processed.image_info.size.y, processed.mips_to_copy, tido_result.value().streamer_data.bin_source.string()));

        TidoTextureCookResult artifact = tido_result.value();
        artifact.source_modified = item.current_mtime;
        artifact.content_hash = content_hash;
        context->store_texture(artifact);
        importer->push_result(ImporterTaskResult{.data = ImporterTaskResult::CookedAsset{
            .streamer_data = tido_result.value().streamer_data,
            .manifest_index = item.manifest_index,
        }});
    }

    void callback(u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        run_cook(chunk_index);
        context->outstanding_asset_imports.fetch_sub(1, std::memory_order_acq_rel);
        importer->notify();
    }
};

// Cooks a batch's cache-missed mesh artifacts, one chunk per artifact.
struct MeshCookTask final : Task
{
    struct Item
    {
        u32 gltf_mesh_index = {};
        u32 gltf_primitive_index = {};
        u64 cache_key = {};
        u32 manifest_index = {};
        i64 current_mtime = {}; // current max source mtime, stamped onto the (re)cooked or refreshed artifact
        // Pre-seeded cache entry that failed the mtime fast path; reused if its content hash still matches.
        std::optional<TidoMeshCookResult> cached = {};
    };

    // Immutable for the run of the task: set once at construction (before dispatch) and only ever read
    // from callback, which may run concurrently across chunks - nothing here is mutated after dispatch.
    // keep_alive holds the batch task owning the parsed asset for as long as any chunk is still dispatched.
    fastgltf::Asset const * const asset;
    std::filesystem::path const asset_path;
    std::filesystem::path const cache_dir; // per-import output folder for the .tido_bin data files
    std::vector<Item> const items;
    std::shared_ptr<SourceContext> const context;
    Importer * const importer;
    std::shared_ptr<Task> const keep_alive;

    MeshCookTask(fastgltf::Asset const * asset, std::filesystem::path asset_path, std::vector<Item> items,
        std::shared_ptr<SourceContext> context, Importer * importer, std::shared_ptr<Task> keep_alive)
        : asset{asset}, asset_path{std::move(asset_path)}, cache_dir{context->cache_dir},
          items{std::move(items)}, context{std::move(context)}, importer{importer}, keep_alive{std::move(keep_alive)}
    {
        chunk_count = s_cast<u32>(this->items.size());
    }

    void push_cooked_mesh(TidoMeshCookResult const & artifact, u32 manifest_index)
    {
        context->store_mesh(artifact);
        importer->push_result(ImporterTaskResult{.data = ImporterTaskResult::CookedAsset{
            .streamer_data = artifact.streamer_data, // .tido_bin reference; streamed in by the scene.
            .manifest_index = manifest_index,
        }});
    }

    void run_cook(u32 chunk_index)
    {
        Item const & item = items.at(chunk_index);
        std::string const mesh_name = std::string(asset->meshes[item.gltf_mesh_index].name.c_str()) + "." + std::to_string(item.gltf_primitive_index);
        // On a failed (re)cook, fall back to the last good cook (the pre-seeded cache entry, if any) and
        // refresh its stored mtime so an un-processable source is not re-flagged "out of date" every
        // import. No fallback (first cook) -> nothing is reported -> the entry stays permanently un-streamed.
        auto keep_cached_fallback = [&]
        {
            if (item.cached.has_value())
            {
                TidoMeshCookResult artifact = item.cached.value();
                artifact.source_modified = item.current_mtime;
                push_cooked_mesh(artifact, item.manifest_index);
            }
        };

        // Part 1: extract the glTF accessors into a format-neutral RawMesh (importer).
        auto raw = extract_raw_mesh(*asset, asset_path, item.gltf_mesh_index, item.gltf_primitive_index);
        if (!raw.has_value())
        {
            DEBUG_MSG(fmt::format("[ERROR] Failed to extract mesh group {} mesh {}",
                item.gltf_mesh_index, item.gltf_primitive_index));
            keep_cached_fallback();
            return;
        }
        // Content hash of the raw extracted geometry: the authoritative change detector.
        u64 const content_hash = raw_mesh_content_hash(raw.value());
        if (item.cached.has_value() && item.cached->content_hash == content_hash && std::filesystem::exists(item.cached->streamer_data.bin_source))
        {
            // Geometry unchanged (only the mtime moved): keep the existing .tido_bin, just refresh the
            // stored mtime so the next import fast-paths without reading the source again.
            TidoMeshCookResult artifact = item.cached.value();
            artifact.source_modified = item.current_mtime;
            push_cooked_mesh(artifact, item.manifest_index);
            return;
        }
        // Part 2: cook the raw streams into the runtime form (optimizer).
        ProcessedMesh const processed = optimize_mesh(raw.value());
        // Part 3: write the cooked mesh out as a .tido_bin artifact.
        auto tido_result = write_mesh_tido(processed, cache_dir, mesh_name, item.cache_key);
        if (!tido_result.has_value())
        {
            DEBUG_MSG(fmt::format("[WARN][write_mesh_tido] failed to write .tido_bin for mesh '{}'", mesh_name));
            keep_cached_fallback();
            return;
        }
        DEBUG_MSG(fmt::format("[write_mesh_tido] cooked '{}' ({} LODs) -> '{}'",
            mesh_name, tido_result.value().streamer_data.descriptor.lod_count, tido_result.value().streamer_data.bin_source.string()));
        TidoMeshCookResult artifact = tido_result.value();
        artifact.source_modified = item.current_mtime;
        artifact.content_hash = content_hash;
        push_cooked_mesh(artifact, item.manifest_index);
    }

    void callback(u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        run_cook(chunk_index);
        context->outstanding_asset_imports.fetch_sub(1, std::memory_order_acq_rel);
        importer->notify();
    }
};

// One source's grouped ImportAsset tasks. Parses the source once, then resolves every artifact: an
// mtime fast path serves cache hits straight from the SourceContext, everything else fans out into
// per-artifact cook chunks that keep this task (and thus the parsed asset) alive via shared_ptr.
struct GltfAssetImportTask final : Task, std::enable_shared_from_this<GltfAssetImportTask>
{
    struct TextureItem
    {
        u32 gltf_image_index = {};
        u32 manifest_index = {};
    };
    struct MeshItem
    {
        u32 gltf_mesh_index = {};
        u32 gltf_primitive_index = {};
        u32 manifest_index = {};
    };

    Importer * importer = {};
    std::shared_ptr<SourceContext> context = {};
    std::vector<TextureItem> texture_items = {};
    std::vector<MeshItem> mesh_items = {};
    // Parsed in callback before any cook chunk is dispatched; cook chunks read it through keep_alive.
    fastgltf::Asset asset;

    GltfAssetImportTask(Importer * importer, std::shared_ptr<SourceContext> context,
        std::vector<TextureItem> texture_items, std::vector<MeshItem> mesh_items)
        : importer{importer}, context{std::move(context)},
          texture_items{std::move(texture_items)}, mesh_items{std::move(mesh_items)}
    {
        chunk_count = 1;
    }

    void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        auto parse_result = parse_gltf_file(context->source_path);
        if (auto const * error = std::get_if<Scene::LoadManifestErrorCode>(&parse_result))
        {
            // The whole batch dies with the parse (e.g. the source changed or vanished since the scene
            // import); its manifest entries simply never become resident.
            DEBUG_MSG(fmt::format("[WARN][GltfAssetImportTask::callback] Loading \"{}\" Error: {}", context->source_path.string(), Scene::to_string(*error)));
            importer->push_result(ImporterTaskResult{.data = ImporterTaskResult::Error{
                .kind = ImporterTaskResult::Error::TaskKind::IMPORT_ASSET,
                .source = context->source_path,
                .message = std::string{Scene::to_string(*error)},
            }});
            context->outstanding_asset_imports.fetch_sub(s_cast<u32>(texture_items.size() + mesh_items.size()), std::memory_order_acq_rel);
            importer->notify();
            return;
        }
        asset = std::move(std::get<fastgltf::Asset>(parse_result));

        resolve_texture_items();
        resolve_mesh_items();
        importer->notify(); // fast-path hits may have decremented the outstanding count / dirtied the cache.
    }

  private:
    void resolve_texture_items()
    {
        // For each texture item decide:
        //   - mtime fast path: a cached entry whose stored source mtime still matches (and whose .tido_bin
        //     exists) is reused WITHOUT reading the source at all.
        //   - otherwise it becomes a chunk of the cook task, which reads the source, content-hashes it, and
        //     either reuses the cached .tido_bin (bytes unchanged, refresh the mtime) or recooks the image.
        u32 fast_hits = 0;
        std::vector<TextureCookTask::Item> items_to_cook = {};
        for (TextureItem const & item : texture_items)
        {
            u64 const key = image_cache_key(asset, context->source_path, item.gltf_image_index, item.type);
            std::optional<TidoTextureCookResult> cached = context->lookup_texture(key);
            std::filesystem::path const src_path = image_source_path(asset, item.gltf_image_index, context->source_path);
            std::optional<i64> const src_mtime = src_path.empty() ? std::nullopt : file_mtime(src_path);

            // mtime fast path: reuse the cached .tido_bin without reading the source.
            if (cached.has_value() && src_mtime.has_value() && cached->source_modified == src_mtime.value() && std::filesystem::exists(cached->streamer_data.bin_source))
            {
                importer->push_result(ImporterTaskResult{.data = ImporterTaskResult::CookedAsset{
                    .streamer_data = cached->streamer_data,
                    .manifest_index = item.manifest_index,
                }});
                ++fast_hits;
                continue;
            }
            items_to_cook.push_back(TextureCookTask::Item{
                .gltf_image_index = item.gltf_image_index,
                .cache_key = key,
                .manifest_index = item.manifest_index,
                .current_mtime = src_mtime.value_or(0),
                .cached = std::move(cached),
            });
        }

        if (fast_hits > 0)
        {
            context->outstanding_asset_imports.fetch_sub(fast_hits, std::memory_order_acq_rel);
        }
        // One chunk per artifact; the chunks own the remaining outstanding decrements.
        u32 const cook_count = s_cast<u32>(items_to_cook.size());
        if (cook_count > 0)
        {
            auto task = std::make_shared<TextureCookTask>(&asset, context->source_path, std::move(items_to_cook), context, importer, shared_from_this());
            importer->thread_pool->async_dispatch(task, TaskPriority::LOW);
        }

        DEBUG_MSG(fmt::format("[GltfAssetImportTask::resolve_texture_items] '{}': {} textures ({} mtime-hit, {} read)",
            context->source_path.filename().string(), fast_hits + cook_count, fast_hits, cook_count));
    }

    void resolve_mesh_items()
    {
        // Same decision per mesh item as resolve_texture_items: mtime fast path or a cook chunk.
        u32 fast_hits = 0;
        std::vector<MeshCookTask::Item> items_to_cook = {};
        for (MeshItem const & item : mesh_items)
        {
            u64 const key = mesh_cache_key(asset, context->source_path, item.gltf_mesh_index, item.gltf_primitive_index);
            std::optional<TidoMeshCookResult> cached = context->lookup_mesh(key);
            std::vector<std::filesystem::path> const src_paths = mesh_source_paths(asset, context->source_path, item.gltf_mesh_index, item.gltf_primitive_index);
            std::optional<i64> const src_mtime = max_source_mtime(src_paths);

            // mtime fast path: reuse the cached .tido_bin without reading the source.
            if (cached.has_value() && src_mtime.has_value() && cached->source_modified == src_mtime.value() &&
                std::filesystem::exists(cached->streamer_data.bin_source))
            {
                importer->push_result(ImporterTaskResult{.data = ImporterTaskResult::CookedAsset{
                    .streamer_data = cached->streamer_data,
                    .manifest_index = item.manifest_index,
                }});
                ++fast_hits;
                continue;
            }
            items_to_cook.push_back(MeshCookTask::Item{
                .gltf_mesh_index = item.gltf_mesh_index,
                .gltf_primitive_index = item.gltf_primitive_index,
                .cache_key = key,
                .manifest_index = item.manifest_index,
                .current_mtime = src_mtime.value_or(0),
                .cached = std::move(cached),
            });
        }

        if (fast_hits > 0)
        {
            context->outstanding_asset_imports.fetch_sub(fast_hits, std::memory_order_acq_rel);
        }
        // One chunk per artifact; the chunks own the remaining outstanding decrements.
        u32 const cook_count = s_cast<u32>(items_to_cook.size());
        if (cook_count > 0)
        {
            auto task = std::make_shared<MeshCookTask>(&asset, context->source_path, std::move(items_to_cook), context, importer, shared_from_this());
            importer->thread_pool->async_dispatch(task, TaskPriority::LOW);
        }

        DEBUG_MSG(fmt::format("[GltfAssetImportTask::resolve_mesh_items] '{}': {} meshes ({} mtime-hit, {} read)",
            context->source_path.filename().string(), fast_hits + cook_count, fast_hits, cook_count));
    }
};
} // namespace

// ======================================= GltfImporter =============================================

GltfImporter::GltfImporter(Importer * importer)
    : _importer{importer}, _cache_registry{importer}
{
}

void GltfImporter::update(std::vector<ImporterTask> & tasks)
{
    // Group this drain's asset tasks by source, so one batch shares one parse and one cache.
    // SceneRuntime pushes a whole source's tasks under one lock, so a drain sees the group together.
    struct PendingBatch
    {
        std::filesystem::path source_path = {};
        std::vector<GltfAssetImportTask::TextureItem> texture_items = {};
        std::vector<GltfAssetImportTask::MeshItem> mesh_items = {};
    };
    std::vector<PendingBatch> pending_batches = {};
    auto batch_for = [&](std::filesystem::path const & source_path) -> PendingBatch &
    {
        for (PendingBatch & pending_batch : pending_batches)
        {
            if (pending_batch.source_path == source_path) { return pending_batch; }
        }
        pending_batches.push_back(PendingBatch{.source_path = source_path});
        return pending_batches.back();
    };

    auto consume_task = [&](ImporterTask & task) -> bool
    {
        if (auto const * import_scene = std::get_if<ImporterTask::ImportScene>(&task.data))
        {
            _importer->thread_pool->async_dispatch(std::make_shared<SceneParseTask>(import_scene->path, _importer), TaskPriority::LOW);
            return true;
        }
        if (auto * import_texture = std::get_if<ImporterTask::ImportTextureAsset>(&task.data))
        {
            auto const * gltf_data = std::get_if<ImageManifestEntry::GltfImporterData>(&import_texture->importer_data);
            if (gltf_data == nullptr) { return false; } // Not glTF provenance - another importer's task.
            batch_for(gltf_data->src_gltf).texture_items.push_back(GltfAssetImportTask::TextureItem{
                .gltf_image_index = gltf_data->image_index,
                .type = import_texture->type,
                .manifest_index = import_texture->image_manifest_index,
            });
            return true;
        }
        if (auto * import_mesh = std::get_if<ImporterTask::ImportMeshAsset>(&task.data))
        {
            auto const * gltf_data = std::get_if<MeshLodGroupManifestEntry::GltfImporterData>(&import_mesh->importer_data);
            if (gltf_data == nullptr) { return false; } // Not glTF provenance - another importer's task.
            batch_for(gltf_data->src_gltf).mesh_items.push_back(GltfAssetImportTask::MeshItem{
                .gltf_mesh_index = gltf_data->mesh_index,
                .gltf_primitive_index = gltf_data->primitive_index,
                .manifest_index = import_mesh->mesh_manifest_index,
            });
            return true;
        }
        return false;
    };
    std::erase_if(tasks, consume_task);

    for (PendingBatch & pending_batch : pending_batches)
    {
        std::filesystem::path const cache_dir = gltf_cache_dir(pending_batch.source_path);
        std::shared_ptr<SourceContext> context = _cache_registry.find_or_create(pending_batch.source_path,
            cache_dir, cache_dir / gltf_cache_file_name(pending_batch.source_path), GLTF_TEXTURE_COOK_VERSION, GLTF_MESH_COOK_VERSION);
        // Taken before dispatch so the count can never cross zero while the batch's items are unresolved;
        // each item's resolution (fast path, cook chunk, or parse failure) releases exactly one.
        context->outstanding_asset_imports.fetch_add(
            s_cast<u32>(pending_batch.texture_items.size() + pending_batch.mesh_items.size()), std::memory_order_relaxed);
        auto task = std::make_shared<GltfAssetImportTask>(
            _importer, std::move(context), std::move(pending_batch.texture_items), std::move(pending_batch.mesh_items));
        _importer->thread_pool->async_dispatch(task, TaskPriority::LOW);
    }

    _cache_registry.run_upkeep();
}
