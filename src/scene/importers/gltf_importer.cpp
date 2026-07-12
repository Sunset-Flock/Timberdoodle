#include "gltf_importer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <span>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fmt/format.h>
#include <glm/gtx/quaternion.hpp>

#include "../asset_processor.hpp"
#include "../optimizers/image_optimizer.hpp"
#include "../optimizers/geometry_optimizer.hpp"
#include "../streamer.hpp"
#include "../tido_format/tido_cache.hpp"
#include "../tido_format/tido_mesh.hpp"
#include "../tido_format/tido_util.hpp"
#include "../../json_utils/tido_cache.hpp" // read_tido_cache + incremental .tido_cache write API (JSON, simdjson)

// Per-kind cook versions, stamped into every .tido_cache this importer writes. Bump the relevant one
// whenever that pipeline's cook output or .tido layout changes; on re-import a mismatching version marks
// all of that kind's cached artifacts stale and recooks them (T5). Versioned independently so a texture-
// cook change does not needlessly recook meshes and vice versa.
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
// This is the authoritative change detector: identical bytes => identical cook, so the .tido is reused.
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
static auto load_raw_image(fastgltf::Asset const & asset, u32 gltf_image_index, std::filesystem::path const & asset_path, TextureMaterialType type) -> std::optional<OptimizeImageInfo>
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

    return OptimizeImageInfo{
        .data = std::move(raw_image_data.raw_data),
        .format = format,
        .type = type,
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
} // namespace

GltfImporter::GltfImporter(Scene & scene, Scene::LoadManifestInfo const & info)
    : scene{scene}, info{info}
{
}

auto GltfImporter::import() -> std::variant<ImporterTaskResult::SceneMetadataBatch, Scene::LoadManifestErrorCode>
{
    if (auto const error = parse(); error.has_value())
    {
        return error.value();
    }

    // The pipeline cooks each leaf resource and only then adds it to the local batch, so every entry is
    // already streamable the moment SceneRuntime appends it; each "translate" pass that references those
    // leaves runs after they are added. Textures and meshes mirror each other: collect/cook/add the leaf,
    // then a translate pass wires the consumer (materials reference images; mesh groups reference meshes).
    collect_referenced_images(); // pass 1: resolve each used texture to its image (+ type); skip unreferenced.
    load_cache();                // load the shared .tido_cache once (used by load_images + load_meshes).
    // A fully valid cache is reused verbatim and left untouched - the optimal path opens nothing. Otherwise
    // something must be (re)cooked, so open the cache for a fresh rewrite: load_images / load_meshes then
    // stream every artifact's record into it as its cook drains.
    rewriting_cache = !validate_cache();
    if (rewriting_cache) { open_cache_writer(); }
    load_images();               // pass 2: cook every referenced image, then add it to the local batch.
    translate_materials();       // pass 3: a material's images are in the batch -> its entry is complete -> add it.
    load_meshes();               // pass 4: cook every mesh, then add it to the local batch.
    translate_mesh_groups();     // pass 5: a group's meshes are in the batch -> add the group over them.
    batch.root_entity_index = translate_entities(); // pass 6: entities reference mesh groups.

    return std::move(batch);
}

void GltfImportTask::callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index)
{
    result = GltfImporter{*scene, info}.import();
    finished.store(true, std::memory_order_release);
}

auto GltfImporter::parse() -> std::optional<Scene::LoadManifestErrorCode>
{
    file_path = info.root_path / info.asset_name;

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
            asset = std::move(result.get());
            break;
        }
        case fastgltf::GltfType::GLB:
        {
            fastgltf::Expected<fastgltf::Asset> result = parser.loadGltfBinary(data, file_path.parent_path(), gltf_options);
            if (result.error() != fastgltf::Error::None)
            {
                return Scene::LoadManifestErrorCode::COULD_NOT_LOAD_ASSET;
            }
            asset = std::move(result.get());
            break;
        }
        default:
            return Scene::LoadManifestErrorCode::INVALID_GLTF_FILE_TYPE;
    }

    // Only used as a suffix for the root entity name; not a manifest offset.
    import_index = scene.lock().root_entity_count();
    return std::nullopt;
}

void GltfImporter::collect_referenced_images()
{
    // First pass: walk every material, resolve each referenced texture to its IMAGE, and record the
    // image's type. An image left as NONE is unreferenced and will be skipped (not added/loaded). The
    // resolved type is also what the cook uses to pick a BC format, so it must be known before load.
    // Keying on images (not textures) means an image shared by several textures is loaded only once.
    image_types.assign(asset.images.size(), TextureMaterialType::NONE);
    image_needs_opacity_split.assign(asset.images.size(), false);
    auto set_image_type = [&](u32 const gltf_texture_index, TextureMaterialType const type)
    {
        auto const gltf_image_idx_opt = gltf_texture_to_image_index(gltf_texture_index);
        if (!gltf_image_idx_opt.has_value())
        {
            return; // Texture references no supported image - nothing to load.
        }
        TextureMaterialType & current = image_types.at(gltf_image_idx_opt.value());
        if (current != type)
        {
            DBG_ASSERT_TRUE_M(current == TextureMaterialType::NONE, "ERROR: Found an image used by different materials as DIFFERENT types!");
            current = type;
        }
    };
    for (auto const & material : asset.materials)
    {
        if (material.pbrData.baseColorTexture.has_value())
        {
            u32 const gltf_texture_index = s_cast<u32>(material.pbrData.baseColorTexture.value().textureIndex);
            set_image_type(gltf_texture_index, TextureMaterialType::DIFFUSE);
            // A material samples this image as a Mask-mode alpha cutoff, not merely for its color -
            // only masked usage earns a dedicated opacity artifact (an image with a real alpha channel
            // that no material actually masks against stays diffuse-only; see image_needs_opacity_split).
            if (material.alphaMode == fastgltf::AlphaMode::Mask)
            {
                auto const gltf_image_idx_opt = gltf_texture_to_image_index(gltf_texture_index);
                if (gltf_image_idx_opt.has_value())
                {
                    image_needs_opacity_split.at(gltf_image_idx_opt.value()) = true;
                }
            }
        }
        if (material.normalTexture.has_value())
        {
            set_image_type(s_cast<u32>(material.normalTexture.value().textureIndex), TextureMaterialType::NORMAL);
        }
        if (material.pbrData.metallicRoughnessTexture.has_value())
        {
            set_image_type(s_cast<u32>(material.pbrData.metallicRoughnessTexture.value().textureIndex), TextureMaterialType::ROUGHNESS_METALNESS);
        }
    }
}

auto GltfImporter::gltf_texture_to_image_index(u32 const gltf_texture_index) -> std::optional<u32>
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
    return std::nullopt;
}

void GltfImporter::load_cache()
{
    // Load this source file's shared .tido_cache. It is LOCATED by the source-path hash; whether its
    // entries are usable is decided per kind by the cook version (a bumped version stales all of that
    // kind), and then per artifact by mtime/content hash in load_images/load_meshes. The source file
    // merely changing (a new entity, reordered nodes) no longer invalidates anything on its own. Stored as
    // members so both load_images and load_meshes reuse the one loaded cache.
    TidoCacheKey const current_key = tido_make_cache_key(file_path, GLTF_TEXTURE_COOK_VERSION, GLTF_MESH_COOK_VERSION);
    cache_output_dir = tido_cache_dir(info.asset_name.string(), current_key.source_hash);
    loaded_cache = read_tido_cache(cache_output_dir / tido_cache_file_name(info.asset_name.string(), current_key.source_hash));
    texture_cache_valid = loaded_cache.has_value() && loaded_cache->key.texture_cook_version == current_key.texture_cook_version;
    mesh_cache_valid = loaded_cache.has_value() && loaded_cache->key.mesh_cook_version == current_key.mesh_cook_version;

    if (!loaded_cache.has_value())
    {
        DEBUG_MSG(fmt::format("[GltfImporter::load_cache] '{}': no cache - cooking everything", info.asset_name.string()));
    }
    else
    {
        DEBUG_MSG(fmt::format("[GltfImporter::load_cache] '{}': cache loaded ({} texture + {} mesh entries); textures {}, meshes {}",
            info.asset_name.string(), loaded_cache->textures.size(), loaded_cache->meshes.size(),
            texture_cache_valid ? "valid" : "stale (cook version changed) - recooking",
            mesh_cache_valid ? "valid" : "stale (cook version changed) - recooking"));
    }
}

// Stable per-image source-identity key. Recomputed identically on re-import so a valid cache can be
// hit without cooking. It also forms the .tido file stem, so it folds in the source path to stay
// unique across ALL imported files (the .tido files share one cache dir) - identical scheme to
// mesh_cache_key (tido_source_identity_key), with the gltf image index as the in-file disambiguator.
auto GltfImporter::image_cache_key(u32 gltf_image_index) -> u64
{
    return tido_source_identity_key(file_path, asset.images[gltf_image_index].name.c_str(), fmt::format("{}", gltf_image_index));
}

// Same source image as image_cache_key, disambiguated with an "_opacity" suffix so the split opacity
// artifact gets its own cache entry and .tido stem instead of colliding with the color artifact's.
auto GltfImporter::image_opacity_cache_key(u32 gltf_image_index) -> u64
{
    return tido_source_identity_key(file_path, asset.images[gltf_image_index].name.c_str(), fmt::format("{}_opacity", gltf_image_index));
}

// Stable per-primitive source-identity key. It both indexes the mesh in the .tido_cache AND forms the
// .tido data-file stem, so it must be unique across ALL imported files (the .tido files share one
// cache dir): the source file path is folded in alongside the (mesh, primitive) index. Without the
// path, two scenes that both have e.g. an unnamed mesh at index 0 would hash to the same key, write to
// the same .tido, and clobber each other (→ "LOD blob out of .tido bounds" when loading the second).
// The path also disambiguates byte-identical primitives within a file (so parallel cook tasks never
// race on one .tido). Textures use the identical scheme (tido_source_identity_key) for their stem.
auto GltfImporter::mesh_cache_key(u32 gltf_mesh_index, u32 gltf_primitive_index) -> u64
{
    return tido_source_identity_key(file_path, asset.meshes[gltf_mesh_index].name.c_str(),
        fmt::format("{}.{}", gltf_mesh_index, gltf_primitive_index));
}

auto GltfImporter::validate_cache() -> bool
{
    // No cache, or a bumped cook version, stales a whole kind - the cache cannot be reused as-is.
    if (!loaded_cache.has_value() || !texture_cache_valid || !mesh_cache_valid) { return false; }

    // Every referenced artifact must have a cached entry whose source is unchanged (same mtime) and whose
    // .tido is still on disk - the same fast-path test load_images/load_meshes apply per artifact. A single
    // miss (absent entry, moved mtime, deleted .tido) means something will be (re)cooked, so the cache is
    // rewritten rather than appended to.
    for (u32 image_index = 0; image_index < s_cast<u32>(asset.images.size()); ++image_index)
    {
        if (image_types.at(image_index) == TextureMaterialType::NONE) { continue; } // unreferenced - not cooked
        std::optional<TidoTextureCookResult> const cached = loaded_cache->lookup_texture(image_cache_key(image_index));
        std::filesystem::path const src_path = image_source_path(asset, image_index, file_path);
        std::optional<i64> const src_mtime = src_path.empty() ? std::nullopt : file_mtime(src_path);
        if (!cached.has_value() || !src_mtime.has_value() || cached->source_modified != src_mtime.value() ||
            !std::filesystem::exists(cached->tido_path))
        {
            return false;
        }
    }
    for (u32 mesh_group_index = 0; mesh_group_index < s_cast<u32>(asset.meshes.size()); ++mesh_group_index)
    {
        for (u32 primitive_index = 0; primitive_index < s_cast<u32>(asset.meshes[mesh_group_index].primitives.size()); ++primitive_index)
        {
            std::optional<TidoMeshCookResult> const cached = loaded_cache->lookup_mesh(mesh_cache_key(mesh_group_index, primitive_index));
            std::optional<i64> const src_mtime = max_source_mtime(mesh_source_paths(asset, file_path, mesh_group_index, primitive_index));
            if (!cached.has_value() || !src_mtime.has_value() || cached->source_modified != src_mtime.value() ||
                !std::filesystem::exists(cached->tido_path))
            {
                return false;
            }
        }
    }
    return true;
}

void GltfImporter::open_cache_writer()
{
    // Truncates the cache file and writes the header record; load_images / load_meshes then append one record
    // per artifact as its cook drains.
    DBG_ASSERT_TRUE_M(rewriting_cache, "[ERROR][GltfImporter::open_cache_writer] called without a rewrite decided (rewriting_cache is false)");
    TidoCacheKey const key = tido_make_cache_key(file_path, GLTF_TEXTURE_COOK_VERSION, GLTF_MESH_COOK_VERSION);
    std::filesystem::path const cache_path = cache_output_dir / tido_cache_file_name(info.asset_name.string(), key.source_hash);
    std::error_code ec = {};
    std::filesystem::create_directories(cache_path.parent_path(), ec); // ignore "already exists"
    cache_stream.open(cache_path, std::ios::binary | std::ios::trunc);
    if (!cache_stream)
    {
        DEBUG_MSG(fmt::format("[WARN][GltfImporter::open_cache_writer] failed to open .tido_cache for '{}'",
            file_path.string()));
        return;
    }
    // The file leads with the header record; load_images / load_meshes append one record per artifact after it.
    write_cache_record(serialize_tido_cache_header(key));
}

void GltfImporter::write_cache_record(std::string const & record)
{
    DBG_ASSERT_TRUE_M(!record.empty(), "write_cache_record: record serialization failed - caller must not write an empty record");
    std::lock_guard<std::mutex> lock{cache_write_mutex};
    // Not open means open_cache_writer's file-open failed - a real I/O failure it already logged (WARN), not
    // a programming error. Skip the write; the artifact simply recooks next import.
    if (!cache_stream) { return; }
    // Records are separated by a single newline so simdjson's document-stream reader sees distinct top-level
    // documents; flush per record so an interrupted import leaves a valid partial cache.
    cache_stream.write(record.data(), static_cast<std::streamsize>(record.size()));
    cache_stream.write("\n", 1);
    cache_stream.flush();
}

void GltfImporter::load_images()
{
    // Second pass: cook every referenced image, then add each to the local batch with its cooked result
    // in hand. An image is added ONLY after it is cooked, so the moment SceneRuntime appends it, it is
    // already streamable. Each cook runs extract (load_raw_image, importer) -> process_image (optimizer)
    // -> write_texture_tido (.tido on disk) - no GPU work (exactly like load_meshes). One task holds a
    // chunk per image that needs cooking; the chunks run in parallel and each adds its own image on
    // completion (add_cooked is thread-safe), so there is no separate collection pass.
    image_manifest_indices.assign(asset.images.size(), INVALID_MANIFEST_INDEX);
    opacity_manifest_indices.assign(asset.images.size(), INVALID_MANIFEST_INDEX);

    // Records a cooked texture (fresh cook or cache hit) into the local batch and, when a cache rewrite is
    // underway, appends its record to the open .tido_cache. Thread-safe: batch_mutex guards the batch and
    // write_cache_record guards the cache stream, and each call writes a distinct image_manifest_indices/
    // opacity_manifest_indices slot, so cook chunks add concurrently. cache_key is already set on the
    // artifact. Used only by the cook task's completion callback below - the fast path (single-threaded, no
    // cook chunks in flight) adds its batch directly further down instead of going through this per entry.
    // opacity_artifact is only present for a DIFFUSE image that genuinely had alpha; it becomes its own
    // OPACITY-typed manifest entry only when image_needs_opacity_split also says a Mask material uses it.
    auto add_cooked_texture = [&](u32 gltf_image_index, TidoTextureCookResult const & artifact, std::optional<TidoTextureCookResult> const & opacity_artifact)
    {
        if (rewriting_cache) { write_cache_record(serialize_tido_cache_texture(artifact)); }
        std::lock_guard<std::mutex> lock{batch_mutex};
        u32 const image_manifest_index = s_cast<u32>(batch.textures.size());
        batch.textures.push_back(TextureManifestEntry{
            .type = image_types.at(gltf_image_index),
            .material_manifest_indices = {},          // Back-refs are filled by SceneRuntime when it appends materials (pass 3).
            .name = asset.images[gltf_image_index].name.c_str(),
            .streamer_data = make_texture_streamer_data(artifact), // .tido reference; streamed in by the scene.
            .importer_data = TextureManifestEntry::GltfImporterData{.src_gltf = file_path, .image_index = gltf_image_index},
        });
        image_manifest_indices.at(gltf_image_index) = image_manifest_index;

        if (opacity_artifact.has_value() && image_needs_opacity_split.at(gltf_image_index))
        {
            if (rewriting_cache) { write_cache_record(serialize_tido_cache_texture(opacity_artifact.value())); }
            u32 const opacity_manifest_index = s_cast<u32>(batch.textures.size());
            batch.textures.push_back(TextureManifestEntry{
                .type = TextureMaterialType::OPACITY,
                .material_manifest_indices = {},
                .name = std::string(asset.images[gltf_image_index].name.c_str()) + "_opacity",
                .streamer_data = make_texture_streamer_data(opacity_artifact.value()),
                .importer_data = TextureManifestEntry::GltfImporterData{.src_gltf = file_path, .image_index = gltf_image_index},
            });
            opacity_manifest_indices.at(gltf_image_index) = opacity_manifest_index;
        }
    };

    struct LoadImagesTask final : Task
    {
        struct Item
        {
            u32 gltf_image_index = {};
            TextureMaterialType type = {};
            u64 cache_key = {};
            u64 opacity_cache_key = {}; // only meaningful when type == DIFFUSE
            bool needs_opacity_split = {}; // only meaningful when type == DIFFUSE; see image_needs_opacity_split
            i64 current_mtime = {}; // current max source mtime, stamped onto the (re)cooked or refreshed artifact
            // Pre-seeded cache entries that failed the mtime fast path; reused if their content hash still
            // matches. cached_opacity is absent when the previous cook found no alpha for this image.
            std::optional<TidoTextureCookResult> cached_color = {};
            std::optional<TidoTextureCookResult> cached_opacity = {};
        };

        // Immutable for the run of the task: set once at construction (before dispatch) and only ever read
        // from callback, which may run concurrently across chunks - nothing here is mutated after dispatch.
        fastgltf::Asset const * const asset;
        std::filesystem::path const asset_path;
        std::filesystem::path const cache_dir; // per-import output folder for the .tido data files
        std::vector<Item> const items;
        std::function<void(u32 gltf_image_index, TidoTextureCookResult const & artifact, std::optional<TidoTextureCookResult> const & opacity_artifact)> const add_cooked;

        LoadImagesTask(fastgltf::Asset const * asset, std::filesystem::path asset_path, std::filesystem::path cache_dir,
            std::vector<Item> items, std::function<void(u32 gltf_image_index, TidoTextureCookResult const & artifact, std::optional<TidoTextureCookResult> const & opacity_artifact)> add_cooked)
            : asset{asset}, asset_path{std::move(asset_path)}, cache_dir{std::move(cache_dir)},
              items{std::move(items)}, add_cooked{std::move(add_cooked)}
        {
            chunk_count = s_cast<u32>(this->items.size());
        }

        void callback(u32 chunk_index, [[maybe_unused]] u32 thread_index) override
        {
            Item const & item = items.at(chunk_index);

            // Part 1: read the raw image bytes (+ tag their source format).
            auto raw = load_raw_image(*asset, item.gltf_image_index, asset_path, item.type);
            if (!raw.has_value())
            {
                DEBUG_MSG(fmt::format("[ERROR] Failed to load image index {} name {}", item.gltf_image_index, asset->images.at(item.gltf_image_index).name));
                return;
            }

            // Verify whether the content of the image actually changed using a content hash over the raw data.
            // The split opacity artifact derives from these same raw bytes, so it shares this content hash.
            u64 const content_hash = tido_fnv1a(std::as_bytes(std::span{raw.value().data}));
            if (item.cached_color.has_value() && item.cached_color->content_hash == content_hash && std::filesystem::exists(item.cached_color->tido_path))
            {
                // Bytes unchanged (only the mtime moved): reuse the cached .tido(s), just refresh their mtime.
                TidoTextureCookResult color_artifact = item.cached_color.value();
                color_artifact.source_modified = item.current_mtime;
                std::optional<TidoTextureCookResult> opacity_artifact = {};
                if (item.cached_opacity.has_value() && std::filesystem::exists(item.cached_opacity->tido_path))
                {
                    opacity_artifact = item.cached_opacity.value();
                    opacity_artifact->source_modified = item.current_mtime;
                }
                add_cooked(item.gltf_image_index, color_artifact, opacity_artifact);
                return;
            }

            // Part 2: process the raw bytes into GPU-ready cooked CPU memory (decode/transcode/compress).
            auto processed_ret = process_image(raw.value());
            if (std::holds_alternative<ImageOptimizeError>(processed_ret))
            {
                DEBUG_MSG(fmt::format("[ERROR] Failed to process image index {} name {}", item.gltf_image_index, asset->images.at(item.gltf_image_index).name));
                return;
            }

            ProcessedImageResult const & processed = std::get<ProcessedImageResult>(processed_ret);

            // Part 3: write the cooked color image out as a .tido artifact.
            auto tido_result = write_texture_tido(processed.color, cache_dir, raw.value().name, item.cache_key);
            if (!tido_result.has_value())
            {
                DEBUG_MSG(fmt::format("[WARN][write_texture_tido] failed to write .tido for image '{}'", raw.value().name));
                return;
            }

            DEBUG_MSG(fmt::format("[write_texture_tido] cooked '{}' -> {} ({}x{}, {} mips) -> '{}'",
                raw.value().name, s_cast<u32>(processed.color.image_info.format), processed.color.image_info.size.x,
                processed.color.image_info.size.y, processed.color.mips_to_copy, tido_result.value().tido_path.string()));

            TidoTextureCookResult color_artifact = tido_result.value();
            color_artifact.source_modified = item.current_mtime;
            color_artifact.content_hash = content_hash;

            // Part 3b: if this image is Mask-sampled by some material AND the cook split off an opacity
            // image (DIFFUSE + genuine source alpha), write its own .tido artifact too. Skipping the write
            // entirely when needs_opacity_split is false avoids cooking + storing an artifact nothing will
            // ever reference. A failed write here does not fail the whole cook - the color output stays
            // usable and the material simply renders without a dedicated opacity texture.
            std::optional<TidoTextureCookResult> opacity_artifact = {};
            if (item.needs_opacity_split && processed.opacity.has_value())
            {
                std::string const opacity_name = raw.value().name + "_opacity";
                auto opacity_tido_result = write_texture_tido(processed.opacity.value(), cache_dir, opacity_name, item.opacity_cache_key);
                if (opacity_tido_result.has_value())
                {
                    DEBUG_MSG(fmt::format("[write_texture_tido] cooked '{}' -> {} ({}x{}, {} mips) -> '{}'",
                        opacity_name, s_cast<u32>(processed.opacity->image_info.format), processed.opacity->image_info.size.x,
                        processed.opacity->image_info.size.y, processed.opacity->mips_to_copy, opacity_tido_result.value().tido_path.string()));
                    opacity_artifact = opacity_tido_result.value();
                    opacity_artifact->source_modified = item.current_mtime;
                    opacity_artifact->content_hash = content_hash;
                }
                else
                {
                    DEBUG_MSG(fmt::format("[WARN][write_texture_tido] failed to write opacity .tido for image '{}'", opacity_name));
                }
            }

            add_cooked(item.gltf_image_index, color_artifact, opacity_artifact);
        }
    };

    // For each referenced image (skip the unreferenced ones found in pass 1) decide:
    //   - mtime fast path: a cached color entry whose stored source mtime still matches (and whose .tido
    //     exists) is reused WITHOUT reading the source at all - collected here and added in one batch below,
    //     under a single lock, rather than reacquiring the scene lock per image. Its opacity counterpart (if
    //     any - only present when a previous cook found this image had alpha) rides along on the same mtime.
    //   - otherwise it becomes a chunk of the cook task, which reads the source, content-hashes it, and either
    //     reuses the cached .tido(s) (bytes unchanged, refresh the mtime) or recooks the image.
    u32 fast_hits = 0;
    struct FastHitEntry
    {
        u32 gltf_image_index = {};
        TidoTextureCookResult color = {};
        std::optional<TidoTextureCookResult> opacity = {};
    };
    std::vector<FastHitEntry> fast_hit_entries = {};
    std::vector<LoadImagesTask::Item> items_to_cook = {};
    for (u32 image_index = 0; image_index < s_cast<u32>(asset.images.size()); ++image_index)
    {
        if (image_types.at(image_index) == TextureMaterialType::NONE)
        {
            continue; // Unreferenced image - do not cook or add it.
        }
        u64 const key = image_cache_key(image_index);
        u64 const opacity_key = image_opacity_cache_key(image_index);
        std::optional<TidoTextureCookResult> cached_color = texture_cache_valid ? loaded_cache->lookup_texture(key) : std::nullopt;
        std::optional<TidoTextureCookResult> cached_opacity = texture_cache_valid ? loaded_cache->lookup_texture(opacity_key) : std::nullopt;
        std::filesystem::path const src_path = image_source_path(asset, image_index, file_path);
        std::optional<i64> const src_mtime = src_path.empty() ? std::nullopt : file_mtime(src_path);

        // mtime fast path: reuse the cached color .tido without reading the source.
        if (cached_color.has_value() && src_mtime.has_value() && cached_color->source_modified == src_mtime.value() && std::filesystem::exists(cached_color->tido_path))
        {
            std::optional<TidoTextureCookResult> opacity_hit = {};
            if (cached_opacity.has_value() && cached_opacity->source_modified == src_mtime.value() && std::filesystem::exists(cached_opacity->tido_path))
            {
                opacity_hit = cached_opacity;
            }
            fast_hit_entries.push_back(FastHitEntry{.gltf_image_index = image_index, .color = cached_color.value(), .opacity = opacity_hit});
            ++fast_hits;
            continue;
        }
        items_to_cook.push_back(LoadImagesTask::Item{
            .gltf_image_index = image_index,
            .type = image_types.at(image_index),
            .cache_key = key,
            .opacity_cache_key = opacity_key,
            .needs_opacity_split = image_needs_opacity_split.at(image_index),
            .current_mtime = src_mtime.value_or(0),
            .cached_color = std::move(cached_color),
            .cached_opacity = std::move(cached_opacity),
        });
    }

    // Fast path: append every mtime-hit entry's cache record(s) first (no batch lock held - just the cache
    // stream), then add all of them to the local batch under a single lock (no cook chunks are in
    // flight yet, so this loop is entirely sequential and safe to batch).
    if (rewriting_cache)
    {
        for (auto const & entry : fast_hit_entries)
        {
            write_cache_record(serialize_tido_cache_texture(entry.color));
            if (entry.opacity.has_value()) { write_cache_record(serialize_tido_cache_texture(entry.opacity.value())); }
        }
    }
    if (!fast_hit_entries.empty())
    {
        std::lock_guard<std::mutex> lock{batch_mutex};
        for (auto const & entry : fast_hit_entries)
        {
            u32 const image_manifest_index = s_cast<u32>(batch.textures.size());
            batch.textures.push_back(TextureManifestEntry{
                .type = image_types.at(entry.gltf_image_index),
                .material_manifest_indices = {},          // Back-refs are filled by SceneRuntime when it appends materials (pass 3).
                .name = asset.images[entry.gltf_image_index].name.c_str(),
                .streamer_data = make_texture_streamer_data(entry.color), // .tido reference; streamed in by the scene.
                .importer_data = TextureManifestEntry::GltfImporterData{.src_gltf = file_path, .image_index = entry.gltf_image_index},
            });
            image_manifest_indices.at(entry.gltf_image_index) = image_manifest_index;

            if (entry.opacity.has_value() && image_needs_opacity_split.at(entry.gltf_image_index))
            {
                u32 const opacity_manifest_index = s_cast<u32>(batch.textures.size());
                batch.textures.push_back(TextureManifestEntry{
                    .type = TextureMaterialType::OPACITY,
                    .material_manifest_indices = {},
                    .name = std::string(asset.images[entry.gltf_image_index].name.c_str()) + "_opacity",
                    .streamer_data = make_texture_streamer_data(entry.opacity.value()),
                    .importer_data = TextureManifestEntry::GltfImporterData{.src_gltf = file_path, .image_index = entry.gltf_image_index},
                });
                opacity_manifest_indices.at(entry.gltf_image_index) = opacity_manifest_index;
            }
        }
    }

    // One task, one chunk per image that needs cooking; the caller participates as a worker until every chunk
    // is done (blocking_dispatch, so this is safe to call from a worker thread once import runs off the main
    // thread). Each chunk cooks its image and adds it via add_cooked on completion.
    u32 const cook_count = s_cast<u32>(items_to_cook.size());
    if (cook_count > 0)
    {
        auto task = std::make_shared<LoadImagesTask>(&asset, file_path, cache_output_dir, std::move(items_to_cook), add_cooked_texture);
        info.thread_pool->blocking_dispatch(task, TaskPriority::LOW);
    }

    DEBUG_MSG(fmt::format("[GltfImporter::load_images] '{}': {} textures ({} mtime-hit, {} read)",
        info.asset_name.string(), fast_hits + cook_count, fast_hits, cook_count));
}

void GltfImporter::load_meshes()
{
    /// NOTE: fastgltf::Mesh is a MeshGroup, fastgltf::Primitive is a Mesh (MeshLodGroup).
    // Fourth pass (mirrors load_images): cook every mesh, then add each to the local batch with its cooked
    // result in hand. A mesh is added ONLY after it is cooked, so the moment SceneRuntime appends it, it is
    // already streamable. Each cook runs extract (importer) -> optimize_mesh (optimizer) -> write_mesh_tido
    // (.tido on disk) - no GPU work. One task holds a chunk per mesh that needs cooking; the chunks run in
    // parallel and each adds its own mesh on completion (add_cooked is thread-safe), so there is no separate
    // collection pass.

    // [gltf mesh-group index][in-group primitive index] -> batch-local index, INVALID until that mesh is
    // cooked + added (so a failed cook leaves a hole that translate_mesh_groups skips).
    mesh_manifest_indices.assign(asset.meshes.size(), {});
    for (u32 mesh_group_index = 0; mesh_group_index < s_cast<u32>(asset.meshes.size()); ++mesh_group_index)
    {
        mesh_manifest_indices[mesh_group_index].assign(asset.meshes[mesh_group_index].primitives.size(), INVALID_MANIFEST_INDEX);
    }

    // Records a cooked mesh (fresh cook or cache hit) into the local batch and, when a cache rewrite is
    // underway, appends its record to the open .tido_cache. Thread-safe: batch_mutex guards the batch and
    // write_cache_record guards the cache stream, and each call writes a distinct mesh_manifest_indices
    // slot, so cook chunks add concurrently. cache_key is already set on the artifact. Used only by the
    // cook task's completion callback below - the fast path (single-threaded, no cook chunks in flight)
    // adds its batch directly further down instead of going through this per entry.
    auto add_cooked_mesh = [&](u32 gltf_mesh_index, u32 gltf_primitive_index, TidoMeshCookResult artifact)
    {
        if (rewriting_cache) { write_cache_record(serialize_tido_cache_mesh(artifact)); }
        auto const & gltf_mesh = asset.meshes.at(gltf_mesh_index);
        auto const & gltf_primitive = gltf_mesh.primitives.at(gltf_primitive_index);
        std::optional<u32> const material_index =
            gltf_primitive.materialIndex.has_value()
                ? std::optional{material_manifest_indices.at(s_cast<u32>(gltf_primitive.materialIndex.value()))}
                : std::nullopt;
        std::lock_guard<std::mutex> lock{batch_mutex};
        u32 const mesh_manifest_index = s_cast<u32>(batch.mesh_lod_groups.size());
        batch.mesh_lod_groups.push_back(MeshLodGroupManifestEntry{
            .material_index = material_index,
            .name = gltf_mesh.name.c_str(),
            .streamer_data = make_mesh_streamer_data(artifact),   // .tido reference; streamed in by the scene.
            .importer_data = MeshLodGroupManifestEntry::GltfImporterData{.src_gltf = file_path, .mesh_index = gltf_mesh_index, .primitive_index = gltf_primitive_index},
        });
        mesh_manifest_indices.at(gltf_mesh_index).at(gltf_primitive_index) = mesh_manifest_index;
    };

    struct LoadMeshesTask final : Task
    {
        struct Item
        {
            u32 gltf_mesh_index = {};
            u32 gltf_primitive_index = {};
            u64 cache_key = {};
            i64 current_mtime = {}; // current max source mtime, stamped onto the (re)cooked or refreshed artifact
            // Pre-seeded cache entry that failed the mtime fast path; reused if its content hash still matches.
            std::optional<TidoMeshCookResult> cached = {};
        };

        // Immutable for the run of the task: set once at construction (before dispatch) and only ever read
        // from callback, which may run concurrently across chunks - nothing here is mutated after dispatch.
        fastgltf::Asset const * const asset;
        std::filesystem::path const asset_path;
        std::filesystem::path const cache_dir; // per-import output folder for the .tido data files
        std::vector<Item> const items;
        std::function<void(u32 gltf_mesh_index, u32 gltf_primitive_index, TidoMeshCookResult artifact)> const add_cooked;

        LoadMeshesTask(fastgltf::Asset const * asset, std::filesystem::path asset_path, std::filesystem::path cache_dir,
            std::vector<Item> items, std::function<void(u32 gltf_mesh_index, u32 gltf_primitive_index, TidoMeshCookResult artifact)> add_cooked)
            : asset{asset}, asset_path{std::move(asset_path)}, cache_dir{std::move(cache_dir)},
              items{std::move(items)}, add_cooked{std::move(add_cooked)}
        {
            chunk_count = s_cast<u32>(this->items.size());
        }

        void callback(u32 chunk_index, [[maybe_unused]] u32 thread_index) override
        {
            Item const & item = items.at(chunk_index);
            std::string const mesh_name = std::string(asset->meshes[item.gltf_mesh_index].name.c_str()) + "." + std::to_string(item.gltf_primitive_index);
            // On a failed (re)cook, fall back to the last good cook (the pre-seeded cache entry, if any) and
            // refresh its stored mtime so an un-processable source is not re-flagged "out of date" every
            // import. No fallback (first cook) -> nothing is added -> translate_mesh_groups skips the hole.
            auto keep_cached_fallback = [&]
            {
                if (item.cached.has_value())
                {
                    TidoMeshCookResult artifact = item.cached.value();
                    artifact.source_modified = item.current_mtime;
                    add_cooked(item.gltf_mesh_index, item.gltf_primitive_index, artifact);
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
            if (item.cached.has_value() && item.cached->content_hash == content_hash && std::filesystem::exists(item.cached->tido_path))
            {
                // Geometry unchanged (only the mtime moved): keep the existing .tido, just refresh the
                // stored mtime so the next import fast-paths without reading the source again.
                TidoMeshCookResult artifact = item.cached.value();
                artifact.source_modified = item.current_mtime;
                add_cooked(item.gltf_mesh_index, item.gltf_primitive_index, artifact);
                return;
            }
            // Part 2: cook the raw streams into the runtime form (optimizer).
            ProcessedMesh const processed = optimize_mesh(raw.value());
            // Part 3: write the cooked mesh out as a .tido artifact.
            auto tido_result = write_mesh_tido(processed, cache_dir, mesh_name, item.cache_key);
            if (!tido_result.has_value())
            {
                DEBUG_MSG(fmt::format("[WARN][write_mesh_tido] failed to write .tido for mesh '{}'", mesh_name));
                keep_cached_fallback();
                return;
            }
            DEBUG_MSG(fmt::format("[write_mesh_tido] cooked '{}' ({} LODs) -> '{}'",
                mesh_name, tido_result.value().descriptor.lod_count, tido_result.value().tido_path.string()));
            TidoMeshCookResult artifact = tido_result.value();
            artifact.source_modified = item.current_mtime;
            artifact.content_hash = content_hash;
            add_cooked(item.gltf_mesh_index, item.gltf_primitive_index, artifact);
        }
    };

    // For each mesh decide per artifact (mirrors load_images):
    //   - mtime fast path: a cached entry whose stored source mtime still matches (and whose .tido exists)
    //     is reused WITHOUT reading the source at all - collected here and added in one batch below, under
    //     a single lock, rather than reacquiring the scene lock per mesh.
    //   - otherwise it becomes a chunk of the cook task, which extracts the geometry, content-hashes it, and
    //     either reuses the cached .tido (bytes unchanged, refresh the mtime) or recooks (bytes changed / no entry).
    u32 fast_hits = 0;
    std::vector<std::tuple<u32, u32, TidoMeshCookResult>> fast_hit_entries = {}; // gltf_mesh_index, gltf_primitive_index -> artifact
    std::vector<LoadMeshesTask::Item> items_to_cook = {};
    for (u32 mesh_group_index = 0; mesh_group_index < s_cast<u32>(asset.meshes.size()); ++mesh_group_index)
    {
        for (u32 primitive_index = 0; primitive_index < s_cast<u32>(asset.meshes[mesh_group_index].primitives.size()); ++primitive_index)
        {
            u64 const key = mesh_cache_key(mesh_group_index, primitive_index);
            std::optional<TidoMeshCookResult> cached =
                mesh_cache_valid ? loaded_cache->lookup_mesh(key) : std::nullopt;
            std::vector<std::filesystem::path> const src_paths = mesh_source_paths(asset, file_path, mesh_group_index, primitive_index);
            std::optional<i64> const src_mtime = max_source_mtime(src_paths);

            // mtime fast path: reuse the cached .tido without reading the source.
            if (cached.has_value() && src_mtime.has_value() && cached->source_modified == src_mtime.value() &&
                std::filesystem::exists(cached->tido_path))
            {
                fast_hit_entries.emplace_back(mesh_group_index, primitive_index, std::move(cached.value()));
                ++fast_hits;
                continue;
            }
            items_to_cook.push_back(LoadMeshesTask::Item{
                .gltf_mesh_index = mesh_group_index,
                .gltf_primitive_index = primitive_index,
                .cache_key = key,
                .current_mtime = src_mtime.value_or(0),
                .cached = std::move(cached),
            });
        }
    }

    // Fast path: append every mtime-hit entry's cache record first (no batch lock held - just the cache
    // stream), then add all of them to the local batch under a single lock (no cook chunks are in
    // flight yet, so this loop is entirely sequential and safe to batch).
    if (rewriting_cache)
    {
        for (auto const & [gltf_mesh_index, gltf_primitive_index, artifact] : fast_hit_entries)
        {
            write_cache_record(serialize_tido_cache_mesh(artifact));
        }
    }
    if (!fast_hit_entries.empty())
    {
        std::lock_guard<std::mutex> lock{batch_mutex};
        for (auto & [gltf_mesh_index, gltf_primitive_index, artifact] : fast_hit_entries)
        {
            auto const & gltf_mesh = asset.meshes.at(gltf_mesh_index);
            auto const & gltf_primitive = gltf_mesh.primitives.at(gltf_primitive_index);
            std::optional<u32> const material_index =
                gltf_primitive.materialIndex.has_value()
                    ? std::optional{material_manifest_indices.at(s_cast<u32>(gltf_primitive.materialIndex.value()))}
                    : std::nullopt;
            u32 const mesh_manifest_index = s_cast<u32>(batch.mesh_lod_groups.size());
            batch.mesh_lod_groups.push_back(MeshLodGroupManifestEntry{
                .material_index = material_index,
                .name = gltf_mesh.name.c_str(),
                .streamer_data = make_mesh_streamer_data(artifact),   // .tido reference; streamed in by the scene.
                .importer_data = MeshLodGroupManifestEntry::GltfImporterData{.src_gltf = file_path, .mesh_index = gltf_mesh_index, .primitive_index = gltf_primitive_index},
            });
            mesh_manifest_indices.at(gltf_mesh_index).at(gltf_primitive_index) = mesh_manifest_index;
        }
    }

    // One task, one chunk per mesh that needs cooking; the caller participates as a worker until every chunk
    // is done (blocking_dispatch, so this is safe to call from a worker thread once import runs off the main
    // thread). Each chunk cooks its mesh and adds it via add_cooked on completion.
    u32 const cook_count = s_cast<u32>(items_to_cook.size());
    if (cook_count > 0)
    {
        auto task = std::make_shared<LoadMeshesTask>(&asset, file_path, cache_output_dir, std::move(items_to_cook), add_cooked_mesh);
        info.thread_pool->blocking_dispatch(task, TaskPriority::LOW);
    }

    DEBUG_MSG(fmt::format("[GltfImporter::load_meshes] '{}': {} meshes ({} mtime-hit, {} read)",
        info.asset_name.string(), fast_hits + cook_count, fast_hits, cook_count));
}


void GltfImporter::translate_materials()
{
    // Images are already in the local batch; a material references them by resolving each texture to its
    // image and looking up the image's batch-local index. SceneRuntime fills the texture -> material
    // back-refs when it appends materials (sampler_index is a placeholder until samplers are translated).
    // An image whose cook failed keeps INVALID_MANIFEST_INDEX, so its textures resolve to nullopt and the
    // material renders without them rather than referencing a texture that will never become resident.
    auto resolve_texture_info = [&](u32 const gltf_texture_index, u32 const sampler_index) -> std::optional<MaterialManifestEntry::TextureInfo>
    {
        u32 const manifest_index = image_manifest_indices.at(gltf_texture_to_image_index(gltf_texture_index).value());
        if (manifest_index == INVALID_MANIFEST_INDEX) { return std::nullopt; }
        return MaterialManifestEntry::TextureInfo{.tex_manifest_index = manifest_index, .sampler_index = sampler_index};
    };
    material_manifest_indices.reserve(asset.materials.size());
    for (u32 material_index = 0; material_index < s_cast<u32>(asset.materials.size()); material_index++)
    {
        auto const & material = asset.materials.at(material_index);
        std::optional<MaterialManifestEntry::TextureInfo> diffuse_texture_info = {};
        std::optional<MaterialManifestEntry::TextureInfo> opacity_texture_info = {};
        std::optional<MaterialManifestEntry::TextureInfo> normal_texture_info = {};
        std::optional<MaterialManifestEntry::TextureInfo> roughness_metalness_info = {};
        if (material.pbrData.baseColorTexture.has_value())
        {
            u32 const gltf_texture_index = s_cast<u32>(material.pbrData.baseColorTexture.value().textureIndex);
            diffuse_texture_info = resolve_texture_info(gltf_texture_index, {});
            // The split opacity texture (see image_needs_opacity_split / opacity_manifest_indices) only
            // exists for images a Mask material actually samples, and only when the cook found real alpha
            // to split; otherwise the material simply has no opacity texture.
            auto const gltf_image_index = gltf_texture_to_image_index(gltf_texture_index);
            if (gltf_image_index.has_value())
            {
                u32 const opacity_manifest_index = opacity_manifest_indices.at(gltf_image_index.value());
                if (opacity_manifest_index != INVALID_MANIFEST_INDEX)
                {
                    opacity_texture_info = MaterialManifestEntry::TextureInfo{.tex_manifest_index = opacity_manifest_index, .sampler_index = {}};
                }
            }
        }
        if (material.normalTexture.has_value())
        {
            normal_texture_info = resolve_texture_info(s_cast<u32>(material.normalTexture.value().textureIndex), 0);
        }
        if (material.pbrData.metallicRoughnessTexture.has_value())
        {
            roughness_metalness_info = resolve_texture_info(s_cast<u32>(material.pbrData.metallicRoughnessTexture.value().textureIndex), 0);
        }

        bool const alpha_discard_enabled = material.alphaMode == fastgltf::AlphaMode::Mask && opacity_texture_info.has_value();

        if(material.alphaMode == fastgltf::AlphaMode::Mask && !opacity_texture_info.has_value())
        {
            DEBUG_MSG(fmt::format("[WARN] Material '{}' has alphaMode=MASK but no opacity texture (diffuse image had no alpha)", material.name));
        }


        u32 const material_manifest_index = s_cast<u32>(batch.materials.size());
        batch.materials.push_back(MaterialManifestEntry{
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
        material_manifest_indices.push_back(material_manifest_index);
    }
}

void GltfImporter::translate_mesh_groups()
{
    /// NOTE: fastgltf::Mesh is a MeshGroup, fastgltf::Primitive is a Mesh (MeshLodGroup).
    // Mirrors translate_materials: every mesh is already cooked + added (load_meshes), so each gltf mesh
    // (= mesh group) is simply added to the local batch over its meshes' batch-local indices. SceneRuntime
    // resolves them + back-links the meshes when it appends the batch. Holes left by failed cooks
    // (INVALID_MANIFEST_INDEX) are skipped so a group never references a mesh that will never become resident.
    mesh_group_manifest_indices.reserve(asset.meshes.size());
    std::vector<u32> group_mesh_manifest_indices = {};
    for (u32 mesh_group_index = 0; mesh_group_index < s_cast<u32>(asset.meshes.size()); mesh_group_index++)
    {
        auto const & gltf_mesh = asset.meshes.at(mesh_group_index);

        group_mesh_manifest_indices.clear();
        for (u32 const mesh_manifest_index : mesh_manifest_indices.at(mesh_group_index))
        {
            if (mesh_manifest_index != INVALID_MANIFEST_INDEX)
            {
                group_mesh_manifest_indices.push_back(mesh_manifest_index);
            }
        }

        u32 const mesh_group_manifest_index = s_cast<u32>(batch.mesh_groups.size());
        batch.mesh_groups.push_back(ImporterTaskResult::SceneMetadataBatch::MeshGroup{
            .mesh_lod_group_indices = group_mesh_manifest_indices,
            .name = gltf_mesh.name.c_str(),
        });
        mesh_group_manifest_indices.push_back(mesh_group_manifest_index);
    }
}

auto GltfImporter::translate_light(fastgltf::Light const & light) -> u32
{
    f32 const LUMENS_PER_WATT = 683.0f;
    // Defines the minimum energy of a light before cutoff.
    // TODO(msakmary) hook this up to UI?
    f32 const E_min = 1.0f;

    switch (light.type)
    {
        case fastgltf::LightType::Point:
        {
            PointLight cpu_point_light = {};
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
            SpotLight cpu_spot_light = {};
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

auto GltfImporter::translate_entities() -> u32
{
    /// NOTE: fastgltf::Node is Entity
    DBG_ASSERT_TRUE_M(asset.nodes.size() != 0, "[ERROR][GltfImporter::translate_entities()] Empty node array - what to do now?");

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
        RenderEntity & r_ent = node_entities[node_index].entity;
        r_ent.mesh_group_manifest_index = node.meshIndex.has_value() ? std::optional<u32>(mesh_group_manifest_indices.at(s_cast<u32>(node.meshIndex.value()))) : std::optional<u32>(std::nullopt);
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
    RenderEntity & root_r_ent = node_entities[root_entity_index].entity;
    root_r_ent = RenderEntity{
        .transform = glm::mat4x3(glm::identity<glm::mat4x3>()),
        .type = EntityType::ROOT,
        .name = info.asset_name.filename().replace_extension("").string() + "_" + std::to_string(import_index),
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


