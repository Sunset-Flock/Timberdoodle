#include "gltf_importer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
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
    auto const t = std::filesystem::last_write_time(path, ec);
    if (ec) { return std::nullopt; }
    return t.time_since_epoch().count();
}

// Max file_mtime over `paths` (a mesh's geometry can span several buffer files); nullopt if none stat'd.
static auto max_source_mtime(std::span<std::filesystem::path const> paths) -> std::optional<i64>
{
    std::optional<i64> newest = std::nullopt;
    for (auto const & p : paths)
    {
        if (auto const t = file_mtime(p)) { newest = newest.has_value() ? std::max(newest.value(), t.value()) : t.value(); }
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

auto GltfImporter::import() -> std::variant<RenderEntityId, Scene::LoadManifestErrorCode>
{
    if (auto const error = parse(); error.has_value())
    {
        return error.value();
    }

    // The pipeline cooks each leaf resource and only then adds it to the manifest, so every entry is
    // already streamable the moment it exists; each "translate" pass that references those leaves runs
    // after they are added. Textures and meshes mirror each other: collect/cook/add the leaf, then a
    // translate pass wires the consumer (materials reference images; mesh groups reference meshes).
    collect_referenced_images(); // pass 1: resolve each used texture to its image (+ type); skip unreferenced.
    load_cache();                // load the shared .tido_cache once (used by load_images + load_meshes).
    bool const textures_dirtied_cache = load_images(); // pass 2: cook every referenced image, then add it.
    translate_materials();       // pass 3: a material's images are loaded -> its entry is complete -> add it.
    bool const meshes_dirtied_cache = load_meshes();   // pass 4: cook every mesh, then add it to the manifest.
    translate_mesh_groups();     // pass 5: a group's meshes are added -> add the group over them.
    RenderEntityId const root_r_ent_id = translate_entities(); // pass 6: entities reference mesh groups.
    scene._root_render_entities.push_back(root_r_ent_id);

    // Persist the cook once every artifact (textures + meshes) is final: record the key + all cooked
    // artifacts into the shared .tido_cache - but only if a cook/refresh actually changed it.
    write_cache_manifest(textures_dirtied_cache || meshes_dirtied_cache);

    return root_r_ent_id;
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
    import_index = s_cast<u32>(scene._root_render_entities.size());
    return std::nullopt;
}

void GltfImporter::collect_referenced_images()
{
    // First pass: walk every material, resolve each referenced texture to its IMAGE, and record the
    // image's type. An image left as NONE is unreferenced and will be skipped (not added/loaded). The
    // resolved type is also what the cook uses to pick a BC format, so it must be known before load.
    // Keying on images (not textures) means an image shared by several textures is loaded only once.
    image_types.assign(asset.images.size(), TextureMaterialType::NONE);
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
            set_image_type(s_cast<u32>(material.pbrData.baseColorTexture.value().textureIndex), TextureMaterialType::DIFFUSE);
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
    loaded_cache = read_tido_cache(TIDO_ASSET_CACHE_DIR / tido_cache_file_name(current_key.source_hash));
    texture_cache_valid = loaded_cache.has_value() && loaded_cache->key.texture_cook_version == current_key.texture_cook_version;
    mesh_cache_valid = loaded_cache.has_value() && loaded_cache->key.mesh_cook_version == current_key.mesh_cook_version;

    if (!loaded_cache.has_value())
    {
        DEBUG_MSG(fmt::format("[GltfImporter::load_cache] '{}': no cache - cooking everything", info.asset_name.string()));
    }
    else
    {
        DEBUG_MSG(fmt::format("[GltfImporter::load_cache] '{}': cache loaded ({} texture + {} mesh entries); textures {}, meshes {}",
            info.asset_name.string(), loaded_cache->texture_index.size(), loaded_cache->mesh_index.size(),
            texture_cache_valid ? "valid" : "stale (cook version changed) - recooking",
            mesh_cache_valid ? "valid" : "stale (cook version changed) - recooking"));
    }
}

auto GltfImporter::load_images() -> bool
{
    // Second pass: cook every referenced image, then add each to the manifest with its cooked result
    // in hand. An image is added ONLY after it is cooked, so the moment it is in the manifest it is
    // already streamable. Each task runs extract (load_raw_image, importer) -> process_image (optimizer)
    // -> write_texture_tido (.tido on disk) - no GPU work (exactly like load_meshes). Tasks run in
    // parallel; we wait for all before adding.
    struct LoadImageTask final : Task
    {
        struct TaskInfo
        {
            fastgltf::Asset const * asset = {};
            std::filesystem::path asset_path = {};
            u32 gltf_image_index = {};
            TextureMaterialType type = {};
            u64 cache_key = {};
            // Current max source mtime, stamped onto the (re)cooked or refreshed artifact.
            i64 current_mtime = {};
        };

        TaskInfo info = {};
        // At the start can hold an artifact loaded from .tido_cache which failed the staleness check.
        // At the end holds the result of the task.
        std::optional<TidoTextureCookResult> cooked_artifact = {};

        LoadImageTask(TaskInfo const & info)
            : info{info}
        {
            chunk_count = 1;
        }

        virtual void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override
        {
            // Part 1: read the raw image bytes (+ tag their source format).
            auto raw = load_raw_image(*info.asset, info.gltf_image_index, info.asset_path, info.type);
            if (!raw.has_value())
            {
                DEBUG_MSG(fmt::format("[ERROR] Failed to load image index {} name {}",
                    info.gltf_image_index, info.asset->images.at(info.gltf_image_index).name));
                cooked_artifact = std::nullopt;
                return;
            }

            // Verify whether the content of the image actually changed using a content hash over the raw data.
            u64 const content_hash = tido_fnv1a(std::as_bytes(std::span{raw.value().data}));
            if (cooked_artifact.has_value() && cooked_artifact->content_hash == content_hash && std::filesystem::exists(cooked_artifact->tido_path))
            {
                cooked_artifact->source_modified = info.current_mtime;
                return;
            }
            // Part 2: process the raw bytes into GPU-ready cooked CPU memory (decode/transcode/compress).
            auto processed_ret = process_image(raw.value());
            if (std::holds_alternative<ImageOptimizeError>(processed_ret))
            {
                DEBUG_MSG(fmt::format("[ERROR] Failed to process image index {} name {}",
                    info.gltf_image_index, info.asset->images.at(info.gltf_image_index).name));
                cooked_artifact = std::nullopt;
                return;
            }
            ProcessedImage const & processed = std::get<ProcessedImage>(processed_ret);
            // Part 3: write the cooked image out as a .tido artifact.
            auto tido_result = write_texture_tido(processed, TIDO_ASSET_CACHE_DIR, raw.value().name, info.cache_key);
            if (!tido_result.has_value())
            {
                DEBUG_MSG(fmt::format("[WARN][write_texture_tido] failed to write .tido for image '{}'", raw.value().name));
                cooked_artifact = std::nullopt;
                return;
            }

            DEBUG_MSG(fmt::format("[write_texture_tido] cooked '{}' -> {} ({}x{}, {} mips) -> '{}'",
                raw.value().name, s_cast<u32>(processed.image_info.format), processed.image_info.size.x,
                processed.image_info.size.y, processed.mips_to_copy, tido_result.value().tido_path.string()));
            cooked_artifact = tido_result.value();
            cooked_artifact->source_modified = info.current_mtime;
            cooked_artifact->content_hash = content_hash;
        };
    };

    image_manifest_indices.assign(asset.images.size(), INVALID_MANIFEST_INDEX);

    // Stable per-image source-identity key. Recomputed identically on re-import so a valid cache can be
    // hit without cooking. It also forms the .tido file stem, so it folds in the source path to stay
    // unique across ALL imported files (the .tido files share one cache dir) - identical scheme to
    // mesh_cache_key (tido_source_identity_key), with the gltf image index as the in-file disambiguator.
    auto image_cache_key = [&](u32 gltf_image_index) -> u64
    {
        return tido_source_identity_key(file_path, asset.images[gltf_image_index].name.c_str(), fmt::format("{}", gltf_image_index));
    };

    // Records a cooked texture (fresh cook or cache hit) into the .tido_cache collection + the scene
    // manifest (which queues it for async streaming). cache_key must already be set on the artifact.
    auto add_cooked_texture = [&](u32 gltf_image_index, const TidoTextureCookResult & artifact)
    {
        cooked_texture_artifacts.push_back(artifact);
        u32 const image_manifest_index = scene.add_texture(TextureManifestEntry{
            .type = image_types.at(gltf_image_index),
            .material_manifest_indices = {},          // Back-refs are filled by Scene::add_material (pass 3).
            .cooked_artifact = artifact,              // .tido reference; streamed in by the scene.
            .name = asset.images[gltf_image_index].name.c_str(),
        });
        image_manifest_indices.at(gltf_image_index) = image_manifest_index;
    };

    // For each referenced image (skip the unreferenced ones found in pass 1) decide:
    //   - mtime fast path: a cached entry whose stored source mtime still matches (and whose .tido exists)
    //     is reused WITHOUT reading the source at all.
    //   - otherwise dispatch a task that reads the source, content-hashes it, and either reuses the cached
    //     .tido and only updates the metadata or recooks the image.
    u32 fast_hits = 0;
    std::vector<std::shared_ptr<LoadImageTask>> image_cook_tasks = {};
    for (u32 image_index = 0; image_index < s_cast<u32>(asset.images.size()); ++image_index)
    {
        if (image_types.at(image_index) == TextureMaterialType::NONE)
        {
            continue; // Unreferenced image - do not cook or add it.
        }
        std::optional<TidoTextureCookResult> cached = texture_cache_valid ? loaded_cache->lookup_texture(image_cache_key(image_index)) : std::nullopt;
        std::filesystem::path const src_path = image_source_path(asset, image_index, file_path);
        std::optional<i64> const src_mtime = src_path.empty() ? std::nullopt : file_mtime(src_path);

        // mtime fast path: reuse the cached .tido without reading the source.
        if (cached.has_value() && src_mtime.has_value() && cached->source_modified == src_mtime.value() && std::filesystem::exists(cached->tido_path))
        {
            add_cooked_texture(image_index, cached.value());
            ++fast_hits;
            continue;
        }
        auto task = std::make_shared<LoadImageTask>(LoadImageTask::TaskInfo{
            .asset = &asset,
            .asset_path = file_path,
            .gltf_image_index = image_index,
            .type = image_types.at(image_index),
            .cache_key = image_cache_key(image_index),
            .current_mtime = src_mtime.value_or(0),
        });
        task->cooked_artifact = cached; // May be nullopt if no cache entry exists.
        info.thread_pool->async_dispatch(task, TaskPriority::LOW);
        image_cook_tasks.push_back(std::move(task));
    }

    // Wait for the dispatched images and add them. Each task set source_modified + content_hash; a task may
    // have reused the cached .tido on a content match rather than recooking.
    for (auto const & task : image_cook_tasks)
    {
        info.thread_pool->block_on(task);
        if (!task->cooked_artifact.has_value())
        {
            continue;
        }
        add_cooked_texture(task->info.gltf_image_index, task->cooked_artifact.value());
    }

    // The cache needs rewriting if anything changed: any dispatched task (a recook OR an mtime refresh
    // both change the persisted metadata) or a stale/absent texture cache. A pure fast-path run does not.
    bool const dirtied_cache = !image_cook_tasks.empty() || !texture_cache_valid;

    DEBUG_MSG(fmt::format("[GltfImporter::load_images] '{}': {} textures ({} mtime-hit, {} read)",
        info.asset_name.string(), fast_hits + s_cast<u32>(image_cook_tasks.size()), fast_hits, image_cook_tasks.size()));
    return dirtied_cache;
}

auto GltfImporter::load_meshes() -> bool
{
    /// NOTE: fastgltf::Mesh is a MeshGroup, fastgltf::Primitive is a Mesh (MeshLodGroup).
    // Fourth pass (mirrors load_images): cook every mesh, then add each to the manifest with its cooked
    // result in hand. A mesh is added ONLY after it is cooked, so the moment it is in the manifest it is
    // already streamable. Each task runs extract (importer) -> optimize_mesh (optimizer) -> write_mesh_tido
    // (.tido on disk) - no GPU work. Tasks run in parallel; we wait for all before adding.
    struct LoadMeshTask final : Task
    {
        struct TaskInfo
        {
            fastgltf::Asset const * asset = {};
            std::filesystem::path asset_path = {};
            u32 gltf_mesh_index = {};
            u32 gltf_primitive_index = {};
            u64 cache_key = {};
            // Current max source mtime, stamped onto the (re)cooked or refreshed artifact.
            i64 current_mtime = {};
        };

        TaskInfo info = {};
        std::optional<TidoMeshCookResult> cooked_artifact = {};

        LoadMeshTask(TaskInfo const & info)
            : info{info}
        {
            chunk_count = 1;
        }

        virtual void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override
        {
            std::string const mesh_name = std::string(info.asset->meshes[info.gltf_mesh_index].name.c_str()) + "." + std::to_string(info.gltf_primitive_index);
            // On a failed (re)cook, fall back to the last good cook (the pre-seeded `cached`, if any) and
            // refresh its stored mtime so an un-processable source is not re-flagged "out of date" every
            // import. No fallback (first cook) -> stays nullopt -> the collect loop skips it. Mirrors load_images.
            auto keep_cached_fallback = [&] { if (cooked_artifact.has_value()) { cooked_artifact->source_modified = info.current_mtime; } };

            // Part 1: extract the glTF accessors into a format-neutral RawMesh (importer).
            auto raw = extract_raw_mesh(*info.asset, info.asset_path, info.gltf_mesh_index, info.gltf_primitive_index);
            if (!raw.has_value())
            {
                DEBUG_MSG(fmt::format("[ERROR] Failed to extract mesh group {} mesh {}",
                    info.gltf_mesh_index, info.gltf_primitive_index));
                keep_cached_fallback();
                return;
            }
            // Content hash of the raw extracted geometry: the authoritative change detector.
            u64 const content_hash = raw_mesh_content_hash(raw.value());
            if (cooked_artifact.has_value() && cooked_artifact->content_hash == content_hash && std::filesystem::exists(cooked_artifact->tido_path))
            {
                // Geometry unchanged (only the mtime moved): keep the existing .tido, just refresh the
                // stored mtime so the next import fast-paths without reading the source again.
                cooked_artifact->source_modified = info.current_mtime;
                return;
            }
            // Part 2: cook the raw streams into the runtime form (optimizer).
            ProcessedMesh const processed = optimize_mesh(raw.value());
            // Part 3: write the cooked mesh out as a .tido artifact.
            auto tido_result = write_mesh_tido(processed, TIDO_ASSET_CACHE_DIR, mesh_name, info.cache_key);
            if (!tido_result.has_value())
            {
                DEBUG_MSG(fmt::format("[WARN][write_mesh_tido] failed to write .tido for mesh '{}'", mesh_name));
                keep_cached_fallback();
                return;
            }
            DEBUG_MSG(fmt::format("[write_mesh_tido] cooked '{}' ({} LODs) -> '{}'",
                mesh_name, tido_result.value().descriptor.lod_count, tido_result.value().tido_path.string()));
            cooked_artifact = tido_result.value();
            cooked_artifact->source_modified = info.current_mtime;
            cooked_artifact->content_hash = content_hash;
        };
    };

    // [gltf mesh-group index][in-group primitive index] -> manifest index, INVALID until that mesh is
    // cooked + added (so a failed cook leaves a hole that translate_mesh_groups skips).
    mesh_manifest_indices.assign(asset.meshes.size(), {});
    for (u32 g = 0; g < s_cast<u32>(asset.meshes.size()); ++g)
    {
        mesh_manifest_indices[g].assign(asset.meshes[g].primitives.size(), INVALID_MANIFEST_INDEX);
    }

    // Stable per-primitive source-identity key. It both indexes the mesh in the .tido_cache AND forms the
    // .tido data-file stem, so it must be unique across ALL imported files (the .tido files share one
    // cache dir): the source file path is folded in alongside the (mesh, primitive) index. Without the
    // path, two scenes that both have e.g. an unnamed mesh at index 0 would hash to the same key, write to
    // the same .tido, and clobber each other (→ "LOD blob out of .tido bounds" when loading the second).
    // The path also disambiguates byte-identical primitives within a file (so parallel cook tasks never
    // race on one .tido). Textures use the identical scheme (tido_source_identity_key) for their stem.
    auto mesh_cache_key = [&](u32 gltf_mesh_index, u32 gltf_primitive_index) -> u64
    {
        return tido_source_identity_key(file_path, asset.meshes[gltf_mesh_index].name.c_str(),
            fmt::format("{}.{}", gltf_mesh_index, gltf_primitive_index));
    };

    // Records a cooked mesh (fresh cook or cache hit) into the .tido_cache collection + the scene manifest
    // (which marks it for async streaming). cache_key is already set on the artifact. Mirrors add_cooked_texture.
    auto add_cooked_mesh = [&](u32 gltf_mesh_index, u32 gltf_primitive_index, TidoMeshCookResult artifact)
    {
        cooked_mesh_artifacts.push_back(artifact);
        auto const & gltf_mesh = asset.meshes.at(gltf_mesh_index);
        auto const & gltf_primitive = gltf_mesh.primitives.at(gltf_primitive_index);
        std::optional<u32> const material_index =
            gltf_primitive.materialIndex.has_value()
                ? std::optional{material_manifest_indices.at(s_cast<u32>(gltf_primitive.materialIndex.value()))}
                : std::nullopt;
        u32 const mesh_manifest_index = scene.add_mesh(MeshLodGroupManifestEntry{
            .material_index = material_index,
            .name = gltf_mesh.name.c_str(),
            .cooked_artifact = std::move(artifact),   // .tido reference; streamed in by the scene.
        });
        mesh_manifest_indices.at(gltf_mesh_index).at(gltf_primitive_index) = mesh_manifest_index;
    };

    // For each mesh decide per artifact (mirrors load_images):
    //   - mtime fast path: a cached entry whose stored source mtime still matches (and whose .tido exists)
    //     is reused WITHOUT reading the source at all.
    //   - otherwise dispatch a task that extracts the geometry, content-hashes it, and either reuses the
    //     cached .tido (bytes unchanged, just refresh the mtime) or recooks (bytes changed / no entry).
    u32 fast_hits = 0;
    std::vector<std::shared_ptr<LoadMeshTask>> mesh_cook_tasks = {};
    for (u32 g = 0; g < s_cast<u32>(asset.meshes.size()); ++g)
    {
        for (u32 p = 0; p < s_cast<u32>(asset.meshes[g].primitives.size()); ++p)
        {
            u64 const key = mesh_cache_key(g, p);
            std::optional<TidoMeshCookResult> cached =
                mesh_cache_valid ? loaded_cache->lookup_mesh(key) : std::nullopt;
            std::vector<std::filesystem::path> const src_paths = mesh_source_paths(asset, file_path, g, p);
            std::optional<i64> const src_mtime = max_source_mtime(src_paths);

            // mtime fast path: reuse the cached .tido without reading the source.
            if (cached.has_value() && src_mtime.has_value() && cached->source_modified == src_mtime.value() &&
                std::filesystem::exists(cached->tido_path))
            {
                add_cooked_mesh(g, p, std::move(cached.value()));
                ++fast_hits;
                continue;
            }
            auto task = std::make_shared<LoadMeshTask>(LoadMeshTask::TaskInfo{
                .asset = &asset,
                .asset_path = file_path,
                .gltf_mesh_index = g,
                .gltf_primitive_index = p,
                .cache_key = key,
                .current_mtime = src_mtime.value_or(0),
            });
            task->cooked_artifact = cached;
            info.thread_pool->async_dispatch(task, TaskPriority::LOW);
            mesh_cook_tasks.push_back(std::move(task));
        }
    }

    // Wait for the dispatched meshes and add them. Each task set source_modified + content_hash; a task may
    // have reused the cached .tido on a content match rather than recooking.
    for (auto const & task : mesh_cook_tasks)
    {
        info.thread_pool->block_on(task);
        if (!task->cooked_artifact.has_value())
        {
            continue;
        }
        add_cooked_mesh(task->info.gltf_mesh_index, task->info.gltf_primitive_index, task->cooked_artifact.value());
    }

    // The cache needs rewriting if anything changed: any dispatched task (a recook OR an mtime refresh
    // both change the persisted metadata) or a stale/absent mesh cache. A pure fast-path run does not.
    bool const dirtied_cache = !mesh_cook_tasks.empty() || !mesh_cache_valid;

    DEBUG_MSG(fmt::format("[GltfImporter::load_meshes] '{}': {} meshes ({} mtime-hit, {} read)",
        info.asset_name.string(), fast_hits + s_cast<u32>(mesh_cook_tasks.size()), fast_hits, mesh_cook_tasks.size()));
    return dirtied_cache;
}

void GltfImporter::write_cache_manifest(bool cache_dirty)
{
    // Nothing to do if every artifact came from an already-current cache (all fast-path hits, versions ok).
    if (!cache_dirty) { return; }
    // Key the cache on the source path + the per-kind cook versions, and record every cooked texture +
    // mesh artifact (each carrying its own source mtime + content hash for per-artifact staleness). The
    // file is named by the hashed source path so a future import can find it deterministically (T5). IO
    // failure is non-fatal: the cook already ran this session.
    TidoCacheKey const key = tido_make_cache_key(file_path, GLTF_TEXTURE_COOK_VERSION, GLTF_MESH_COOK_VERSION);
    std::filesystem::path const cache_path = TIDO_ASSET_CACHE_DIR / tido_cache_file_name(key.source_hash);
    if (write_tido_cache(cache_path, key, cooked_texture_artifacts, cooked_mesh_artifacts))
    {
        DEBUG_MSG(fmt::format("[GltfImporter::write_cache_manifest] wrote {} texture + {} mesh entries to '{}'",
            cooked_texture_artifacts.size(), cooked_mesh_artifacts.size(), cache_path.string()));
    }
    else
    {
        DEBUG_MSG(fmt::format("[WARN][GltfImporter::write_cache_manifest] failed to write .tido_cache for '{}'",
            file_path.string()));
    }
}

void GltfImporter::translate_materials()
{
    // Images are already added; a material references them by resolving each texture to its image and
    // looking up the image's manifest index. Scene::add_material fills the texture -> material back-refs.
    // (sampler_index is a placeholder until samplers are translated - see T6.6.)
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
            diffuse_texture_info = resolve_texture_info(s_cast<u32>(material.pbrData.baseColorTexture.value().textureIndex), {});
            opacity_texture_info = diffuse_texture_info;
        }
        if (material.normalTexture.has_value())
        {
            normal_texture_info = resolve_texture_info(s_cast<u32>(material.normalTexture.value().textureIndex), 0);
        }
        if (material.pbrData.metallicRoughnessTexture.has_value())
        {
            roughness_metalness_info = resolve_texture_info(s_cast<u32>(material.pbrData.metallicRoughnessTexture.value().textureIndex), 0);
        }
        u32 const material_manifest_index = scene.add_material(MaterialManifestEntry{
            .diffuse_info = diffuse_texture_info,
            .opacity_mask_info = opacity_texture_info,
            .normal_info = normal_texture_info,
            .roughness_metalness_info = roughness_metalness_info,
            .alpha_discard_enabled = material.alphaMode == fastgltf::AlphaMode::Mask,
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
    // (= mesh group) is simply added over its meshes' manifest indices. Scene::add_mesh_group records
    // them + back-links the meshes. Holes left by failed cooks (INVALID_MANIFEST_INDEX) are skipped so a
    // group never references a mesh that will never become resident.
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

        u32 const mesh_group_manifest_index = scene.add_mesh_group(
            MeshGroupManifestEntry{.name = gltf_mesh.name.c_str()},
            group_mesh_manifest_indices);
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
            return scene.add_point_light(cpu_point_light);
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
            return scene.add_spot_light(cpu_spot_light);
        }
        case fastgltf::LightType::Directional:
        {
            // TODO(msakmary) add handling of directional lights.
            DBG_ASSERT_TRUE_M(false, "TODO(msakmary) implement directional lights");
            break;
        }
    }
    return s_cast<u32>(-1);
}

auto GltfImporter::translate_entities() -> RenderEntityId
{
    /// NOTE: fastgltf::Node is Entity
    DBG_ASSERT_TRUE_M(asset.nodes.size() != 0, "[ERROR][GltfImporter::translate_entities()] Empty node array - what to do now?");
    std::vector<RenderEntityId> node_index_to_entity_id = {};
    /// NOTE: Here we allocate space for each entity and create a translation table between node index and entity id
    for (u32 node_index = 0; node_index < s_cast<u32>(asset.nodes.size()); node_index++)
    {
        node_index_to_entity_id.push_back(scene._render_entities.create_slot());
        scene._dirty_render_entities.push_back(node_index_to_entity_id.back());
    }
    for (u32 node_index = 0; node_index < s_cast<u32>(asset.nodes.size()); node_index++)
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
        RenderEntityId const parent_r_ent_id = node_index_to_entity_id[node_index];
        RenderEntity & r_ent = *scene._render_entities.slot(parent_r_ent_id);
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
            r_ent.first_child = node_index_to_entity_id[node.children[0]];
        }

        for (u32 curr_child_vec_idx = 0; curr_child_vec_idx < node.children.size(); curr_child_vec_idx++)
        {
            u64 const curr_child_node_idx = node.children[curr_child_vec_idx];
            RenderEntityId const curr_child_r_ent_id = node_index_to_entity_id[curr_child_node_idx];
            RenderEntity & curr_child_r_ent = *scene._render_entities.slot(curr_child_r_ent_id);
            curr_child_r_ent.parent = parent_r_ent_id;
            bool const has_next_sibling = curr_child_vec_idx < (node.children.size() - 1ull);
            if (has_next_sibling)
            {
                RenderEntityId const next_r_ent_child_id = node_index_to_entity_id[node.children[curr_child_vec_idx + 1]];
                curr_child_r_ent.next_sibling = next_r_ent_child_id;
            }
        }
    }

    /// NOTE: Find all root render entities (aka render entities that have no parent) and store them as
    //        Child root entites under scene root node
    RenderEntityId root_r_ent_id = scene._render_entities.create_slot({
        .transform = glm::mat4x3(glm::identity<glm::mat4x3>()),
        .first_child = std::nullopt,
        .next_sibling = std::nullopt,
        .parent = std::nullopt,
        .mesh_group_manifest_index = std::nullopt,
        .name = info.asset_name.filename().replace_extension("").string() + "_" + std::to_string(import_index),
    });

    scene._dirty_render_entities.push_back(root_r_ent_id);
    RenderEntity & root_r_ent = *scene._render_entities.slot(root_r_ent_id);
    root_r_ent.type = EntityType::ROOT;
    std::optional<RenderEntityId> root_r_ent_prev_child = {};
    for (u32 node_index = 0; node_index < s_cast<u32>(asset.nodes.size()); node_index++)
    {
        RenderEntityId const r_ent_id = node_index_to_entity_id[node_index];
        RenderEntity & r_ent = *scene._render_entities.slot(r_ent_id);
        if (!r_ent.parent.has_value())
        {
            r_ent.parent = root_r_ent_id;
            if (!root_r_ent_prev_child.has_value()) // First child
            {
                root_r_ent.first_child = r_ent_id;
            }
            else // We have other root children already
            {
                scene._render_entities.slot(root_r_ent_prev_child.value())->next_sibling = r_ent_id;
            }
            root_r_ent_prev_child = r_ent_id;
        }
    }
    return root_r_ent_id;
}


