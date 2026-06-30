#include "gltf_importer.hpp"

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

    // Pass 1: resolve each used texture to its image; find which images are referenced (+ their type).
    collect_referenced_images();
    // Pass 2: add + load/optimize every referenced image; returns once they are all loaded.
    load_images();
    // Pass 3: a material's images are now loaded, so its entry is complete -> add it.
    translate_materials();
    // Meshes reference materials (added above); entities reference mesh groups (added below).
    translate_meshes_and_mesh_groups();
    RenderEntityId const root_r_ent_id = translate_entities();
    scene._root_render_entities.push_back(root_r_ent_id);

    // Mesh cook tasks borrow `asset` (owned by this importer). Dispatch them in parallel, then wait
    // for all to finish before returning so none can outlive the importer / the parsed asset.
    dispatch_async_mesh_loads();
    for (auto const & task : mesh_cook_tasks)
    {
        info.thread_pool->block_on(task);
    }

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

void GltfImporter::load_images()
{
    // Second pass: cook every referenced image, then add each to the manifest with its cooked result
    // in hand. An image is added ONLY after it is optimized, so the moment it is in the manifest it is
    // already streamable/resident. Each task runs the full pipeline: part 1 (load_raw_image, importer)
    // -> part 2 (optimize_image, optimizer) -> make resident (streamer). Tasks run in parallel; we wait
    // for all before adding.
    struct LoadImageTask final : Task
    {
        struct TaskInfo
        {
            fastgltf::Asset const * asset = {};
            std::filesystem::path asset_path = {};
            u32 gltf_image_index = {};
            TextureMaterialType type = {};
            daxa::Device device = {};
        };

        TaskInfo info = {};
        // Outputs, valid iff `succeeded`.
        daxa::ImageId image = {};
        TidoTextureCookResult cooked_artifact = {};
        bool succeeded = {};

        LoadImageTask(TaskInfo const & info)
            : info{info}
        {
            chunk_count = 1;
        }

        virtual void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override
        {
            // Part 1: read the raw image bytes (+ tag their source format). No decoding here.
            auto raw = load_raw_image(*info.asset, info.gltf_image_index, info.asset_path, info.type);
            if (!raw.has_value())
            {
                DEBUG_MSG(fmt::format("[ERROR] Failed to load image index {} name {}",
                    info.gltf_image_index, info.asset->images.at(info.gltf_image_index).name));
                return;
            }
            // Part 2: optimizer decodes/transcodes/compresses and writes the cooked .tido data file,
            // returning the artifact reference (descriptor + subresource offset table + path).
            auto cooked_ret = optimize_image(raw.value());
            if (std::holds_alternative<ImageOptimizeError>(cooked_ret))
            {
                DEBUG_MSG(fmt::format("[ERROR] Failed to optimize image index {} name {}",
                    info.gltf_image_index, info.asset->images.at(info.gltf_image_index).name));
                return;
            }
            cooked_artifact = std::move(std::get<TidoTextureCookResult>(cooked_ret));
            // Make resident on the GPU (streamer) by reading the cooked .tido back from disk.
            image = make_resident_image(info.device, cooked_artifact);
            succeeded = true;
        };
    };

    image_manifest_indices.assign(asset.images.size(), INVALID_MANIFEST_INDEX);

    // Dispatch a cook for every referenced image (skip the unreferenced ones found in pass 1).
    std::vector<std::shared_ptr<LoadImageTask>> image_cook_tasks = {};
    for (u32 i = 0; i < s_cast<u32>(asset.images.size()); ++i)
    {
        if (image_types.at(i) == TextureMaterialType::NONE)
        {
            continue; // Unreferenced image - do not cook or add it.
        }
        auto task = std::make_shared<LoadImageTask>(LoadImageTask::TaskInfo{
            .asset = &asset,
            .asset_path = file_path,
            .gltf_image_index = i,
            .type = image_types.at(i),
            .device = scene._device,
        });
        info.thread_pool->async_dispatch(task, TaskPriority::LOW);
        image_cook_tasks.push_back(std::move(task));
    }

    // Wait for all cooks, then add each optimized image to the manifest with its result.
    for (auto const & task : image_cook_tasks)
    {
        info.thread_pool->block_on(task);
        if (!task->succeeded)
        {
            continue;
        }
        u32 const gltf_image_index = task->info.gltf_image_index;
        u32 const image_manifest_index = scene.add_texture(TextureManifestEntry{
            .type = image_types.at(gltf_image_index),
            .material_manifest_indices = {},          // Back-refs are filled by Scene::add_material (pass 3).
            .runtime_texture = task->image,           // Already cooked + resident: immediately streamable.
            .cooked_artifact = std::move(task->cooked_artifact), // .tido reference for later re-streaming.
            .name = asset.images[gltf_image_index].name.c_str(),
        });
        image_manifest_indices.at(gltf_image_index) = image_manifest_index;
    }
}

void GltfImporter::translate_materials()
{
    // Images are already added; a material references them by resolving each texture to its image and
    // looking up the image's manifest index. Scene::add_material fills the texture -> material back-refs.
    // (sampler_index is a placeholder until samplers are translated - see T6.6.)
    auto image_manifest_index_of = [&](u32 const gltf_texture_index) -> u32
    {
        return image_manifest_indices.at(gltf_texture_to_image_index(gltf_texture_index).value());
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
            u32 const manifest_index = image_manifest_index_of(s_cast<u32>(material.pbrData.baseColorTexture.value().textureIndex));
            diffuse_texture_info = {.tex_manifest_index = manifest_index, .sampler_index = {}};
            opacity_texture_info = {.tex_manifest_index = manifest_index, .sampler_index = {}};
        }
        if (material.normalTexture.has_value())
        {
            u32 const gltf_image_index = gltf_texture_to_image_index(s_cast<u32>(material.normalTexture.value().textureIndex)).value();
            normal_texture_info = {.tex_manifest_index = image_manifest_indices.at(gltf_image_index), .sampler_index = 0};
        }
        if (material.pbrData.metallicRoughnessTexture.has_value())
        {
            u32 const manifest_index = image_manifest_index_of(s_cast<u32>(material.pbrData.metallicRoughnessTexture.value().textureIndex));
            roughness_metalness_info = {.tex_manifest_index = manifest_index, .sampler_index = 0};
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

void GltfImporter::translate_meshes_and_mesh_groups()
{
    /// NOTE: fastgltf::Mesh is a MeshGroup, fastgltf::Primitive is a Mesh (MeshLodGroup).
    mesh_group_manifest_indices.reserve(asset.meshes.size());
    std::vector<u32> group_mesh_manifest_indices = {};
    for (u32 mesh_group_index = 0; mesh_group_index < s_cast<u32>(asset.meshes.size()); mesh_group_index++)
    {
        auto const & gltf_mesh = asset.meshes.at(mesh_group_index);

        // Add all of the group's meshes first, collecting their returned manifest indices. The group
        // is then added over those indices (Scene::add_mesh_group records them + back-links the meshes).
        group_mesh_manifest_indices.clear();
        for (u32 in_group_index = 0; in_group_index < s_cast<u32>(gltf_mesh.primitives.size()); in_group_index++)
        {
            auto const & gltf_primitive = gltf_mesh.primitives.at(in_group_index);
            std::optional<u32> const material_manifest_index =
                gltf_primitive.materialIndex.has_value()
                    ? std::optional{material_manifest_indices.at(s_cast<u32>(gltf_primitive.materialIndex.value()))}
                    : std::nullopt;

            u32 const mesh_manifest_index = scene.add_mesh(MeshLodGroupManifestEntry{
                .material_index = material_manifest_index,
                .name = gltf_mesh.name.c_str(),
            });
            group_mesh_manifest_indices.push_back(mesh_manifest_index);

            pending_mesh_loads.push_back(PendingMeshLoad{
                .mesh_manifest_index = mesh_manifest_index,
                .gltf_mesh_index = mesh_group_index,
                .gltf_primitive_index = in_group_index,
                .material_manifest_index = material_manifest_index.value_or(INVALID_MANIFEST_INDEX),
            });
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

void GltfImporter::dispatch_async_mesh_loads()
{
    // Mirrors the texture pipeline: each task runs extract (importer) -> optimize_mesh (optimizer) ->
    // make_resident_mesh (streamer), then enqueues the GPU-resident result for the next manifest update.
    struct LoadMeshTask final : Task
    {
        struct TaskInfo
        {
            fastgltf::Asset const * asset = {};
            std::filesystem::path asset_path = {};
            u32 gltf_mesh_index = {};
            u32 gltf_primitive_index = {};
            u32 mesh_lod_manifest_index = {};
            u32 material_manifest_index = {};
            daxa::Device device = {};
            Scene * scene = {};
        };

        TaskInfo info = {};
        LoadMeshTask(TaskInfo const & info)
            : info{info}
        {
            chunk_count = 1;
        }

        virtual void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override
        {
            // Part 1: extract the glTF accessors into a format-neutral RawMesh (importer).
            auto raw = extract_raw_mesh(*info.asset, info.asset_path, info.gltf_mesh_index, info.gltf_primitive_index);
            if (!raw.has_value())
            {
                DEBUG_MSG(fmt::format("[ERROR] Failed to extract mesh group {} mesh {}",
                    info.gltf_mesh_index, info.gltf_primitive_index));
                return;
            }
            // Part 2: cook the raw streams into the runtime form (optimizer).
            ProcessedMesh const processed = optimize_mesh(raw.value());
            // Part 3: pack + upload each cooked LOD to the GPU (streamer).
            MeshLodGroupUploadInfo const upload = make_resident_mesh(info.device, {
                .processed = processed,
                .mesh_lod_manifest_index = info.mesh_lod_manifest_index,
                .material_manifest_index = info.material_manifest_index,
                .name = std::string(info.asset->meshes[info.gltf_mesh_index].name.c_str()) + "." + std::to_string(info.gltf_primitive_index),
            });
            // Hand the resident mesh straight to the scene: sets its runtime data + marks it dirty so the
            // next record_gpu_manifest_update uploads it (mirrors add_texture handing over a resident texture).
            info.scene->set_mesh_runtime(upload.mesh_lod_manifest_index, upload.lods, upload.lod_count);
        };
    };

    for (PendingMeshLoad const & pending : pending_mesh_loads)
    {
        auto task = std::make_shared<LoadMeshTask>(LoadMeshTask::TaskInfo{
            .asset = &asset,
            .asset_path = file_path,
            .gltf_mesh_index = pending.gltf_mesh_index,
            .gltf_primitive_index = pending.gltf_primitive_index,
            .mesh_lod_manifest_index = pending.mesh_manifest_index,
            .material_manifest_index = pending.material_manifest_index,
            .device = scene._device,
            .scene = &scene,
        });
        info.thread_pool->async_dispatch(task, TaskPriority::LOW);
        mesh_cook_tasks.push_back(std::move(task));
    }
}

