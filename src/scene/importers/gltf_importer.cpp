#include "importer.hpp"

#include <cmath>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fmt/format.h>
#include <glm/gtx/quaternion.hpp>

#include "../optimizers/image_processor.hpp"
#include "../tido_format/tido_util.hpp"

namespace
{
static constexpr std::string_view VERT_ATTRIB_POSITION_NAME = "POSITION";
static constexpr std::string_view VERT_ATTRIB_TEXCOORD0_NAME = "TEXCOORD_0";
static constexpr std::string_view VERT_ATTRIB_NORMAL_NAME = "NORMAL";

// Byte range an accessor's tightly-packed data occupies in its external buffer file. nullopt if the buffer
// is embedded/unsupported or the accessor has no buffer view. Asserts the source is tightly packed - the
// generic cook reads a contiguous range and reinterprets it as a tight array, so interleaved sources aren't
// supported.
static auto locate_accessor_range(fastgltf::Asset const & asset, std::filesystem::path const & root_path, fastgltf::Accessor const & accessor) -> std::optional<SourceLocation>
{
    if (!accessor.bufferViewIndex.has_value()) { return std::nullopt; }
    fastgltf::BufferView const & view = asset.bufferViews.at(accessor.bufferViewIndex.value());
    fastgltf::Buffer const & buffer = asset.buffers.at(view.bufferIndex);
    if (!std::holds_alternative<fastgltf::sources::URI>(buffer.data)) { return std::nullopt; }
    fastgltf::sources::URI const & uri = std::get<fastgltf::sources::URI>(buffer.data);

    u64 const element_byte_size = fastgltf::getElementByteSize(accessor.type, accessor.componentType);
    DBG_ASSERT_TRUE_M(!view.byteStride.has_value() || view.byteStride.value() == element_byte_size, "Mesh accessor source is not tightly packed");
    return SourceLocation{
        .file = root_path / uri.uri.fspath(),
        .slice = ByteSlice{
            .byte_offset = view.byteOffset + accessor.byteOffset + uri.fileByteOffset,
            .byte_length = accessor.count * element_byte_size,
        },
    };
}

// A resolved mesh stream plus its element count, kept so callers can cross-check vertex/index counts.
struct ResolvedMeshStream
{
    MeshAttribSource source = {};
    u32 element_count = {};
};

// Resolve where one primitive's tightly-packed vertex/index streams live without reading them. The accepted
// formats match extract_raw_mesh's accessor validation (F32 vec3 positions/normals, F32 vec2 uvs, U16|U32
// scalar indices). nullopt if a required stream is missing/invalid or a source is embedded/unsupported.
static auto resolve_mesh_source(fastgltf::Asset const & asset, std::filesystem::path const & asset_path, u32 gltf_mesh_index, u32 gltf_primitive_index) -> std::optional<MeshImporterData>
{
    std::filesystem::path const root_path = std::filesystem::path{asset_path}.remove_filename();
    fastgltf::Mesh const & gltf_mesh = asset.meshes.at(gltf_mesh_index);
    fastgltf::Primitive const & gltf_prim = gltf_mesh.primitives.at(gltf_primitive_index);

    // Resolve one accessor of expected_type, mapping its glTF component type to ours (nullopt rejects the
    // stream). nullopt if the accessor is absent, the wrong type, or backed by an embedded/unsupported source.
    auto resolve_stream = [&](std::optional<std::size_t> accessor_index, fastgltf::AccessorType expected_type) -> std::optional<ResolvedMeshStream>
    {
        if (!accessor_index.has_value()) { return std::nullopt; }
        fastgltf::Accessor const & accessor = asset.accessors.at(accessor_index.value());

        std::optional<ComponentType> const component_type = [&]() -> std::optional<ComponentType>
        {
            switch(accessor.componentType)
            {
                case fastgltf::ComponentType::UnsignedShort: return ComponentType::U16;
                case fastgltf::ComponentType::UnsignedInt: return ComponentType::U32;
                case fastgltf::ComponentType::Float: return ComponentType::F32;
                default: return std::nullopt;
            }
        }();

        if (accessor.type != expected_type || !component_type.has_value()) { return std::nullopt; }

        auto const location = locate_accessor_range(asset, root_path, accessor);
        if (!location.has_value()) { return std::nullopt; }
        return ResolvedMeshStream{
            .source = MeshAttribSource{.location = location.value(), .component_type = component_type.value()},
            .element_count = s_cast<u32>(accessor.count),
        };
    };

    auto attribute_index = [&](std::string_view attrib_name) -> std::optional<std::size_t>
    {
        auto const attrib_iter = gltf_prim.findAttribute(attrib_name);
        if (attrib_iter == gltf_prim.attributes.end()) { return std::nullopt; }
        return attrib_iter->accessorIndex;
    };

    auto const position_accessor_index = attribute_index(VERT_ATTRIB_POSITION_NAME);
    auto const normal_accessor_index = attribute_index(VERT_ATTRIB_NORMAL_NAME);
    auto const uv_accessor_index = attribute_index(VERT_ATTRIB_TEXCOORD0_NAME);
    auto const indices_accessor_index = gltf_prim.indicesAccessor;

    auto const indices = resolve_stream(indices_accessor_index, fastgltf::AccessorType::Scalar);
    if (!indices.has_value()) { return std::nullopt; }
    DBG_ASSERT_TRUE_M(indices->source.component_type == ComponentType::U16 || indices->source.component_type == ComponentType::U32, "Mesh indices must be U16 or U32");

    auto const positions = resolve_stream(position_accessor_index, fastgltf::AccessorType::Vec3);
    if (!positions.has_value()) { return std::nullopt; }
    DBG_ASSERT_TRUE_M(positions->source.component_type == ComponentType::F32, "Mesh positions must be F32");

    auto const normals = resolve_stream(normal_accessor_index, fastgltf::AccessorType::Vec3);
    if (!normals.has_value()) { return std::nullopt; }
    DBG_ASSERT_TRUE_M(normals->source.component_type == ComponentType::F32, "Mesh normals must be F32");
    DBG_ASSERT_TRUE_M(normals->element_count == positions->element_count, "Mesh normals must have the same element count as positions");

    auto const uvs = resolve_stream(uv_accessor_index, fastgltf::AccessorType::Vec2);
    if (uv_accessor_index.has_value() && !uvs.has_value()) { DEBUG_MSG("Mesh uvs found but failed to parse - dropping uvs"); }
    DBG_ASSERT_TRUE_M(!uvs.has_value() || uvs->source.component_type == ComponentType::F32, "Mesh uvs must be F32");
    DBG_ASSERT_TRUE_M(!uvs.has_value() || uvs->element_count == positions->element_count, "Mesh uvs must have the same element count as positions");


    return MeshImporterData{
        .indices = indices->source,
        .positions = positions->source,
        .normals = normals->source,
        .uvs = uvs.has_value() ? std::optional<MeshAttribSource>{uvs->source} : std::nullopt,
        .vertex_count = positions->element_count,
        .index_count = indices->element_count,
    };
}
} // namespace

namespace
{
static auto mime_type_to_image_format(fastgltf::MimeType mime_type) -> std::optional<ImageFileFormat>
{
    if (mime_type == fastgltf::MimeType::KTX2) { return ImageFileFormat::KTX2; }
    if (mime_type == fastgltf::MimeType::PNG) { return ImageFileFormat::PNG; }
    return std::nullopt; // Unsupported source format.
}

struct ImageSourceLocate
{
    SourceLocation location = {};
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
            .location = SourceLocation{.file = full_image_path},
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
        fastgltf::sources::URI const & buffer_uri = std::get<fastgltf::sources::URI>(gltf_buffer.data);
        std::filesystem::path const full_buffer_path = scene_dir_path / buffer_uri.uri.fspath();
        auto const format = mime_type_to_image_format(buffer_view->mimeType);
        if (!format.has_value())
        {
            return std::nullopt;
        }
        return ImageSourceLocate{
            .location = SourceLocation{
                .file = full_buffer_path,
                .slice = ByteSlice{
                    .byte_offset = gltf_buffer_view.byteOffset + buffer_uri.fileByteOffset,
                    .byte_length = gltf_buffer_view.byteLength,
                },
            },
            .format = format.value(),
        };
    }
    return std::nullopt;
}

// Collapses the images of one parse into manifest entries. Keyed on location plus recipe, not on content: the
// artifact key needs the source bytes, which are only read at cook time. Two entries whose content turns out
// to be identical still share one artifact, they just each resolve to it. Parse-local and never stored - no
// durable name may depend on a location.
auto image_parse_dedup_key(ImageImporterData const & importer_data) -> u64
{
    std::vector<std::byte> key_bytes;
    std::string const path_string = importer_data.source_location.file.generic_string();
    tido_append_bytes(key_bytes, path_string.data(), path_string.size());
    tido_append_pod(key_bytes, importer_data.source_location.slice.has_value());
    if (importer_data.source_location.slice.has_value())
    {
        tido_append_pod(key_bytes, importer_data.source_location.slice->byte_offset);
        tido_append_pod(key_bytes, importer_data.source_location.slice->byte_length);
    }

    tido_append_pod(key_bytes, importer_data.container_format);
    for (auto const & mapped_channel : importer_data.channel_mapping)
    {
        key_bytes.push_back(static_cast<std::byte>(mapped_channel));
    }
    tido_append_pod(key_bytes, importer_data.target_format);

    return tido_fnv1a(key_bytes, 0);
}
} // namespace

// ============================ Shared glTF parse ===========================
namespace
{
// Feeds fastgltf from our own file I/O rather than handing it the path, so the share mode, the bounds checks
// and the FileIoResult taxonomy all stay in file_io. Only what fastgltf asks for is buffered: for a .glb that
// is the JSON chunk, while the binary chunk is read straight into the storage the asset keeps it in.
struct GltfFileReaderDataGetter final : fastgltf::GltfDataGetter
{
    explicit GltfFileReaderDataGetter(FileReader && reader) : reader{std::move(reader)} {}

    void read(void * destination, std::size_t byte_count) override
    {
        if (reader.read_into(destination, byte_count) != FileIoResult::SUCCESS) { read_failed = true; }
    }

    auto read(std::size_t byte_count, std::size_t padding) -> fastgltf::span<std::byte> override
    {
        // The span has to expose byte_count + padding bytes: simdjson reads past the data it is given and
        // does not care what the padding holds.
        scratch.resize(byte_count + padding);
        if (reader.read_into(scratch.data(), byte_count) != FileIoResult::SUCCESS)
        {
            read_failed = true;
            // fastgltf has no way to report the failure back, so hand it zeroes rather than stale bytes.
            std::fill(scratch.begin(), scratch.end(), std::byte{});
        }
        return fastgltf::span<std::byte>(scratch.data(), scratch.size());
    }

    void reset() override
    {
        if (reader.seek(0) != FileIoResult::SUCCESS) { read_failed = true; }
    }

    auto bytesRead() -> std::size_t override { return s_cast<std::size_t>(reader.read_byte_offset()); }
    auto totalSize() -> std::size_t override { return s_cast<std::size_t>(reader.file_byte_size()); }

    FileReader reader = {};
    std::vector<std::byte> scratch = {};
    // The interface returns void and spans, so a failed read can only be reported after the parse finishes.
    bool read_failed = false;
};

// Parses a .gltf/.glb into a fastgltf::Asset. Only SceneParseTask calls this - fastgltf is touched at
// scene-parse only; asset cooks run off the resolved ImporterData and never re-parse.
static auto parse_gltf_file(std::filesystem::path const & file_path) -> std::variant<Scene::LoadManifestErrorCode, fastgltf::Asset>
{
    fastgltf::Parser parser{
        fastgltf::Extensions::KHR_texture_basisu |
        fastgltf::Extensions::KHR_lights_punctual};

    constexpr auto gltf_options =
        fastgltf::Options::DontRequireValidAssetMember |
        fastgltf::Options::AllowDouble;

    auto [io_result, reader] = FileReader::open(file_path);
    if (io_result != FileIoResult::SUCCESS)
    {
        return io_result == FileIoResult::NOT_FOUND ? Scene::LoadManifestErrorCode::FILE_NOT_FOUND : Scene::LoadManifestErrorCode::COULD_NOT_LOAD_ASSET;
    }
    if (reader.file_byte_size() == 0) { return Scene::LoadManifestErrorCode::COULD_NOT_LOAD_ASSET; }

    GltfFileReaderDataGetter data = GltfFileReaderDataGetter(std::move(reader));
    auto const type = fastgltf::determineGltfFileType(data);

    switch (type)
    {
        case fastgltf::GltfType::glTF:
        {
            fastgltf::Expected<fastgltf::Asset> result = parser.loadGltf(data, file_path.parent_path(), gltf_options);
            if (result.error() != fastgltf::Error::None || data.read_failed)
            {
                return Scene::LoadManifestErrorCode::COULD_NOT_LOAD_ASSET;
            }
            return std::move(result.get());
        }
        case fastgltf::GltfType::GLB:
        {
            fastgltf::Expected<fastgltf::Asset> result = parser.loadGltfBinary(data, file_path.parent_path(), gltf_options);
            if (result.error() != fastgltf::Error::None || data.read_failed)
            {
                return Scene::LoadManifestErrorCode::COULD_NOT_LOAD_ASSET;
            }
            return std::move(result.get());
        }
        case fastgltf::GltfType::Invalid:
            return Scene::LoadManifestErrorCode::INVALID_GLTF_FILE_TYPE;
        default:
            DBG_ASSERT_TRUE_M(false, "Unhandled glTF file type");
            return Scene::LoadManifestErrorCode::INVALID_GLTF_FILE_TYPE;
    }
}


} // namespace

// ====================== gltf backend: parse -> ParsedSource (no cooking) =====================
namespace
{

// Parses one glTF/GLB file on a ThreadPool worker and translates it into a ParsedSource: the scene
// metadata the engine should apply, and the cook requests that will produce the artifacts it is missing.
// Never touches the Scene or the cache; the Importer publishes the parse and queues the cooks.
//
// Materials, mesh groups and entities are translated in that order because each wires up references to
// what the previous one produced. Image cooks are deduplicated on image_parse_dedup_key - several textures
// can share one image, differing only by sampler, and that image must be cooked only once - which is why a
// single cook carries a list of the material slots consuming it.

struct SceneParseTask final : SourceParseTask
{
    explicit SceneParseTask(SourceImportRequest request)
        : request{std::move(request)}
    {
        chunk_count = 1;
        // Set here rather than in the callback so a parse that fails before emitting anything still names
        // the source it belongs to.
        parsed.source_index = this->request.source_index;
    }

    void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override;

  private:
    SourceImportRequest request = {};

    fastgltf::Asset asset;

    void translate_materials();
    void translate_mesh_groups();
    void translate_entities();
    auto translate_light(fastgltf::Light const & light) -> u32;
};

void SceneParseTask::callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index)
{
    // A container discovers its own slots, so it has nothing to apply a recipe to. Overriding one needs a
    // recipe that can name the slot it applies to, which does not exist yet.
    if (!request.recipes.empty())
    {
        DEBUG_MSG(fmt::format("[WARN][SceneParseTask] '{}' imported with recipes; a container discovers its own slots and cannot apply one",
            request.path.string()));
        failed = true;
        return;
    }

    auto parse_result = parse_gltf_file(request.path);
    if (auto const * error = std::get_if<Scene::LoadManifestErrorCode>(&parse_result))
    {
        DEBUG_MSG(fmt::format("[WARN][SceneParseTask::callback] Loading \"{}\" Error: {}", request.path.string(), Scene::to_string(*error)));
        failed = true;
        return;
    }
    asset = std::move(std::get<fastgltf::Asset>(parse_result));

    translate_materials();
    translate_mesh_groups();
    translate_entities();
}

void SceneParseTask::translate_materials()
{
    // image_parse_dedup_key -> index into parsed.images
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

    auto default_image_import_info = [&](MaterialTextureSlot const texture_type, u32 const image_index) -> ImageImporterData
    {
        auto const source = resolve_image_source(asset, image_index, request.path);
        DBG_ASSERT_TRUE_M(source.has_value(), "Unsupported or unresolvable image source");
        ImageSourceLocate const resolved_source = source.value_or(ImageSourceLocate{});

        // Shared location; only the recipe (channel_mapping + target_format) varies per material slot.
        ImageImporterData import_info = {
            .source_location = resolved_source.location,
            .container_format = resolved_source.format,
        };
        switch(texture_type)
        {
            case MaterialTextureSlot::DIFFUSE:
                import_info.channel_mapping = {0, 1, 2};    import_info.target_format = daxa::Format::BC1_RGB_SRGB_BLOCK;   break;
            case MaterialTextureSlot::OPACITY:
                import_info.channel_mapping = {3};          import_info.target_format = daxa::Format::BC4_UNORM_BLOCK;  break;
            case MaterialTextureSlot::NORMAL:
                import_info.channel_mapping = {0, 1};       import_info.target_format = daxa::Format::BC5_UNORM_BLOCK;  break;
            case MaterialTextureSlot::ROUGHNESS_METALNESS:
                import_info.channel_mapping = {0, 1, 2, 3}; import_info.target_format = daxa::Format::BC7_UNORM_BLOCK;  break;
            case MaterialTextureSlot::COUNT:
            default:
                DBG_ASSERT_TRUE_M(false, "Unhandled texture type in default_image_import_info");
                break;
        }
        return import_info;
    };

    // Register one material texture slot as a consumer of an image's cook, creating that cook the first time
    // the image is seen under this recipe. The binding itself is left empty: what a slot samples until its
    // own image is cooked is a stand-in the parse knows nothing about, so publishing fills it in.
    auto bind_material_texture = [&](u32 const local_material_index, MaterialTextureSlot const texture_type, u32 const gltf_texture_index) -> bool
    {
        auto const gltf_image_index = gltf_texture_to_image_index(gltf_texture_index);
        if (!gltf_image_index.has_value()) { return false; }

        ImageImporterData importer_data = default_image_import_info(texture_type, gltf_image_index.value());
        u64 const dedup_key = image_parse_dedup_key(importer_data);
        auto const [iterator, inserted] = image_manifest_map.try_emplace(dedup_key, s_cast<u32>(parsed.images.size()));
        u32 const cook_index = iterator->second;
        if (inserted)
        {
            parsed.images.push_back(ParsedSource::Image{
                .name = asset.images.at(gltf_image_index.value()).name.c_str(),
            });
            parsed.image_cook_inputs.push_back(std::move(importer_data));
        }

        // Sanity check: the same image under the same recipe must always resolve to the same cook.
        // Every slot this backend emits is an encoded 2D image, never a vdb one.
        DBG_ASSERT_TRUE_M(
            image_parse_dedup_key(std::get<ImageImporterData>(parsed.image_cook_inputs.at(cook_index))) == dedup_key,
            "Image cook request mismatch");

        parsed.images.at(cook_index).bound_materials.push_back(ImageMaterialBinding{
            .material_index = local_material_index,
            .slot = texture_type,
        });
        return true;
    };

    for (u32 material_index = 0; material_index < s_cast<u32>(asset.materials.size()); material_index++)
    {
        auto const & material = asset.materials.at(material_index);
        u32 const local_material_index = s_cast<u32>(parsed.materials.size());
        bool has_opacity_texture = false;
        if (material.pbrData.baseColorTexture.has_value())
        {
            bind_material_texture(local_material_index, MaterialTextureSlot::DIFFUSE, s_cast<u32>(material.pbrData.baseColorTexture.value().textureIndex));
        }
        if(material.alphaMode == fastgltf::AlphaMode::Mask && material.pbrData.baseColorTexture.has_value())
        {
            has_opacity_texture = bind_material_texture(local_material_index, MaterialTextureSlot::OPACITY, s_cast<u32>(material.pbrData.baseColorTexture.value().textureIndex));
        }
        if (material.normalTexture.has_value())
        {
            bind_material_texture(local_material_index, MaterialTextureSlot::NORMAL, s_cast<u32>(material.normalTexture.value().textureIndex));
        }
        // if (material.pbrData.metallicRoughnessTexture.has_value())
        // {
        //     bind_material_texture(local_material_index, MaterialTextureSlot::ROUGHNESS_METALNESS, s_cast<u32>(material.pbrData.metallicRoughnessTexture.value().textureIndex));
        // }

        bool const alpha_discard_enabled = material.alphaMode == fastgltf::AlphaMode::Mask && has_opacity_texture;

        // Texture bindings stay empty here and are filled with their stand-ins at publish; the cooks
        // registered above carry which slots to rebind once the real images exist.
        parsed.materials.push_back(MaterialWrite{
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
    parsed.mesh_groups.reserve(asset.meshes.size());
    for (u32 mesh_group_index = 0; mesh_group_index < s_cast<u32>(asset.meshes.size()); ++mesh_group_index)
    {
        auto const & gltf_mesh = asset.meshes.at(mesh_group_index);
        auto & mesh_group = parsed.mesh_groups.emplace_back(std::vector<u32>{}, gltf_mesh.name.c_str());
        mesh_group.local_mesh_indices.reserve(gltf_mesh.primitives.size());

        for (u32 primitive_index = 0; primitive_index < s_cast<u32>(gltf_mesh.primitives.size()); ++primitive_index)
        {
            auto const & gltf_primitive = gltf_mesh.primitives.at(primitive_index);

            u32 const local_mesh_index = s_cast<u32>(parsed.meshes.size());
            auto mesh_importer_data_opt = resolve_mesh_source(asset, request.path, mesh_group_index, primitive_index);
            DBG_ASSERT_TRUE_M(mesh_importer_data_opt.has_value(), "Unresolvable or unsupported mesh primitive source");
            parsed.meshes.push_back(ParsedSource::Mesh{
                .local_material_index = gltf_primitive.materialIndex.has_value()
                    ? std::optional<u32>(s_cast<u32>(gltf_primitive.materialIndex.value()))
                    : std::nullopt,
                .name = gltf_mesh.name.c_str(),
            });
            parsed.mesh_cook_inputs.push_back(std::move(mesh_importer_data_opt).value_or(MeshImporterData{}));
            mesh_group.local_mesh_indices.push_back(local_mesh_index);
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
            PointLightWrite cpu_point_light = {};
            cpu_point_light.position = f32vec3{0.0f, 0.0f, 0.0f}; // Filled/updated later when processing scene graph
            cpu_point_light.color = f32vec3{light.color.x(), light.color.y(), light.color.z()};
            // Converting candella to watt - blender (https://projects.blender.org/blender/blender-addons/issues/91035).
            cpu_point_light.intensity = (light.intensity * 4.0f * glm::pi<f32>()) / LUMENS_PER_WATT;
            // When the cutoff is not specified attempt to calculate one based on a minimum energy.
            cpu_point_light.cutoff = light.range.value_or(std::sqrt(light.intensity / E_min));
            u32 const index = s_cast<u32>(parsed.point_lights.size());
            parsed.point_lights.push_back(cpu_point_light);
            return index;
        }
        case fastgltf::LightType::Spot:
        {
            SpotLightWrite cpu_spot_light = {};
            cpu_spot_light.transform = {}; // Filled/updated later when processing scene graph
            cpu_spot_light.color = f32vec3{light.color.x(), light.color.y(), light.color.z()};
            // Converting candella to watt - https://google.github.io/filament/Filament.md.html#lighting
            cpu_spot_light.intensity = (light.intensity * glm::pi<f32>()) / LUMENS_PER_WATT;
            cpu_spot_light.inner_cone_angle = light.innerConeAngle.value();
            cpu_spot_light.outer_cone_angle = light.outerConeAngle.value();
            DBG_ASSERT_TRUE_M(light.range.has_value(), "Currently no auto deduce of range from intensity for spot lights");
            cpu_spot_light.cutoff = light.range.value();
            u32 const index = s_cast<u32>(parsed.spot_lights.size());
            parsed.spot_lights.push_back(cpu_spot_light);
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

void SceneParseTask::translate_entities()
{
    /// NOTE: fastgltf::Node is Entity
    DBG_ASSERT_TRUE_M(asset.nodes.size() != 0, "[ERROR][SceneParseTask::translate_entities()] Empty node array - what to do now?");

    u32 const node_count = s_cast<u32>(asset.nodes.size());
    // node index == local index. Nodes left unparented here are adopted by the root publish gives the source.
    std::vector<ParsedSource::Entity> node_entities(node_count);

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
        ParsedSource::Entity & r_ent = node_entities[node_index];
        r_ent.local_mesh_group_index = node.meshIndex.has_value()
            ? std::optional<u32>(s_cast<u32>(node.meshIndex.value()))
            : std::nullopt;
        r_ent.transform = fastgltf_to_glm_mat4x3_transform(node.transform);
        r_ent.name = node.name.c_str();

        r_ent.local_light_index = std::nullopt;

        DBG_ASSERT_TRUE_M(
            s_cast<u32>(node.lightIndex.has_value()) +
                    s_cast<u32>(node.meshIndex.has_value()) +
                    s_cast<u32>(node.cameraIndex.has_value()) <=
                1u,
            "Node can only be of one type");

        if (node.lightIndex.has_value())
        {
            fastgltf::Light const & light = asset.lights.at(node.lightIndex.value());
            r_ent.local_light_index = translate_light(light);
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

    parsed.entities = std::move(node_entities);
}
} // namespace


// ==================================== gltf backend dispatch =======================================

auto parse_gltf_source(SourceImportRequest const & request) -> std::shared_ptr<SourceParseTask>
{
    // The gltf backend only parses sources; the fastgltf-free asset cooks are dispatched by the Importer from
    // the slots the parse emits.
    return std::make_shared<SceneParseTask>(request);
}
