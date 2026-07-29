#include "importer.hpp"

#include <cmath>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fmt/format.h>
#include <glm/gtx/quaternion.hpp>

#include "../optimizers/image_processor.hpp"

namespace
{
static constexpr std::string_view VERT_ATTRIB_POSITION_NAME = "POSITION";
static constexpr std::string_view VERT_ATTRIB_TEXCOORD0_NAME = "TEXCOORD_0";
static constexpr std::string_view VERT_ATTRIB_NORMAL_NAME = "NORMAL";

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

// A resolved mesh stream plus its element count, kept so callers can cross-check vertex/index counts.
struct ResolvedMeshStream
{
    MeshAttribSource source = {};
    u32 element_count = {};
};

// Resolve where one primitive's tightly-packed vertex/index streams live without reading them. The accepted
// formats match extract_raw_mesh's accessor validation (F32 vec3 positions/normals, F32 vec2 uvs, U16|U32
// scalar indices). nullopt if a required stream is missing/invalid or a source is embedded/unsupported.
// cache_path is left defaulted (routing-only, set by the caller).
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

        auto const range = locate_accessor_range(asset, root_path, accessor);
        if (!range.has_value()) { return std::nullopt; }
        return ResolvedMeshStream{
            .source = MeshAttribSource{.range = range.value(), .component_type = component_type.value()},
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
        fastgltf::sources::URI const & buffer_uri = std::get<fastgltf::sources::URI>(gltf_buffer.data);
        std::filesystem::path const full_buffer_path = scene_dir_path / buffer_uri.uri.fspath();
        auto const format = mime_type_to_image_format(buffer_view->mimeType);
        if (!format.has_value())
        {
            return std::nullopt;
        }
        return ImageSourceLocate{
            .range = FileByteRange{
                .file = full_buffer_path,
                .byte_offset = gltf_buffer_view.byteOffset + buffer_uri.fileByteOffset,
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

    auto [io_result, file_bytes] = read_file(file_path);
    if (io_result != FileIoResult::SUCCESS)
    {
        return io_result == FileIoResult::NOT_FOUND ? Scene::LoadManifestErrorCode::FILE_NOT_FOUND : Scene::LoadManifestErrorCode::COULD_NOT_LOAD_ASSET;
    }
    if (file_bytes.empty()) { return Scene::LoadManifestErrorCode::COULD_NOT_LOAD_ASSET; }

    auto data_opt = fastgltf::GltfDataBuffer::FromBytes(file_bytes.data(), file_bytes.size());
    if (data_opt.error() != fastgltf::Error::None) { return Scene::LoadManifestErrorCode::COULD_NOT_LOAD_ASSET; }

    // Free the file bytes now that fastgltf has copied them into its own padded buffer.
    file_bytes.clear();

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
        ImageSourceLocate const resolved_source = source.value_or(ImageSourceLocate{});

        // Shared location; only the recipe (channel_mapping + target_format) varies per material slot.
        ImageImporterData import_info = {
            .cache_path = file_path,
            .source_bytes = resolved_source.range,
            .container_format = resolved_source.format,
        };
        switch(texture_type)
        {
            case GLTFTextureMaterialType::DIFFUSE:
                import_info.channel_mapping = {0, 1, 2};    import_info.target_format = daxa::Format::BC1_RGB_SRGB_BLOCK;   break;
            case GLTFTextureMaterialType::OPACITY:
                import_info.channel_mapping = {3};          import_info.target_format = daxa::Format::BC4_UNORM_BLOCK;  break;
            case GLTFTextureMaterialType::NORMAL:
                import_info.channel_mapping = {0, 1};       import_info.target_format = daxa::Format::BC5_UNORM_BLOCK;  break;
            case GLTFTextureMaterialType::ROUGHNESS_METALNESS:
                import_info.channel_mapping = {0, 1, 2, 3}; import_info.target_format = daxa::Format::BC7_UNORM_BLOCK;  break;
            case GLTFTextureMaterialType::NONE:
            default:
                DBG_ASSERT_TRUE_M(false, "Unhandled texture type in default_image_import_info");
                break;
        }
        return import_info;
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

    // Resolve one material texture slot: gltf texture -> image -> deduped manifest image tagged with its name.
    auto resolve_material_texture = [&](u32 const gltf_texture_index, GLTFTextureMaterialType const texture_type) -> std::optional<MaterialManifestEntry::ImageInfo>
    {
        auto const gltf_image_index = gltf_texture_to_image_index(gltf_texture_index);
        auto const gltf_texture_name = asset.images.at(gltf_image_index.value()).name;
        return resolve_image_info({.name = gltf_texture_name.c_str(), .importer_data = default_image_import_info(texture_type, gltf_image_index.value())}, 0);
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
            diffuse_texture_info = resolve_material_texture(s_cast<u32>(material.pbrData.baseColorTexture.value().textureIndex), GLTFTextureMaterialType::DIFFUSE);
        }
        if(material.alphaMode == fastgltf::AlphaMode::Mask && material.pbrData.baseColorTexture.has_value())
        {
            opacity_texture_info = resolve_material_texture(s_cast<u32>(material.pbrData.baseColorTexture.value().textureIndex), GLTFTextureMaterialType::OPACITY);
        }
        if (material.normalTexture.has_value())
        {
            normal_texture_info = resolve_material_texture(s_cast<u32>(material.normalTexture.value().textureIndex), GLTFTextureMaterialType::NORMAL);
        }
        // if (material.pbrData.metallicRoughnessTexture.has_value())
        // {
        //     roughness_metalness_info = resolve_material_texture(s_cast<u32>(material.pbrData.metallicRoughnessTexture.value().textureIndex), GLTFTextureMaterialType::ROUGHNESS_METALNESS);
        // }

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
                .material_index = gltf_primitive.materialIndex.has_value() ? std::optional<u32>(s_cast<u32>(gltf_primitive.materialIndex.value())) : std::nullopt,
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
        r_ent.mesh_group_manifest_index = node.meshIndex.has_value() ? std::optional<u32>(s_cast<u32>(node.meshIndex.value())) : std::nullopt;
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


// ==================================== Scene-parse dispatch =======================================

void dispatch_scene_parses(Importer & importer, std::vector<ImporterTask> & tasks)
{
    // The gltf backend only parses scenes into a SceneMetadataBatch; the fastgltf-free asset cooks are
    // dispatched by Importer (from the fully resolved ImporterData) before this runs.
    auto consume_scene_task = [&](ImporterTask & task) -> bool
    {
        if (auto const * import_scene = std::get_if<ImporterTask::ImportScene>(&task.data))
        {
            importer.thread_pool->async_dispatch(std::make_shared<SceneParseTask>(import_scene->path, &importer), TaskPriority::LOW);
            return true;
        }
        return false;
    };
    std::erase_if(tasks, consume_scene_task);
}
