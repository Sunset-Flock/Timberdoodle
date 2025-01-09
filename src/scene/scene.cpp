#include "scene.hpp"

#include <fstream>

#include <fastgltf/core.hpp>

#include <fmt/format.h>
#include <glm/gtx/quaternion.hpp>
#include <thread>
#include <chrono>
#include <ktx.h>
#include "../daxa_helper.hpp"

#include "../shader_shared/raytracing.inl"
#include "../shader_shared/scene.inl"
#include "mesh_lod.hpp"

Scene::Scene(daxa::Device device, GPUContext * gpu_context)
    : _device{std::move(device)}, gpu_context{gpu_context}

{
    /// TODO: THIS IS TEMPORARY! Make manifest and entity buffers growable!
    _gpu_entity_meta = tido::make_task_buffer(_device, sizeof(GPUEntityMetaData), "_gpu_entity_meta");
    _gpu_entity_parents = tido::make_task_buffer(_device, sizeof(RenderEntityId) * MAX_ENTITIES, "_gpu_entity_parents");
    _gpu_entity_transforms = tido::make_task_buffer(_device, sizeof(daxa_f32mat4x3) * MAX_ENTITIES, "_gpu_entity_transforms");
    _gpu_entity_combined_transforms = tido::make_task_buffer(_device, sizeof(daxa_f32mat4x3) * MAX_ENTITIES, "_gpu_entity_combined_transforms");
    _gpu_entity_mesh_groups = tido::make_task_buffer(_device, sizeof(u32) * MAX_ENTITIES, "_gpu_entity_mesh_groups");
    _gpu_mesh_manifest = tido::make_task_buffer(_device, sizeof(GPUMesh) * MAX_MESHES, "_gpu_mesh_manifest");
    _gpu_mesh_lod_group_manifest = tido::make_task_buffer(_device, sizeof(GPUMeshLodGroup) * MAX_MESH_LOD_GROUPS, "_gpu_mesh_lod_group_manifest");
    _gpu_mesh_group_manifest = tido::make_task_buffer(_device, sizeof(GPUMeshGroup) * MAX_MESH_LOD_GROUPS, "_gpu_mesh_group_manifest");
    _gpu_material_manifest = tido::make_task_buffer(_device, sizeof(GPUMaterial) * MAX_MATERIALS, "_gpu_material_manifest");
    _gpu_point_lights = tido::make_task_buffer(_device, sizeof(GPUPointLight) * MAX_POINT_LIGHTS, "_gpu_point_lights", daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE);
    _gpu_scratch_buffer = tido::make_task_buffer(_device, _gpu_scratch_buffer_size, "_gpu_scratch_buffer");
    _gpu_mesh_acceleration_structure_build_scratch_buffer = tido::make_task_buffer(_device, _gpu_mesh_acceleration_structure_build_scratch_buffer_size, "_gpu_mesh_acceleration_structure_build_scratch_buffer");
    _gpu_tlas_build_scratch_buffer = tido::make_task_buffer(_device, _gpu_tlas_build_scratch_buffer_size, "_gpu_tlas_build_scratch_buffer");
    mesh_instances_buffer = daxa::TaskBuffer{daxa::TaskBufferInfo{.name = "mesh_instances"}};
    // _scene_tlas = daxa::TaskTlas{
    //     {
    //         .initial_tlas = {
    //             .tlas = std::array{gpu_context->dummy_tlas_id},
    //         },
    //         .name = "scene tlas",
    //     },
    // };
    _scene_as_indirections = tido::make_task_buffer(_device, _indirections_count, "_scene_as_indirections", daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE);
}

Scene::~Scene()
{
    if (!_gpu_mesh_group_indices_array_buffer.is_empty()) { _device.destroy_buffer(_gpu_mesh_group_indices_array_buffer); }
    if (!_scene_blas.is_empty()) { _device.destroy_blas(_scene_blas); }

    for (auto & mesh_group : _mesh_group_manifest)
    {
        if(!mesh_group.blas.is_empty())
        {
            _device.destroy_blas(mesh_group.blas);
        }
    }

    for (auto & mesh : _mesh_lod_group_manifest)
    {
        if (mesh.runtime.has_value())
        {
            for (daxa_u32 lod = 0; lod < mesh.runtime.value().lod_count; ++lod)
            {
                _device.destroy_buffer(std::bit_cast<daxa::BufferId>(mesh.runtime.value().lods[lod].mesh_buffer));
                if (!mesh.runtime.value().blas_lods[lod].is_empty())
                {
                    _device.destroy_blas(mesh.runtime.value().blas_lods[lod]);
                }
            }
        }
    }

    for (auto & texture : _material_texture_manifest)
    {
        if (texture.runtime_texture.has_value())
        {
            _device.destroy_image(std::bit_cast<daxa::ImageId>(texture.runtime_texture.value()));
        }
        if (texture.secondary_runtime_texture.has_value())
        {
            _device.destroy_image(std::bit_cast<daxa::ImageId>(texture.secondary_runtime_texture.value()));
        }
    }
}
// TODO: Loading god function.
struct LoadManifestFromFileContext
{
    std::filesystem::path file_path = {};
    fastgltf::Asset asset;
    u32 gltf_asset_manifest_index = {};
    u32 texture_manifest_offset = {};
    u32 material_manifest_offset = {};
    u32 mesh_group_manifest_offset = {};
    u32 mesh_manifest_offset = {};
};
static auto get_load_manifest_data_from_gltf(Scene & scene, Scene::LoadManifestInfo const & info) -> std::variant<LoadManifestFromFileContext, Scene::LoadManifestErrorCode>;
static void update_material_manifest_from_gltf(Scene & scene, Scene::LoadManifestInfo const & info, LoadManifestFromFileContext & load_ctx);
static void update_texture_manifest_from_gltf(Scene & scene, Scene::LoadManifestInfo const & info, LoadManifestFromFileContext & load_ctx);
static void update_meshgroup_and_mesh_manifest_from_gltf(Scene & scene, Scene::LoadManifestInfo const & info, LoadManifestFromFileContext & load_ctx);
static void start_async_loads_of_dirty_meshes(Scene & scene, Scene::LoadManifestInfo const & info);
static void start_async_loads_of_dirty_textures(Scene & scene, Scene::LoadManifestInfo const & info);
// Returns root entity of loaded asset.
static auto update_entities_from_gltf(Scene & scene, Scene::LoadManifestInfo const & info, LoadManifestFromFileContext & gpu_context) -> RenderEntityId;
static void update_lights_from_gltf(Scene & scene, Scene::LoadManifestInfo const & info);

auto Scene::load_manifest_from_gltf(LoadManifestInfo const & info) -> std::variant<RenderEntityId, LoadManifestErrorCode>
{
    RenderEntityId root_r_ent_id = {};
    {
        auto load_result = get_load_manifest_data_from_gltf(*this, info);
        if (std::holds_alternative<LoadManifestErrorCode>(load_result))
        {
            update_lights_from_gltf(*this, info);
            return std::get<LoadManifestErrorCode>(load_result);
        }
        LoadManifestFromFileContext load_ctx = std::get<LoadManifestFromFileContext>(std::move(load_result));
        update_texture_manifest_from_gltf(*this, info, load_ctx);
        update_material_manifest_from_gltf(*this, info, load_ctx);
        update_meshgroup_and_mesh_manifest_from_gltf(*this, info, load_ctx);
        root_r_ent_id = update_entities_from_gltf(*this, info, load_ctx);
        update_lights_from_gltf(*this, info);
        _gltf_asset_manifest.push_back(GltfAssetManifestEntry{
            .path = load_ctx.file_path,
            .gltf_asset = std::make_unique<fastgltf::Asset>(std::move(load_ctx.asset)),
            .texture_manifest_offset = load_ctx.texture_manifest_offset,
            .material_manifest_offset = load_ctx.material_manifest_offset,
            .mesh_group_manifest_offset = load_ctx.mesh_group_manifest_offset,
            .mesh_manifest_offset = load_ctx.mesh_manifest_offset,
            .root_render_entity = root_r_ent_id,
        });
    }
    start_async_loads_of_dirty_meshes(*this, info);
    start_async_loads_of_dirty_textures(*this, info);

    return root_r_ent_id;
}

static auto get_load_manifest_data_from_gltf(Scene & scene, Scene::LoadManifestInfo const & info) -> std::variant<LoadManifestFromFileContext, Scene::LoadManifestErrorCode>
{
    auto file_path = info.root_path / info.asset_name;

    fastgltf::Parser parser{fastgltf::Extensions::KHR_texture_basisu};

    constexpr auto gltf_options =
        fastgltf::Options::DontRequireValidAssetMember |
        fastgltf::Options::AllowDouble;

    fastgltf::GltfDataBuffer data;
    bool const worked = data.loadFromFile(file_path);
    if (!worked)
    {
        return Scene::LoadManifestErrorCode::FILE_NOT_FOUND;
    }
    auto type = fastgltf::determineGltfFileType(&data);
    LoadManifestFromFileContext load_ctx;
    switch (type)
    {
        case fastgltf::GltfType::glTF:
        {
            fastgltf::Expected<fastgltf::Asset> result = parser.loadGltf(&data, file_path.parent_path(), gltf_options);
            if (result.error() != fastgltf::Error::None)
            {
                return Scene::LoadManifestErrorCode::COULD_NOT_LOAD_ASSET;
            }
            load_ctx.asset = std::move(result.get());
            break;
        }
        case fastgltf::GltfType::GLB:
        {
            fastgltf::Expected<fastgltf::Asset> result = parser.loadGltfBinary(&data, file_path.parent_path(), gltf_options);
            if (result.error() != fastgltf::Error::None)
            {
                return Scene::LoadManifestErrorCode::COULD_NOT_LOAD_ASSET;
            }
            load_ctx.asset = std::move(result.get());
            break;
        }
        default:
            return Scene::LoadManifestErrorCode::INVALID_GLTF_FILE_TYPE;
    }
    load_ctx.file_path = std::move(file_path);
    load_ctx.gltf_asset_manifest_index = s_cast<u32>(scene._gltf_asset_manifest.size());
    load_ctx.texture_manifest_offset = s_cast<u32>(scene._material_texture_manifest.size());
    load_ctx.material_manifest_offset = s_cast<u32>(scene._material_manifest.size());
    load_ctx.mesh_group_manifest_offset = s_cast<u32>(scene._mesh_group_manifest.size());
    load_ctx.mesh_manifest_offset = s_cast<u32>(scene._mesh_lod_group_manifest.size());
    return load_ctx;
}

static void update_texture_manifest_from_gltf(Scene & scene, Scene::LoadManifestInfo const & info, LoadManifestFromFileContext & load_ctx)
{
    auto gltf_texture_to_image_index = [&](u32 const texture_index) -> std::optional<u32>
    {
        fastgltf::Asset const & asset = load_ctx.asset;
        if (asset.textures.at(texture_index).basisuImageIndex.has_value())
        {
            return s_cast<u32>(asset.textures.at(texture_index).basisuImageIndex.value());
        }
        else if (asset.textures.at(texture_index).imageIndex.has_value())
        {
            return s_cast<u32>(asset.textures.at(texture_index).imageIndex.value());
        }
        else
        {
            return std::nullopt;
        }
    };
    /// NOTE: GLTF texture = image + sampler we collapse the sampler into the material itself here we thus only iterate over the images
    //        Later when we load in the materials which reference the textures rather than images we just
    //        translate the textures image index and store that in the material

    for (u32 i = 0; i < s_cast<u32>(load_ctx.asset.textures.size()); ++i)
    {
        u32 const texture_manifest_index = s_cast<u32>(scene._material_texture_manifest.size());
        auto gltf_image_idx_opt = gltf_texture_to_image_index(texture_manifest_index);
        DBG_ASSERT_TRUE_M(
            gltf_image_idx_opt.has_value(),
            fmt::format(
                "[ERROR] Texture \"{}\" has no supported gltf image index!\n",
                load_ctx.asset.textures[i].name.c_str()));
        u32 gltf_image_index = gltf_image_idx_opt.value();
        DEBUG_MSG(
            fmt::format("[INFO] Loading texture meta data into manifest:\n  name: {}\n  asset local index: {}\n  manifest index:  {}",
                load_ctx.asset.images[gltf_image_index].name, i, texture_manifest_index));
        // KTX_TTF_BC7_RGBA
        scene._material_texture_manifest.push_back(TextureManifestEntry{
            .type = TextureMaterialType::NONE, // Set by material manifest.
            .gltf_asset_manifest_index = load_ctx.gltf_asset_manifest_index,
            .asset_local_index = i,
            .asset_local_image_index = gltf_image_index,
            .material_manifest_indices = {}, // Filled when reading in materials
            .runtime_texture = {},           // Filled when the texture data are uploaded to the GPU
            .secondary_runtime_texture = {}, // Filled when the texture data are uploaded to the GPU
            .name = load_ctx.asset.textures[i].name.c_str(),
        });
        scene._new_texture_manifest_entries += 1;
    }
}

static void update_material_manifest_from_gltf(Scene & scene, Scene::LoadManifestInfo const & info, LoadManifestFromFileContext & load_ctx)
{
    for (u32 material_index = 0; material_index < s_cast<u32>(load_ctx.asset.materials.size()); material_index++)
    {
        auto const & material = load_ctx.asset.materials.at(material_index);
        u32 const material_manifest_index = scene._material_manifest.size();
        bool const has_normal_texture = material.normalTexture.has_value();
        bool const has_diffuse_texture = material.pbrData.baseColorTexture.has_value();
        bool const has_roughness_metalness_texture = material.pbrData.metallicRoughnessTexture.has_value();
        std::optional<MaterialManifestEntry::TextureInfo> diffuse_texture_info = {};
        std::optional<MaterialManifestEntry::TextureInfo> opacity_texture_info = {};
        std::optional<MaterialManifestEntry::TextureInfo> normal_texture_info = {};
        std::optional<MaterialManifestEntry::TextureInfo> roughness_metalness_info = {};
        if (has_diffuse_texture)
        {
            u32 const gltf_texture_index = s_cast<u32>(material.pbrData.baseColorTexture.value().textureIndex);
            diffuse_texture_info = {
                .tex_manifest_index = gltf_texture_index + load_ctx.texture_manifest_offset,
                .sampler_index = {}, // TODO(msakmary) ADD SAMPLERS
            };
            opacity_texture_info = {
                .tex_manifest_index = gltf_texture_index + load_ctx.texture_manifest_offset,
                .sampler_index = {}, // TODO(msakmary) ADD SAMPLERS
            };
            TextureManifestEntry & tmenty = scene._material_texture_manifest.at(diffuse_texture_info->tex_manifest_index);
            if (tmenty.type != TextureMaterialType::DIFFUSE)
            {
                DBG_ASSERT_TRUE_M(tmenty.type == TextureMaterialType::NONE, "ERROR: Found a texture used by different materials as DIFFERENT types!");
                tmenty.type = TextureMaterialType::DIFFUSE;
            }
            tmenty.material_manifest_indices.push_back({
                .material_manifest_index = material_manifest_index,
            });
        }
        if (has_normal_texture)
        {
            u32 const gltf_texture_index = s_cast<u32>(material.normalTexture.value().textureIndex);
            normal_texture_info = {
                .tex_manifest_index = gltf_texture_index + load_ctx.texture_manifest_offset,
                .sampler_index = 0, // TODO(msakmary) ADD SAMPLERS
            };
            TextureManifestEntry & tmenty = scene._material_texture_manifest.at(normal_texture_info->tex_manifest_index);
            if (tmenty.type != TextureMaterialType::NORMAL)
            {
                DBG_ASSERT_TRUE_M(tmenty.type == TextureMaterialType::NONE, "ERROR: Found a texture used by different materials as DIFFERENT types!");
                tmenty.type = TextureMaterialType::NORMAL;
            }
            tmenty.material_manifest_indices.push_back({
                .material_manifest_index = material_manifest_index,
            });
        }
        if (has_roughness_metalness_texture)
        {
            u32 const gltf_texture_index = s_cast<u32>(material.pbrData.metallicRoughnessTexture.value().textureIndex);
            roughness_metalness_info = {
                .tex_manifest_index = gltf_texture_index + load_ctx.texture_manifest_offset,
                .sampler_index = 0, // TODO(msakmary) ADD SAMPLERS
            };
            TextureManifestEntry & tmenty = scene._material_texture_manifest.at(roughness_metalness_info->tex_manifest_index);
            if (tmenty.type != TextureMaterialType::ROUGHNESS_METALNESS)
            {
                DBG_ASSERT_TRUE_M(tmenty.type == TextureMaterialType::NONE, "ERROR: Found a texture used by different materials as DIFFERENT types!");
                tmenty.type = TextureMaterialType::ROUGHNESS_METALNESS;
            }
            tmenty.material_manifest_indices.push_back({
                .material_manifest_index = material_manifest_index,
            });
        }
        scene._material_manifest.push_back(MaterialManifestEntry{
            .diffuse_info = diffuse_texture_info,
            .opacity_mask_info = opacity_texture_info,
            .normal_info = normal_texture_info,
            .roughness_metalness_info = roughness_metalness_info,
            .gltf_asset_manifest_index = load_ctx.gltf_asset_manifest_index,
            .asset_local_index = material_index,
            .alpha_discard_enabled = material.alphaMode == fastgltf::AlphaMode::Mask, // || material.alphaMode == fastgltf::AlphaMode::Blend,
            .double_sided = material.doubleSided,
            .base_color = f32vec3(material.pbrData.baseColorFactor[0], material.pbrData.baseColorFactor[1], material.pbrData.baseColorFactor[2]),
            .name = material.name.c_str(),
        });
        scene._new_material_manifest_entries += 1;
        DBG_ASSERT_TRUE_M(scene._new_material_manifest_entries < MAX_MATERIALS, "EXCEEDED MAX_MATERIALS");
    }
}

static void update_meshgroup_and_mesh_manifest_from_gltf(Scene & scene, Scene::LoadManifestInfo const & info, LoadManifestFromFileContext & load_ctx)
{
    /// NOTE: fastgltf::Mesh is a MeshGroup
    // std::array<u32, MAX_MESHES_PER_MESHGROUP> mesh_manifest_indices;
    for (u32 mesh_group_index = 0; mesh_group_index < s_cast<u32>(load_ctx.asset.meshes.size()); mesh_group_index++)
    {
        auto const & gltf_mesh = load_ctx.asset.meshes.at(mesh_group_index);
        // Linearly allocate chunk from indices array:
        u32 const mesh_lod_group_manifest_indices_array_offset = static_cast<u32>(scene._mesh_lod_group_manifest_indices.size());
        scene._mesh_lod_group_manifest_indices.resize(scene._mesh_lod_group_manifest_indices.size() + gltf_mesh.primitives.size());

        u32 const mesh_group_manifest_index = s_cast<u32>(scene._mesh_group_manifest.size());
        /// NOTE: fastgltf::Primitive is Mesh
        for (u32 in_group_index = 0; in_group_index < s_cast<u32>(gltf_mesh.primitives.size()); in_group_index++)
        {
            u32 const mesh_manifest_entry = scene._mesh_lod_group_manifest.size();
            auto const & gltf_primitive = gltf_mesh.primitives.at(in_group_index);
            scene._mesh_lod_group_manifest_indices.at(mesh_lod_group_manifest_indices_array_offset + in_group_index) = mesh_manifest_entry;
            std::optional<u32> material_manifest_index =
                gltf_primitive.materialIndex.has_value() ? std::optional{s_cast<u32>(gltf_primitive.materialIndex.value()) + load_ctx.material_manifest_offset} : std::nullopt;
            scene._mesh_lod_group_manifest.push_back(MeshLodGroupManifestEntry{
                .gltf_asset_manifest_index = load_ctx.gltf_asset_manifest_index,
                // Gltf calls a meshgroup a mesh because these local indices are only used for loading we use the gltf naming
                .asset_local_mesh_index = mesh_group_index,
                // Same as above Gltf calls a mesh a primitive
                .asset_local_primitive_index = in_group_index,
                .mesh_group_manifest_index = mesh_group_manifest_index,
                .material_index = material_manifest_index,
            });
            scene._new_mesh_lod_group_manifest_entries += 1;
        }

        scene._mesh_group_manifest.push_back(MeshGroupManifestEntry{
            .mesh_lod_group_manifest_indices_array_offset = mesh_lod_group_manifest_indices_array_offset,
            .mesh_lod_group_count = s_cast<u32>(gltf_mesh.primitives.size()),
            .gltf_asset_manifest_index = load_ctx.gltf_asset_manifest_index,
            .asset_local_index = mesh_group_index,
            .name = gltf_mesh.name.c_str(),
        });
        scene._new_mesh_group_manifest_entries += 1;
    }
}

static auto update_entities_from_gltf(Scene & scene, Scene::LoadManifestInfo const & info, LoadManifestFromFileContext & load_ctx) -> RenderEntityId
{
    /// NOTE: fastgltf::Node is Entity
    DBG_ASSERT_TRUE_M(load_ctx.asset.nodes.size() != 0, "[ERROR][load_manifest_from_gltf()] Empty node array - what to do now?");
    std::vector<RenderEntityId> node_index_to_entity_id = {};
    /// NOTE: Here we allocate space for each entity and create a translation table between node index and entity id
    for (u32 node_index = 0; node_index < s_cast<u32>(load_ctx.asset.nodes.size()); node_index++)
    {
        node_index_to_entity_id.push_back(scene._render_entities.create_slot());
        scene._dirty_render_entities.push_back(node_index_to_entity_id.back());
    }
    for (u32 node_index = 0; node_index < s_cast<u32>(load_ctx.asset.nodes.size()); node_index++)
    {
        // TODO: For now store transform as a matrix - later should be changed to something else (TRS: translation, rotor, scale).
        auto fastgltf_to_glm_mat4x3_transform = [](std::variant<fastgltf::TRS, fastgltf::Node::TransformMatrix> const & trans) -> glm::mat4x3
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
            else if (auto const * trs = std::get_if<fastgltf::Node::TransformMatrix>(&trans))
            {
                // Gltf and glm matrices are column major.
                ret_trans = glm::mat4x3(std::bit_cast<glm::mat4x4>(*trs));
            }
            return ret_trans;
        };

        fastgltf::Node const & node = load_ctx.asset.nodes[node_index];
        RenderEntityId const parent_r_ent_id = node_index_to_entity_id[node_index];
        RenderEntity & r_ent = *scene._render_entities.slot(parent_r_ent_id);
        r_ent.mesh_group_manifest_index = node.meshIndex.has_value() ? std::optional<u32>(s_cast<u32>(node.meshIndex.value()) + load_ctx.mesh_group_manifest_offset) : std::optional<u32>(std::nullopt);
        r_ent.transform = fastgltf_to_glm_mat4x3_transform(node.transform);
        r_ent.name = node.name.c_str();
        if (node.meshIndex.has_value())
        {
            r_ent.type = EntityType::MESHGROUP;
        }
        else if (node.cameraIndex.has_value())
        {
            r_ent.type = EntityType::CAMERA;
        }
        else if (node.lightIndex.has_value())
        {
            r_ent.type = EntityType::LIGHT;
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
            u32 const curr_child_node_idx = node.children[curr_child_vec_idx];
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

    bool const grid_copy_scene = false;
    if (grid_copy_scene)
    {
        u32 const grid_size = 50;
        f32 const grid_cell_offset_x = 20.0f;
        f32 const grid_cell_offset_y = 5.0f;
        f32 const grid_cell_offset_z = 5.0f;
        for (u32 x = 0; x < grid_size; ++x)
        {
            for (u32 y = 0; y < grid_size; ++y)
            {
                for (u32 z = 0; z < grid_size; ++z)
                {    
                    std::vector<RenderEntityId> node_index_to_entity_id = {};
                    for (u32 node_index = 0; node_index < s_cast<u32>(load_ctx.asset.nodes.size()); node_index++)
                    {
                        node_index_to_entity_id.push_back(scene._render_entities.create_slot());
                        scene._dirty_render_entities.push_back(node_index_to_entity_id.back());
                    }
                    for (u32 node_index = 0; node_index < s_cast<u32>(load_ctx.asset.nodes.size()); node_index++)
                    {
                        // TODO: For now store transform as a matrix - later should be changed to something else (TRS: translation, rotor, scale).
                        auto fastgltf_to_glm_mat4x3_transform = [](std::variant<fastgltf::TRS, fastgltf::Node::TransformMatrix> const & trans) -> glm::mat4x3
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
                            else if (auto const * trs = std::get_if<fastgltf::Node::TransformMatrix>(&trans))
                            {
                                // Gltf and glm matrices are column major.
                                ret_trans = glm::mat4x3(std::bit_cast<glm::mat4x4>(*trs));
                            }
                            return ret_trans;
                        };

                        fastgltf::Node const & node = load_ctx.asset.nodes[node_index];
                        RenderEntityId const parent_r_ent_id = node_index_to_entity_id[node_index];
                        RenderEntity & r_ent = *scene._render_entities.slot(parent_r_ent_id);
                        r_ent.mesh_group_manifest_index = node.meshIndex.has_value() ? std::optional<u32>(s_cast<u32>(node.meshIndex.value()) + load_ctx.mesh_group_manifest_offset) : std::optional<u32>(std::nullopt);
                        r_ent.transform = fastgltf_to_glm_mat4x3_transform(node.transform);
                        r_ent.transform[3].x += x * grid_cell_offset_x;
                        r_ent.transform[3].y += y * grid_cell_offset_y;
                        r_ent.transform[3].z += z * grid_cell_offset_z;
                        r_ent.name = node.name.c_str();
                        if (node.meshIndex.has_value())
                        {
                            r_ent.type = EntityType::MESHGROUP;
                        }
                        else if (node.cameraIndex.has_value())
                        {
                            r_ent.type = EntityType::CAMERA;
                        }
                        else if (node.lightIndex.has_value())
                        {
                            r_ent.type = EntityType::LIGHT;
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
                            u32 const curr_child_node_idx = node.children[curr_child_vec_idx];
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
                }
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
        .name = info.asset_name.filename().replace_extension("").string() + "_" + std::to_string(load_ctx.gltf_asset_manifest_index),
    });

    scene._dirty_render_entities.push_back(root_r_ent_id);
    RenderEntity & root_r_ent = *scene._render_entities.slot(root_r_ent_id);
    root_r_ent.type = EntityType::ROOT;
    std::optional<RenderEntityId> root_r_ent_prev_child = {};
    for (u32 node_index = 0; node_index < s_cast<u32>(load_ctx.asset.nodes.size()); node_index++)
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

static void update_lights_from_gltf(Scene & scene, Scene::LoadManifestInfo const & info)
{
    // TODO(msakmary) Hook this into a scene, this sucks!
    scene._active_point_lights.push_back({
        .position = {-2.8f, -11.4f, 3.5f},
        .color = {1.0f, 0.55f, 0.15f}, 
        .intensity = 5.0f,
        .cutoff = 20.0f,
        .point_light_ptr = scene._device.buffer_device_address(scene._gpu_point_lights.get_state().buffers[0]).value(),
    });

    scene._active_point_lights.push_back({
        .position = {-15.1f, 1.4f, 4.0f},
        .color = {1.0f, 0.2f, 0.15f}, 
        .intensity = 4.0f,
        .cutoff = 10.0f,
        .point_light_ptr = scene._device.buffer_device_address(scene._gpu_point_lights.get_state().buffers[0]).value() + sizeof(GPUPointLight),
    });

    scene._active_point_lights.push_back({
        .position = {-5.51f, 16.5f, 3.0f},
        .color = {0.2f, 0.3f, 0.15f}, 
        .intensity = 3.5f,
        .cutoff = 13.0f,
        .point_light_ptr = scene._device.buffer_device_address(scene._gpu_point_lights.get_state().buffers[0]).value() + (2 * sizeof(GPUPointLight)),
    });

    auto * const gpu_point_lights_write_ptr = scene._device.buffer_host_address_as<GPUPointLight>(scene._gpu_point_lights.get_state().buffers[0]).value();

    for(i32 light_idx = 0; light_idx < MAX_POINT_LIGHTS; ++light_idx)
    {
        auto const & light = scene._active_point_lights.at(light_idx);
        gpu_point_lights_write_ptr[light_idx] = GPUPointLight{
            .position = std::bit_cast<daxa_f32vec3>(light.position),
            .color = std::bit_cast<daxa_f32vec3>(light.color),
            .intensity = light.intensity,
            .cutoff = light.cutoff,
        };
    }
}

static void start_async_loads_of_dirty_meshes(Scene & scene, Scene::LoadManifestInfo const & info)
{
    struct LoadMeshTask : Task
    {
        struct TaskInfo
        {
            AssetProcessor::LoadMeshLodGroupInfo load_info = {};
            AssetProcessor * asset_processor = {};
            u32 manifest_index = {};
        };

        TaskInfo info = {};
        LoadMeshTask(TaskInfo const & info)
            : info{info}
        {
            chunk_count = 1;
        }

        virtual void callback(u32 chunk_index, u32 thread_index) override
        {
            auto const ret_status = info.asset_processor->load_mesh(info.load_info);
            if (ret_status != AssetProcessor::AssetLoadResultCode::SUCCESS)
            {
                DEBUG_MSG(fmt::format("[ERROR]Failed to load mesh group {} mesh {} - error {}",
                    info.load_info.gltf_mesh_index, info.load_info.gltf_primitive_index, AssetProcessor::to_string(ret_status)));
            }
            else
            {
                // DEBUG_MSG(fmt::format("[SUCCESS] Successfuly loaded mesh group {} mesh {}",
                //     info.load_info.gltf_mesh_index, info.load_info.gltf_primitive_index));
            }
        };
    };

    for (u32 mesh_lod_group_manifest_index = 0; mesh_lod_group_manifest_index < scene._new_mesh_lod_group_manifest_entries; mesh_lod_group_manifest_index++)
    {
        auto const & curr_asset = scene._gltf_asset_manifest.back();
        auto const & mesh_manifest_lod_group_entry = scene._mesh_lod_group_manifest.at(curr_asset.mesh_manifest_offset + mesh_lod_group_manifest_index);
        // Launch loading of this mesh
        // TODO: ADD DUMMY MATERIAL INDEX!
        info.thread_pool->async_dispatch(
            std::make_shared<LoadMeshTask>(LoadMeshTask::TaskInfo{
                .load_info = {
                    .asset_path = curr_asset.path,
                    .asset = curr_asset.gltf_asset.get(),
                    .gltf_mesh_index = mesh_manifest_lod_group_entry.asset_local_mesh_index,
                    .gltf_primitive_index = mesh_manifest_lod_group_entry.asset_local_primitive_index,
                    .global_material_manifest_offset = curr_asset.material_manifest_offset,
                    .mesh_lod_manifest_index = mesh_lod_group_manifest_index,
                    .material_manifest_index = mesh_manifest_lod_group_entry.material_index.value_or(INVALID_MANIFEST_INDEX),
                },
                .asset_processor = info.asset_processor.get(),
            }),
            TaskPriority::LOW);
    }
}

static void start_async_loads_of_dirty_textures(Scene & scene, Scene::LoadManifestInfo const & info)
{
    struct LoadTextureTask : Task
    {
        struct TaskInfo
        {
            AssetProcessor::LoadTextureInfo load_info = {};
            AssetProcessor * asset_processor = {};
            u32 manifest_index = {};
        };

        TaskInfo info = {};
        LoadTextureTask(TaskInfo const & info)
            : info{info}
        {
            chunk_count = 1;
        }

        virtual void callback(u32 chunk_index, u32 thread_index) override
        {
            auto const ret_status = info.asset_processor->load_texture(info.load_info);
            auto const texture_name = info.load_info.asset->images.at(info.load_info.gltf_image_index).name;
            if (ret_status != AssetProcessor::AssetLoadResultCode::SUCCESS)
            {
                DEBUG_MSG(fmt::format("[ERROR] Failed to load texture index {} name {} - error {}",
                    info.load_info.gltf_texture_index, texture_name, AssetProcessor::to_string(ret_status)));
            }
            else
            {
                // DEBUG_MSG(fmt::format("[SUCCESS] Successfuly loaded texture index {} name {}",
                //     info.load_info.gltf_texture_index, texture_name));
            }
        };
    };
    auto gltf_texture_to_image_index = [&](u32 const texture_index) -> std::optional<u32>
    {
        std::unique_ptr<fastgltf::Asset> const & asset =
            scene._gltf_asset_manifest.at(scene._material_texture_manifest.at(texture_index).gltf_asset_manifest_index).gltf_asset;
        if (asset->textures.at(texture_index).basisuImageIndex.has_value())
        {
            return s_cast<u32>(asset->textures.at(texture_index).basisuImageIndex.value());
        }
        else if (asset->textures.at(texture_index).imageIndex.has_value())
        {
            return s_cast<u32>(asset->textures.at(texture_index).imageIndex.value());
        }
        else
        {
            return std::nullopt;
        }
    };

    for (u32 gltf_texture_index = 0; gltf_texture_index < scene._new_texture_manifest_entries; gltf_texture_index++)
    {
        auto const & curr_asset = scene._gltf_asset_manifest.back();
        auto const texture_manifest_index = curr_asset.texture_manifest_offset + gltf_texture_index;
        auto const & texture_manifest_entry = scene._material_texture_manifest.at(texture_manifest_index);
        auto gpu_compression_format = KTX_TTF_BC7_RGBA;
        auto gltf_image_idx_opt = gltf_texture_to_image_index(texture_manifest_index);
        DBG_ASSERT_TRUE_M(
            gltf_image_idx_opt.has_value(),
            fmt::format(
                "[ERROR] Texture \"{}\" has no supported gltf image index!\n",
                texture_manifest_entry.name));
        if (!texture_manifest_entry.material_manifest_indices.empty())
        {
            // Launch loading of this texture
            info.thread_pool->async_dispatch(
                std::make_shared<LoadTextureTask>(LoadTextureTask::TaskInfo{
                    .load_info = {
                        .asset_path = curr_asset.path,
                        .asset = curr_asset.gltf_asset.get(),
                        .gltf_texture_index = texture_manifest_entry.asset_local_index,
                        .gltf_image_index = texture_manifest_entry.asset_local_image_index,
                        .texture_manifest_index = texture_manifest_index,
                        .texture_material_type = texture_manifest_entry.type,
                    },
                    .asset_processor = info.asset_processor.get(),
                }),
                TaskPriority::LOW);
        }
        else
        {
            DEBUG_MSG(
                fmt::format("[WARNING] Texture \"{}\" can not be loaded because it is not referenced by any material", texture_manifest_entry.name));
        }
    }
    scene._new_texture_manifest_entries = 0;
}

static void update_mesh_and_mesh_lod_group_manifest(Scene & scene, Scene::RecordGPUManifestUpdateInfo const & info, daxa::CommandRecorder & recorder);
static void update_material_and_texture_manifest(Scene & scene, Scene::RecordGPUManifestUpdateInfo const & info, daxa::CommandRecorder & recorder);
auto Scene::record_gpu_manifest_update(RecordGPUManifestUpdateInfo const & info) -> daxa::ExecutableCommandList
{
    auto recorder = _device.create_command_recorder({});
    /// TODO: Make buffers resize.

    // Calculate required staging buffer size:
    daxa::BufferId staging_buffer = {};
    usize staging_offset = 0;
    std::byte * host_ptr = {};
    if (_dirty_render_entities.size() > 0 || _modified_render_entities.size() > 0)
    {
        usize required_staging_size = 0;
        required_staging_size += sizeof(GPUEntityMetaData);                                                                   // _gpu_entity_meta
        required_staging_size += sizeof(daxa_f32mat4x3) * (_dirty_render_entities.size() + _modified_render_entities.size()); // _gpu_entity_transforms
        required_staging_size += sizeof(daxa_f32mat4x3) * (_dirty_render_entities.size() + _modified_render_entities.size()); // _gpu_entity_combined_transforms
        required_staging_size += sizeof(GPUMeshGroup) * (_dirty_render_entities.size() + _modified_render_entities.size());   // _gpu_entity_mesh_groups
        staging_buffer = _device.create_buffer({
            .size = required_staging_size,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "entities update staging",
        });
        recorder.destroy_buffer_deferred(staging_buffer);
        host_ptr = _device.buffer_host_address(staging_buffer).value();
        *r_cast<GPUEntityMetaData *>(host_ptr) = {.entity_count = s_cast<u32>(_render_entities.size())};
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _gpu_entity_meta.get_state().buffers[0],
            .src_offset = staging_offset,
            .size = sizeof(GPUEntityMetaData),
        });
        staging_offset += sizeof(GPUEntityMetaData);
    }

    /**
     * TODO:
     * - replace with compute shader
     * - write two arrays, one containing entity ids other containing update data
     * - write compute shader that reads both arrays, they then write the updates from staging to entity arrays
     */
    /// NOTE: Update dirty entities.
    auto update_entity = [&](i32 i, RenderEntity * entity, u32 entity_index) -> glm::mat4
    {
        usize offset = (staging_offset + (sizeof(glm::mat4x3) * 2 + sizeof(u32)) * i);
        glm::mat4 transform4 = glm::mat4(
            glm::vec4(entity->transform[0], 0.0f),
            glm::vec4(entity->transform[1], 0.0f),
            glm::vec4(entity->transform[2], 0.0f),
            glm::vec4(entity->transform[3], 1.0f));
        glm::mat4 combined_transform4 = transform4;
        glm::mat4 combined_parent_transform4 = glm::identity<glm::mat4>();
        std::optional<RenderEntityId> parent = entity->parent;
        while (parent.has_value())
        {
            glm::mat4x3 parent_transform4 = glm::mat4(
                glm::vec4(_render_entities.slot(parent.value())->transform[0], 0.0f),
                glm::vec4(_render_entities.slot(parent.value())->transform[1], 0.0f),
                glm::vec4(_render_entities.slot(parent.value())->transform[2], 0.0f),
                glm::vec4(_render_entities.slot(parent.value())->transform[3], 1.0f));
            combined_transform4 = parent_transform4 * combined_transform4;
            combined_parent_transform4 = parent_transform4 * combined_parent_transform4;
            parent = _render_entities.slot(parent.value())->parent;
        }
        entity->combined_transform = combined_transform4;
        u32 mesh_group_manifest_index = entity->mesh_group_manifest_index.value_or(INVALID_MANIFEST_INDEX);
        struct RenderEntityUpdateStagingMemoryView
        {
            glm::mat4x3 transform;
            glm::mat4x3 combined_transform;
            u32 mesh_group_manifest_index;
        };
        *r_cast<RenderEntityUpdateStagingMemoryView *>(host_ptr + offset) = {
            .transform = transform4,
            .combined_transform = combined_transform4,
            .mesh_group_manifest_index = mesh_group_manifest_index,
        };
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _gpu_entity_transforms.get_state().buffers[0],
            .src_offset = offset + offsetof(RenderEntityUpdateStagingMemoryView, transform),
            .dst_offset = sizeof(glm::mat4x3) * entity_index,
            .size = sizeof(glm::mat4x3),
        });
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _gpu_entity_combined_transforms.get_state().buffers[0],
            .src_offset = offset + offsetof(RenderEntityUpdateStagingMemoryView, combined_transform),
            .dst_offset = sizeof(glm::mat4x3) * entity_index,
            .size = sizeof(glm::mat4x3),
        });
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _gpu_entity_mesh_groups.get_state().buffers[0],
            .src_offset = offset + offsetof(RenderEntityUpdateStagingMemoryView, mesh_group_manifest_index),
            .dst_offset = sizeof(u32) * entity_index,
            .size = sizeof(u32),
        });
        return combined_parent_transform4;
    };
    for (u32 i = 0; i < _dirty_render_entities.size(); ++i)
    {
        u32 entity_index = _dirty_render_entities[i].index;
        auto * entity = _render_entities.slot(_dirty_render_entities[i]);
        entity->dirty = true;
        update_entity(i, entity, entity_index);
    }

    _dirty_render_entities.clear();
    _modified_render_entities.clear();

    // Add new mesh group manifest entries
    if (_new_mesh_group_manifest_entries > 0)
    {
        if (!_gpu_mesh_group_indices_array_buffer.is_empty())
        {
            recorder.destroy_buffer_deferred(_gpu_mesh_group_indices_array_buffer);
        }
        usize mesh_group_indices_mem_size = sizeof(daxa_u32) * _mesh_lod_group_manifest_indices.size();
        _gpu_mesh_group_indices_array_buffer = _device.create_buffer({
            .size = mesh_group_indices_mem_size,
            .name = "_gpu_mesh_group_indices_array_buffer",
        });

        daxa::BufferId mesh_groups_indices_staging = _device.create_buffer({
            .size = mesh_group_indices_mem_size,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "mesh group update staging buffer",
        });
        recorder.destroy_buffer_deferred(mesh_groups_indices_staging);
        u32 * indices_staging_ptr = _device.buffer_host_address_as<u32>(mesh_groups_indices_staging).value();
        std::memcpy(indices_staging_ptr, _mesh_lod_group_manifest_indices.data(), _mesh_lod_group_manifest_indices.size() * sizeof(_mesh_lod_group_manifest_indices[0]));
        recorder.copy_buffer_to_buffer({
            .src_buffer = mesh_groups_indices_staging,
            .dst_buffer = _gpu_mesh_group_indices_array_buffer,
            .size = mesh_group_indices_mem_size,
        });

        auto mesh_group_indices_array_addr = _device.buffer_device_address(_gpu_mesh_group_indices_array_buffer).value();

        u32 const mesh_group_staging_buffer_size = sizeof(GPUMeshGroup) * _new_mesh_group_manifest_entries;
        daxa::BufferId mesh_group_staging_buffer = _device.create_buffer({
            .size = mesh_group_staging_buffer_size,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "mesh group update staging buffer",
        });
        recorder.destroy_buffer_deferred(mesh_group_staging_buffer);
        GPUMeshGroup * staging_ptr = _device.buffer_host_address_as<GPUMeshGroup>(mesh_group_staging_buffer).value();
        u32 const mesh_group_manifest_offset = _mesh_group_manifest.size() - _new_mesh_group_manifest_entries;
        for (u32 new_mesh_group_idx = 0; new_mesh_group_idx < _new_mesh_group_manifest_entries; new_mesh_group_idx++)
        {
            u32 const mesh_group_manifest_idx = mesh_group_manifest_offset + new_mesh_group_idx;
            staging_ptr[new_mesh_group_idx].mesh_lod_group_indices =
                mesh_group_indices_array_addr +
                sizeof(daxa_u32) * _mesh_group_manifest.at(mesh_group_manifest_idx).mesh_lod_group_manifest_indices_array_offset;
            staging_ptr[new_mesh_group_idx].mesh_lod_group_count = _mesh_group_manifest.at(mesh_group_manifest_idx).mesh_lod_group_count;
        }
        recorder.copy_buffer_to_buffer({
            .src_buffer = mesh_group_staging_buffer,
            .dst_buffer = _gpu_mesh_group_manifest.get_state().buffers[0],
            .src_offset = 0,
            .dst_offset = mesh_group_manifest_offset * sizeof(GPUMeshGroup),
            .size = sizeof(GPUMeshGroup) * _new_mesh_group_manifest_entries,
        });
    }

    // Add new mesh lod group manifest entries.
    // Zero out new lod group manifest entries entries.
    if (_new_mesh_lod_group_manifest_entries > 0)
    {
        u32 const mesh_lod_group_update_staging_buffer_size = sizeof(GPUMeshLodGroup) * _new_mesh_lod_group_manifest_entries;
        u32 const mesh_manifest_offset = _mesh_lod_group_manifest.size() - _new_mesh_lod_group_manifest_entries;
        daxa::BufferId mesh_lod_group_staging_buffer = _device.create_buffer({
            .size = mesh_lod_group_update_staging_buffer_size,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "mesh lod group manifest update staging buffer",
        });
        recorder.destroy_buffer_deferred(mesh_lod_group_staging_buffer);
        std::byte * staging_ptr = _device.buffer_host_address(mesh_lod_group_staging_buffer).value();
        std::memset(staging_ptr, 0u, _new_mesh_lod_group_manifest_entries * sizeof(GPUMeshLodGroup));

        recorder.copy_buffer_to_buffer({
            .src_buffer = mesh_lod_group_staging_buffer,
            .dst_buffer = _gpu_mesh_lod_group_manifest.get_state().buffers[0],
            .src_offset = 0,
            .dst_offset = sizeof(GPUMeshLodGroup) * mesh_manifest_offset,
            .size = sizeof(GPUMeshLodGroup) * _new_mesh_lod_group_manifest_entries,
        });
    }

    // Add new mesh manifest entries.
    // Zero out new manifest entries entries.
    if (_new_mesh_lod_group_manifest_entries > 0)
    {
        u32 const mesh_update_staging_buffer_size = sizeof(GPUMesh) * _new_mesh_lod_group_manifest_entries * MAX_MESHES_PER_LOD_GROUP;
        u32 const mesh_manifest_offset = _mesh_lod_group_manifest.size() - _new_mesh_lod_group_manifest_entries;
        daxa::BufferId mesh_staging_buffer = _device.create_buffer({
            .size = mesh_update_staging_buffer_size,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "mesh manifest update staging buffer",
        });
        recorder.destroy_buffer_deferred(mesh_staging_buffer);
        std::byte * staging_ptr = _device.buffer_host_address(mesh_staging_buffer).value();
        std::memset(staging_ptr, 0u, _new_mesh_lod_group_manifest_entries * sizeof(GPUMesh));

        recorder.copy_buffer_to_buffer({
            .src_buffer = mesh_staging_buffer,
            .dst_buffer = _gpu_mesh_manifest.get_state().buffers[0],
            .src_offset = 0,
            .dst_offset = sizeof(GPUMesh) * mesh_manifest_offset * MAX_MESHES_PER_LOD_GROUP,
            .size = sizeof(GPUMesh) * _new_mesh_lod_group_manifest_entries * MAX_MESHES_PER_LOD_GROUP,
        });
    }

    // Add new material manifest entries
    if (_new_material_manifest_entries > 0)
    {
        u32 const material_update_staging_buffer_size = sizeof(GPUMaterial) * _new_material_manifest_entries;
        u32 const material_manifest_offset = _material_manifest.size() - _new_material_manifest_entries;
        daxa::BufferId material_staging_buffer = _device.create_buffer({
            .size = material_update_staging_buffer_size,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "material update staging buffer",
        });
        recorder.destroy_buffer_deferred(material_staging_buffer);
        GPUMaterial * staging_ptr = _device.buffer_host_address_as<GPUMaterial>(material_staging_buffer).value();
        std::vector<GPUMaterial> tmp_materials(_new_material_manifest_entries);
        for (i32 i = 0; i < _new_material_manifest_entries; i++)
        {
            tmp_materials.at(i).base_color = std::bit_cast<daxa_f32vec3>(_material_manifest.at(i + material_manifest_offset).base_color);
        }
        std::memcpy(staging_ptr, tmp_materials.data(), _new_material_manifest_entries * sizeof(GPUMaterial));

        recorder.copy_buffer_to_buffer({
            .src_buffer = material_staging_buffer,
            .dst_buffer = _gpu_material_manifest.get_state().buffers[0],
            .src_offset = 0,
            .dst_offset = material_manifest_offset * sizeof(GPUMaterial),
            .size = sizeof(GPUMaterial) * _new_material_manifest_entries,
        });
    }

    /// TODO: Taskgraph this shit.
    recorder.pipeline_barrier({
        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_access = daxa::AccessConsts::READ_WRITE,
    });

    // updating material & texture manifest
    update_material_and_texture_manifest(*this, info, recorder);

    // updating mesh manifest
    update_mesh_and_mesh_lod_group_manifest(*this, info, recorder);

    /// TODO: Taskgraph this shit.
    recorder.pipeline_barrier({
        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_access = daxa::AccessConsts::READ_WRITE,
    });

    _new_material_manifest_entries = 0;
    _new_mesh_lod_group_manifest_entries = 0;
    _new_mesh_group_manifest_entries = 0;
    return recorder.complete_current_commands();
}

/// NOTE: As the mesh group manifest entries never change after loading them into the scene, we do not need to upload them here.
static void update_mesh_and_mesh_lod_group_manifest(Scene & scene, Scene::RecordGPUManifestUpdateInfo const & info, daxa::CommandRecorder & recorder)
{
    if (info.uploaded_meshes.size() > 0)
    {
        usize const meshes_staging_size = info.uploaded_meshes.size() * sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP;
        usize const mesh_lod_group_staging_size = info.uploaded_meshes.size() * sizeof(GPUMeshLodGroup);
        daxa::BufferId staging_buffer = scene._device.create_buffer({
            .size = meshes_staging_size + mesh_lod_group_staging_size,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "mesh manifest and mesh lod group manifest upload staging buffer",
        });

        recorder.destroy_buffer_deferred(staging_buffer);
        GPUMesh * mesh_staging_ptr = scene._device.buffer_host_address_as<GPUMesh>(staging_buffer).value();
        GPUMeshLodGroup * mesh_lod_group_staging_ptr = reinterpret_cast<GPUMeshLodGroup*>(scene._device.buffer_host_address(staging_buffer).value() + meshes_staging_size);
        for (i32 upload_index = 0; upload_index < info.uploaded_meshes.size(); upload_index++)
        {
            auto const & upload = info.uploaded_meshes[upload_index];
            scene._mesh_lod_group_manifest.at(upload.mesh_lod_manifest_index).runtime = MeshLodGroupManifestEntry::Runtime{
                .lods = upload.lods,
                .lod_count = upload.lod_count,
            };
            for (u32 i = 0; i < upload.lod_count; ++i)
            {
                u32 mesh_index = upload.mesh_lod_manifest_index * MAX_MESHES_PER_LOD_GROUP + i;
                scene._mesh_as_build_queue.push_back(mesh_index);
            }
            // Check if all meshes in a meshgroup are loaded
            auto & mesh_lod_group = scene._mesh_lod_group_manifest.at(upload.mesh_lod_manifest_index);
            // Incrementing loaded mesh count in mesh group
            {
                auto & mesh_group = scene._mesh_group_manifest.at(mesh_lod_group.mesh_group_manifest_index);
                mesh_group.loaded_mesh_lod_groups += 1;
                bool is_completely_loaded = true;
                u32 range[] = {mesh_group.mesh_lod_group_manifest_indices_array_offset, mesh_group.mesh_lod_group_manifest_indices_array_offset + mesh_group.mesh_lod_group_count};
                for (u32 mesh_idx_array_idx = range[0]; mesh_idx_array_idx < range[1]; mesh_idx_array_idx++)
                {
                    auto & checked_mesh = scene._mesh_lod_group_manifest.at(scene._mesh_lod_group_manifest_indices.at(mesh_idx_array_idx));
                    // Early out when we encounter a single unloaded mesh -> enough for us to know the meshgroup is not loaded
                    if (!checked_mesh.runtime.has_value())
                    {
                        is_completely_loaded = false;
                        break;
                    }
                }
                // the meshgroup is not fully loaded -> do not add it to
                if (is_completely_loaded)
                {
                    scene._newly_completed_mesh_groups.push_back(mesh_lod_group.mesh_group_manifest_index);
                }
            }
            std::memcpy(mesh_staging_ptr + upload_index * MAX_MESHES_PER_LOD_GROUP, &upload.lods, sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP);
            recorder.copy_buffer_to_buffer({
                .src_buffer = staging_buffer,
                .dst_buffer = scene._gpu_mesh_manifest.get_state().buffers[0],
                .src_offset = upload_index * sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP,
                .dst_offset = upload.mesh_lod_manifest_index * sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP,
                .size = sizeof(GPUMesh) * MAX_MESHES_PER_LOD_GROUP,
            });
            *(mesh_lod_group_staging_ptr + upload_index) = {
                .lod_count = upload.lod_count,
            };
            recorder.copy_buffer_to_buffer({
                .src_buffer = staging_buffer,
                .dst_buffer = scene._gpu_mesh_lod_group_manifest.get_state().buffers[0],
                .src_offset = upload_index * sizeof(GPUMeshLodGroup) + meshes_staging_size,
                .dst_offset = upload.mesh_lod_manifest_index * sizeof(GPUMeshLodGroup),
                .size = sizeof(GPUMeshLodGroup),
            });
        }
    }
}

static void update_material_and_texture_manifest(Scene & scene, Scene::RecordGPUManifestUpdateInfo const & info, daxa::CommandRecorder & recorder)
{
    if (info.uploaded_textures.size() > 0)
    {
        /// NOTE: We need to propagate each loaded texture image ID into the material manifest This will be done in two steps:
        //        1) We update the CPU manifest with the correct values and remember the materials that were updated
        //        2) For each dirty material we generate a copy buffer to buffer comand to update the GPU manifest
        for(auto const dirty_material_index : scene.dirty_material_entry_indices) 
        {
            scene._material_manifest.at(dirty_material_index).alpha_dirty = false;
        }
        scene.dirty_material_entry_indices.clear();

        // 1) Update CPU Manifest
        for (AssetProcessor::LoadedTextureInfo const & texture_upload : info.uploaded_textures)
        {
            if (texture_upload.secondary_texture)
            {
                scene._material_texture_manifest.at(texture_upload.texture_manifest_index).secondary_runtime_texture = texture_upload.dst_image;
            }
            else
            {
                scene._material_texture_manifest.at(texture_upload.texture_manifest_index).runtime_texture = texture_upload.dst_image;
            }
            TextureManifestEntry const & texture_manifest_entry = scene._material_texture_manifest.at(texture_upload.texture_manifest_index);
            for (auto const material_using_texture_info : texture_manifest_entry.material_manifest_indices)
            {
                MaterialManifestEntry & material_entry = scene._material_manifest.at(material_using_texture_info.material_manifest_index);
                switch (texture_manifest_entry.type)
                {
                    case TextureMaterialType::DIFFUSE:
                    {
                        if (texture_upload.secondary_texture)
                        {
                            material_entry.alpha_dirty = material_entry.alpha_discard_enabled;
                            material_entry.opacity_mask_info->tex_manifest_index = texture_upload.texture_manifest_index;
                        }
                        else
                        {
                            material_entry.diffuse_info->tex_manifest_index = texture_upload.texture_manifest_index;
                        }
                    }
                    break;
                    case TextureMaterialType::DIFFUSE_OPACITY:
                    {
                        material_entry.alpha_dirty = material_entry.alpha_discard_enabled;
                        material_entry.diffuse_info->tex_manifest_index = texture_upload.texture_manifest_index;
                    }
                    break;
                    case TextureMaterialType::NORMAL:
                    {
                        material_entry.normal_info->tex_manifest_index = texture_upload.texture_manifest_index;
                        material_entry.normal_compressed_bc5_rg = texture_upload.compressed_bc5_rg;
                    }
                    break;
                    case TextureMaterialType::ROUGHNESS_METALNESS:
                    {
                        material_entry.roughness_metalness_info->tex_manifest_index = texture_upload.texture_manifest_index;
                    }
                    break;
                    default: DBG_ASSERT_TRUE_M(false, "unimplemented"); break;
                }
                /// NOTE: Add material index only if it was not added previously
                if (std::find(
                        scene.dirty_material_entry_indices.begin(),
                        scene.dirty_material_entry_indices.end(),
                        material_using_texture_info.material_manifest_index) ==
                    scene.dirty_material_entry_indices.end())
                {
                    scene.dirty_material_entry_indices.push_back(material_using_texture_info.material_manifest_index);
                }
            }
        }
        // // 2) Update GPU manifest
        daxa::BufferId materials_update_staging_buffer = {};
        GPUMaterial * staging_origin_ptr = {};
        if (scene.dirty_material_entry_indices.size())
        {
            materials_update_staging_buffer = scene._device.create_buffer({
                .size = sizeof(GPUMaterial) * scene.dirty_material_entry_indices.size(),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "gpu materials update staging",
            });
            recorder.destroy_buffer_deferred(materials_update_staging_buffer);
            staging_origin_ptr = scene._device.buffer_host_address_as<GPUMaterial>(materials_update_staging_buffer).value();
        }
        for (u32 dirty_materials_index = 0; dirty_materials_index < scene.dirty_material_entry_indices.size(); dirty_materials_index++)
        {
            MaterialManifestEntry const & material = scene._material_manifest.at(scene.dirty_material_entry_indices.at(dirty_materials_index));
            daxa::ImageId diffuse_id = {};
            daxa::ImageId opacity_id = {};
            daxa::ImageId normal_id = {};
            daxa::ImageId roughness_metalness_id = {};
            /// NOTE: We check if material even has diffuse info, if it does we need to check if the runtime value of this
            //        info is present - It might be that diffuse texture was uploaded marking this material as dirty, but
            //        the normal texture is not yet present thus we don't yet have the runtime info
            if (material.diffuse_info.has_value())
            {
                auto const & texture_entry = scene._material_texture_manifest.at(material.diffuse_info.value().tex_manifest_index);
                diffuse_id = texture_entry.runtime_texture.value_or(daxa::ImageId{});
            }
            if (material.opacity_mask_info.has_value())
            {
                auto const & texture_entry = scene._material_texture_manifest.at(material.opacity_mask_info.value().tex_manifest_index);
                opacity_id = texture_entry.secondary_runtime_texture.value_or(daxa::ImageId{});
            }
            if (material.normal_info.has_value())
            {
                auto const & texture_entry = scene._material_texture_manifest.at(material.normal_info.value().tex_manifest_index);
                normal_id = texture_entry.runtime_texture.value_or(daxa::ImageId{});
            }
            if (material.roughness_metalness_info.has_value())
            {
                auto const & texture_entry = scene._material_texture_manifest.at(material.roughness_metalness_info.value().tex_manifest_index);
                roughness_metalness_id = texture_entry.runtime_texture.value_or(daxa::ImageId{});
            }
            staging_origin_ptr[dirty_materials_index].diffuse_texture_id = diffuse_id.default_view();
            staging_origin_ptr[dirty_materials_index].opacity_texture_id = opacity_id.default_view();
            staging_origin_ptr[dirty_materials_index].normal_texture_id = normal_id.default_view();
            staging_origin_ptr[dirty_materials_index].roughnes_metalness_id = roughness_metalness_id.default_view();
            staging_origin_ptr[dirty_materials_index].alpha_discard_enabled = material.alpha_discard_enabled;
            staging_origin_ptr[dirty_materials_index].normal_compressed_bc5_rg = material.normal_compressed_bc5_rg;
            staging_origin_ptr[dirty_materials_index].base_color = std::bit_cast<daxa_f32vec3>(material.base_color);
            staging_origin_ptr[dirty_materials_index].double_sided_enabled = static_cast<daxa_b32>(material.double_sided);

            daxa::BufferId gpu_material_manifest = scene._gpu_material_manifest.get_state().buffers[0];
            recorder.copy_buffer_to_buffer({
                .src_buffer = materials_update_staging_buffer,
                .dst_buffer = gpu_material_manifest,
                .src_offset = sizeof(GPUMaterial) * dirty_materials_index,
                .dst_offset = sizeof(GPUMaterial) * scene.dirty_material_entry_indices.at(dirty_materials_index),
                .size = sizeof(GPUMaterial),
            });
        }
        recorder.pipeline_barrier({
            .src_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_access = daxa::AccessConsts::READ,
        });
    }
}

auto Scene::create_mesh_acceleration_structures() -> daxa::ExecutableCommandList
{
    auto const scratch_buffer_offset_alignment =
        _device.properties().acceleration_structure_properties.value().min_acceleration_structure_scratch_offset_alignment;

    u32 current_scratch_buffer_offset = 0;
    auto const scratch_device_address = _device.buffer_device_address(_gpu_mesh_acceleration_structure_build_scratch_buffer.get_state().buffers[0]).value();
    std::vector<daxa::BlasTriangleGeometryInfo> build_geometries = {};
    // Reserve is nessecary to avoid memory resising.
    // We store pointers to the vector memory elsewhere, IT MUST NOT REALLOCATE!
    build_geometries.reserve(MAX_MESH_BLAS_BUILDS_PER_FRAME);
    std::vector<daxa::BlasBuildInfo> build_infos = {};
    while (!_mesh_as_build_queue.empty() && build_geometries.size() < MAX_MESH_BLAS_BUILDS_PER_FRAME)
    {
        auto const mesh_index = _mesh_as_build_queue.back();
        auto const lod = mesh_index % MAX_MESHES_PER_LOD_GROUP;
        auto const lod_group_index = mesh_index / MAX_MESHES_PER_LOD_GROUP;
        MeshLodGroupManifestEntry & mesh_lod_group = _mesh_lod_group_manifest.at(lod_group_index);

        bool is_alpha_discard = false;
        if (mesh_lod_group.material_index.has_value())
        {
            is_alpha_discard = _material_manifest.at(mesh_lod_group.material_index.value()).alpha_discard_enabled;
        }

        GPUMesh const & mesh = mesh_lod_group.runtime.value().lods[lod];

        // Must store geometries in vector as the memory address must persist for outside of the loop!
        build_geometries.push_back(daxa::BlasTriangleGeometryInfo{
            .vertex_data = mesh.vertex_positions,
            .max_vertex = mesh.vertex_count - 1,
            .index_data = mesh.primitive_indices,
            .count = static_cast<daxa_u32>(mesh.primitive_count),
            .flags = is_alpha_discard ? daxa::GeometryFlagBits::NONE : daxa::GeometryFlagBits::OPAQUE,
        });
        auto& geometry = build_geometries.back();
        daxa::BlasBuildInfo blas_build_info = daxa::BlasBuildInfo{
            .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE |
                     daxa::AccelerationStructureBuildFlagBits::ALLOW_DATA_ACCESS,
            .geometries = daxa::Span<daxa::BlasTriangleGeometryInfo const>(
                &geometry, 1ull),
        };

        auto const build_size_info = _device.blas_build_sizes(blas_build_info);
        u64 aligned_scratch_size = round_up_to_multiple(build_size_info.build_scratch_size, scratch_buffer_offset_alignment);
        DBG_ASSERT_TRUE_M(aligned_scratch_size < _gpu_mesh_acceleration_structure_build_scratch_buffer_size,
            "[ERROR][Scene::create_and_record_build_as()] Mesh group too big for the scratch buffer - increase scratch buffer size");
 
        bool const fits_scratch = (current_scratch_buffer_offset + aligned_scratch_size <= _gpu_mesh_acceleration_structure_build_scratch_buffer_size);
        if (!fits_scratch) { break; }

        blas_build_info.scratch_data = scratch_device_address + current_scratch_buffer_offset;
        current_scratch_buffer_offset += aligned_scratch_size;
        auto const aligned_accel_structure_size = round_up_to_multiple(build_size_info.acceleration_structure_size, 256);
        auto blas = _device.create_blas({
            .size = aligned_accel_structure_size,
            .name = mesh_lod_group.name,
        });
        blas_build_info.dst_blas = blas;
        mesh_lod_group.runtime->blas_lods[lod] = blas;

        build_infos.push_back(std::move(blas_build_info));
        _mesh_as_build_queue.pop_back();
    }

    auto recorder = _device.create_command_recorder({});
    if (!build_infos.empty())
    {
        DEBUG_MSG(fmt::format("[DEBUG][Scene::create_and_record_build_as()] Building {} blases this frame", build_infos.size()));
        recorder.build_acceleration_structures({.blas_build_infos = {build_infos.data(), build_infos.size()}});
    }
    recorder.pipeline_barrier({
        .src_access = daxa::AccessConsts::ACCELERATION_STRUCTURE_BUILD_WRITE,
        .dst_access = daxa::AccessConsts::ACCELERATION_STRUCTURE_BUILD_READ,
    });

    return recorder.complete_current_commands();
}

auto Scene::create_tlas_from_mesh_instances(CPUMeshInstances const& mesh_instances) -> daxa::ExecutableCommandList
{
    auto recorder = _device.create_command_recorder({});
#if 0

    std::vector<daxa_BlasInstanceData> blas_instances = {};
    for (u32 mesh_inst_i = 0; mesh_inst_i < mesh_instances.mesh_instances.size(); ++mesh_inst_i)
    {
        MeshInstance const& mesh_instance =  mesh_instances.mesh_instances[mesh_inst_i];
        auto const lod = mesh_instance.mesh_index % MAX_MESHES_PER_LOD_GROUP;
        auto const lod_group = mesh_instance.mesh_index / MAX_MESHES_PER_LOD_GROUP;

        if (!_mesh_lod_group_manifest[lod_group].runtime.has_value()) { continue; }
        if (_mesh_lod_group_manifest[lod_group].runtime.value().blas_lods[lod].is_empty()) { continue; }

        RenderEntity const* render_entity = _render_entities.slot_by_index(mesh_instance.entity_index);
        auto const& t = render_entity->combined_transform;
        blas_instances.push_back(daxa_BlasInstanceData{
            .transform = {
                {t[0][0], t[1][0], t[2][0], t[3][0]},
                {t[0][1], t[1][1], t[2][1], t[3][1]},
                {t[0][2], t[1][2], t[2][2], t[3][2]},
            },
            .instance_custom_index = mesh_inst_i,
            .mask = 0xFF,
            .instance_shader_binding_table_record_offset = ((mesh_instance.flags & MESH_INSTANCE_FLAG_MASKED) != 0) ? 1u : 0u,
            .flags = 0,
            .blas_device_address = _device.blas_device_address(_mesh_lod_group_manifest[lod_group].runtime.value().blas_lods[lod]).value(),
        });
    }

    daxa::BufferId blas_instances_buffer = {};
    if (!blas_instances.empty())
    {
        blas_instances_buffer = _device.create_buffer({.size = sizeof(daxa_BlasInstanceData) * blas_instances.size(),
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "blas instances buffer"});
        recorder.destroy_buffer_deferred(blas_instances_buffer);

        std::memcpy(_device.buffer_host_address_as<daxa_BlasInstanceData>(blas_instances_buffer).value(),
            blas_instances.data(),
            blas_instances.size() * sizeof(daxa_BlasInstanceData));
    }
    else
    {
        return recorder.complete_current_commands();
    }

    auto tlas_blas_instances_infos = std::array{daxa::TlasInstanceInfo{
        .data = blas_instances.empty() ? daxa::DeviceAddress{0ull} : _device.buffer_device_address(blas_instances_buffer).value(),
        .count = s_cast<u32>(blas_instances.size()),
        .is_data_array_of_pointers = false,
        .flags = daxa::GeometryFlagBits::NONE,
    }};

    auto tlas_build_info = daxa::TlasBuildInfo{
        .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE,
        .instances = tlas_blas_instances_infos,
    };

    daxa::AccelerationStructureBuildSizesInfo const tlas_build_sizes = _device.tlas_build_sizes(tlas_build_info);

    if (_scene_tlas.get_state().tlas.size() != 0)
    {
        _device.destroy_tlas(_scene_tlas.get_state().tlas[0]);
    }

    auto scene_tlas_id = _device.create_tlas({
        .size = tlas_build_sizes.acceleration_structure_size,
        .name = "scene tlas",
    });

    _scene_tlas.set_tlas({
        .tlas = std::array{scene_tlas_id},
        .latest_access = daxa::AccessConsts::NONE,
    });

    DBG_ASSERT_TRUE_M(tlas_build_sizes.build_scratch_size < _gpu_tlas_build_scratch_buffer_size,
        "[ERROR][Scene::create_tlas_from_mesh_instances] Tlas too big for scratch buffer - create bigger scratch buffer");

    daxa::DeviceAddress scratch_device_address = _device.buffer_device_address(_gpu_tlas_build_scratch_buffer.get_state().buffers[0]).value();
    tlas_build_info.dst_tlas = scene_tlas_id;
    tlas_build_info.scratch_data = scratch_device_address;

    recorder.build_acceleration_structures({.tlas_build_infos = std::array{tlas_build_info}});
    recorder.pipeline_barrier({
        .src_access = daxa::AccessConsts::ACCELERATION_STRUCTURE_BUILD_READ_WRITE,
        .dst_access = daxa::AccessConsts::READ_WRITE,
    });

#endif
    return recorder.complete_current_commands();
}

auto Scene::process_entities(RenderGlobalData & render_data) -> CPUMeshInstances
{
    // Go over all entities every frame.
    // Check if entity changed, if so, queue appropriate update events.
    // Populate GPUMeshInstances buffer from scratch, sort entities into draw lists.
    CPUMeshInstances ret = {};

    u32 active_entity_offset = 0;
    for (u32 entity_i = 0; entity_i < _render_entities.capacity(); ++entity_i)
    {
        RenderEntity * r_ent = _render_entities.slot_by_index(entity_i);
        bool const is_entity_dirty = r_ent->dirty;
        r_ent->dirty = false;

        if (r_ent != nullptr && r_ent->mesh_group_manifest_index.has_value())
        {
            usize mesh_group_index = r_ent->mesh_group_manifest_index.value();
            MeshGroupManifestEntry & mesh_group = _mesh_group_manifest.at(mesh_group_index);
            bool const is_mesh_group_loaded = (mesh_group.loaded_mesh_lod_groups == mesh_group.mesh_lod_group_count);
            bool const is_mesh_group_just_loaded = !mesh_group.fully_loaded_last_frame && is_mesh_group_loaded;
            mesh_group.fully_loaded_last_frame = is_mesh_group_loaded;

            entities_changed |= is_mesh_group_just_loaded;

            // Process all fully loaded mesh groups
            if (is_mesh_group_loaded)
            {
                if (mesh_group.blas.is_empty())
                {
                    blas_build_requests.push_back(_render_entities.id_from_index(entity_i));
                }

                auto const mesh_lod_group_indices_meshgroup_offset = mesh_group.mesh_lod_group_manifest_indices_array_offset;
                for (u32 in_mesh_group_index = 0; in_mesh_group_index < mesh_group.mesh_lod_group_count; in_mesh_group_index++)
                {
                    u32 const mesh_lod_group_manifest_index = _mesh_lod_group_manifest_indices.at(mesh_lod_group_indices_meshgroup_offset + in_mesh_group_index);
                    auto const & mesh_lod_group = _mesh_lod_group_manifest.at(mesh_lod_group_manifest_index);
                    bool is_alpha_discard = false;
                    bool is_alpha_dirty = false;
                    if (mesh_lod_group.material_index.has_value())
                    {
                        is_alpha_discard = _material_manifest.at(mesh_lod_group.material_index.value()).alpha_discard_enabled;
                        is_alpha_dirty = _material_manifest.at(mesh_lod_group.material_index.value()).alpha_dirty;
                    }

                    // Put this mesh into appropriate drawlist for prepass
                    u32 const draw_list_type = is_alpha_discard ? PREPASS_DRAW_LIST_MASKED : PREPASS_DRAW_LIST_OPAQUE;
                    
                    ret.prepass_draw_lists[draw_list_type].push_back(static_cast<u32>(ret.mesh_instances.size()));

                    // If the mesh loaded for the first time, it needs to invalidate VSM
                    // We also need to invalidate when the alpha texture just got streamed in
                    //   - this is because previously the shadows were drawn without alpha discard and so may be cached incorrectly
                    if (is_mesh_group_just_loaded || is_alpha_dirty || is_entity_dirty ) { ret.vsm_invalidate_draw_list.push_back(static_cast<u32>(ret.mesh_instances.size())); }

                    u32 mesh_index = select_lod(render_data, mesh_lod_group, mesh_lod_group_manifest_index, r_ent);

                    // Because this mesh will be referenced by the prepass drawlist, we need also need it's appropriate mesh instance data
                    ret.mesh_instances.push_back({
                        .entity_index = entity_i,
                        .mesh_index = mesh_index,
                        .in_mesh_group_index = in_mesh_group_index,
                        .mesh_group_index = static_cast<u32>(mesh_group_index),
                        .flags = s_cast<daxa_u32>(is_alpha_discard ? MESH_INSTANCE_FLAG_MASKED : MESH_INSTANCE_FLAG_OPAQUE),
                    });
                }
            }
        }
    }
    return ret;
}

void Scene::write_gpu_mesh_instances_buffer(CPUMeshInstances const& cpu_mesh_instances)
{
    // Calculate offsets into buffer and required size:
    usize offset = {};
    MeshInstancesBufferHead buffer_head = {};
    offset += sizeof(MeshInstancesBufferHead);
    buffer_head.instances = offset;
    buffer_head.count = static_cast<u32>(cpu_mesh_instances.mesh_instances.size());
    offset += sizeof(MeshInstance) * cpu_mesh_instances.mesh_instances.size();
    for (int i = 0; i < PREPASS_DRAW_LIST_TYPE_COUNT; ++i)
    {
        buffer_head.prepass_draw_lists[i].instances = offset;
        buffer_head.prepass_draw_lists[i].count = static_cast<u32>(cpu_mesh_instances.prepass_draw_lists[i].size());
        offset += sizeof(daxa_u32) * cpu_mesh_instances.prepass_draw_lists[i].size();
    }
    buffer_head.vsm_invalidate_draw_list.instances = offset;
    buffer_head.vsm_invalidate_draw_list.count = static_cast<u32>(cpu_mesh_instances.vsm_invalidate_draw_list.size());
    offset += sizeof(daxa_u32) * cpu_mesh_instances.vsm_invalidate_draw_list.size();
    usize const required_size = offset;

    // TODO: Allocate this into a ring buffer.
    // Allocate buffer
    if (!mesh_instances_buffer.get_state().buffers.empty())
    {
        _device.destroy_buffer(mesh_instances_buffer.get_state().buffers[0]);
    }
    mesh_instances_buffer.set_buffers({
        .buffers = std::array{_device.create_buffer({
            .size = required_size,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
            .name = "cpu_mesh_instances_buffer",
        })},
    });
    daxa::DeviceAddress device_address = _device.buffer_device_address(mesh_instances_buffer.get_state().buffers[0]).value();
    std::byte * host_address = _device.buffer_host_address(mesh_instances_buffer.get_state().buffers[0]).value();

    // Write Buffer, add address on offsets and counts:
    cpu_mesh_instance_counts = {};
    usize const mesh_instances_size = sizeof(MeshInstance) * cpu_mesh_instances.mesh_instances.size();
    std::memcpy(host_address + buffer_head.instances, cpu_mesh_instances.mesh_instances.data(), mesh_instances_size);

    buffer_head.instances += device_address;
    cpu_mesh_instance_counts.mesh_instance_count = cpu_mesh_instances.mesh_instances.size();
    for (int draw_list_type = 0; draw_list_type < PREPASS_DRAW_LIST_TYPE_COUNT; ++draw_list_type)
    {
        std::memcpy(
            host_address + buffer_head.prepass_draw_lists[draw_list_type].instances,
            cpu_mesh_instances.prepass_draw_lists[draw_list_type].data(),
            sizeof(u32) * cpu_mesh_instances.prepass_draw_lists[draw_list_type].size());
        buffer_head.prepass_draw_lists[draw_list_type].instances += device_address;
        cpu_mesh_instance_counts.prepass_instance_counts[draw_list_type] = cpu_mesh_instances.prepass_draw_lists[draw_list_type].size();
    }
    std::memcpy(
        host_address + buffer_head.vsm_invalidate_draw_list.instances,
        cpu_mesh_instances.vsm_invalidate_draw_list.data(),
        sizeof(u32) * cpu_mesh_instances.vsm_invalidate_draw_list.size());
    buffer_head.vsm_invalidate_draw_list.instances += device_address;

    std::memcpy(host_address, &buffer_head, sizeof(MeshInstancesBufferHead));

    cpu_mesh_instance_counts.vsm_invalidate_instance_count = cpu_mesh_instances.vsm_invalidate_draw_list.size();
}