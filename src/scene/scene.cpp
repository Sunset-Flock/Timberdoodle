#include "scene.hpp"

#include <fastgltf/parser.hpp>
#include <fstream>

#include <fmt/format.h>
#include <glm/gtx/quaternion.hpp>
#include <thread>
#include <chrono>

Scene::Scene(daxa::Device device)
    : _device{std::move(device)}
{
    /// TODO: THIS IS TEMPORARY! Make manifest and entity buffers growable!
    _gpu_entity_meta.set_buffers(daxa::TrackedBuffers{
        .buffers = std::array{
            _device.create_buffer({
                .size = sizeof(GPUEntityMetaData),
                .name = "_gpu_entity_meta",
            }),
        },
    });
    _gpu_entity_transforms.set_buffers(daxa::TrackedBuffers{
        .buffers = std::array{
            _device.create_buffer({
                .size = sizeof(daxa_f32mat4x3) * MAX_ENTITY_COUNT,
                .name = "_gpu_entity_transforms",
            }),
        },
    });
    _gpu_entity_combined_transforms.set_buffers(daxa::TrackedBuffers{
        .buffers = std::array{
            _device.create_buffer({
                .size = sizeof(daxa_f32mat4x3) * MAX_ENTITY_COUNT,
                .name = "_gpu_entity_combined_transforms",
            }),
        },
    });
    _gpu_entity_mesh_groups.set_buffers(daxa::TrackedBuffers{
        .buffers = std::array{
            _device.create_buffer({
                .size = sizeof(u32) * MAX_ENTITY_COUNT,
                .name = "_gpu_entity_mesh_groups",
            }),
        },
    });
    _gpu_mesh_manifest.set_buffers(daxa::TrackedBuffers{
        .buffers = std::array{
            _device.create_buffer({
                .size = sizeof(GPUMesh) * MAX_ENTITY_COUNT,
                .name = "_gpu_mesh_manifest",
            }),
        },
    });
    _gpu_mesh_group_manifest.set_buffers(daxa::TrackedBuffers{
        .buffers = std::array{
            _device.create_buffer({
                .size = sizeof(GPUMeshGroup) * MAX_ENTITY_COUNT,
                .name = "_gpu_mesh_group_manifest",
            }),
        },
    });
    _gpu_material_manifest.set_buffers(daxa::TrackedBuffers{
        .buffers = std::array{
            _device.create_buffer({.size = sizeof(GPUMaterial) * MAX_MATERIAL_COUNT,
                .name = "_gpu_material_manifest"}),
        },
    });
}

Scene::~Scene()
{
    _device.destroy_buffer(_gpu_entity_meta.get_state().buffers[0]);
    _device.destroy_buffer(_gpu_entity_transforms.get_state().buffers[0]);
    _device.destroy_buffer(_gpu_entity_combined_transforms.get_state().buffers[0]);
    _device.destroy_buffer(_gpu_entity_mesh_groups.get_state().buffers[0]);
    _device.destroy_buffer(_gpu_mesh_manifest.get_state().buffers[0]);
    _device.destroy_buffer(_gpu_mesh_group_manifest.get_state().buffers[0]);
    _device.destroy_buffer(_gpu_material_manifest.get_state().buffers[0]);

    for (auto & mesh : _mesh_manifest)
    {
        if (mesh.runtime.has_value())
        {
            _device.destroy_buffer(std::bit_cast<daxa::BufferId>(mesh.runtime.value().mesh_buffer));
        }
    }
    for(auto & texture : _material_texture_manifest)
    {
        if(texture.runtime.has_value())
        {
            _device.destroy_image(std::bit_cast<daxa::ImageId>(texture.runtime.value()));
        }
    }
}

// TODO: Loading god function.
auto Scene::load_manifest_from_gltf(LoadManifestInfo const & info) -> std::variant<RenderEntityId, LoadManifestErrorCode>
{
#pragma region SETUP
    auto file_path = info.root_path / info.asset_name;

    fastgltf::Parser parser{};

    constexpr auto gltf_options =
        fastgltf::Options::DontRequireValidAssetMember |
        fastgltf::Options::AllowDouble;
    // fastgltf::Options::LoadGLBBuffers |
    // fastgltf::Options::LoadExternalBuffers |
    // fastgltf::Options::LoadExternalImages

    fastgltf::GltfDataBuffer data;
    bool const worked = data.loadFromFile(file_path);
    if (!worked)
    {
        return LoadManifestErrorCode::FILE_NOT_FOUND;
    }
    auto type = fastgltf::determineGltfFileType(&data);
    fastgltf::Asset asset;
    switch (type)
    {
        case fastgltf::GltfType::glTF:
        {
            fastgltf::Expected<fastgltf::Asset> result = parser.loadGLTF(&data, file_path.parent_path(), gltf_options);
            if (result.error() != fastgltf::Error::None)
            {
                return LoadManifestErrorCode::COULD_NOT_LOAD_ASSET;
            }
            asset = std::move(result.get());
            break;
        }
        case fastgltf::GltfType::GLB:
        {
            fastgltf::Expected<fastgltf::Asset> result = parser.loadBinaryGLTF(&data, file_path.parent_path(), gltf_options);
            if (result.error() != fastgltf::Error::None)
            {
                return LoadManifestErrorCode::COULD_NOT_LOAD_ASSET;
            }
            asset = std::move(result.get());
            break;
        }
        default:
            return LoadManifestErrorCode::INVALID_GLTF_FILE_TYPE;
    }

    u32 const gltf_asset_manifest_index = s_cast<u32>(_gltf_asset_manifest.size());
    u32 const texture_manifest_offset = s_cast<u32>(_material_texture_manifest.size());
    u32 const material_manifest_offset = s_cast<u32>(_material_manifest.size());
    u32 const mesh_group_manifest_offset = s_cast<u32>(_mesh_group_manifest.size());
    u32 const mesh_manifest_offset = s_cast<u32>(_mesh_manifest.size());
#pragma endregion

#pragma region POPULATE_TEXTURE_MANIFEST
    /// NOTE: GLTF texture = image + sampler we collapse the sampler into the material itself here we thus only iterate over the images
    //        Later when we load in the materials which reference the textures rather than images we just
    //        translate the textures image index and store that in the material
    for (u32 i = 0; i < s_cast<u32>(asset.images.size()); ++i)
    {
        u32 const texture_manifest_index = s_cast<u32>(_material_texture_manifest.size());
        _material_texture_manifest.push_back(TextureManifestEntry{
            .gltf_asset_manifest_index = gltf_asset_manifest_index,
            .asset_local_index = i,
            .material_manifest_indices = {}, // Filled when reading in materials
            .runtime = {},                   // Filled when the texture data are uploaded to the GPU
            .name = asset.images[i].name.c_str(),
        });
        _new_texture_manifest_entries += 1;
        DEBUG_MSG(fmt::format("[INFO] Loading texture meta data into manifest:\n  name: {}\n  asset local index: {}\n  manifest index:  {}",
            asset.images[i].name, i, texture_manifest_index));
    }
#pragma endregion

#pragma region POPULATE_MATERIAL_MANIFEST
    /// NOTE: Because we previously only loaded the images and not textures we now need to translate
    //        the texture indices into image indeces and store that
    auto gltf_texture_to_manifest_texture_index = [&](u32 const texture_index) -> std::optional<u32>
    {
        const bool gltf_texture_has_image_index = asset.textures.at(texture_index).imageIndex.has_value();
        if (!gltf_texture_has_image_index)
        {
            return std::nullopt;
        }
        else
        {
            return s_cast<u32>(asset.textures.at(texture_index).imageIndex.value()) + texture_manifest_offset;
        }
    };
    for (u32 material_index = 0; material_index < s_cast<u32>(asset.materials.size()); material_index++)
    {
        auto const & material = asset.materials.at(material_index);
        u32 const material_manifest_index = _material_manifest.size();
        bool const has_normal_texture = material.normalTexture.has_value();
        bool const has_diffuse_texture = material.pbrData.baseColorTexture.has_value();
        bool const has_roughness_metalness_texture = material.pbrData.metallicRoughnessTexture.has_value();
        std::optional<MaterialManifestEntry::TextureInfo> diffuse_texture_info = {};
        std::optional<MaterialManifestEntry::TextureInfo> normal_texture_info = {};
        std::optional<MaterialManifestEntry::TextureInfo> roughnes_metalness_info = {};
        if (has_diffuse_texture)
        {
            u32 const texture_index = s_cast<u32>(material.pbrData.baseColorTexture.value().textureIndex);
            auto const has_index = gltf_texture_to_manifest_texture_index(texture_index).has_value();
            if (has_index)
            {
                diffuse_texture_info = {
                    .tex_manifest_index = gltf_texture_to_manifest_texture_index(texture_index).value(),
                    .sampler_index = 0, // TODO(msakmary) ADD SAMPLERS
                };
                _material_texture_manifest.at(diffuse_texture_info->tex_manifest_index).material_manifest_indices.push_back({
                    .diffuse = true,
                    .material_manifest_index = material_manifest_index,
                });
            }
        }
        if (has_normal_texture)
        {
            u32 const texture_index = s_cast<u32>(material.normalTexture.value().textureIndex);
            bool const has_index = gltf_texture_to_manifest_texture_index(texture_index).has_value();
            if (has_index)
            {
                normal_texture_info = {
                    .tex_manifest_index = gltf_texture_to_manifest_texture_index(texture_index).value(),
                    .sampler_index = 0, // TODO(msakmary) ADD SAMPLERS
                };
                _material_texture_manifest.at(normal_texture_info->tex_manifest_index).material_manifest_indices.push_back({
                    .normal = true,
                    .material_manifest_index = material_manifest_index,
                });
            }
        }
        if (has_roughness_metalness_texture)
        {
            u32 const texture_index = s_cast<u32>(material.pbrData.metallicRoughnessTexture.value().textureIndex);
            bool const has_index = gltf_texture_to_manifest_texture_index(texture_index).has_value();
            if (has_index)
            {
                roughnes_metalness_info = {
                    .tex_manifest_index = gltf_texture_to_manifest_texture_index(texture_index).value(),
                    .sampler_index = 0, // TODO(msakmary) ADD SAMPLERS
                };
                _material_texture_manifest.at(roughnes_metalness_info->tex_manifest_index).material_manifest_indices.push_back({
                    .roughness_metalness = true,
                    .material_manifest_index = material_manifest_index,
                });
            }
        }
        _material_manifest.push_back(MaterialManifestEntry{
            .diffuse_info = diffuse_texture_info,
            .normal_info = normal_texture_info,
            .roughness_metalness_info = roughnes_metalness_info,
            .gltf_asset_manifest_index = gltf_asset_manifest_index,
            .asset_local_index = material_index,
            .alpha_discard_enabled = material.alphaMode == fastgltf::AlphaMode::Mask || material.alphaMode == fastgltf::AlphaMode::Blend,
            .name = material.name.c_str(),
        });
        _new_material_manifest_entries += 1;
    }
#pragma endregion

#pragma region POPULATE_MESHGROUP_AND_MESH_MANIFEST
    /// NOTE: fastgltf::Mesh is a MeshGroup
    std::array<u32, MAX_MESHES_PER_MESHGROUP> mesh_manifest_indices;
    for (u32 mesh_group_index = 0; mesh_group_index < s_cast<u32>(asset.meshes.size()); mesh_group_index++)
    {
        auto const & mesh_group = asset.meshes.at(mesh_group_index);
        u32 const mesh_group_manifest_index = s_cast<u32>(_mesh_group_manifest.size());
        /// NOTE: fastgltf::Primitive is Mesh
        for (u32 mesh_index = 0; mesh_index < s_cast<u32>(mesh_group.primitives.size()); mesh_index++)
        {
            u32 const mesh_manifest_entry = _mesh_manifest.size();
            auto const & mesh = mesh_group.primitives.at(mesh_index);
            mesh_manifest_indices.at(mesh_index) = mesh_manifest_entry;
            std::optional<u32> material_manifest_index = mesh.materialIndex.has_value() ? std::optional{s_cast<u32>(mesh.materialIndex.value()) + material_manifest_offset} : std::nullopt;
            _mesh_manifest.push_back(MeshManifestEntry{
                .gltf_asset_manifest_index = gltf_asset_manifest_index,
                // Gltf calls a meshgroup a mesh because these local indices are only used for loading we use the gltf naming
                .asset_local_mesh_index = mesh_group_index,
                // Same as above Gltf calls a mesh a primitive
                .asset_local_primitive_index = mesh_index,
            });
            _new_mesh_manifest_entries += 1;
        }

        _mesh_group_manifest.push_back(MeshGroupManifestEntry{
            .mesh_manifest_indices = std::move(mesh_manifest_indices),
            .mesh_count = s_cast<u32>(mesh_group.primitives.size()),
            .gltf_asset_manifest_index = gltf_asset_manifest_index,
            .asset_local_index = mesh_group_index,
            .name = mesh_group.name.c_str(),
        });
        _new_mesh_group_manifest_entries += 1;
        mesh_manifest_indices.fill(0u);
    }
#pragma endregion

#pragma region POPULATE_RENDER_ENTITIES
    /// NOTE: fastgltf::Node is Entity
    DBG_ASSERT_TRUE_M(asset.nodes.size() != 0, "[ERROR][load_manifest_from_gltf()] Empty node array - what to do now?");
    std::vector<RenderEntityId> node_index_to_entity_id = {};
    /// NOTE: Here we allocate space for each entity and create a translation table between node index and entity id
    for (u32 node_index = 0; node_index < s_cast<u32>(asset.nodes.size()); node_index++)
    {
        node_index_to_entity_id.push_back(_render_entities.create_slot());
        _dirty_render_entities.push_back(node_index_to_entity_id.back());
    }
    for (u32 node_index = 0; node_index < s_cast<u32>(asset.nodes.size()); node_index++)
    {
        // TODO: For now store transform as a matrix - later should be changed to something else (TRS: translation, rotor, scale).
        auto fastgltf_to_glm_mat4x3_transform = [](std::variant<fastgltf::Node::TRS, fastgltf::Node::TransformMatrix> const & trans) -> glm::mat4x3
        {
            glm::mat4x3 ret_trans;
            if (auto const * trs = std::get_if<fastgltf::Node::TRS>(&trans))
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

        fastgltf::Node const & node = asset.nodes[node_index];
        RenderEntityId const parent_r_ent_id = node_index_to_entity_id[node_index];
        RenderEntity & r_ent = *_render_entities.slot(parent_r_ent_id);
        r_ent.mesh_group_manifest_index = node.meshIndex.has_value() ? std::optional<u32>(s_cast<u32>(node.meshIndex.value()) + mesh_group_manifest_offset) : std::optional<u32>(std::nullopt);
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
            RenderEntity & curr_child_r_ent = *_render_entities.slot(curr_child_r_ent_id);
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
    RenderEntityId root_r_ent_id = _render_entities.create_slot({
        .transform = glm::mat4x3(glm::identity<glm::mat4x3>()),
        .first_child = std::nullopt,
        .next_sibling = std::nullopt,
        .parent = std::nullopt,
        .mesh_group_manifest_index = std::nullopt,
        .name = info.asset_name.filename().replace_extension("").string() + "_" + std::to_string(gltf_asset_manifest_index),
    });

    _dirty_render_entities.push_back(root_r_ent_id);
    RenderEntity & root_r_ent = *_render_entities.slot(root_r_ent_id);
    root_r_ent.type = EntityType::ROOT;
    std::optional<RenderEntityId> root_r_ent_prev_child = {};
    for (u32 node_index = 0; node_index < s_cast<u32>(asset.nodes.size()); node_index++)
    {
        RenderEntityId const r_ent_id = node_index_to_entity_id[node_index];
        RenderEntity & r_ent = *_render_entities.slot(r_ent_id);
        if (!r_ent.parent.has_value())
        {
            r_ent.parent = root_r_ent_id;
            if (!root_r_ent_prev_child.has_value()) // First child
            {
                root_r_ent.first_child = r_ent_id;
            }
            else // We have other root children already
            {
                _render_entities.slot(root_r_ent_prev_child.value())->next_sibling = r_ent_id;
            }
            root_r_ent_prev_child = r_ent_id;
        }
    }

#pragma endregion

    _gltf_asset_manifest.push_back(GltfAssetManifestEntry{
        .path = file_path,
        .gltf_asset = std::make_unique<fastgltf::Asset>(std::move(asset)),
        .texture_manifest_offset = texture_manifest_offset,
        .material_manifest_offset = material_manifest_offset,
        .mesh_group_manifest_offset = mesh_group_manifest_offset,
        .mesh_manifest_offset = mesh_manifest_offset,
        .root_render_entity = root_r_ent_id,
    });

#pragma region LOAD_MESHES_ASYNC
    struct LoadMeshTask : Task
    {
        struct TaskInfo
        {
            AssetProcessor::LoadMeshInfo load_info = {};
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
            // std::this_thread::sleep_for(std::chrono::milliseconds(100));
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

    for (u32 mesh_manifest_index = 0; mesh_manifest_index < _new_mesh_manifest_entries; mesh_manifest_index++)
    {
        auto const & curr_asset = _gltf_asset_manifest.back();
        auto const & mesh_manifest_entry = _mesh_manifest.at(curr_asset.mesh_manifest_offset + mesh_manifest_index);
        // Launch loading of this mesh
        info.thread_pool->async_dispatch(
            std::make_shared<LoadMeshTask>(LoadMeshTask::TaskInfo{
                .load_info = {
                    .asset_path = curr_asset.path,
                    .asset = curr_asset.gltf_asset.get(),
                    .gltf_mesh_index = mesh_manifest_entry.asset_local_mesh_index,
                    .gltf_primitive_index = mesh_manifest_entry.asset_local_primitive_index,
                    .global_material_manifest_offset = curr_asset.material_manifest_offset,
                    .manifest_index = mesh_manifest_index,
                },
                .asset_processor = info.asset_processor.get(),
            }),
            TaskPriority::LOW);
    }
#pragma endregion
#pragma region LOAD_TEXTURES_ASYNC
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
            auto const texture_name = info.load_info.asset->images.at(info.load_info.gltf_texture_index).name;
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

    for (u32 gltf_texture_index = 0; gltf_texture_index < _new_texture_manifest_entries; gltf_texture_index++)
    {
        auto const & curr_asset = _gltf_asset_manifest.back();
        auto const texture_manifest_index = curr_asset.texture_manifest_offset + gltf_texture_index;
        auto const & texture_manifest_entry = _material_texture_manifest.at(texture_manifest_index);
        bool used_as_diffuse = false;
        for (auto const & material_manifest_index : texture_manifest_entry.material_manifest_indices)
        {
            used_as_diffuse |= material_manifest_index.diffuse;
            DBG_ASSERT_TRUE_M(!(used_as_diffuse && material_manifest_index.normal),
                "[ERROR] Texture used as diffuse and normal is not supported");
            DBG_ASSERT_TRUE_M(!(used_as_diffuse && material_manifest_index.roughness_metalness),
                "[ERROR] Texture used as diffuse and roughness metalness is not supported");
        }
        if (!texture_manifest_entry.material_manifest_indices.empty())
        {
            // Launch loading of this mesh
            info.thread_pool->async_dispatch(
                std::make_shared<LoadTextureTask>(LoadTextureTask::TaskInfo{
                    .load_info = {
                        .asset_path = curr_asset.path,
                        .asset = curr_asset.gltf_asset.get(),
                        .gltf_texture_index = gltf_texture_index,
                        .texture_manifest_index = texture_manifest_index,
                        .load_as_srgb = used_as_diffuse,
                    },
                    .asset_processor = info.asset_processor.get(),
                }),
                TaskPriority::LOW);
        }
        else
        {
            auto const texture_name = curr_asset.gltf_asset->images.at(gltf_texture_index).name;
            // DEBUG_MSG(fmt::format("[INFO] Skipping load of texture index {} name {} - not used by any material",
            //     gltf_texture_index, texture_name));
        }
    }
    _new_texture_manifest_entries = 0;
#pragma endregion

    return root_r_ent_id;
}

auto Scene::record_gpu_manifest_update(RecordGPUManifestUpdateInfo const & info) -> daxa::ExecutableCommandList
{
    auto recorder = _device.create_command_recorder({});
    /// TODO: Make buffers resize.

    // Calculate required staging buffer size:
    daxa::BufferId staging_buffer = {};
    usize staging_offset = 0;
    std::byte * host_ptr = {};
    if (_dirty_render_entities.size() > 0)
    {
        usize required_staging_size = 0;
        required_staging_size += sizeof(GPUEntityMetaData);                              // _gpu_entity_meta
        required_staging_size += sizeof(daxa_f32mat4x3) * _dirty_render_entities.size(); // _gpu_entity_transforms
        required_staging_size += sizeof(daxa_f32mat4x3) * _dirty_render_entities.size(); // _gpu_entity_combined_transforms
        required_staging_size += sizeof(GPUMeshGroup) * _dirty_render_entities.size();   // _gpu_entity_mesh_groups
        staging_buffer = _device.create_buffer({
            .size = required_staging_size,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "entities update staging",
        });
        recorder.destroy_buffer_deferred(staging_buffer);
        host_ptr = _device.get_host_address(staging_buffer).value();
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
    for (u32 i = 0; i < _dirty_render_entities.size(); ++i)
    {
        usize offset = (staging_offset + (sizeof(glm::mat4x3) * 2 + sizeof(u32)) * i);
        u32 entity_index = _dirty_render_entities[i].index;
        glm::mat4 transform4 = glm::mat4(
            glm::vec4(_render_entities.slot(_dirty_render_entities[i])->transform[0], 0.0f),
            glm::vec4(_render_entities.slot(_dirty_render_entities[i])->transform[1], 0.0f),
            glm::vec4(_render_entities.slot(_dirty_render_entities[i])->transform[2], 0.0f),
            glm::vec4(_render_entities.slot(_dirty_render_entities[i])->transform[3], 1.0f));
        glm::mat4 combined_transform4 = transform4;
        std::optional<RenderEntityId> parent = _render_entities.slot(_dirty_render_entities[i])->parent;
        while (parent.has_value())
        {
            glm::mat4x3 parent_transform4 = glm::mat4(
                glm::vec4(_render_entities.slot(parent.value())->transform[0], 0.0f),
                glm::vec4(_render_entities.slot(parent.value())->transform[1], 0.0f),
                glm::vec4(_render_entities.slot(parent.value())->transform[2], 0.0f),
                glm::vec4(_render_entities.slot(parent.value())->transform[3], 1.0f));
            combined_transform4 = parent_transform4 * combined_transform4;
            parent = _render_entities.slot(parent.value())->parent;
        }
        u32 mesh_group_manifest_index = _render_entities.slot(_dirty_render_entities[i])->mesh_group_manifest_index.value_or(INVALID_MANIFEST_INDEX);
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
    }
    _dirty_render_entities.clear();

    if (_new_mesh_group_manifest_entries > 0)
    {
        u32 const mesh_group_staging_buffer_size = sizeof(GPUMeshGroup) * _new_mesh_group_manifest_entries;
        daxa::BufferId mesh_group_staging_buffer = _device.create_buffer({
            .size = mesh_group_staging_buffer_size,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "mesh group update staging buffer",
        });
        recorder.destroy_buffer_deferred(mesh_group_staging_buffer);
        GPUMeshGroup * staging_ptr = _device.get_host_address_as<GPUMeshGroup>(mesh_group_staging_buffer).value();
        u32 const mesh_group_manifest_offset = _mesh_group_manifest.size() - _new_mesh_group_manifest_entries;
        for (u32 new_mesh_group_idx = 0; new_mesh_group_idx < _new_mesh_group_manifest_entries; new_mesh_group_idx++)
        {
            u32 const mesh_group_manifest_idx = mesh_group_manifest_offset + new_mesh_group_idx;
            staging_ptr[new_mesh_group_idx].count = _mesh_group_manifest.at(mesh_group_manifest_idx).mesh_count;
            std::memcpy(
                &staging_ptr[new_mesh_group_idx].mesh_manifest_indices,
                _mesh_group_manifest.at(mesh_group_manifest_idx).mesh_manifest_indices.data(),
                sizeof(_mesh_group_manifest.at(mesh_group_manifest_idx).mesh_manifest_indices));
        }
        recorder.copy_buffer_to_buffer({
            .src_buffer = mesh_group_staging_buffer,
            .dst_buffer = _gpu_mesh_group_manifest.get_state().buffers[0],
            .src_offset = 0,
            .dst_offset = mesh_group_manifest_offset * sizeof(GPUMeshGroup),
            .size = sizeof(GPUMeshGroup) * _new_mesh_group_manifest_entries,
        });
    }
    if (_new_mesh_manifest_entries > 0)
    {
        u32 const mesh_update_staging_buffer_size = sizeof(GPUMesh) * _new_mesh_manifest_entries;
        u32 const mesh_manifest_offset = _mesh_manifest.size() - _new_mesh_manifest_entries;
        daxa::BufferId mesh_staging_buffer = _device.create_buffer({
            .size = mesh_update_staging_buffer_size,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "mesh update staging buffer",
        });
        recorder.destroy_buffer_deferred(mesh_staging_buffer);
        GPUMesh * staging_ptr = _device.get_host_address_as<GPUMesh>(mesh_staging_buffer).value();
        std::vector<GPUMesh> tmp_meshes(_new_mesh_manifest_entries);
        std::memcpy(staging_ptr, tmp_meshes.data(), _new_mesh_manifest_entries);

        recorder.copy_buffer_to_buffer({
            .src_buffer = mesh_staging_buffer,
            .dst_buffer = _gpu_mesh_manifest.get_state().buffers[0],
            .src_offset = 0,
            .dst_offset = mesh_manifest_offset * sizeof(GPUMesh),
            .size = sizeof(GPUMesh) * _new_mesh_manifest_entries,
        });
    }
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
        GPUMaterial * staging_ptr = _device.get_host_address_as<GPUMaterial>(material_staging_buffer).value();
        std::vector<GPUMaterial> tmp_materials(_new_material_manifest_entries);
        std::memcpy(staging_ptr, tmp_materials.data(), _new_material_manifest_entries);

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
    {
        if (info.uploaded_textures.size() > 0)
        {
            /// NOTE: We need to propagate each loaded texture image ID into the material manifest This will be done in two steps:
            //        1) We update the CPU manifest with the correct values and remember the materials that were updated
            //        2) For each dirty material we generate a copy buffer to buffer comand to update the GPU manifest
            std::vector<u32> dirty_material_entry_indices = {};
            // 1) Update CPU Manifest
            for (AssetProcessor::TextureUploadInfo const & texture_upload : info.uploaded_textures)
            {
                _material_texture_manifest.at(texture_upload.texture_manifest_index).runtime = texture_upload.dst_image;
                TextureManifestEntry const & texture_manifest_entry = _material_texture_manifest.at(texture_upload.texture_manifest_index);
                for (auto const material_using_texture_info : texture_manifest_entry.material_manifest_indices)
                {
                    MaterialManifestEntry & material_entry = _material_manifest.at(material_using_texture_info.material_manifest_index);
                    if (material_using_texture_info.diffuse)
                    {
                        material_entry.diffuse_info->tex_manifest_index = texture_upload.texture_manifest_index;
                    }
                    if (material_using_texture_info.normal)
                    {
                        material_entry.normal_info->tex_manifest_index = texture_upload.texture_manifest_index;
                    }
                    if (material_using_texture_info.roughness_metalness)
                    {
                        material_entry.roughness_metalness_info->tex_manifest_index = texture_upload.texture_manifest_index;
                    }
                    /// NOTE: Add material index only if it was not added previously
                    if (std::find(
                            dirty_material_entry_indices.begin(),
                            dirty_material_entry_indices.end(),
                            material_using_texture_info.material_manifest_index) ==
                        dirty_material_entry_indices.end())
                    {
                        dirty_material_entry_indices.push_back(material_using_texture_info.material_manifest_index);
                    }
                }
            }
            // // 2) Update GPU manifest
            daxa::BufferId materials_update_staging_buffer = {};
            GPUMaterial * staging_origin_ptr = {};
            if (dirty_material_entry_indices.size())
            {
                materials_update_staging_buffer = _device.create_buffer({
                    .size = sizeof(GPUMaterial) * dirty_material_entry_indices.size(),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "gpu materials update staging",
                });
                recorder.destroy_buffer_deferred(materials_update_staging_buffer);
                staging_origin_ptr = _device.get_host_address_as<GPUMaterial>(materials_update_staging_buffer).value();
            }
            for (u32 dirty_materials_index = 0; dirty_materials_index < dirty_material_entry_indices.size(); dirty_materials_index++)
            {
                MaterialManifestEntry const & material = _material_manifest.at(dirty_material_entry_indices.at(dirty_materials_index));
                daxa::ImageId diffuse_id = {};
                daxa::ImageId normal_id = {};
                daxa::ImageId roughness_metalness_id = {};
                /// NOTE: We check if material even has diffuse info, if it does we need to check if the runtime value of this 
                //        info is present - It might be that diffuse texture was uploaded marking this material as dirty, but 
                //        the normal texture is not yet present thus we don't yet have the runtime info
                if (material.diffuse_info.has_value())
                {
                    auto const & texture_entry = _material_texture_manifest.at(material.diffuse_info.value().tex_manifest_index);
                    diffuse_id = texture_entry.runtime.value_or(daxa::ImageId{});
                }
                if (material.normal_info.has_value())
                {
                    auto const & texture_entry = _material_texture_manifest.at(material.normal_info.value().tex_manifest_index);
                    normal_id = texture_entry.runtime.value_or(daxa::ImageId{});
                }
                if (material.roughness_metalness_info.has_value())
                {
                    auto const & texture_entry = _material_texture_manifest.at(material.roughness_metalness_info.value().tex_manifest_index);
                    roughness_metalness_id = texture_entry.runtime.value_or(daxa::ImageId{});
                }
                staging_origin_ptr[dirty_materials_index].diffuse_texture_id = diffuse_id.default_view();
                staging_origin_ptr[dirty_materials_index].normal_texture_id = normal_id.default_view();
                staging_origin_ptr[dirty_materials_index].roughnes_metalness_id = roughness_metalness_id.default_view();
                staging_origin_ptr[dirty_materials_index].alpha_discard_enabled = material.alpha_discard_enabled;

                daxa::BufferId gpu_material_manifest = _gpu_material_manifest.get_state().buffers[0];
                recorder.copy_buffer_to_buffer({
                    .src_buffer = materials_update_staging_buffer,
                    .dst_buffer = gpu_material_manifest,
                    .src_offset = sizeof(GPUMaterial) * dirty_materials_index,
                    .dst_offset = sizeof(GPUMaterial) * dirty_material_entry_indices.at(dirty_materials_index),
                    .size = sizeof(GPUMaterial),
                });
            }
            recorder.pipeline_barrier({
                .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                .dst_access = daxa::AccessConsts::READ,
            });
        }
    }

    // updating mesh manifest
    {
        if (info.uploaded_meshes.size() > 0)
        {
            daxa::BufferId staging_buffer = _device.create_buffer({
                .size = info.uploaded_meshes.size() * sizeof(GPUMesh),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "mesh manifest upload staging buffer",
            });

            recorder.destroy_buffer_deferred(staging_buffer);
            GPUMesh * staging_ptr = _device.get_host_address_as<GPUMesh>(staging_buffer).value();
            for (i32 upload_index = 0; upload_index < info.uploaded_meshes.size(); upload_index++)
            {
                auto const & upload = info.uploaded_meshes[upload_index];
                _mesh_manifest.at(upload.manifest_index).runtime = upload.mesh;
                std::memcpy(staging_ptr + upload_index, &upload.mesh, sizeof(GPUMesh));
                recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_buffer,
                    .dst_buffer = _gpu_mesh_manifest.get_state().buffers[0],
                    .src_offset = upload_index * sizeof(GPUMesh),
                    .dst_offset = upload.manifest_index * sizeof(GPUMesh),
                    .size = sizeof(GPUMesh),
                });
            }
        }
    }

    /// TODO: Taskgraph this shit.
    recorder.pipeline_barrier({
        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_access = daxa::AccessConsts::READ_WRITE,
    });

    _new_material_manifest_entries = 0;
    _new_mesh_manifest_entries = 0;
    _new_mesh_group_manifest_entries = 0;
    return recorder.complete_current_commands();
}