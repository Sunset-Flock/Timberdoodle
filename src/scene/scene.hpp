#pragma once

#include <optional>

#include "fastgltf/parser.hpp"

#include "../sandbox.hpp"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/scene.inl"
#include "../slot_map.hpp"

/**
 * DESCRIPTION:
 * Scenes are described by entities and their resources.
 * These resources can have complex dependencies between each other.
 * We want to be able to load AND UNLOAD the resources asynchronously.
 * BUT we want to remember unloaded resources. We never delete metadata.
 * The metadata tracks all the complex dependencies. Never deleting them makes the lifetimes for dependencies trivial.
 * It also allows us to have a better tracking of when a resource was unloaded how it was used etc. .
 * We store the metadata in manifest arrays.
 * The only data that can change in the manifests are in leaf nodes of the dependencies, eg texture data, mesh data.
*/

struct SceneFileManifestEntry
{
    std::filesystem::path path = {};
    std::unique_ptr<fastgltf::glTF> gltf_info = {};
    std::unique_ptr<fastgltf::Asset> gltf_asset = {};
    /// @brief  Offsets of the gltf indices to the loaded manifest indices.
    ///         Subtracting the scene offset from the manifest index gives you the gltf index.
    u32 texture_manifest_offset = {};
    u32 material_manifest_offset = {};
    u32 mesh_group_manifest_offset = {};
    u32 mesh_manifest_offset = {};
};

struct TextureManifestEntry
{
    u32 scene_file_manifest_index = {};
    u32 in_scene_file_index = {};
    // List of materials that use this texture.
    // The GPUMaterialDescriptor contrains ImageIds directly,
    // So the GPUMaterialDescriptors Need to be updated when the texture changes.
    std::vector<u32> material_manifest_indices = {};
    std::optional<daxa::ImageId> runtime = {};
    std::string name = {};
};

struct MaterialManifestEntry
{
    std::optional<u32> diffuse_tex_index = {};
    std::optional<u32> normal_tex_index = {};
    u32 scene_file_manifest_index = {};
    u32 in_scene_file_index = {};
    std::string name = {};
};

struct MeshManifestEntry
{
    std::optional<u32> material_manifest_index = {};
    u32 scene_file_manifest_index = {};
    u32 scene_file_mesh_index = {};
    u32 scene_file_primitive_index = {};
    std::optional<GPUMeshDescriptor> runtime = {};
};

struct MeshGroupManifestEntry
{
    std::array<u32, MAX_MESHES_PER_MESHGROUP> mesh_manifest_indices = {};
    u32 mesh_count = {};
    u32 scene_file_manifest_index = {};
    u32 in_scene_file_index = {};
    std::string name = {};
};

struct RenderEntity;
using RenderEntityId = SlotMap<RenderEntity>::Id;

struct RenderEntity
{
    glm::mat4x3 transform = {};
    std::optional<RenderEntityId> first_child = {};
    std::optional<RenderEntityId> next_sibling = {};
    std::optional<RenderEntityId> parent = {};
    std::optional<u32> mesh_group_manifest_index = {};
    std::string name = {};
};

using RenderEntitySlotMap = SlotMap<RenderEntity>;



struct Scene
{
    /**
     * NOTES:
     * - On the cpu, the entities are stored in a slotmap
     * - On the gpu, render entities are stored in an 'soa' slotmap
     * - the slotmaps capacity (and its underlying arrays) will only grow with time, it never shrinks
     * - all entity buffer updates are recorded within the scenes record commands function
     * - WARNING: FOR NOW THE RENDERER ASSUMES TIGHTLY PACKED ENTITIES!
     * - TODO: Upload sparse set to gpu so gpu can tightly iterate!
     * - TODO: Make the task buffers real buffers grow with time, unfix their size!
     * - TODO: Combine all into one task buffer when task graph gets array uses.
    */
    daxa::TaskBuffer _gpu_entity_meta = daxa::TaskBufferInfo{.name = "_gpu_entity_meta"};
    daxa::TaskBuffer _gpu_entity_transforms = daxa::TaskBufferInfo{.name = "_gpu_entity_transforms"};
    daxa::TaskBuffer _gpu_entity_combined_transforms = daxa::TaskBufferInfo{.name = "_gpu_entity_combined_transforms"};
    // UNUSED, but later we wanna do 
    // the compined transform calculation on the gpu!
    daxa::TaskBuffer _gpu_entity_parents = daxa::TaskBufferInfo{.name = "_gpu_entity_parents"};                
    daxa::TaskBuffer _gpu_entity_mesh_groups = daxa::TaskBufferInfo{.name = "_gpu_entity_mesh_groups"};            
    RenderEntitySlotMap _render_entities = {};
    std::vector<RenderEntityId> _dirty_render_entities = {}; 

    /**
     * NOTES:
     * -    growing and initializing the manifest on the gpu is recorded in the scene,
     *      following UPDATES to the manifests are recorded from the asset processor
     * - growing and initializing the manifest on the cpu is done when recording scene commands
     * - the manifests only grow and are largely immutable on the cpu
     * - specific cpu manifests will have 'runtime' data that is not immutable
     * - the asset processor may update the immutable runtime data within the manifests
     * - the cpu and gpu versions of the manifest will be different to reduce indirections on the gpu
     * - TODO: Make the task buffers real buffers grow with time, unfix their size!
     * */
    daxa::TaskBuffer _gpu_mesh_manifest = daxa::TaskBufferInfo{.name = "_gpu_mesh_manifest"};  
    daxa::TaskBuffer _gpu_mesh_group_manifest = daxa::TaskBufferInfo{.name = "_gpu_mesh_group_manifest"};
    daxa::TaskBuffer _gpu_material_manifest = daxa::TaskBufferInfo{.name = "_gpu_material_manifest"};       
    std::vector<SceneFileManifestEntry> _scene_file_manifest = {};
    std::vector<TextureManifestEntry> _material_texture_manifest = {};
    std::vector<MaterialManifestEntry> _material_manifest = {};
    std::vector<MeshManifestEntry> _mesh_manifest = {};
    std::vector<MeshGroupManifestEntry> _mesh_group_manifest = {};
    // Count the added meshes and meshgroups when loading.
    // Used to do the initialization of these on the gpu when recording manifest update.
    u32 _new_mesh_manifest_entries = {};
    u32 _new_mesh_group_manifest_entries = {};

    Scene(daxa::Device device);
    ~Scene();

    enum struct LoadManifestErrorCode
    {
        FILE_NOT_FOUND,
        COULD_NOT_LOAD_ASSET,
        INVALID_GLTF_FILE_TYPE,
        COULD_NOT_PARSE_ASSET_NODES,
    };
    static auto to_string(LoadManifestErrorCode result) -> std::string_view
    {
        switch(result)
        {
            case LoadManifestErrorCode::FILE_NOT_FOUND: return "FILE_NOT_FOUND";
            case LoadManifestErrorCode::COULD_NOT_LOAD_ASSET: return "COULD_NOT_LOAD_ASSET";
            case LoadManifestErrorCode::INVALID_GLTF_FILE_TYPE: return "INVALID_GLTF_FILE_TYPE";
            case LoadManifestErrorCode::COULD_NOT_PARSE_ASSET_NODES: return "COULD_NOT_PARSE_ASSET_NODES";
            default: return "UNKNOWN";
        }
        return "UNKNOWN";
    }
    auto load_manifest_from_gltf(std::filesystem::path const& root_path, std::filesystem::path const& glb_name) -> std::variant<RenderEntityId, LoadManifestErrorCode>;

    auto record_gpu_manifest_update() -> daxa::ExecutableCommandList;

    daxa::Device _device = {};
};