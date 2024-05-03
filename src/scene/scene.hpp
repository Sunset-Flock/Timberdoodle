#pragma once

#include <optional>
#include <variant>

#include <fastgltf/types.hpp>
#include "../timberdoodle.hpp"

#include "../shader_shared/geometry.inl"
#include "../shader_shared/scene.inl"
#include "../slot_map.hpp"
#include "../multithreading/thread_pool.hpp"
#include "../rendering/scene_renderer_context.hpp"
#include "asset_processor.hpp"
using namespace tido::types;
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
struct TextureManifestEntry
{
    // The type is determined by the materials that reference it.
    TextureMaterialType type = {};
    struct MaterialManifestIndex
    {
        u32 material_manifest_index = {};
    };
    u32 gltf_asset_manifest_index = {};
    u32 asset_local_index = {};
    u32 asset_local_image_index = {};
    // List of materials that use this texture and how they use it
    // The GPUMaterial contrains ImageIds directly,
    // So the GPUMaterial Need to be updated when the texture changes.
    std::vector<MaterialManifestIndex> material_manifest_indices = {};
    std::optional<daxa::ImageId> runtime_texture = {};
    // This is used for separate oppacity mask when we generate one
    std::optional<daxa::ImageId> secondary_runtime_texture = {};
    std::string name = {};
};

struct MaterialManifestEntry
{
    struct TextureInfo
    {
        u32 tex_manifest_index = {};
        u32 sampler_index = {};
    };
    std::optional<TextureInfo> diffuse_info = {};
    std::optional<TextureInfo> opacity_mask_info = {};
    std::optional<TextureInfo> normal_info = {};
    std::optional<TextureInfo> roughness_metalness_info = {};
    u32 gltf_asset_manifest_index = {};
    u32 asset_local_index = {};
    bool alpha_discard_enabled = {};
    bool normal_compressed_bc5_rg = {}; 
    f32vec3 base_color = {};
    std::string name = {};
};

struct MeshManifestEntry
{
    u32 gltf_asset_manifest_index = {};
    u32 asset_local_mesh_index = {};
    u32 asset_local_primitive_index = {};
    std::optional<u32> material_index = {};
    std::optional<GPUMesh> runtime = {};
};

struct MeshGroupManifestEntry
{
    u32 mesh_manifest_indices_array_offset = {};
    u32 mesh_count = {};
    u32 gltf_asset_manifest_index = {};
    u32 asset_local_index = {};
    std::string name = {};
};

struct RenderEntity;
using RenderEntityId = tido::SlotMap<RenderEntity>::Id;

// TODO(msakmary) This assumes entity is only one of these types exclusively however this is not true
//                for example, an entity can be both Transform (aka parent to other entities) and
//                Meshgroup (aka have a meshgroup index and represent mesh)
enum struct EntityType
{
    ROOT,
    TRANSFORM,
    LIGHT,
    CAMERA,
    MESHGROUP,
    UNKNOWN
};

struct RenderEntity
{
    glm::mat4x3 transform = {};
    std::optional<RenderEntityId> first_child = {};
    std::optional<RenderEntityId> next_sibling = {};
    std::optional<RenderEntityId> parent = {};
    std::optional<u32> mesh_group_manifest_index = {};
    EntityType type = EntityType::UNKNOWN;
    std::string name = {};
};

struct GltfAssetManifestEntry
{
    std::filesystem::path path = {};
    std::unique_ptr<fastgltf::Asset> gltf_asset = {};
    /// @brief  Offsets of the gltf asset local indices that is applied when storing the data into the global manifest.
    ///         For example, when a meshgroup has asset_local_index = 4 it is the 5th meshgroup in its gltf asset. 
    ///         meshgroup.asset_local_index + mesh_group_manifest_offset then gives the global index into the meshgroup manifest
    u32 texture_manifest_offset = {};
    u32 material_manifest_offset = {};
    u32 mesh_group_manifest_offset = {};
    u32 mesh_manifest_offset = {};
    RenderEntityId root_render_entity = {};
};

using RenderEntitySlotMap = tido::SlotMap<RenderEntity>;

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
    daxa::TaskBuffer _gpu_entity_meta = {};
    daxa::TaskBuffer _gpu_entity_transforms = {};
    daxa::TaskBuffer _gpu_entity_combined_transforms = {};
    // UNUSED, but later we wanna do
    // the compined transform calculation on the gpu!
    daxa::TaskBuffer _gpu_entity_parents = {};
    daxa::TaskBuffer _gpu_entity_mesh_groups = {};
    RenderEntitySlotMap _render_entities = {};
    std::vector<RenderEntityId> _dirty_render_entities = {};
    struct ModifiedEntityInfo
    {
        RenderEntityId entity = {};
        glm::mat4x4 prev_transform = {};
        glm::mat4x4 curr_transform = {};
    };
    std::vector<ModifiedEntityInfo> _modified_render_entities = {};

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
    daxa::TaskBuffer _gpu_mesh_manifest = {};
    daxa::TaskBuffer _gpu_mesh_group_manifest = {};
    daxa::BufferId _gpu_mesh_group_indices_array_buffer = {};
    daxa::TaskBuffer _gpu_material_manifest = {};
    std::vector<GltfAssetManifestEntry> _gltf_asset_manifest = {};
    std::vector<TextureManifestEntry> _material_texture_manifest = {};
    std::vector<MaterialManifestEntry> _material_manifest = {};
    std::vector<MeshManifestEntry> _mesh_manifest = {};
    std::vector<u32> _mesh_manifest_indices_new = {};
    std::vector<MeshGroupManifestEntry> _mesh_group_manifest = {};
    // Count the added meshes and meshgroups when loading.
    // Used to do the initialization of these on the gpu when recording manifest update.
    u32 _new_mesh_manifest_entries = {};
    u32 _new_mesh_group_manifest_entries = {};
    u32 _new_material_manifest_entries = {};
    u32 _new_texture_manifest_entries = {};

    SceneDraw _scene_draw = {};

    // TODO(msakmary) REMOVE ME - this is a giant hack
    std::vector<AABB> REMOVE_ME_dynamic_object_aabbs_REMOVE_ME = {};

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
        switch (result)
        {
            case LoadManifestErrorCode::FILE_NOT_FOUND:              return "FILE_NOT_FOUND";
            case LoadManifestErrorCode::COULD_NOT_LOAD_ASSET:        return "COULD_NOT_LOAD_ASSET";
            case LoadManifestErrorCode::INVALID_GLTF_FILE_TYPE:      return "INVALID_GLTF_FILE_TYPE";
            case LoadManifestErrorCode::COULD_NOT_PARSE_ASSET_NODES: return "COULD_NOT_PARSE_ASSET_NODES";
            default:                                                 return "UNKNOWN";
        }
        return "UNKNOWN";
    }
    struct LoadManifestInfo
    {
        std::filesystem::path root_path;
        std::filesystem::path asset_name;
        std::unique_ptr<ThreadPool> & thread_pool;
        std::unique_ptr<AssetProcessor> & asset_processor;
    };
    auto load_manifest_from_gltf(LoadManifestInfo const & info) -> std::variant<RenderEntityId, LoadManifestErrorCode>;

    struct RecordGPUManifestUpdateInfo
    {
        std::span<const AssetProcessor::MeshUploadInfo> uploaded_meshes = {};
        std::span<const AssetProcessor::LoadedTextureInfo> uploaded_textures = {};
    };
    auto record_gpu_manifest_update(RecordGPUManifestUpdateInfo const & info) -> daxa::ExecutableCommandList;

    daxa::Device _device = {};
};