#pragma once

#include <optional>
#include <variant>
#include <span>
#include <string_view>

#include "../timberdoodle.hpp"

#include "../shader_shared/geometry.inl"
#include "../shader_shared/geometry_pipeline.inl"
#include "../shader_shared/scene.inl"
#include "../slot_map.hpp"
#include "asset_processor.hpp"
#include "importers/openvdb_importer.hpp"
#include "tido_format/tido_format.hpp"
#include "streamer.hpp"
using namespace tido::types;

struct CPUMeshInstanceCounts
{
    u32 mesh_instance_count = {};
    u32 prepass_instance_counts[PREPASS_DRAW_LIST_TYPE_COUNT] = {};
    u32 vsm_invalidate_instance_count = {};
};

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


/// ================================================== IMAGE ==================================================
struct ImageRuntimeData
{
    // The live GPU handle for this texture; empty until the streamer has made it resident.
    daxa::ImageId image = {};
};

struct ImageImporterData
{
    // source + cook recipe: everything importer needs to re-cook this image.
    std::filesystem::path file = {};
    u64 image_index = {};
    std::vector<u8> channel_mapping = {};
    daxa::Format target_format = {};
};

struct ImageManifestEntry
{
    // List of materials that use this texture.
    // GPUMaterial contrains ImageIds directly.
    // Used to update the GPUMaterial when the image is loaded/unloaded.
    std::vector<u32> material_manifest_indices = {};
    std::string name = {};

    ImageStreamerData streamer_data = {};
    ImageImporterData importer_data = {};
    std::optional<ImageRuntimeData> runtime_data = {};

    auto loaded() const -> bool{ return runtime_data.has_value(); }
};

/// ================================================== MESH ==================================================
struct MeshRuntimeData
{
    // The live per-LOD GPU handles; empty until the streamer has made the mesh resident.
    std::array<GPUMesh, MAX_MESHES_PER_LOD_GROUP> lods = {};
    std::array<daxa::BlasId, MAX_MESHES_PER_LOD_GROUP> blas_lods = {};
    daxa_u32 lod_count = {};
};

struct MeshImporterData
{
    // source + cook recipe: everything importer needs to re-cook this mesh.
    TidoMeshDescriptor descriptor = {};
    std::filesystem::path bin_source = {};
    u64 mesh_index = {};
    u64 file_data_offset = {};
};

struct MeshLodGroupManifestEntry
{
    std::optional<u32> mesh_group_manifest_index = {};
    std::optional<u32> material_index = {};
    std::string name = {};

    MeshStreamerData streamer_data = {};
    MeshImporterData importer_data = {};
    std::optional<MeshRuntimeData> runtime_data = {};

    auto loaded() const -> bool{ return runtime_data.has_value(); }
};

struct MaterialManifestEntry
{
    struct TextureInfo
    {
        u32 image_manifest_index = {};
        u32 sampler_index = {};
    };
    std::optional<TextureInfo> diffuse_info = {};
    std::optional<TextureInfo> opacity_mask_info = {};
    std::optional<TextureInfo> normal_info = {};
    std::optional<TextureInfo> roughness_metalness_info = {};
    bool alpha_discard_enabled = {};
    bool double_sided = {};
    bool blend_enabled = {};
    bool is_metal = {};
    f32vec3 base_color = {};
    f32vec3 emissive_color = {};
    std::string name = {};
};

struct MeshGroupManifestEntry
{
    u32 mesh_lod_group_manifest_indices_array_offset = {};
    u32 mesh_lod_group_count = {};
    u32 loaded_mesh_lod_groups = {};
    bool fully_loaded_last_frame = {};
    daxa::BlasId blas = {};
    std::string name = {};
};

struct PointLight
{
    f32vec3 position;
    f32vec3 color;
    f32 intensity;
    f32 cutoff;
    daxa_BufferPtr(GPUPointLight) point_light_ptr;
};

struct SpotLight
{
    f32mat4x3 transform;
    f32vec3 color;
    f32 intensity;
    f32 cutoff;
    f32 inner_cone_angle;
    f32 outer_cone_angle;
    daxa_BufferPtr(GPUSpotLight) spot_light_ptr;
};

struct CloudVolume
{
    std::string cloud_volume_data_path;
    std::string detail_noise_path;

    u32 data_image_manifest_index = {};
    u32 sdf_image_manifest_index = {};
    u32 detail_noise_image_manifest_index = {};
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
    POINT_LIGHT,
    SPOT_LIGHT,
    CAMERA,
    MESHGROUP,
    CLOUD_VOLUME,
    UNKNOWN
};

struct RenderEntity
{
    glm::mat4x3 transform = {};
    glm::mat4x3 combined_transform = {};
    std::optional<RenderEntityId> first_child = {};
    std::optional<RenderEntityId> next_sibling = {};
    std::optional<RenderEntityId> parent = {};
    std::optional<u32> mesh_group_manifest_index = {};
    std::optional<u32> cloud_volume_index = {};
    EntityType type = EntityType::UNKNOWN;
    std::string name = {};
    std::optional<u32> light_index = {};
    bool dirty = {};
};

using RenderEntitySlotMap = tido::SlotMap<RenderEntity>;

struct CPUMeshInstances
{
    std::vector<MeshInstance> mesh_instances = {};
    std::vector<u32> prepass_draw_lists[PREPASS_DRAW_LIST_TYPE_COUNT] = {{}, {}};
    std::vector<u32> vsm_invalidate_draw_list = {};
};

struct CPUCloudVolumeInstaces
{
    std::vector<AABB> instance_aabbs = {};
    std::vector<CloudVolumeInstance> instances = {};
};

struct CPUSceneInstances
{
    CPUMeshInstances mesh_instances = {};
    CPUCloudVolumeInstaces cloud_volume_instances = {};
};

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
    daxa::ExternalTaskBuffer _gpu_entity_meta = {};
    daxa::ExternalTaskBuffer _gpu_entity_transforms = {};
    daxa::ExternalTaskBuffer _gpu_entity_combined_transforms = {};
    // UNUSED, but later we wanna do
    // the compined transform calculation on the gpu!
    daxa::ExternalTaskBuffer _gpu_entity_parents = {};
    daxa::ExternalTaskBuffer _gpu_entity_mesh_groups = {};
    daxa::ExternalTaskBuffer _gpu_point_lights = {};
    daxa::ExternalTaskBuffer _gpu_spot_lights = {};

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
    daxa::ExternalTaskBuffer _gpu_mesh_manifest = {};
    daxa::ExternalTaskBuffer _gpu_mesh_lod_group_manifest = {};
    daxa::ExternalTaskBuffer _gpu_mesh_group_manifest = {};
    daxa::BufferId _gpu_mesh_group_indices_array_buffer = {};
    daxa::ExternalTaskBuffer _gpu_material_manifest = {};
    daxa::ExternalTaskBuffer _gpu_scratch_buffer = {};
    daxa::ExternalTaskBuffer _gpu_mesh_acceleration_structure_build_scratch_buffer = {};
    daxa::ExternalTaskBuffer _gpu_tlas_build_scratch_buffer = {};
    static constexpr u32 _gpu_scratch_buffer_size = 1u << 24u;
    static constexpr u32 _gpu_mesh_acceleration_structure_build_scratch_buffer_size = 1u << 29u;
    static constexpr u32 _gpu_tlas_build_scratch_buffer_size = 1u << 24u;
    static constexpr u32 _indirections_count = (1 << 26);
    static constexpr u32 MAX_MESH_BLAS_BUILDS_PER_FRAME = 64;

    daxa::BlasId _scene_blas = {};
    daxa::ExternalTaskBuffer _scene_as_indirections = {};

    daxa::Device _device = {};
    GPUContext * gpu_context = {};

    Scene(daxa::Device device, GPUContext * gpu_context);
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
            default:
                DBG_ASSERT_TRUE_M(false, "Unhandled LoadManifestErrorCode");
                return "UNKNOWN";
        }
    }
    void build_tlas_from_mesh_instances(daxa::CommandRecorder & recorder, daxa::TlasId tlas);

    /// --- Transient Processes ---

    // Populated by process entities every frame
    CPUMeshInstanceCounts cpu_mesh_instance_counts = {};                            // Useful for cpu driven dispatches and draws. Only really need counts on cpu.
    CPUMeshInstances current_frame_mesh_instances = {};
    daxa::ExternalTaskBuffer mesh_instances_buffer = {};

    auto process_entities(RenderGlobalData & render_data) -> CPUSceneInstances;
    void write_gpu_mesh_instances_buffer(CPUMeshInstances const& mesh_instances);

    CPUCloudVolumeInstaces current_frame_cloud_volume_instances = {};
    daxa::ExternalTaskBuffer cloud_volume_instances_buffer = {};
    void write_gpu_cloud_volume_instances_buffer(CPUCloudVolumeInstaces const& cloud_volume_instances);

    void clear(std::unique_ptr<ThreadPool> & thread_pool, std::unique_ptr<AssetProcessor> & asset_processor);

    RenderEntitySlotMap _render_entities = {};
    std::vector<RenderEntityId> _dirty_render_entities = {};

    std::vector<u32> _newly_completed_mesh_groups = {};

    // Root entity of each imported asset's entity sub-tree.
    std::vector<RenderEntityId> _root_render_entities = {};
    std::vector<ImageManifestEntry> _image_manifest = {};
    std::vector<MaterialManifestEntry> _material_manifest = {};
    std::vector<MeshLodGroupManifestEntry> _mesh_lod_group_manifest = {};
    std::vector<MeshGroupManifestEntry> _mesh_group_manifest = {};
    std::vector<u32> _mesh_lod_group_manifest_indices = {};

    std::vector<PointLight> _point_lights = {};
    std::vector<SpotLight> _spot_lights = {};
    std::vector<CloudVolume> _cloud_volumes = {};

    // Manifest indices that changed (were added, or had their runtime data updated) since the last GPU
    // manifest sync. Every manifest that is mirrored on the GPU owns one; SceneRuntime is the only thing
    // that touches these, always from the main thread, so plain vectors are safe - SceneRuntime::update
    // drains each one to re-upload exactly those entries (instead of assuming a contiguous tail of new
    // entries). Manifests that are not shared with the GPU (e.g. the image manifest) do not need one.
    std::vector<u32> _dirty_material_indices = {};
    std::vector<u32> _dirty_mesh_lod_group_indices = {};
    std::vector<u32> _dirty_mesh_group_indices = {};
    std::vector<u32> _dirty_texture_indices = {};

    // Mesh-lod-group indices whose cooked artifact still needs streaming in (separate from the GPU-sync
    // dirty list above: this only drives async residency). SceneRuntime::update drains it to spawn stream
    // tasks; a finished stream then marks _dirty_mesh_lod_group_indices so the GPU sync uploads the real data.
    std::vector<u32> _dirty_mesh_lod_group_streaming_indices = {};

    std::vector<u32> _cloud_volumes_requesting_load = {};

    // Dispatches an async load task for every cloud volume in _cloud_volumes_requesting_load, then
    // clears the request list.
    void start_async_loads_of_dirty_cloud_volumes(AssetProcessor * asset_processor, ThreadPool * thread_pool);
};