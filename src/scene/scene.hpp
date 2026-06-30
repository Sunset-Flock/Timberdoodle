#pragma once

#include <optional>
#include <variant>
#include <mutex>
#include <span>

#include "../timberdoodle.hpp"

#include "../shader_shared/geometry.inl"
#include "../shader_shared/geometry_pipeline.inl"
#include "../shader_shared/scene.inl"
#include "../slot_map.hpp"
#include "../multithreading/thread_pool.hpp"
#include "asset_processor.hpp"
#include "tido_format/tido_texture.hpp"
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
struct TextureManifestEntry
{
    struct MaterialManifestIndex
    {
        u32 material_manifest_index = {};
    };

    // The type is determined by the materials that reference it.
    TextureMaterialType type = {};
    // List of materials that use this texture and how they use it
    // The GPUMaterial contrains ImageIds directly,
    // So the GPUMaterial Need to be updated when the texture changes.
    std::vector<MaterialManifestIndex> material_manifest_indices = {};  // Would prefer some other allocation scheme here.
    std::optional<daxa::ImageId> runtime_texture = {};
    // Reference to the cooked .tido artifact (descriptor + subresource offset table + path). The
    // streamer reads this to make runtime_texture resident; kept so the texture can be re-streamed
    // (loaded/unloaded) later without the source file. Empty for non-gltf textures (e.g. cloud volumes).
    TidoTextureCookResult cooked_artifact = {};
    std::string name = {};

    auto loaded() const -> bool{ return runtime_texture.has_value(); }
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
    bool alpha_discard_enabled = {};
    bool double_sided = {};
    bool blend_enabled = {};
    bool is_metal = {};
    f32vec3 base_color = {};
    f32vec3 emissive_color = {};
    std::string name = {};
};

struct MeshLodGroupManifestEntry
{
    u32 mesh_group_manifest_index = {};
    std::optional<u32> material_index = {};
    std::string name = {}; // TODO(pahrens): fill out.
    // Reference to the cooked .tido artifact (descriptor + per-LOD offset table + path). The streamer
    // reads this to make `runtime` resident; kept so the mesh can be re-streamed (loaded/unloaded) later
    // without the source file. Attached at add_mesh time (the mesh is added only after it is cooked).
    // Mirrors TextureManifestEntry.
    TidoMeshCookResult cooked_artifact = {};
    struct Runtime
    {
        std::array<GPUMesh, MAX_MESHES_PER_LOD_GROUP> lods = {};
        std::array<daxa::BlasId, MAX_MESHES_PER_LOD_GROUP> blas_lods = {};
        daxa_u32 lod_count = {};
    };
    std::optional<Runtime> runtime = {};

    auto loaded() const -> bool{ return runtime.has_value(); }
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

    u32 data_texture_manifest_index = {};
    u32 sdf_texture_manifest_index = {};
    u32 detail_noise_texture_manifest_index = {};
};

// A thread-safe list of manifest indices that changed (were added, or had their runtime data updated)
// since the last GPU manifest sync. Every manifest that is mirrored on the GPU owns one; importers
// mark indices from worker threads as they populate the scene, and update_scene drains
// it to re-upload exactly those entries (instead of assuming a contiguous tail of new entries).
// Manifests that are not shared with the GPU (e.g. the texture manifest) do not need one.
struct DirtyManifestList
{
    // Records that `index` changed. Safe to call concurrently from multiple importer threads.
    void mark(u32 index)
    {
        std::lock_guard<std::mutex> lock{*_mutex};
        _indices.push_back(index);
    }
    // Moves the accumulated indices out and leaves the list empty. Called by the GPU sync.
    auto drain() -> std::vector<u32>
    {
        std::lock_guard<std::mutex> lock{*_mutex};
        std::vector<u32> out = std::move(_indices);
        _indices.clear();
        return out;
    }

    std::vector<u32> _indices = {};
    // unique_ptr so the owning Scene stays movable (std::mutex is not movable), matching _manifest_mutex.
    std::unique_ptr<std::mutex> _mutex = std::make_unique<std::mutex>();
};

// An async texture residency job. update_scene spawns one per newly dirtied texture: on a worker
// thread it reads the cooked .tido off disk and uploads it to the GPU (the streamer). update_scene
// polls `finished` each frame; once set it publishes `result` as the texture's runtime image and
// re-marks the materials referencing that texture dirty so their GPUMaterial gets the resolved id.
struct TextureStreamTask : Task
{
    daxa::Device device = {};
    // Copied (not referenced) so it stays valid if _material_texture_manifest reallocates mid-stream.
    TidoTextureCookResult artifact = {};
    u32 texture_manifest_index = {};
    daxa::ImageId result = {};
    std::atomic<bool> finished = false;

    void callback(u32 chunk_index, u32 thread_index) override;
};

// An async mesh residency job. update_scene spawns one per newly requested mesh: on a worker thread it
// reads the cooked .tido off disk and uploads each LOD into its per-LOD GPU buffer (the streamer).
// update_scene polls `finished` each frame; once set it publishes `result` as the mesh's runtime and
// marks the mesh-lod-group manifest dirty so the GPU sync uploads it + does the BLAS / mesh-group
// completeness bookkeeping. Mirrors TextureStreamTask.
struct MeshStreamTask : Task
{
    daxa::Device device = {};
    // Copied (not referenced) so it stays valid if _mesh_lod_group_manifest reallocates mid-stream.
    TidoMeshCookResult artifact = {};
    u32 mesh_lod_manifest_index = {};
    u32 material_manifest_index = {};
    std::string name = {};
    MeshLodGroupUploadInfo result = {};
    std::atomic<bool> finished = false;

    void callback(u32 chunk_index, u32 thread_index) override;
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

    RenderEntitySlotMap _render_entities = {};
    std::vector<RenderEntityId> _dirty_render_entities = {};
    struct ModifiedEntityInfo
    {
        RenderEntityId entity = {};
        glm::mat4x4 prev_transform = {};
        glm::mat4x4 curr_transform = {};
    };
    std::vector<ModifiedEntityInfo> _modified_render_entities = {};

    std::vector<u32> _newly_completed_mesh_groups = {};
    std::vector<u32> _mesh_as_build_queue = {};
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
    // Root entity of each imported asset's entity sub-tree (file-agnostic; replaces the old per-file session list).
    std::vector<RenderEntityId> _root_render_entities = {};
    std::vector<TextureManifestEntry> _material_texture_manifest = {};
    std::vector<MaterialManifestEntry> _material_manifest = {};
    std::vector<MeshLodGroupManifestEntry> _mesh_lod_group_manifest = {};
    std::vector<u32> _mesh_lod_group_manifest_indices = {};
    std::vector<MeshGroupManifestEntry> _mesh_group_manifest = {};
    // Per GPU-resident manifest: the indices changed since the last GPU sync (see DirtyManifestList).
    DirtyManifestList _dirty_material_manifest = {};
    DirtyManifestList _dirty_mesh_lod_group_manifest = {};
    DirtyManifestList _dirty_mesh_group_manifest = {};
    // Texture indices whose cooked artifact still needs streaming in. update_scene drains this to spawn
    // async stream tasks; it is not a GPU-resident manifest itself (textures reach the GPU only as
    // ImageIds inside GPUMaterial), but it drives the residency + the material re-dirtying.
    DirtyManifestList _dirty_material_texture_manifest = {};
    // Texture stream jobs currently in flight (spawned by update_scene, collected once finished).
    std::vector<std::shared_ptr<TextureStreamTask>> _inflight_texture_streams = {};
    // Mesh-lod-group indices whose cooked artifact still needs streaming in (separate from the GPU-sync
    // dirty list above: this only drives async residency). update_scene drains it to spawn stream tasks;
    // a finished stream then marks _dirty_mesh_lod_group_manifest so the GPU sync uploads the real data.
    DirtyManifestList _dirty_mesh_lod_group_streaming = {};
    // Mesh stream jobs currently in flight (spawned by update_scene, collected once finished).
    std::vector<std::shared_ptr<MeshStreamTask>> _inflight_mesh_streams = {};
    std::vector<PointLight> _point_lights = {};
    std::vector<SpotLight> _spot_lights = {};
    std::vector<CloudVolume> _cloud_volumes = {};

    std::vector<u32> _cloud_volumes_requesting_load = {};


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

    /// --- Generic, format-agnostic scene builder API ---
    // Each add_* appends a fully-built (format-neutral) entry and returns its global manifest index.
    // Importers reference already-added entries by these returned indices (never by a captured base
    // offset), so multiple importers can populate the scene concurrently. All adds are internally
    // synchronized via _manifest_mutex; the scene maintains all cross-references (texture<->material
    // back-refs, mesh<->mesh-group links).
    std::unique_ptr<std::mutex> _manifest_mutex = std::make_unique<std::mutex>();
    auto add_texture(TextureManifestEntry texture) -> u32;
    auto add_material(MaterialManifestEntry material) -> u32;
    auto add_mesh(MeshLodGroupManifestEntry mesh) -> u32;
    // Adds a mesh group over the given (already-added) mesh manifest indices. Fills the group's
    // index range + count and back-links each mesh to this group.
    auto add_mesh_group(MeshGroupManifestEntry mesh_group, std::span<u32 const> mesh_manifest_indices) -> u32;
    auto add_point_light(PointLight light) -> u32;
    auto add_spot_light(SpotLight light) -> u32;
    auto add_entity(RenderEntity entity) -> RenderEntityId;

    auto add_cloud_volume(std::string const & cloud_volume_data_path, std::string const & detail_noise_path, AssetProcessor * asset_processor, ThreadPool * thread_pool) -> u32;

    struct UpdateSceneInfo
    {
        // Used to spawn async texture- and mesh-streaming jobs (reading cooked .tido off disk). May be
        // null (e.g. at shutdown after the pool is gone) - then no new streams are spawned this call.
        ThreadPool * thread_pool = {};
        // Cloud volume textures still arrive via the AssetProcessor queue (their load path is not yet
        // ported); gltf meshes/textures now go straight into the manifest via add_mesh/add_texture.
        std::span<const AssetProcessor::LoadedTextureInfo> uploaded_textures = {};
    };
    // Streams in newly added textures, collects finished streams (publishing their runtime image + re-
    // marking referencing materials dirty), then records the GPU manifest updates for the frame.
    auto update_scene(UpdateSceneInfo const & info) -> daxa::ExecutableCommandList;

    auto create_mesh_acceleration_structures() -> daxa::ExecutableCommandList;
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
};