#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "../../timberdoodle.hpp"
#include "../scene.hpp"
#include "../tido_format/tido_format.hpp"
using namespace tido::types;

/// --- Importer task results ---
/// What an importer hands back to SceneRuntime instead of touching the Scene directly. An Import-scene
/// task (parse) emits a SceneMetadataBatch; an Import-asset task (cook) emits a CookedAsset; a failed
/// task emits an Error. SceneRuntime drains a thread-safe queue of these each frame and applies them to
/// the Scene's manifests on the main thread - the only place the Scene is ever written.
struct ImporterTaskResult
{
    struct SceneMetadataBatch
    {
        struct Image
        {
            std::string name = {};
            // The slot's recipe, whose alternative is also its type tag: encoded 2D image bytes, or grids
            // densified out of a .vdb. Both cook into the image manifest.
            std::variant<ImageImporterData, VdbImporterData> importer_data = {};
        };

        struct Material
        {
            std::optional<MaterialManifestEntry::ImageInfo> diffuse_info = {};
            std::optional<MaterialManifestEntry::ImageInfo> opacity_mask_info = {};
            std::optional<MaterialManifestEntry::ImageInfo> normal_info = {};
            std::optional<MaterialManifestEntry::ImageInfo> roughness_metalness_info = {};
            bool alpha_discard_enabled = {};
            bool double_sided = {};
            bool blend_enabled = {};
            f32vec3 base_color = {};
            f32vec3 emissive_color = {};
            std::string name = {};
        };

        struct MeshLodGroup
        {
            std::optional<u32> material_index = {};
            std::string name = {};
            MeshImporterData importer_data = {};
        };

        struct MeshGroup
        {
            std::vector<u32> mesh_lod_group_indices = {};
            std::string name = {};
        };

        struct PointLight
        {
            f32vec3 position = {};
            f32vec3 color = {};
            f32 intensity = {};
            f32 cutoff = {};
        };

        struct SpotLight
        {
            f32mat4x3 transform = {};
            f32vec3 color = {};
            f32 intensity = {};
            f32 cutoff = {};
            f32 inner_cone_angle = {};
            f32 outer_cone_angle = {};
        };

        // Indices into this batch's `images`; the three volumes a cloud entity samples.
        struct CloudVolume
        {
            u32 data_image_index = {};
            u32 sdf_image_index = {};
            u32 detail_noise_image_index = {};
        };

        struct Entity
        {
            glm::mat4x3 transform = {};
            EntityType type = EntityType::UNKNOWN;
            std::string name = {};
            std::optional<u32> mesh_group_manifest_index = {};
            std::optional<u32> cloud_volume_index = {};
            std::optional<u32> light_index = {};
            std::optional<u32> parent_index = {};
            std::optional<u32> first_child_index = {};
            std::optional<u32> next_sibling_index = {};
        };

        std::vector<Image> images = {};
        std::vector<Material> materials = {};
        std::vector<MeshLodGroup> mesh_lod_groups = {};
        std::vector<MeshGroup> mesh_groups = {};
        std::vector<PointLight> point_lights = {};
        std::vector<SpotLight> spot_lights = {};
        std::vector<CloudVolume> cloud_volumes = {};
        std::vector<Entity> entities = {};
        // Index into `entities` of the synthetic subtree root that parents every parentless node.
        u32 root_entity_index = {};
    };

    struct CookedAsset
    {
        std::variant<ImageStreamerData, MeshStreamerData> streamer_data = {};
        u32 manifest_index = {};
    };

    struct Error
    {
        enum struct TaskKind
        {
            IMPORT_SOURCE,
            IMPORT_ASSET,
        };
        TaskKind kind = {};
        std::filesystem::path source = {};
        std::string message = {};
    };

    std::variant<SceneMetadataBatch, CookedAsset, Error> data = {};
};
