#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "../../timberdoodle.hpp"
#include "../scene.hpp"
#include "../tido_format/tido_mesh.hpp"
#include "../tido_format/tido_texture.hpp"
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
        struct Texture
        {
            TextureMaterialType type = {};
            std::string name = {};
            std::variant<TextureManifestEntry::GltfImporterData, TextureManifestEntry::RawImporterData> importer_data = {};
        };

        struct Material
        {
            std::optional<MaterialManifestEntry::TextureInfo> diffuse_info = {};
            std::optional<MaterialManifestEntry::TextureInfo> opacity_mask_info = {};
            std::optional<MaterialManifestEntry::TextureInfo> normal_info = {};
            std::optional<MaterialManifestEntry::TextureInfo> roughness_metalness_info = {};
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
            std::variant<MeshLodGroupManifestEntry::GltfImporterData, MeshLodGroupManifestEntry::RawImporterData> importer_data = {};
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

        struct Entity
        {
            glm::mat4x3 transform = {};
            EntityType type = EntityType::UNKNOWN;
            std::string name = {};
            std::optional<u32> mesh_group_manifest_index = {};
            std::optional<u32> light_index = {};
            std::optional<u32> parent_index = {};
            std::optional<u32> first_child_index = {};
            std::optional<u32> next_sibling_index = {};
        };

        std::vector<Texture> textures = {};
        std::vector<Material> materials = {};
        std::vector<MeshLodGroup> mesh_lod_groups = {};
        std::vector<MeshGroup> mesh_groups = {};
        std::vector<PointLight> point_lights = {};
        std::vector<SpotLight> spot_lights = {};
        std::vector<Entity> entities = {};
        // Index into `entities` of the synthetic subtree root that parents every parentless node.
        u32 root_entity_index = {};
    };

    struct CookedAsset
    {
        std::variant<TidoTextureStreamerData, TidoMeshStreamerData> streamer_data = {};
        u32 manifest_index = {};
    };

    // Generic failure scaffolding: which task kind failed for which source, with a log-friendly reason.
    // A proper error taxonomy (specific codes + recovery) is a later pass; SceneRuntime only logs these
    // and clears its pending-import state for a failed ImportScene.
    struct Error
    {
        enum struct TaskKind
        {
            IMPORT_SCENE,
            IMPORT_ASSET,
        };
        TaskKind kind = {};
        std::filesystem::path source = {};
        std::string message = {};
    };

    std::variant<SceneMetadataBatch, CookedAsset, Error> data = {};
};
