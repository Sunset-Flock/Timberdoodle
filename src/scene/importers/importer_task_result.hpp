#pragma once

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
/// task (parse) emits a SceneMetadataBatch; an Import-asset task (cook) emits a CookedAsset.
/// SceneRuntime drains a thread-safe queue of these each frame and applies them to the Scene's
/// manifests on the main thread - the only place the Scene is ever written.
struct ImporterTaskResult
{
    // Every manifest entry describing one source file, applied as a single batch so a partially-linked
    // scene is never observable. Cross-references between entries (a material's texture, a mesh-lod-
    // group's material, an entity's mesh group / light, a mesh group's meshes) are indices local to this
    // batch, not global manifest indices - resolve them to global indices when appending each entry.
    struct SceneMetadataBatch
    {
        // A mesh group's member meshes, by index into `mesh_lod_groups`. Self-contained (unlike
        // MeshGroupManifestEntry's offset into Scene's shared indices array) since the batch has no such
        // array of its own yet.
        struct MeshGroup
        {
            std::vector<u32> mesh_lod_group_indices = {};
            std::string name = {};
        };

        // One imported node. `entity` carries every field except the tree links, which are indices into
        // this batch's own `entities` (RenderEntityId cannot be minted off the main thread, since it names
        // a slot in Scene's slotmap).
        struct Entity
        {
            RenderEntity entity = {};
            std::optional<u32> parent_index = {};
            std::optional<u32> first_child_index = {};
            std::optional<u32> next_sibling_index = {};
        };

        std::vector<TextureManifestEntry> textures = {};
        std::vector<MaterialManifestEntry> materials = {};
        std::vector<MeshLodGroupManifestEntry> mesh_lod_groups = {};
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

    std::variant<SceneMetadataBatch, CookedAsset> data = {};
};
