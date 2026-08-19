#pragma once

#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "../timberdoodle.hpp"
#include "scene.hpp"
#include "importer_types.hpp"
using namespace tido::types;

/// --- Writing the Scene ---
/// The only way the Scene is ever written, and the whole of the engine's write vocabulary. Main thread only.
///
/// Identity belongs to the producer: it creates entries, is handed the manifest indices they were created at,
/// and names them in every later write. There is no batch, no local index space and no resolution step - an
/// index in an argument is always a manifest index. An entry is created empty and filled by the writes below,
/// which is what lets one exist before its artifact does.
///
/// This is engine-side vocabulary. The producer is the editor's Importer today and a scene document loader
/// tomorrow, and neither is named here.

// Creates `count` entries and returns the index of the first. They are contiguous, so a producer holding
// local indices needs only the base to name any of them.
auto create_images(Scene & scene, u32 count) -> u32;
auto create_mesh_lod_groups(Scene & scene, u32 count) -> u32;

void write_image_name(Scene & scene, u32 image_manifest_index, std::string name);
// Makes the entry streamable; until this lands it has no artifact and never becomes resident.
void set_image_artifact(Scene & scene, u32 image_manifest_index, ImageStreamerData streamer_data);

// Everything about a material except its textures, which are set one slot at a time - a rebind changes one
// binding and must not have to restate the others.
struct SurfaceMaterialWrite
{
    bool alpha_discard_enabled = {};
    bool double_sided = {};
    bool blend_enabled = {};
    f32vec3 base_color = {};
    f32vec3 emissive_color = {};
};

struct CloudMaterialWrite
{
    f32 albedo = {};
    f32 density_scale = {};
};

struct MaterialWrite
{
    std::variant<SurfaceMaterialWrite, CloudMaterialWrite> payload = SurfaceMaterialWrite{};
    std::string name = {};
};
auto create_materials(Scene & scene, std::span<MaterialWrite const> materials) -> u32;
void set_material_texture(Scene & scene, u32 material_manifest_index, MaterialTextureSlot slot, std::optional<MaterialManifestEntry::ImageInfo> binding);

struct MeshLodGroupWrite
{
    std::optional<u32> material_manifest_index = {};
    std::string name = {};
};
void write_mesh_lod_group(Scene & scene, u32 mesh_manifest_index, MeshLodGroupWrite mesh);
void set_mesh_artifact(Scene & scene, u32 mesh_manifest_index, MeshStreamerData streamer_data);

// A group claims a contiguous range of the mesh index array, so its members are given at creation and never
// changed.
struct MeshGroupWrite
{
    std::vector<u32> mesh_manifest_indices = {};
    std::string name = {};
};
auto create_mesh_groups(Scene & scene, std::span<MeshGroupWrite const> mesh_groups) -> u32;

struct PointLightWrite
{
    f32vec3 position = {};
    f32vec3 color = {};
    f32 intensity = {};
    f32 cutoff = {};
};
auto create_point_lights(Scene & scene, std::span<PointLightWrite const> lights) -> u32;

struct SpotLightWrite
{
    f32mat4x3 transform = {};
    f32vec3 color = {};
    f32 intensity = {};
    f32 cutoff = {};
    f32 inner_cone_angle = {};
    f32 outer_cone_angle = {};
};
auto create_spot_lights(Scene & scene, std::span<SpotLightWrite const> lights) -> u32;

/// One import's entity hierarchy, created as a unit. Entities are identified by a slotmap id rather than a
/// manifest index, so every id has to exist before the parent/child/sibling links can reference them - which
/// is why this is the one write that takes a whole subtree rather than a single element. Its links stay
/// local indices into `entities` for the same reason: a link to an entity outside the subtree is not a u32.
struct EntitySubtreeWrite
{
    struct Entity
    {
        glm::mat4x3 transform = {};
        EntityType type = EntityType::UNKNOWN;
        std::string name = {};
        std::optional<u32> mesh_group_manifest_index = {};
        std::optional<u32> material_manifest_index = {};
        std::optional<u32> light_index = {};
        std::optional<u32> parent_index = {};
        std::optional<u32> first_child_index = {};
        std::optional<u32> next_sibling_index = {};
    };

    std::vector<Entity> entities = {};
    // Index into `entities` of the synthetic subtree root that parents every parentless node.
    u32 root_entity_index = {};
};
// Returns the created ids in `entities` order, so a local index maps to the entity it became. Entities are
// slotmap ids rather than a contiguous run, so this is the only handle a producer gets on what it created.
auto create_entity_subtree(Scene & scene, EntitySubtreeWrite subtree) -> std::vector<RenderEntityId>;
