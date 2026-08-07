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
/// What an importer hands back to SceneRuntime instead of touching the Scene directly. Every result is a
/// SceneMetadataBatch - a parse emits one describing the entries an import creates, and each finished cook
/// emits one modifying the entries its artifact belongs to; a failed task emits an Error. SceneRuntime drains
/// a thread-safe queue of these each frame and applies them to the Scene's manifests on the main thread - the
/// only place the Scene is ever written.
struct ImporterTaskResult
{
    /// A batch is the list of modifications SceneRuntime makes to the manifests. Every element either creates
    /// an entry or modifies one: an absent manifest_index creates, a present one names the entry to modify.
    /// A modification carries the element's whole producer-owned state - there are no partial updates - and
    /// leaves what the engine owns (residency, back-links, mesh group membership) untouched. Every reference
    /// one element makes to another says whether it means an element of this batch or an entry that already
    /// exists; an element's own identity is its manifest_index and is never a reference.
    struct SceneMetadataBatch
    {
        struct SceneRef
        {
            enum struct Kind
            {
                BATCH_ELEMENT,
                MANIFEST_ENTRY,
            };
            Kind kind = {};
            // A BATCH_ELEMENT resolves to whatever the element it names resolved to, which holds even when
            // that element is itself a modification.
            u32 index = {};
        };

        struct TextureBinding
        {
            SceneRef image = {};
            u32 sampler_index = {};
        };

        struct Image
        {
            std::optional<u32> manifest_index = {};
            std::string name = {};
            // Absent while nothing has cooked this entry yet; a later modification supplies it. A volume is
            // created that way because entities reference it from the parse, while a material-bound image is
            // created already cooked - rebinding the materials sampling it is what lets it wait without
            // anything drawing a hole.
            std::optional<ImageStreamerData> streamer_data = {};
        };

        // A material is created bound to the stand-ins its producer already had entries for, and is modified
        // onto its own image once that is cooked - so a rebind names the image the same batch is creating
        // while the slots it leaves alone still name their stand-ins. A slot with no texture, or whose
        // stand-in is unavailable, stays empty until a modification introduces it.
        struct Material
        {
            std::optional<u32> manifest_index = {};
            std::optional<TextureBinding> diffuse_info = {};
            std::optional<TextureBinding> opacity_mask_info = {};
            std::optional<TextureBinding> normal_info = {};
            std::optional<TextureBinding> roughness_metalness_info = {};
            bool alpha_discard_enabled = {};
            bool double_sided = {};
            bool blend_enabled = {};
            f32vec3 base_color = {};
            f32vec3 emissive_color = {};
            std::string name = {};
        };

        struct MeshLodGroup
        {
            std::optional<u32> manifest_index = {};
            std::optional<SceneRef> material = {};
            std::string name = {};
            // Absent on creation and supplied by a later modification: a mesh group holds a contiguous range
            // of mesh entries, so they have to exist from the parse, long before anything has cooked them.
            std::optional<MeshStreamerData> streamer_data = {};
        };

        struct MeshGroup
        {
            std::vector<SceneRef> mesh_lod_groups = {};
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

        // The three volumes a cloud entity samples.
        struct CloudVolume
        {
            SceneRef data_image = {};
            SceneRef sdf_image = {};
            SceneRef detail_noise_image = {};
        };

        struct Entity
        {
            glm::mat4x3 transform = {};
            EntityType type = EntityType::UNKNOWN;
            std::string name = {};
            std::optional<SceneRef> mesh_group = {};
            std::optional<SceneRef> cloud_volume = {};
            std::optional<SceneRef> light = {};
            // Entities are identified by a slotmap id rather than a manifest index, so a link to one outside
            // this batch is not a u32 and cannot be a SceneRef. Linking to an existing entity - reparenting -
            // needs its own reference type and has no producer yet.
            std::optional<u32> parent_index = {};
            std::optional<u32> first_child_index = {};
            std::optional<u32> next_sibling_index = {};
        };

        // One import's entity hierarchy, applied as a unit. Absent on a batch that only touches the
        // manifests, which is every batch a finished cook produces.
        struct EntitySubtree
        {
            std::vector<Entity> entities = {};
            // Index into `entities` of the synthetic subtree root that parents every parentless node.
            u32 root_entity_index = {};
        };

        std::vector<Image> images = {};
        std::vector<Material> materials = {};
        std::vector<MeshLodGroup> mesh_lod_groups = {};
        std::vector<MeshGroup> mesh_groups = {};
        std::vector<PointLight> point_lights = {};
        std::vector<SpotLight> spot_lights = {};
        std::vector<CloudVolume> cloud_volumes = {};
        std::optional<EntitySubtree> entity_subtree = {};
        // Identify this batch to whoever produced it. The engine echoes them back untouched and never
        // interprets them. `source_index` routes the result to its registry row without a separate index;
        // the generation it was published at lives on that row rather than on the wire, where it could go
        // stale against the registry.
        u32 source_index = {};
        u32 batch_id = {};
    };

    // Where a batch's newly created entries landed, handed back so the producer can name them in a later
    // batch's modifications. Elements the batch modified are absent - those already carried their index. The
    // importer is the only consumer today; a game or editor layer wanting to reference what it imported is
    // the reason this is a result rather than an importer-private callback.
    struct AppliedBatch
    {
        // Echoed back from the batch untouched, identifying it to whoever produced it.
        u32 source_index = {};
        u32 batch_id = {};
        // Indexed by the element's index in the batch, holding the manifest index it was created at. An
        // element the batch modified instead of creating holds INVALID_MANIFEST_INDEX.
        std::vector<u32> image_manifest_indices = {};
        std::vector<u32> material_manifest_indices = {};
        std::vector<u32> mesh_manifest_indices = {};
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

    std::variant<SceneMetadataBatch, Error> data = {};
};
