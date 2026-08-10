#pragma once

#include <array>
#include <optional>
#include <unordered_map>
#include <vector>

#include "../../timberdoodle.hpp"
#include "../../multithreading/thread_pool.hpp"
#include "../importer_types.hpp"
#include "../scene_write.hpp"
#include "cook.hpp"
using namespace tido::types;

// The recipe describes how to interpret and cook the data stored in the file.
// For images this is just the target format we wish the cooked image to have and how individual channes of the source map to the target channels.
struct ImageSlotRecipe
{
    std::string name = {};
    std::vector<u8> channel_mapping = {};
    daxa::Format target_format = {};
};

// For VDBs we extract a set of grids and then perform channel mapping by selecting from these grids.
struct VdbSlotRecipe
{
    // Appended to the source's stem to name the manifest entry.
    std::string name = {};
    std::vector<std::string> grid_names = {};
    std::vector<u8> channel_mapping = {};
    daxa::Format target_format = {};
};

using SlotRecipe = std::variant<ImageSlotRecipe, VdbSlotRecipe>;

struct ImageMaterialBinding
{
    // Local or global index.
    u32 material_index = {};
    MaterialTextureSlot slot = {};
};

struct ParsedSource
{
    struct Image
    {
        std::string name = {};
        std::vector<ImageMaterialBinding> bound_materials = {};
    };

    struct Mesh
    {
        std::optional<u32> local_material_index = {};
        std::string name = {};
    };

    struct MeshGroup
    {
        std::vector<u32> local_mesh_indices = {};
        std::string name = {};
    };

    struct Entity
    {
        glm::mat4x3 transform = {};
        EntityType type = EntityType::UNKNOWN;
        std::string name = {};
        std::optional<u32> local_mesh_group_index = {};
        // Which light list this indexes is decided by entity `type`.
        std::optional<u32> local_light_index = {};
        std::optional<u32> parent_index = {};
        std::optional<u32> first_child_index = {};
        std::optional<u32> next_sibling_index = {};
    };

    std::vector<Image> images = {};
    std::vector<Mesh> meshes = {};

    /// --- Cook queue ---
    /// One cook per image/mesh slot, indexed alongside `images` and `meshes` rather than held inside them:
    /// this is the one part of a parse the registry row keeps, so keeping it in its own run lets publish move
    /// the whole thing across instead of lifting it out element by element. Publish asserts the lengths agree.
    std::vector<std::variant<ImageImporterData, VdbImporterData>> image_cook_inputs = {};
    std::vector<MeshImporterData> mesh_cook_inputs = {};

    // Stores only the basic material properties, the images remember which materials bind them.
    // This is done so that when an image fininshes cooking it can update all the materials that depend on it.
    std::vector<MaterialWrite> materials = {};
    std::vector<MeshGroup> mesh_groups = {};
    std::vector<PointLightWrite> point_lights = {};
    std::vector<SpotLightWrite> spot_lights = {};

    std::vector<Entity> entities = {};

    u32 source_index = {};
};

// A common base for all source parse tasks.
struct SourceParseTask : Task
{
    ParsedSource parsed = {};
    bool failed = {};
};

struct ManifestRange
{
    u32 base = {};
    u32 count = {};
};

// Imported source represents one logical source file and everything that has been imported from it.
// An imported source would for example be a glTF file or a VDB file containing three grids.
struct ImportedSource
{
    std::filesystem::path path = {};
    // Increased when the source is reloaded - used to track the version of the source.
    u32 generation = {};

    // Ranges of manifest entries that this source created and thus owns.
    ManifestRange images = {};
    ManifestRange materials = {};
    ManifestRange mesh_lod_groups = {};
    ManifestRange mesh_groups = {};
    ManifestRange point_lights = {};
    ManifestRange spot_lights = {};

    // Entities are slotmap ids rather than a contiguous run, so they are listed rather than ranged. In the
    // order the parse emitted them, so a local index maps to the entity it became.
    std::vector<RenderEntityId> entities = {};

    /// --- What this source cooks ---
    /// The cook input per slot, kept because it is the only thing a publish produces that no manifest entry
    /// holds. Re-cooking one slot - after a recipe edit or a cook version bump - reads it straight from here
    /// rather than re-parsing the source it came from.
    std::vector<std::variant<ImageImporterData, VdbImporterData>> image_cook_inputs = {};
    std::vector<std::vector<ImageMaterialBinding>> image_bindings = {};

    std::vector<MeshImporterData> mesh_cook_inputs = {};

    // Indexed by an image cook's slot_index.
    // How much of this source is still cooking. Nothing reads it yet - it is what a reload will check before
    // replacing entries a cook still in flight is going to write.
    u32 outstanding_cooks = {};
};

// One source's import, as its backend receives it: the registry row's index rides along with the path so the
// slots the backend emits can name where they came from.
struct SourceImportRequest
{
    std::filesystem::path path = {};
    u32 source_index = {};
    std::vector<SlotRecipe> recipes = {};
};

/// --- Source backends ---
/// A backend turns one source file into a list of slots - { location, type, recipe } each. A whole-file
/// source is not a second path through this, it is a one-slot source, and nothing downstream (hashing,
/// artifact keys, cook policies, streamer) learns which backend a slot came from. The table that maps an
/// extension to one is importer-private; these declarations exist only so it can name them across
/// translation units.

// Each returns the parse undispatched, so how it runs stays the Importer's decision.

// Parses the source and emits its N slots - a slice location per image/mesh, recipes from the material bindings.
auto parse_gltf_source(SourceImportRequest const & request) -> std::shared_ptr<SourceParseTask>;
// Emits one whole-file slot per recipe - the same file cooked as many ways as it is asked for.
auto parse_image_source(SourceImportRequest const & request) -> std::shared_ptr<SourceParseTask>;
// Emits one image slot per recipe, all reading the same whole .vdb.
auto parse_vdb_source(SourceImportRequest const & request) -> std::shared_ptr<SourceParseTask>;

/// --- Editor placeholders ---
/// What a material texture slot samples until its own image is cooked. They are ordinary imported images -
/// same sources, same recipes, same content-addressed cook - and the only thing that makes them placeholders
/// is that the editor remembers their manifest entries and binds pending slots to them. Nothing engine-side
/// knows the word: it only ever sees a material bound to an image, and later bound to a different one.
///
/// The set is imported, published and cooked before the Importer constructor returns, so every material
/// published afterwards has a stand-in with an artifact behind it. A placeholder whose cook fails leaves an
/// entry that never becomes resident, which resolves to id 0 and reads as "this material has no such texture".
///
/// They are editor state. A game ships pre-cooked binaries and never cooks, so an artifact it cannot stream
/// is an error rather than a state to draw around, and it needs none of this.

/// --- The Importer ---
/// The Editor layer, and the only thing that writes the Scene. It holds the source registry and the cook
/// queue, runs entirely on the main thread, and drives the engine rather than being called by it: nothing in
/// SceneRuntime or Scene knows it exists, so a build with no importer linked is the same engine code with
/// nothing on the production side.
///
/// The pool is where the work is. Parses and cooks are dispatched to it and report back through one inbox;
/// everything the Importer itself does is bookkeeping against state only it touches.
struct Importer
{
    // Blocks until the editor's stand-in images are cooked, so a material published later always has one.
    Importer(ThreadPool * thread_pool, Scene & scene);
    ~Importer();

    void request_import(std::filesystem::path const & path, std::vector<SlotRecipe> recipes = {});
    void tick(Scene & scene);

    ThreadPool * thread_pool = {};

  private:
    std::vector<std::shared_ptr<SourceParseTask>> _inflight_parses = {};
    std::vector<std::shared_ptr<CookTask>> _inflight_cooks = {};

    std::vector<ImportedSource> _sources = {};
    std::unordered_map<std::filesystem::path, u32> _source_indices = {};

    // The entry each placeholder import created, which is what every pending material slot binds. Absent
    // where no stand-in is available - which includes every 3D image, since nothing can assume what a
    // .vdb represents.
    std::array<std::optional<u32>, s_cast<usize>(MaterialTextureSlot::COUNT)> _placeholder_manifest_indices = {};

    // Creates the import's entries, binds every pending texture slot to its stand-in, and dispatches the cooks.
    void publish_import(Scene & scene, ParsedSource parsed);
    // Dispatches every one of a source's cooks from the inputs on its row. Meshes first: they gate what can
    // be drawn at all, while a pending image only costs a material its stand-in.
    void dispatch_cooks(u32 source_index);
    // Writes a finished cook's artifact into the entry it belongs to and rebinds every slot waiting on it.
    void resolve_cook(Scene & scene, CookTask & cook);
};
