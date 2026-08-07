#pragma once

#include <array>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../../timberdoodle.hpp"
#include "../../multithreading/thread_pool.hpp"
#include "importer_task_result.hpp"
using namespace tido::types;

// Which slot of which published batch a piece of work belongs to. Purely importer-internal: manifest indices
// only exist once a batch has been applied, so a cook is dispatched and reported against this and resolved to
// a manifest entry before anything engine-side sees it.
struct SourceSlot
{
    u32 source_index = {};
    // Placing one source twice repeats every slot index, so this is what pairs work with its own entries.
    u32 batch_id = {};
    // Indexes the publish's cook request list, not anything in the batch.
    u32 slot_index = {};
};

// How to cook one image out of a standalone source. The file carries none of this, so a recipe is authored
// data - supplied by whoever asks for the import, from the project document once that exists. A source
// imported without one falls back to the backend's default.
struct ImageSlotRecipe
{
    std::vector<u8> channel_mapping = {};
    daxa::Format target_format = {};
};

/// --- Cook work ---
/// Cook work travels beside a batch, never inside it: the engine has no use for it and no way to interpret
/// it. A request is split at dispatch - the importer data goes into the cook task, the target stays behind on
/// the registry row, because a finished cook carries only a SourceSlot and nothing else says what its
/// artifact is for.

// One material texture binding a finished image is for. The material is named by its index in the batch that
// created it, because a cook is queued before the engine has said where that batch landed.
struct MaterialTextureConsumer
{
    u32 batch_material_index = {};
    MaterialTextureSlot slot = {};
};

struct ImageCookTarget
{
    std::string name = {};
    // The batch element this artifact belongs to, when the batch created its entry up front. Absent when the
    // cook's own batch creates the entry instead, which is the material-bound case.
    std::optional<u32> batch_image_index = {};
    std::vector<MaterialTextureConsumer> consumers = {};
};

struct ImageCookRequest
{
    ImageCookTarget target = {};
    std::variant<ImageImporterData, VdbImporterData> importer_data = {};
};

struct MeshCookRequest
{
    // Always set: a mesh group holds a contiguous range of mesh entries, so they exist from the parse.
    u32 batch_mesh_index = {};
    MeshImporterData importer_data = {};
};

// What a backend produces: the modifications the engine should make, and the cook work that will produce the
// artifacts they are still missing.
struct SourceImportResult
{
    ImporterTaskResult::SceneMetadataBatch batch = {};
    std::vector<ImageCookRequest> image_cooks = {};
    std::vector<MeshCookRequest> mesh_cooks = {};
};

struct ImporterTask
{
    struct ImportSource
    {
        std::filesystem::path path = {};
        std::optional<ImageSlotRecipe> image_recipe = {};
    };

    struct CookImage
    {
        std::variant<ImageImporterData, VdbImporterData> importer_data = {};
        SourceSlot slot = {};
    };

    struct CookMesh
    {
        MeshImporterData importer_data = {};
        SourceSlot slot = {};
    };

    // A backend's finished import, on its way to the engine. Routed through the queue because a backend may
    // build it on a pool worker, while binding stand-ins and queueing its cooks reads importer-thread state.
    struct PublishImport
    {
        SourceImportResult import_result = {};
    };

    // The engine reporting where a published batch's elements landed. Routed through the task queue like
    // everything else, so the registry it updates stays importer-thread-only.
    struct BatchApplied
    {
        ImporterTaskResult::AppliedBatch applied = {};
    };

    // A finished cook on its way back through the importer thread, which is the only place that can turn a
    // slot into a manifest entry. Never leaves the importer.
    struct CookFinished
    {
        std::variant<ImageStreamerData, MeshStreamerData> streamer_data = {};
        SourceSlot slot = {};
        // A cook that produced no artifact still reports back, so its slot resolves and its publish can
        // retire. The streamer data is then default-constructed and carries nothing but its alternative,
        // which is what says whether slot_index names an image or a mesh cook.
        bool failed = {};
    };

    std::variant<ImportSource, CookImage, CookMesh, PublishImport, BatchApplied, CookFinished> data = {};
};

struct Importer;

/// --- The source registry ---
/// One row per source the importer has been asked to import, in-memory and rebuilt on launch: nothing about
/// resolution is persisted until the project document exists. It answers where an asset comes from - the
/// lookup a reload and the editor's own production bookkeeping both need - and holds the generation counters.
/// Importer thread only, so it takes no lock.
struct ImportedSource
{
    /// One publish of this source: what its cooks still need in order to be resolved, and where its elements
    /// landed. Not a batch - none of it can be said to the engine - and dropped once every cook has resolved
    /// and the result has arrived.
    struct PendingCooks
    {
        u32 batch_id = {};
        // The source generation at publish, compared against the row's current one to drop a cook whose
        // content has since been replaced.
        u32 generation = {};
        // Both indexed by a cook's slot_index; the only thing that says what a finished artifact is for.
        std::vector<ImageCookTarget> image_cook_targets = {};
        std::vector<u32> mesh_cook_targets = {};
        // Authored state for everything a cook will later modify. A modification carries the element whole,
        // so re-emitting one means re-emitting all of its producer-owned fields, and the batch's own copy
        // went to the engine. Their references are normalized to manifest entries once the result arrives,
        // since a batch element index means nothing in the batch that re-emits them.
        std::vector<ImporterTaskResult::SceneMetadataBatch::Material> materials = {};
        std::vector<ImporterTaskResult::SceneMetadataBatch::MeshLodGroup> mesh_lod_groups = {};
        // A rebind that has gone out naming its image as a batch element, waiting on that batch's result to
        // turn into a manifest index the row can keep. A cook for a material holding one of these parks:
        // the copy it would re-emit does not yet name the image the material is already bound to.
        struct PendingRebind
        {
            u32 batch_material_index = {};
            MaterialTextureSlot slot = {};
            u32 image_batch_id = {};
            u32 batch_image_index = {};
        };
        std::vector<PendingRebind> pending_rebinds = {};
        // Absent until the engine reports the apply, which is what a finished cook parks on.
        std::optional<ImporterTaskResult::AppliedBatch> result = {};
        u32 outstanding_cooks = {};
    };

    std::filesystem::path path = {};
    // Bumped by a re-parse and stamped onto every publish it makes, so a cook still in flight for content
    // since replaced is dropped instead of writing into entries that have been tombstoned.
    u32 generation = {};
    std::vector<PendingCooks> pending_cooks = {};
};

// One source's import, as its backend receives it: the registry row's index and generation ride along with
// the path so the slots the backend emits can name where they came from.
struct SourceImportRequest
{
    std::filesystem::path path = {};
    u32 source_index = {};
    u32 generation = {};
    std::optional<ImageSlotRecipe> image_recipe = {};
};

/// --- Source backends ---
/// A backend turns one source file into a list of slots - { location, type, recipe } each - which reach the
/// scene as a SceneMetadataBatch. A whole-file source is not a second path through this, it is a one-slot
/// source, and nothing downstream (hashing, artifact keys, cook policies, streamer) learns which backend a
/// slot came from. The table is an open list: engine-authored materials and future engine-native asset kinds
/// are added to it rather than branched on at the call site.
struct SourceBackend
{
    // Lowercase, dot included. One backend may claim several.
    std::string_view extension = {};
    void (*dispatch)(Importer & importer, SourceImportRequest const & request) = {};
};

// Null when no backend claims the path's extension, which is what request_import rejects on.
auto find_source_backend(std::filesystem::path const & path) -> SourceBackend const *;

// Parses the source and emits its N slots - a slice location per image/mesh, recipes from the material bindings.
void dispatch_gltf_source(Importer & importer, SourceImportRequest const & request);
// Emits the source's single whole-file slot; the recipes stay hardcoded until the project document carries
// authored ones.
void dispatch_standalone_image_source(Importer & importer, SourceImportRequest const & request);

// What one image cooked out of a .vdb is made of: which grids to densify, how their channels map into the
// target and what to cook them to. A .vdb carries none of this, so a recipe is authored data - supplied by
// whoever asks for the import, from the project document once that exists.
struct VdbSlotRecipe
{
    // Appended to the source's stem to name the manifest entry.
    std::string name = {};
    std::vector<std::string> grid_names = {};
    std::vector<u8> channel_mapping = {};
    daxa::Format target_format = {};
};

// One image slot per recipe, all reading the same whole .vdb, plus the synthetic root every batch carries.
// Knows nothing about what the volumes are for; a caller wanting them grouped into something adds that to the
// batch itself, which is why this returns the import result instead of pushing it.
auto build_vdb_source_import(SourceImportRequest const & request, std::span<VdbSlotRecipe const> recipes)
    -> SourceImportResult;

// PROVISIONAL: takes any .vdb to be one cloud volume, hardcoding the three recipes, their grouping and the
// placement. It exists because none of that can be authored yet. Once a cloud material lets a cloud be an
// ordinary entity referencing three cooked volumes, this goes away and .vdb claims a plain recipe-driven
// import instead.
void dispatch_cloud_volume_source(Importer & importer, SourceImportRequest const & request);

/// --- Editor placeholders ---
/// What a material texture slot samples until its own image is cooked. They are ordinary imported images -
/// same sources, same recipes, same content-addressed cook - and the only thing that makes them placeholders
/// is that the editor remembers their manifest entries and binds pending slots to them. Nothing engine-side
/// knows the word: it only ever sees a material bound to an image, and later bound to a different one.
///
/// They are editor state. A game ships pre-cooked binaries and never cooks, so an artifact it cannot stream
/// is an error rather than a state to draw around, and it needs none of this.

struct Importer
{
    explicit Importer(ThreadPool * thread_pool);
    ~Importer();

    // Signals the orchestration thread to exit and joins it; still-queued tasks are dropped. Idempotent.
    // Must run before ~ThreadPool so nothing new is dispatched into a joining pool; the pool's own join
    // then completes the in-flight parse/cook/cache-write tasks while this Importer is still alive.
    void stop();

    // Pushes a group of tasks under one lock, so a single drain of the orchestration loop sees them
    // together (a source's asset tasks then batch into a single parse). Callable from any thread.
    void push_tasks(std::span<ImporterTask> tasks);
    // Moves out every result produced so far; SceneRuntime drains this once per frame.
    auto pop_results() -> std::vector<ImporterTaskResult>;

    // Registers `path` as an imported source, or finds the row it already has, and returns what its backend
    // needs to resolve it. Importer thread only.
    auto register_source_import(std::filesystem::path const & path) -> SourceImportRequest;

    // How every backend emits its result. Queues it for the importer thread, which binds each cook's consumer
    // slots to their stand-ins, publishes the batch and queues the cooks - so what reaches the engine is
    // metadata only, and the cooks are queued here rather than round-tripping through the scene for a
    // manifest index. Callable from any thread, which is why it defers rather than doing the work inline.
    void publish_source_import(SourceImportResult import_result);

    // Worker-side API, called from the ThreadPool tasks the importers dispatch:
    void push_result(ImporterTaskResult result);
    // A finished cook, which cannot be reported to the engine as it stands: only the importer thread can
    // turn its slot into a manifest entry, so this routes through the task queue rather than the results.
    void push_cook_finished(std::variant<ImageStreamerData, MeshStreamerData> streamer_data, SourceSlot slot);
    // A cook that produced no artifact. Reporting it is not optional: a slot that never resolves leaves its
    // publish outstanding forever, and a placeholder whose entry never appears holds every later import. The
    // type parameter carries no data - it only says which of the publish's cook lists the slot indexes.
    template<typename StreamerData>
    void push_cook_failed(SourceSlot slot)
    {
        push_cook_result(std::variant<ImageStreamerData, MeshStreamerData>{std::in_place_type<StreamerData>}, slot, true);
    }

    ThreadPool * thread_pool = {};

  private:
    std::mutex _queue_mutex = {};
    std::condition_variable _wake_signal = {};
    // All three guarded by _queue_mutex.
    std::vector<ImporterTask> _task_queue = {};
    bool _stop_requested = false;

    std::mutex _result_queue_mutex = {};
    std::vector<ImporterTaskResult> _result_queue = {};

    // Both importer-thread-only, and never serialized. A path already registered keeps its row, so importing
    // it twice resolves to one source - the two imports still create two full sets of manifest entries.
    std::vector<ImportedSource> _sources = {};
    std::unordered_map<std::filesystem::path, u32> _source_indices = {};
    // Cooks that finished before the engine reported where their publish landed, which is the common case: a
    // cache hit takes microseconds while the result cannot arrive until the next frame's poll. Drained every
    // time a result lands.
    std::vector<ImporterTask::CookFinished> _parked_cooks = {};

    // Which source each placeholder slot was imported as, and the manifest entry it resolved to once its own
    // cook landed and the batch creating it came back. Sources imported before every slot is settled are held
    // in _parked_source_imports, so nothing can bind to a stand-in that does not exist.
    std::array<std::optional<u32>, s_cast<usize>(MaterialTextureSlot::COUNT)> _placeholder_source_indices = {};
    std::array<std::optional<u32>, s_cast<usize>(MaterialTextureSlot::COUNT)> _placeholder_manifest_indices = {};
    // A slot whose placeholder cannot be imported at all. Settled but unusable: bindings simply stay empty,
    // which is far better than holding every import for the rest of the process.
    std::array<bool, s_cast<usize>(MaterialTextureSlot::COUNT)> _placeholder_failed = {};
    std::vector<ImporterTask> _parked_source_imports = {};
    // Distinguishes two publishes, whose slots are otherwise identical.
    u32 _next_batch_id = {};

    std::thread _thread = {};
    void thread_main();
    void push_cook_result(std::variant<ImageStreamerData, MeshStreamerData> streamer_data, SourceSlot slot, bool failed);
    // Binds stand-ins, hands the batch to the engine and queues the cooks. Importer thread only.
    void publish_import_on_importer_thread(SourceImportResult import_result);
    // Hands every source to the backend claiming its extension, holding back anything that is not itself a
    // placeholder until the placeholder entries exist.
    void dispatch_source_imports(std::vector<ImporterTask> & tasks);
    // Imports the editor's placeholder set. Runs once, before any other source is allowed through.
    void request_placeholder_imports();
    auto placeholders_ready() const -> bool;
    // Settles a slot's placeholder as unusable so it stops holding every later import.
    void fail_placeholder_slot(MaterialTextureSlot slot, std::string_view reason);
    void release_parked_source_imports();
    // The material texture slot `source_index` was imported as the placeholder for, if it is one at all.
    auto placeholder_slot_for_source(u32 source_index) const -> std::optional<MaterialTextureSlot>;
    // The entry a material binds for `slot` until its own image is cooked, or nullopt where no stand-in is
    // available - which includes every 3D image, since nothing can assume what a .vdb represents.
    auto placeholder_for_slot(MaterialTextureSlot slot) const -> std::optional<u32>;
    // Finds the publish a cook belongs to, or nullptr once it has been dropped.
    auto find_pending_cooks(SourceSlot slot) -> ImportedSource::PendingCooks *;
    // Turns a finished cook into batch elements, or parks it if its publish has not been applied yet or the
    // material it rebinds still holds an unresolved one. Silently drops a cook whose generation the source
    // has moved past. Appends into `cook_batches`, keyed by source, so one drain produces one batch per
    // source however many cooks it resolves.
    void resolve_cook(ImporterTask::CookFinished cook,
        std::unordered_map<u32, ImporterTaskResult::SceneMetadataBatch> & cook_batches);
    // Drops every publish with nothing left outstanding; under reload a source republishes repeatedly. Swept
    // across every row rather than the one just touched, because the last thing a publish waits on can be a
    // failed cook, which produces no batch and so never comes back through an apply.
    void retire_settled_publishes();
    void record_applied_batch(ImporterTaskResult::AppliedBatch applied);
    // Turns every rebind waiting on `applied`'s batch into a manifest index the row can keep re-emitting.
    void resolve_pending_rebinds(u32 source_index, ImporterTaskResult::AppliedBatch const & applied);
    void publish_cook_batches(std::unordered_map<u32, ImporterTaskResult::SceneMetadataBatch> & cook_batches);
};
