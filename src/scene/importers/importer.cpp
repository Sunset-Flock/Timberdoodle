#include "importer.hpp"

#include <algorithm>
#include <optional>

#include <fmt/format.h>

#include "../tido_format/tido_util.hpp"

namespace
{

struct SourceBackend
{
    // Lowercase, dot included. One backend may claim several.
    std::string_view extension = {};
    std::shared_ptr<SourceParseTask> (*parse)(SourceImportRequest const & request);
};

// The editor's own assets, deliberately outside the Tido Assets root: the sandbox constrains paths a user
// picks, and these are neither picked nor scene content. They are still imported and cooked like anything
// else, so editing one re-cooks it and a cook version bump reaches it.
struct PlaceholderSource
{
    MaterialTextureSlot slot = {};
    char const * path = {};
    ImageSlotRecipe recipe = {};
};

// The fills are neutral so a pending asset reads as plausibly lit rather than announcing itself. Each recipe
// matches the one the real asset in that slot is cooked with, so a stand-in and the artifact replacing it
// are the same kind of image.
// Only the slots a 2D stand-in makes sense for, so this is a list of the placeholders that exist rather than
// one entry per slot. A slot with none binds nothing until its own artifact arrives.
auto placeholder_sources() -> std::span<PlaceholderSource const>
{
    static std::array<PlaceholderSource, 4> const SOURCES = {
        // Mid grey once the sRGB target decodes it.
        PlaceholderSource{MaterialTextureSlot::DIFFUSE, "editor/placeholders/placeholder_diffuse.png",
            {.channel_mapping = {0, 1, 2}, .target_format = daxa::Format::BC1_RGB_SRGB_BLOCK}},
        // Fully opaque, so nothing is alpha-discarded while it is pending.
        PlaceholderSource{MaterialTextureSlot::OPACITY, "editor/placeholders/placeholder_opacity.png",
            {.channel_mapping = {0}, .target_format = daxa::Format::BC4_UNORM_BLOCK}},
        // Flat +Z. Cooked to a non-sRGB target, so the source stays linear and 128 means 0.5.
        PlaceholderSource{MaterialTextureSlot::NORMAL, "editor/placeholders/placeholder_normal.png",
            {.channel_mapping = {0, 1}, .target_format = daxa::Format::BC5_UNORM_BLOCK}},
        // glTF channel order: mid rough, fully dielectric.
        PlaceholderSource{MaterialTextureSlot::ROUGHNESS_METALNESS, "editor/placeholders/placeholder_rough_metal.png",
            {.channel_mapping = {0, 1, 2, 3}, .target_format = daxa::Format::BC7_UNORM_BLOCK}},
    };
    return SOURCES;
}
} // namespace

Importer::Importer(ThreadPool * thread_pool, Scene & scene)
    : thread_pool{thread_pool}
{
    // Hardcoded placeholder sources - parsed and cooked inline so that they are immediately available.
    for (PlaceholderSource const & source : placeholder_sources())
    {
        auto const [registered, inserted] = _source_indices.try_emplace(source.path, s_cast<u32>(_sources.size()));
        DBG_ASSERT_TRUE_M(inserted, "placeholder source path is not unique");
        _sources.push_back(ImportedSource{.path = source.path});

        u32 const source_index = registered->second;
        auto const parse_task = parse_image_source(SourceImportRequest{ .path = source.path, .source_index = source_index, .recipes = {SlotRecipe{source.recipe}}, });
        thread_pool->blocking_dispatch(parse_task);
        if(parse_task->failed)
        {
            DEBUG_MSG(fmt::format("[Importer] placeholder source parse failed: '{}'", source.path));
            continue;
        }
        // Creates the entries and dispatches the one cook this source has.
        publish_import(scene, std::move(parse_task->parsed));

        ImportedSource const & placeholder_source = _sources.at(source_index);
        DBG_ASSERT_TRUE_M(placeholder_source.images.count == 1,
            fmt::format("placeholder source '{}' parse did not produce exactly one image (produced {} instead)", source.path, placeholder_source.images.count));
        _placeholder_manifest_indices[s_cast<usize>(source.slot)] = placeholder_source.images.base;
    }

    for(auto & cook : _inflight_cooks)
    {
        thread_pool->block_on(cook);
        DBG_ASSERT_TRUE_M(!std::holds_alternative<std::monostate>(cook->streamer_data),
            fmt::format("placeholder source index {} cook failed", cook->identifier.source_index));
    }
}

Importer::~Importer() = default;

auto Importer::request_import(std::filesystem::path const & path, std::vector<SlotRecipe> recipes) -> std::optional<u32>
{
    constexpr std::array<SourceBackend, 5> SOURCE_BACKENDS = {
        SourceBackend{".gltf", parse_gltf_source},
        SourceBackend{".glb",  parse_gltf_source},
        SourceBackend{".png",  parse_image_source},
        SourceBackend{".ktx2", parse_image_source},
        SourceBackend{".vdb",  parse_vdb_source},
    };

    if (!path.has_filename() || !path.has_parent_path()) { return std::nullopt; }

    // Don't allow importing outside the Tido Assets root.
    // TODO(msaky): This is here so that the cache is transferrable so we can key assets by their relative path.
    //              Remove/Rething this once we have a proper Tido project file.
    if (!tido_relative_to_assets_root(path).has_value())
    {
        DEBUG_MSG(fmt::format("[WARN][Importer::request_import] '{}' is outside the Tido Assets root '{}' - rejected",
            path.string(), TIDO_ASSETS_ROOT.string()));
        return std::nullopt;
    }

    std::string const extension = tido_lowercase_extension(path);
    const auto backend_it = std::find_if(SOURCE_BACKENDS.begin(), SOURCE_BACKENDS.end(), [&](SourceBackend const & b) { return b.extension == extension; });
    if (backend_it == SOURCE_BACKENDS.end())
    {
        DEBUG_MSG(fmt::format("[WARN][Importer::request_import] no source backend handles '{}' - rejected", path.string()));
        return std::nullopt;
    }

    // One row per source, imported once. Importing a path twice would create a second full set of manifest
    // entries rather than reloading the first, so it is dropped; replacing a source's entries is reload's
    // job and needs its own entry point.
    auto const [registered, inserted] = _source_indices.try_emplace(path, s_cast<u32>(_sources.size()));
    if (!inserted)
    {
        DEBUG_MSG(fmt::format("[WARN][Importer::request_import] '{}' is already imported - dropped", path.string()));
        return std::nullopt;
    }
    _sources.push_back(ImportedSource{.path = path});
    u32 const source_index = registered->second;

    DEBUG_MSG(fmt::format("[Importer] dispatching source {} generation {} '{}'", source_index, _sources.at(source_index).generation, path.string()));

    auto task = backend_it->parse(SourceImportRequest{
        .path = path,
        .source_index = source_index,
        .recipes = std::move(recipes),
    });
    thread_pool->async_dispatch(task, TaskPriority::LOW);
    _inflight_parses.push_back(std::move(task));
    return source_index;
}

auto Importer::import_stage(u32 source_index) const -> ImportStage
{
    return _sources.at(source_index).stage;
}

auto Importer::source_images(u32 source_index) const -> ManifestRange
{
    return _sources.at(source_index).images;
}

void Importer::publish_import(Scene & scene, ParsedSource parsed)
{
    u32 const source_index = parsed.source_index;

    // Allocate manifest entry ranges.
    u32 const image_base = create_images(scene, s_cast<u32>(parsed.images.size()));
    u32 const material_base = create_materials(scene, parsed.materials);
    u32 const mesh_base = create_mesh_lod_groups(scene, s_cast<u32>(parsed.meshes.size()));
    u32 const point_light_base = create_point_lights(scene, parsed.point_lights);
    u32 const spot_light_base = create_spot_lights(scene, parsed.spot_lights);

    // Validation that all indices fall into the ranges parsed from the source.
    auto local_index_to_manifest = [](u32 base, u32 local_index, [[maybe_unused]] usize local_count) -> u32
    {
        DBG_ASSERT_TRUE_M(local_index < local_count, "Parsed source references an element it did not emit");
        return base + local_index;
    };

    for (u32 local_mesh_index = 0; local_mesh_index < s_cast<u32>(parsed.meshes.size()); ++local_mesh_index)
    {
        ParsedSource::Mesh & mesh = parsed.meshes[local_mesh_index];

        std::optional<u32> const material_index = mesh.local_material_index.has_value()
            ? std::optional{local_index_to_manifest(material_base, mesh.local_material_index.value(), parsed.materials.size())} : std::nullopt;

        MeshLodGroupWrite const mesh_write = MeshLodGroupWrite{
            .material_manifest_index = material_index,
            .name = std::move(mesh.name),
        };
        write_mesh_lod_group(scene, mesh_base + local_mesh_index, mesh_write);
    }

    ImportedSource & source = _sources.at(source_index);
    source.images = {.base = image_base, .count = s_cast<u32>(parsed.images.size())};
    source.materials = {.base = material_base, .count = s_cast<u32>(parsed.materials.size())};
    source.mesh_lod_groups = {.base = mesh_base, .count = s_cast<u32>(parsed.meshes.size())};
    source.point_lights = {.base = point_light_base, .count = s_cast<u32>(parsed.point_lights.size())};
    source.spot_lights = {.base = spot_light_base, .count = s_cast<u32>(parsed.spot_lights.size())};

    DBG_ASSERT_TRUE_M(parsed.image_cook_inputs.size() == parsed.images.size(), "Parsed source has a cook per image slot");
    DBG_ASSERT_TRUE_M(parsed.mesh_cook_inputs.size() == parsed.meshes.size(), "Parsed source has a cook per mesh slot");
    source.image_cook_inputs = std::move(parsed.image_cook_inputs);
    source.mesh_cook_inputs = std::move(parsed.mesh_cook_inputs);

    source.image_bindings.clear();
    source.image_bindings.reserve(parsed.images.size());
    for (u32 local_index = 0; local_index < s_cast<u32>(parsed.images.size()); ++local_index)
    {
        ParsedSource::Image & image = parsed.images[local_index];
        u32 const image_manifest_index = image_base + local_index;
        write_image_name(scene, image_manifest_index, std::move(image.name));

        std::vector<ImageMaterialBinding> bindings = {};
        bindings.reserve(image.bound_materials.size());
        for (ImageMaterialBinding const & bound_material : image.bound_materials)
        {
            u32 const material_manifest_index = local_index_to_manifest(material_base, bound_material.material_index, parsed.materials.size());
            bindings.push_back({.material_index = material_manifest_index, .slot = bound_material.slot});

            // By default we bind a placeholder image.
            // Once the real artifact is ready - the cook finishes - the binding is updated to point to it.
            std::optional<u32> const placeholder = _placeholder_manifest_indices.at(s_cast<usize>(bound_material.slot));
            if (!placeholder.has_value()) { continue; }
            set_material_texture(scene, material_manifest_index, bound_material.slot, MaterialManifestEntry::ImageInfo{.image_manifest_index = placeholder.value()});
        }
        source.image_bindings.push_back(std::move(bindings));
    }

    std::vector<MeshGroupWrite> mesh_group_writes = {};
    mesh_group_writes.reserve(parsed.mesh_groups.size());
    for (ParsedSource::MeshGroup & mesh_group : parsed.mesh_groups)
    {
        // Translate local mesh indices to manifest indices.
        for (u32 & mesh_index : mesh_group.local_mesh_indices) { mesh_index += mesh_base; }

        mesh_group_writes.push_back(MeshGroupWrite{
            .mesh_manifest_indices = std::move(mesh_group.local_mesh_indices),
            .name = std::move(mesh_group.name),
        });
    }
    u32 const mesh_group_base = create_mesh_groups(scene, mesh_group_writes);
    source.mesh_groups = {.base = mesh_group_base, .count = s_cast<u32>(mesh_group_writes.size())};

    // Only a source that placed something gets a subtree. 
    if (!parsed.entities.empty())
    {
        EntitySubtreeWrite subtree_write = {};
        subtree_write.entities.reserve(parsed.entities.size() + 1);
        for (ParsedSource::Entity & entity : parsed.entities)
        {
            EntitySubtreeWrite::Entity entity_write = {
                .transform = entity.transform,
                .type = entity.type,
                .name = std::move(entity.name),
                .parent_index = entity.parent_index,
                .first_child_index = entity.first_child_index,
                .next_sibling_index = entity.next_sibling_index,
            };
            if (entity.local_mesh_group_index.has_value())
            {
                entity_write.mesh_group_manifest_index = local_index_to_manifest(mesh_group_base, entity.local_mesh_group_index.value(), parsed.mesh_groups.size());
            }
            if (entity.local_light_index.has_value())
            {
                switch (entity.type)
                {
                    case EntityType::POINT_LIGHT: entity_write.light_index = local_index_to_manifest(point_light_base, entity.local_light_index.value(), parsed.point_lights.size()); break;
                    case EntityType::SPOT_LIGHT:  entity_write.light_index = local_index_to_manifest(spot_light_base, entity.local_light_index.value(), parsed.spot_lights.size()); break;
                    case EntityType::ROOT:
                    case EntityType::TRANSFORM:
                    case EntityType::CAMERA:
                    case EntityType::MESHGROUP:
                    case EntityType::CLOUD_VOLUME:
                    case EntityType::UNKNOWN:
                        DBG_ASSERT_TRUE_M(false, "Entity has a light index but is not a light type");
                        break;
                }
            }
            subtree_write.entities.push_back(std::move(entity_write));
        }

        // The root adopts whatever the parse left unparented, so one placement of a source is one movable thing however many nodes it brought.
        u32 const root_index = s_cast<u32>(subtree_write.entities.size());
        subtree_write.entities.push_back(EntitySubtreeWrite::Entity{
            .transform = glm::mat4x3(glm::identity<glm::mat4x3>()),
            .type = EntityType::ROOT,
            .name = _sources.at(source_index).path.stem().string(),
        });
        subtree_write.root_entity_index = root_index;

        std::optional<u32> previous_child = {};
        for (u32 local_index = 0; local_index < root_index; ++local_index)
        {
            if (subtree_write.entities[local_index].parent_index.has_value()) { continue; }
            subtree_write.entities[local_index].parent_index = root_index;
            if (previous_child.has_value()) { subtree_write.entities[previous_child.value()].next_sibling_index = local_index; }
            else { subtree_write.entities[root_index].first_child_index = local_index; }
            previous_child = local_index;
        }
        source.entities = create_entity_subtree(scene, std::move(subtree_write));
    }

    DEBUG_MSG(fmt::format("[Importer] source {} created: {} image, {} material, {} mesh entries",
        source_index, parsed.images.size(), parsed.materials.size(), parsed.meshes.size()));

    source.outstanding_cooks = s_cast<u32>(parsed.images.size() + parsed.meshes.size());
    // A source with nothing to cook is finished the moment its entries exist.
    source.stage = source.outstanding_cooks == 0 ? ImportStage::COOKED : ImportStage::PUBLISHED;
    dispatch_cooks(source_index);
}

void Importer::dispatch_cooks(u32 source_index)
{
    ImportedSource const & source = _sources.at(source_index);

    // ============================================= Meshes =============================================
    for (u32 slot_index = 0; slot_index < s_cast<u32>(source.mesh_cook_inputs.size()); ++slot_index)
    {
        MeshImporterData const & importer_data = source.mesh_cook_inputs[slot_index];
        SourceIndentifier const slot = {.source_index = source_index, .slot_index = slot_index, .generation = source.generation};

        DEBUG_MSG(fmt::format("[Importer] dispatching mesh cook for '{}'", importer_data.indices.location.file.string()));
        auto task = cook_mesh(importer_data, slot);
        thread_pool->async_dispatch(task, TaskPriority::LOW);
        _inflight_cooks.push_back(std::move(task));
    }

    // ============================================= Images =============================================
    for (u32 slot_index = 0; slot_index < s_cast<u32>(source.image_cook_inputs.size()); ++slot_index)
    {
        SourceIndentifier const slot = {.source_index = source_index, .slot_index = slot_index, .generation = source.generation};
        std::shared_ptr<CookTask> task = {};

        std::visit([&](auto const & importer_data)
        {
            using T = std::decay_t<decltype(importer_data)>;
            if constexpr (std::is_same_v<T, ImageImporterData>)
            {
                DEBUG_MSG(fmt::format("[Importer] dispatching image cook for '{}'", importer_data.source_location.file.string()));
                task = cook_image(importer_data, slot);
            }
            else if constexpr (std::is_same_v<T, VdbImporterData>)
            {
                DEBUG_MSG(fmt::format("[Importer] dispatching vdb cook for '{}'", importer_data.source_location.file.string()));
                task = cook_vdb(importer_data, slot);
            }
            else
            {
                DBG_ASSERT_TRUE_M(false, "Image has no cook");
            }
        }, source.image_cook_inputs[slot_index]);

        thread_pool->async_dispatch(task, TaskPriority::LOW);
        _inflight_cooks.push_back(std::move(task));
    }
}

void Importer::resolve_cook(Scene & scene, CookTask & cook)
{
    ImportedSource & source = _sources.at(cook.identifier.source_index);

    // The source has been updated and so nothing guarantees this cook is correct.
    if (cook.identifier.generation != source.generation) { return; }

    source.outstanding_cooks -= 1;
    if (source.outstanding_cooks == 0) { source.stage = ImportStage::COOKED; }

    // Which manifest the result belongs in is the alternative it holds, so a cook that produced nothing has
    // nowhere to be written rather than a flag saying not to.
    std::visit([&](auto & streamer_data)
    {
        using T = std::decay_t<decltype(streamer_data)>;
        if constexpr (std::is_same_v<T, MeshStreamerData>)
        {
            set_mesh_artifact(scene, source.mesh_lod_groups.base + cook.identifier.slot_index, std::move(streamer_data));
        }
        else if constexpr (std::is_same_v<T, ImageStreamerData>)
        {
            u32 const image_manifest_index = source.images.base + cook.identifier.slot_index;
            set_image_artifact(scene, image_manifest_index, std::move(streamer_data));

            // Every slot that has been sampling a placeholder moves to the entry that now has its own artifact.
            for (auto const & binding : source.image_bindings.at(cook.identifier.slot_index))
            {
                set_material_texture(scene, binding.material_index, binding.slot, MaterialManifestEntry::ImageInfo{.image_manifest_index = image_manifest_index});
            }
        }
        // A failed cook leaves its entry on the placeholder it already has.
        else if constexpr (std::is_same_v<T, std::monostate>) { }
        else
        {
            DBG_ASSERT_TRUE_M(false, "Cook result has no manifest to write it to");
        }
    }, cook.streamer_data);
}

void Importer::tick(Scene & scene)
{
    // Publishes first, in dispatch order: a source's entries have to exist before any of its cooks can
    // resolve against them.
    std::vector<std::shared_ptr<SourceParseTask>> finished_parses = {};
    std::erase_if(_inflight_parses, [&](std::shared_ptr<SourceParseTask> const & task)
    {
        if (!task->is_finished()) { return false; }
        finished_parses.push_back(task);
        return true;
    });
    for (std::shared_ptr<SourceParseTask> const & task : finished_parses)
    {
        // A failed parse leaves its row empty forever, so it has to be recorded - otherwise nothing can tell
        // it apart from a parse still in flight.
        if (task->failed)
        {
            _sources.at(task->parsed.source_index).stage = ImportStage::PARSE_FAILED;
            continue;
        }
        publish_import(scene, std::move(task->parsed));
    }

    std::vector<std::shared_ptr<CookTask>> finished_cooks = {};
    std::erase_if(_inflight_cooks, [&](std::shared_ptr<CookTask> const & task)
    {
        if (!task->is_finished()) { return false; }
        finished_cooks.push_back(task);
        return true;
    });
    for (std::shared_ptr<CookTask> const & task : finished_cooks)
    {
        resolve_cook(scene, *task);
    }
}
