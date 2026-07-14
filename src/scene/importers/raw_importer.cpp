#include "raw_importer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>

#include <fmt/format.h>

#include "gltf_cache.hpp"
#include "importer.hpp"
#include "openvdb_importer.hpp"
#include "../tido_format/tido_util.hpp"

// Per-kind cook versions, stamped into every .raw_cache this importer writes (see GLTF_*_COOK_VERSION,
// gltf_importer.cpp - same role, independent numbering). No raw mesh cook exists yet, so the mesh version
// is pinned at 0: find_or_create can never see it drift, and MESH_COOK_VERSION never needs bumping.
static constexpr u32 RAW_TEXTURE_COOK_VERSION = 1;
static constexpr u32 RAW_MESH_COOK_VERSION = 0;

namespace
{
using RawTextureRecipe = std::variant<TextureManifestEntry::RawImporterData::Image, TextureManifestEntry::RawImporterData::VdbVolume>;

// Last-write-time of a single file in filesystem-clock ticks; nullopt if it can't be stat'd. Mirrors
// gltf_importer.cpp's file_mtime - a raw source is always exactly one file, so no max_source_mtime needed.
auto file_mtime(std::filesystem::path const & path) -> std::optional<i64>
{
    std::error_code ec = {};
    auto const write_time = std::filesystem::last_write_time(path, ec);
    if (ec) { return std::nullopt; }
    return write_time.time_since_epoch().count();
}

// The on-disk container format from a raw source's extension; nullopt for anything process_image doesn't understand.
auto image_file_format_from_extension(std::filesystem::path const & path) -> std::optional<ImageFileFormat>
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return s_cast<char>(std::tolower(c)); });
    if (extension == ".png") { return ImageFileFormat::PNG; }
    if (extension == ".ktx2") { return ImageFileFormat::KTX2; }
    return std::nullopt;
}

// Recipe-derived disambiguator folded into the artifact's source-identity key (see tido_source_identity_key,
// N6.1's texture_recipe_tag/image_cache_key scheme) - a changed recipe must produce a different key so a
// stale cook is never fast-pathed under a new recipe.
auto raw_recipe_tag(RawTextureRecipe const & recipe, TextureMaterialType type) -> std::string
{
    if (std::holds_alternative<TextureManifestEntry::RawImporterData::Image>(recipe))
    {
        switch (type)
        {
            case TextureMaterialType::DIFFUSE:             return "diffuse";
            case TextureMaterialType::OPACITY:              return "opacity";
            case TextureMaterialType::NORMAL:               return "normal";
            case TextureMaterialType::ROUGHNESS_METALNESS:  return "roughness_metalness";
            case TextureMaterialType::NONE:
            default:
                DBG_ASSERT_TRUE_M(false, "raw_recipe_tag: unhandled TextureMaterialType");
                return "none";
        }
    }
    auto const & volume = std::get<TextureManifestEntry::RawImporterData::VdbVolume>(recipe);
    std::string tag = "vdb";
    for (VDBGridInfo const & grid : volume.grids) { tag += "+" + grid.name; }
    switch (volume.target)
    {
        case Compression::BC6:       tag += "#bc6";     break;
        case Compression::BC1_SDF:   tag += "#bc1_sdf"; break;
        case Compression::UNDEFINED: tag += "#raw";     break;
        case Compression::BC1:
        case Compression::BC4:
        case Compression::BC5:
        case Compression::BC7:
        default:
            DBG_ASSERT_TRUE_M(false, "raw_recipe_tag: unsupported volume compression target");
            break;
    }
    return tag;
}

auto raw_cache_key(std::filesystem::path const & source_path, RawTextureRecipe const & recipe, TextureMaterialType type) -> u64
{
    return tido_source_identity_key(source_path, source_path.stem().string(), raw_recipe_tag(recipe, type));
}

// Cooks one cache-missed raw texture artifact - a plain image or a VDB volume, dispatching on the recipe.
struct RawTextureCookTask final : Task
{
    std::shared_ptr<SourceContext> const context;
    Importer * const importer;
    RawTextureRecipe const recipe;
    TextureMaterialType const type;
    u64 const cache_key;
    u32 const manifest_index;
    i64 const current_mtime; // current source mtime, stamped onto the (re)cooked or refreshed artifact
    // Pre-seeded cache entry that failed the mtime fast path; reused if its content hash still matches.
    std::optional<TidoTextureCookResult> const cached;

    RawTextureCookTask(std::shared_ptr<SourceContext> context, Importer * importer, RawTextureRecipe recipe, TextureMaterialType type,
        u64 cache_key, u32 manifest_index, i64 current_mtime, std::optional<TidoTextureCookResult> cached)
        : context{std::move(context)}, importer{importer}, recipe{std::move(recipe)}, type{type},
          cache_key{cache_key}, manifest_index{manifest_index}, current_mtime{current_mtime}, cached{std::move(cached)}
    {
        chunk_count = 1;
    }

    void push_error(std::string message)
    {
        importer->push_result(ImporterTaskResult{.data = ImporterTaskResult::Error{
            .kind = ImporterTaskResult::Error::TaskKind::IMPORT_ASSET,
            .source = context->source_path,
            .message = std::move(message),
        }});
    }

    void push_cooked(TidoTextureCookResult const & artifact)
    {
        context->store_texture(artifact);
        importer->push_result(ImporterTaskResult{.data = ImporterTaskResult::CookedAsset{
            .streamer_data = artifact.streamer_data,
            .manifest_index = manifest_index,
        }});
    }

    void run_cook_image(TextureManifestEntry::RawImporterData::Image const &)
    {
        std::optional<ImageFileFormat> const format = image_file_format_from_extension(context->source_path);
        if (!format.has_value())
        {
            push_error(fmt::format("unsupported raw image extension '{}'", context->source_path.extension().string()));
            return;
        }
        std::ifstream ifs{context->source_path, std::ios::binary};
        if (!ifs)
        {
            push_error("failed to open source file");
            return;
        }
        ifs.seekg(0, ifs.end);
        i64 const file_size = ifs.tellg();
        ifs.seekg(0, ifs.beg);
        std::vector<std::byte> raw_bytes(file_size);
        if (!ifs.read(r_cast<char *>(raw_bytes.data()), file_size))
        {
            push_error("failed to read source file");
            return;
        }

        u64 const content_hash = tido_fnv1a(std::as_bytes(std::span{raw_bytes}));
        if (cached.has_value() && cached->content_hash == content_hash && std::filesystem::exists(cached->streamer_data.bin_source))
        {
            // Bytes unchanged (only the mtime moved): reuse the cached .tido_bin, just refresh its stored mtime.
            TidoTextureCookResult artifact = cached.value();
            artifact.source_modified = current_mtime;
            push_cooked(artifact);
            return;
        }

        // The cache key (recipe-tag disambiguated) is hashed into the stem, so artifacts sharing a source never collide.
        std::string const artifact_name = context->source_path.stem().string();

        auto processed_ret = process_image(OptimizeImageInfo{
            .data = std::move(raw_bytes),
            .format = format.value(),
            .type = type,
            .name = artifact_name,
        });
        if (std::get_if<ImageOptimizeError>(&processed_ret) != nullptr)
        {
            push_error(fmt::format("failed to process raw image '{}'", context->source_path.string()));
            return;
        }
        ProcessedImage const & processed = std::get<ProcessedImage>(processed_ret);

        auto tido_result = write_texture_tido(processed, context->cache_dir, artifact_name, cache_key);
        if (!tido_result.has_value())
        {
            push_error(fmt::format("failed to write .tido_bin for '{}'", artifact_name));
            return;
        }
        TidoTextureCookResult artifact = tido_result.value();
        artifact.source_modified = current_mtime;
        artifact.content_hash = content_hash;
        push_cooked(artifact);
    }

    void run_cook_volume(TextureManifestEntry::RawImporterData::VdbVolume const & volume)
    {
        auto load_task = std::make_shared<LoadVDBTask>(LoadVDBTaskInfo{.vdb_path = context->source_path, .grids_to_load = volume.grids});
        if (!load_task->initialize())
        {
            push_error(fmt::format("failed to initialize VDB load for '{}': {}", context->source_path.string(), load_task->error_message));
            return;
        }
        importer->thread_pool->blocking_dispatch(load_task);
        if (!load_task->result)
        {
            push_error(fmt::format("failed to load VDB grids for '{}': {}", context->source_path.string(), load_task->error_message));
            return;
        }

        u64 content_hash = tido_fnv1a(std::as_bytes(std::span{load_task->grids_data.at(0)}));
        for (usize grid_index = 1; grid_index < load_task->grids_data.size(); ++grid_index)
        {
            content_hash = tido_fnv1a(std::as_bytes(std::span{load_task->grids_data[grid_index]}), content_hash);
        }
        if (cached.has_value() && cached->content_hash == content_hash && std::filesystem::exists(cached->streamer_data.bin_source))
        {
            // Grids unchanged (only the mtime moved): reuse the cached .tido_bin, just refresh its stored mtime.
            TidoTextureCookResult artifact = cached.value();
            artifact.source_modified = current_mtime;
            push_cooked(artifact);
            return;
        }

        std::string const artifact_name = context->source_path.stem().string();
        ProcessedImage const processed = process_volume(OptimizeVolumeInfo{
            .grids_data = std::move(load_task->grids_data),
            .grid_extents = load_task->grid_extents,
            .grids = volume.grids,
            .target = volume.target,
            .name = artifact_name,
        }, importer->thread_pool);

        auto tido_result = write_texture_tido(processed, context->cache_dir, artifact_name, cache_key);
        if (!tido_result.has_value())
        {
            push_error(fmt::format("failed to write .tido_bin for '{}'", artifact_name));
            return;
        }
        TidoTextureCookResult artifact = tido_result.value();
        artifact.source_modified = current_mtime;
        artifact.content_hash = content_hash;
        push_cooked(artifact);
    }

    void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        if (auto const * image_recipe = std::get_if<TextureManifestEntry::RawImporterData::Image>(&recipe))
        {
            run_cook_image(*image_recipe);
        }
        else
        {
            run_cook_volume(std::get<TextureManifestEntry::RawImporterData::VdbVolume>(recipe));
        }
        context->outstanding_asset_imports.fetch_sub(1, std::memory_order_acq_rel);
        importer->notify();
    }
};
} // namespace

RawImporter::RawImporter(Importer * importer)
    : _importer{importer}, _cache_registry{importer}
{
}

void RawImporter::resolve_texture_items(std::shared_ptr<SourceContext> const & context, std::vector<TextureItem> const & items)
{
    // Same decision per item as GltfImporter::resolve_texture_items: mtime fast path or a cook chunk.
    u32 fast_hits = 0;
    for (TextureItem const & item : items)
    {
        u64 const key = raw_cache_key(context->source_path, item.recipe, item.type);
        std::optional<TidoTextureCookResult> cached = context->lookup_texture(key);
        std::optional<i64> const src_mtime = file_mtime(context->source_path);

        // mtime fast path: reuse the cached .tido_bin without reading the source.
        if (cached.has_value() && src_mtime.has_value() && cached->source_modified == src_mtime.value() && std::filesystem::exists(cached->streamer_data.bin_source))
        {
            _importer->push_result(ImporterTaskResult{.data = ImporterTaskResult::CookedAsset{
                .streamer_data = cached->streamer_data,
                .manifest_index = item.manifest_index,
            }});
            ++fast_hits;
            continue;
        }

        auto cook_task = std::make_shared<RawTextureCookTask>(
            context, _importer, item.recipe, item.type, key, item.manifest_index, src_mtime.value_or(0), std::move(cached));
        _importer->thread_pool->async_dispatch(cook_task, TaskPriority::LOW);
    }

    if (fast_hits > 0)
    {
        context->outstanding_asset_imports.fetch_sub(fast_hits, std::memory_order_acq_rel);
    }
    DEBUG_MSG(fmt::format("[RawImporter::resolve_texture_items] '{}': {} textures ({} mtime-hit, {} cooked)",
        context->source_path.filename().string(), items.size(), fast_hits, items.size() - fast_hits));
}

void RawImporter::update(std::vector<ImporterTask> & tasks)
{
    // Group this drain's raw asset tasks by source, so one source shares one SourceContext lookup.
    struct PendingBatch
    {
        std::filesystem::path source_path = {};
        std::vector<TextureItem> texture_items = {};
    };
    std::vector<PendingBatch> pending_batches = {};
    auto batch_for = [&](std::filesystem::path const & source_path) -> PendingBatch &
    {
        for (PendingBatch & pending_batch : pending_batches)
        {
            if (pending_batch.source_path == source_path) { return pending_batch; }
        }
        pending_batches.push_back(PendingBatch{.source_path = source_path});
        return pending_batches.back();
    };

    auto consume_task = [&](ImporterTask & task) -> bool
    {
        if (auto * import_texture = std::get_if<ImporterTask::ImportTextureAsset>(&task.data))
        {
            auto const * raw_data = std::get_if<TextureManifestEntry::RawImporterData>(&import_texture->importer_data);
            if (raw_data == nullptr) { return false; } // Not raw provenance - another importer's task.
            batch_for(raw_data->src).texture_items.push_back(TextureItem{
                .recipe = raw_data->recipe,
                .type = import_texture->type,
                .manifest_index = import_texture->texture_manifest_index,
            });
            return true;
        }
        if (auto * import_mesh = std::get_if<ImporterTask::ImportMeshAsset>(&task.data))
        {
            auto const * raw_data = std::get_if<MeshLodGroupManifestEntry::RawImporterData>(&import_mesh->importer_data);
            if (raw_data == nullptr) { return false; } // Not raw provenance - another importer's task.
            // No raw mesh cook exists yet - every raw mesh request fails outright.
            _importer->push_result(ImporterTaskResult{.data = ImporterTaskResult::Error{
                .kind = ImporterTaskResult::Error::TaskKind::IMPORT_ASSET,
                .source = raw_data->src,
                .message = "no raw mesh cook exists",
            }});
            return true;
        }
        return false;
    };
    std::erase_if(tasks, consume_task);

    for (PendingBatch & pending_batch : pending_batches)
    {
        std::filesystem::path const cache_dir = gltf_cache_dir(pending_batch.source_path);
        std::shared_ptr<SourceContext> context = _cache_registry.find_or_create(pending_batch.source_path,
            cache_dir, cache_dir / raw_cache_file_name(pending_batch.source_path), RAW_TEXTURE_COOK_VERSION, RAW_MESH_COOK_VERSION);
        // Taken before dispatch so the count can never cross zero while the batch's items are unresolved;
        // each item's resolution (fast path or cook chunk) releases exactly one.
        context->outstanding_asset_imports.fetch_add(s_cast<u32>(pending_batch.texture_items.size()), std::memory_order_relaxed);
        resolve_texture_items(context, pending_batch.texture_items);
    }

    _cache_registry.run_upkeep();
}
