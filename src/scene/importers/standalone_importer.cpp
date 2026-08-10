#include "importer.hpp"

#include <fmt/format.h>

#include "../tido_format/tido_util.hpp"

// ============ whole-file backends: one source -> one image slot per recipe ============
// Resolving a whole-file source is the whole of the work: the location is the file itself, so there is
// nothing to read and nothing to parse.
namespace
{

// The manifest name for one slot. A recipe that names itself is one of several cut from the same file, so it
// says which; a lone slot is just the source.
auto slot_name(std::filesystem::path const & path, std::string const & recipe_name) -> std::string
{
    std::string const stem = path.stem().string();
    return recipe_name.empty() ? stem : fmt::format("{} {}", stem, recipe_name);
}

struct ImageParseTask final : SourceParseTask
{
    SourceImportRequest request = {};
    // source_index is set here rather than in the callback so a parse that fails before emitting anything
    // still names the source it belongs to.
    explicit ImageParseTask(SourceImportRequest request) : request{std::move(request)} { chunk_count = 1; parsed.source_index = this->request.source_index; }
    void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override;
};

struct VdbParseTask final : SourceParseTask
{
    SourceImportRequest request = {};
    explicit VdbParseTask(SourceImportRequest request) : request{std::move(request)} { chunk_count = 1; parsed.source_index = this->request.source_index; }
    void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override;
};

void ImageParseTask::callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index)
{
    std::string const extension = tido_lowercase_extension(request.path);
    std::optional<ImageFileFormat> const container_format = [&]() -> std::optional<ImageFileFormat>
    {
        if (extension == ".png") { return ImageFileFormat::PNG; }
        if (extension == ".ktx2") { return ImageFileFormat::KTX2; }
        return std::nullopt;
    }();
    // Only reachable if the backend table gains an extension this mapping does not know about.
    if (!container_format.has_value())
    {
        DEBUG_MSG(fmt::format("[WARN][StandaloneImageParseTask] no image container format for extension '{}' of '{}'",
            extension, request.path.string()));
        failed = true;
        return;
    }

    auto emit_slot = [&](ImageSlotRecipe const & recipe)
    {
        // A lone image places nothing: it is an entry in the manifest, not something in the scene tree.
        parsed.images.push_back(ParsedSource::Image{
            .name = slot_name(request.path, recipe.name),
        });
        parsed.image_cook_inputs.push_back(ImageImporterData{
            .source_location = SourceLocation{.file = request.path},
            .container_format = container_format.value(),
            .channel_mapping = recipe.channel_mapping,
            .target_format = recipe.target_format,
        });
    };

    if (request.recipes.empty())
    {
        // An import that authored no recipe has no way to say what the image is for, so it gets the sRGB
        // colour one - the only assumption a lone image file supports.
        emit_slot(ImageSlotRecipe{.channel_mapping = {0, 1, 2}, .target_format = daxa::Format::BC1_RGB_SRGB_BLOCK});
        return;
    }
    // Several recipes over one file are the same bytes cooked several ways: one read, and as many artifacts
    // as there are distinct recipes.
    for (SlotRecipe const & recipe : request.recipes)
    {
        auto const * image_recipe = std::get_if<ImageSlotRecipe>(&recipe);
        // Emitting the slots that do match would leave the import short of what it asked for, which surfaces
        // much later as a manifest entry nothing filled in.
        if (image_recipe == nullptr)
        {
            DEBUG_MSG(fmt::format("[WARN][StandaloneImageParseTask] '{}' given a recipe that does not describe an image slot",
                request.path.string()));
            failed = true;
            return;
        }
        emit_slot(*image_recipe);
    }
}

void VdbParseTask::callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index)
{
    // Nothing can assume what a .vdb holds - which grids it has and what they mean is authored, never
    // inferred - so an import that names no slots has nothing to produce.
    if (request.recipes.empty())
    {
        DEBUG_MSG(fmt::format("[WARN][VdbParseTask] '{}' imported with no recipes; a .vdb has no slots until it is told which grids to take",
            request.path.string()));
        failed = true;
        return;
    }

    for (SlotRecipe const & recipe : request.recipes)
    {
        auto const * vdb_recipe = std::get_if<VdbSlotRecipe>(&recipe);
        // Emitting the slots that do match would leave the import short of what it asked for, which surfaces
        // much later as a manifest entry nothing filled in.
        if (vdb_recipe == nullptr)
        {
            DEBUG_MSG(fmt::format("[WARN][VdbParseTask] '{}' given a recipe that does not describe a vdb slot",
                request.path.string()));
            failed = true;
            return;
        }
        // Every slot reads the whole file; only the grid selection and the cook differ.
        parsed.images.push_back(ParsedSource::Image{
            .name = slot_name(request.path, vdb_recipe->name),
        });
        parsed.image_cook_inputs.push_back(VdbImporterData{
            .source_location = SourceLocation{.file = request.path},
            .grid_names = vdb_recipe->grid_names,
            .channel_mapping = vdb_recipe->channel_mapping,
            .target_format = vdb_recipe->target_format,
        });
    }
}

} // namespace

auto parse_image_source(SourceImportRequest const & request) -> std::shared_ptr<SourceParseTask>
{
    return std::make_shared<ImageParseTask>(request);
}

auto parse_vdb_source(SourceImportRequest const & request) -> std::shared_ptr<SourceParseTask>
{
    return std::make_shared<VdbParseTask>(request);
}
