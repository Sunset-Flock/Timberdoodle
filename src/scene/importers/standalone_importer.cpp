#include "importer.hpp"

#include <array>

#include <fmt/format.h>

#include "../tido_format/tido_util.hpp"

// ============ standalone backends: whole-file source -> SceneMetadataBatch (no cooking) ============
namespace
{

// A whole-file source's single slot, plus the subtree root that places it. Resolving it needs no I/O and no
// parse - the location is the file itself - so it runs inline on the importer thread rather than costing a
// ThreadPool dispatch the way the gltf parse does.
auto make_standalone_batch(std::filesystem::path const & path,
    std::variant<ImageImporterData, VdbImporterData> importer_data) -> ImporterTaskResult::SceneMetadataBatch
{
    std::string const stem = path.stem().string();

    ImporterTaskResult::SceneMetadataBatch batch = {};
    batch.images.push_back(ImporterTaskResult::SceneMetadataBatch::Image{
        .name = stem,
        .importer_data = std::move(importer_data),
    });
    // The asset needs a durable place in the scene tree even with nothing under it, so the source still gets
    // the synthetic root every batch carries.
    batch.entities.push_back(ImporterTaskResult::SceneMetadataBatch::Entity{
        .transform = glm::mat4x3(glm::identity<glm::mat4x3>()),
        .type = EntityType::ROOT,
        .name = stem,
    });
    batch.root_entity_index = 0;
    return batch;
}

} // namespace

void dispatch_standalone_image_source(Importer & importer, std::filesystem::path const & path)
{
    std::string const extension = tido_lowercase_extension(path);
    std::optional<ImageFileFormat> const container_format = [&]() -> std::optional<ImageFileFormat>
    {
        if (extension == ".png") { return ImageFileFormat::PNG; }
        if (extension == ".ktx2") { return ImageFileFormat::KTX2; }
        return std::nullopt;
    }();
    // Only reachable if the backend table gains an extension this mapping does not know about.
    if (!container_format.has_value())
    {
        importer.push_result(ImporterTaskResult{.data = ImporterTaskResult::Error{
            .kind = ImporterTaskResult::Error::TaskKind::IMPORT_SOURCE,
            .source = path,
            .message = fmt::format("no image container format for extension '{}'", extension),
        }});
        return;
    }

    // Nothing binds a standalone image to a material slot, so its role is unknown and the recipe is the sRGB
    // colour one until the project document can carry an authored recipe.
    ImageImporterData importer_data = {
        .source_location = SourceLocation{.file = path},
        .container_format = container_format.value(),
        .channel_mapping = {0, 1, 2},
        .target_format = daxa::Format::BC1_RGB_SRGB_BLOCK,
    };
    importer.push_result(ImporterTaskResult{.data = make_standalone_batch(path, std::move(importer_data))});
}

auto build_vdb_source_batch(std::filesystem::path const & path, std::span<VdbSlotRecipe const> recipes)
    -> ImporterTaskResult::SceneMetadataBatch
{
    std::string const stem = path.stem().string();

    ImporterTaskResult::SceneMetadataBatch batch = {};
    for (VdbSlotRecipe const & recipe : recipes)
    {
        batch.images.push_back(ImporterTaskResult::SceneMetadataBatch::Image{
            .name = fmt::format("{} {}", stem, recipe.name),
            .importer_data = VdbImporterData{
                .source_location = SourceLocation{.file = path},
                .grid_names = recipe.grid_names,
                .channel_mapping = recipe.channel_mapping,
                .target_format = recipe.target_format,
            },
        });
    }
    // The asset needs a durable place in the scene tree even with nothing under it, so the source still gets
    // the synthetic root every batch carries.
    batch.root_entity_index = s_cast<u32>(batch.entities.size());
    batch.entities.push_back(ImporterTaskResult::SceneMetadataBatch::Entity{
        .transform = glm::mat4x3(glm::identity<glm::mat4x3>()),
        .type = EntityType::ROOT,
        .name = stem,
    });
    return batch;
}

void dispatch_cloud_volume_source(Importer & importer, std::filesystem::path const & path)
{
    // The three modelling fields the raymarch samples as one BC6 volume, the normalized SDF the custom BC1
    // encoder packs, and the four channel erosion noise. A .vdb naming its grids differently fails the cook.
    static std::array<VdbSlotRecipe, 3> const CLOUD_VOLUME_RECIPES = {
        VdbSlotRecipe{
            .name = "cloud data",
            .grid_names = {"density", "detail_type", "density_scale"},
            .channel_mapping = {0, 1, 2},
            .target_format = daxa::Format::BC6H_UFLOAT_BLOCK,
        },
        VdbSlotRecipe{
            .name = "cloud sdf",
            .grid_names = {"sdf_normalized"},
            .channel_mapping = {0},
            .target_format = daxa::Format::BC1_RGBA_UNORM_BLOCK,
        },
        VdbSlotRecipe{
            .name = "cloud erosion noise",
            .grid_names = {"detail_noise_0", "detail_noise_1", "detail_noise_2", "detail_noise_3"},
            .channel_mapping = {0, 1, 2, 3},
            .target_format = daxa::Format::R16G16B16A16_SFLOAT,
        },
    };
    // Slot order is the recipe order above; the volume samples all three together.
    constexpr u32 CLOUD_DATA_SLOT = 0;
    constexpr u32 CLOUD_SDF_SLOT = 1;
    constexpr u32 CLOUD_DETAIL_NOISE_SLOT = 2;

    ImporterTaskResult::SceneMetadataBatch batch = build_vdb_source_batch(path, CLOUD_VOLUME_RECIPES);
    batch.cloud_volumes.push_back(ImporterTaskResult::SceneMetadataBatch::CloudVolume{
        .data_image_index = CLOUD_DATA_SLOT,
        .sdf_image_index = CLOUD_SDF_SLOT,
        .detail_noise_image_index = CLOUD_DETAIL_NOISE_SLOT,
    });

    // The volume's placement is not in the .vdb, so it keeps the size the runtime has always given the default
    // cloud volume until the project document can carry an authored transform.
    u32 const cloud_entity_index = s_cast<u32>(batch.entities.size());
    batch.entities.push_back(ImporterTaskResult::SceneMetadataBatch::Entity{
        .transform = glm::mat4x3(glm::translate(glm::scale(glm::identity<glm::mat4x4>(), f32vec3(512.0f, 512.0f, 64.0f) * 20.0f), f32vec3(-0.5f, -0.5f, 0.3f))),
        .type = EntityType::CLOUD_VOLUME,
        .name = path.stem().string(),
        .cloud_volume_index = 0,
        .parent_index = batch.root_entity_index,
    });
    batch.entities.at(batch.root_entity_index).first_child_index = cloud_entity_index;

    importer.push_result(ImporterTaskResult{.data = std::move(batch)});
}
