#include "importer.hpp"

#include <array>

#include <fmt/format.h>

#include "../tido_format/tido_util.hpp"

// ============ standalone backends: whole-file source -> SourceImportResult (no cooking) ============
namespace
{

using SceneBatch = ImporterTaskResult::SceneMetadataBatch;

// The subtree root every source gets: the asset needs a durable place in the scene tree even with nothing
// under it.
auto make_source_root(std::string name) -> SceneBatch::EntitySubtree
{
    SceneBatch::EntitySubtree subtree = {};
    subtree.entities.push_back(SceneBatch::Entity{
        .transform = glm::mat4x3(glm::identity<glm::mat4x3>()),
        .type = EntityType::ROOT,
        .name = std::move(name),
    });
    subtree.root_entity_index = 0;
    return subtree;
}

} // namespace

// Resolving a whole-file source needs no I/O and no parse - the location is the file itself - so this runs
// inline on the importer thread rather than costing a ThreadPool dispatch the way the gltf parse does.
void dispatch_standalone_image_source(Importer & importer, SourceImportRequest const & request)
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
        importer.push_result(ImporterTaskResult{.data = ImporterTaskResult::Error{
            .kind = ImporterTaskResult::Error::TaskKind::IMPORT_SOURCE,
            .source = request.path,
            .message = fmt::format("no image container format for extension '{}'", extension),
        }});
        return;
    }

    // An import that authored no recipe has no way to say what the image is for, so it gets the sRGB colour
    // one - the only assumption a lone image file supports.
    ImageSlotRecipe const recipe = request.image_recipe.value_or(ImageSlotRecipe{
        .channel_mapping = {0, 1, 2},
        .target_format = daxa::Format::BC1_RGB_SRGB_BLOCK,
    });

    std::string const stem = request.path.stem().string();
    SourceImportResult import_result = {};
    import_result.batch.source_index = request.source_index;
    import_result.batch.entity_subtree = make_source_root(stem);
    // No image element here: nothing references a lone image, so its entry is created by the batch carrying
    // its finished cook rather than existing empty until then.
    import_result.image_cooks.push_back(ImageCookRequest{
        .target = ImageCookTarget{.name = stem},
        .importer_data = ImageImporterData{
            .source_location = SourceLocation{.file = request.path},
            .container_format = container_format.value(),
            .channel_mapping = recipe.channel_mapping,
            .target_format = recipe.target_format,
        },
    });
    importer.publish_source_import(std::move(import_result));
}

auto build_vdb_source_import(SourceImportRequest const & request, std::span<VdbSlotRecipe const> recipes)
    -> SourceImportResult
{
    std::string const stem = request.path.stem().string();

    SourceImportResult import_result = {};
    import_result.batch.source_index = request.source_index;
    for (VdbSlotRecipe const & recipe : recipes)
    {
        // Created empty and filled by its cook, because a CloudVolume and the entity placing it reference
        // these entries from the parse and so cannot wait for the artifacts.
        u32 const batch_image_index = s_cast<u32>(import_result.batch.images.size());
        import_result.batch.images.push_back(SceneBatch::Image{
            .name = fmt::format("{} {}", stem, recipe.name),
        });
        import_result.image_cooks.push_back(ImageCookRequest{
            .target = ImageCookTarget{.batch_image_index = batch_image_index},
            .importer_data = VdbImporterData{
                .source_location = SourceLocation{.file = request.path},
                .grid_names = recipe.grid_names,
                .channel_mapping = recipe.channel_mapping,
                .target_format = recipe.target_format,
            },
        });
    }
    import_result.batch.entity_subtree = make_source_root(stem);
    return import_result;
}

void dispatch_cloud_volume_source(Importer & importer, SourceImportRequest const & request)
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

    SourceImportResult import_result = build_vdb_source_import(request, CLOUD_VOLUME_RECIPES);
    import_result.batch.cloud_volumes.push_back(SceneBatch::CloudVolume{
        .data_image = {.kind = SceneBatch::SceneRef::Kind::BATCH_ELEMENT, .index = CLOUD_DATA_SLOT},
        .sdf_image = {.kind = SceneBatch::SceneRef::Kind::BATCH_ELEMENT, .index = CLOUD_SDF_SLOT},
        .detail_noise_image = {.kind = SceneBatch::SceneRef::Kind::BATCH_ELEMENT, .index = CLOUD_DETAIL_NOISE_SLOT},
    });

    SceneBatch::EntitySubtree & subtree = import_result.batch.entity_subtree.value();
    // The volume's placement is not in the .vdb, so it keeps the size the runtime has always given the default
    // cloud volume until the project document can carry an authored transform.
    u32 const cloud_entity_index = s_cast<u32>(subtree.entities.size());
    subtree.entities.push_back(SceneBatch::Entity{
        .transform = glm::mat4x3(glm::translate(glm::scale(glm::identity<glm::mat4x4>(), f32vec3(512.0f, 512.0f, 64.0f) * 20.0f), f32vec3(-0.5f, -0.5f, 0.3f))),
        .type = EntityType::CLOUD_VOLUME,
        .name = request.path.stem().string(),
        .cloud_volume = SceneBatch::SceneRef{.kind = SceneBatch::SceneRef::Kind::BATCH_ELEMENT, .index = 0},
        .parent_index = subtree.root_entity_index,
    });
    subtree.entities.at(subtree.root_entity_index).first_child_index = cloud_entity_index;

    importer.publish_source_import(std::move(import_result));
}
