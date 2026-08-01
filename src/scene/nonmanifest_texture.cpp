#include "nonmanifest_texture.hpp"

#include <cstring>

#include <fmt/format.h>

#include "../io/file_io.hpp"
#include "optimizers/image_processor.hpp"
#include "tido_format/tido_format.hpp"

namespace
{

// Layer L of a multi-layer set is the same filename with its trailing digit replaced by L, which is how the
// STBN sets are named on disk ("..._128x128x64_0.png" through "_63.png").
auto layer_file_path(std::filesystem::path const & base_path, u32 layer) -> std::filesystem::path
{
    if (layer == 0) { return base_path; }

    std::string stem = base_path.stem().string();
    stem.pop_back();
    stem += std::to_string(layer);

    std::filesystem::path layer_path = base_path;
    layer_path.replace_filename(stem + base_path.extension().string());
    return layer_path;
}

auto parse_png_file(std::filesystem::path const & path, bool load_as_srgb) -> std::optional<TidoImageWithData>
{
    auto [read_result, file_bytes] = read_file(path);
    if (read_result != FileIoResult::SUCCESS)
    {
        DEBUG_MSG(fmt::format("[ERROR][load_nonmanifest_texture] could not read '{}'", path.string()));
        return std::nullopt;
    }

    auto parsed = image_parse(ImageParseInfo{.src_data = file_bytes, .source_format = ImageFileFormat::PNG});
    if (std::holds_alternative<ImageProcessResult>(parsed))
    {
        DEBUG_MSG(fmt::format("[ERROR][load_nonmanifest_texture] could not parse '{}'", path.string()));
        return std::nullopt;
    }
    TidoImageWithData image = std::move(std::get<TidoImageWithData>(parsed));

    // image_parse always emits a linear tag - PNG's own metadata is ignored, so the caller is the only
    // authority on whether the file holds colour. Vulkan names sRGB only for 8-bit UNORM.
    FormatInfo format_info = get_info_from_format(image.descriptor.info.format);
    if (load_as_srgb && !format_info.is_srgb)
    {
        if (format_info.channel_byte_size != 1 || format_info.numeric_type != FormatNumericType::UNORM)
        {
            DEBUG_MSG(fmt::format("[ERROR][load_nonmanifest_texture] an sRGB load needs an 8-bit UNORM source, got {} byte channels in '{}'",
                format_info.channel_byte_size, path.string()));
            return std::nullopt;
        }
        format_info.is_srgb = true;
        image.descriptor.info.format = get_format_from_info(format_info);
    }
    return image;
}

void upload_layer(daxa::Device & device, daxa::ImageId image, TidoImageWithData const & parsed, u32 layer)
{
    auto recorder = device.create_command_recorder({.name = "upload nonmanifest texture"});

    recorder.pipeline_image_barrier({
        .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
        .image = image,
        .layout_operation = daxa::ImageLayoutOperation::TO_GENERAL,
    });

    daxa::BufferId staging_buffer = device.create_buffer({
        .size = parsed.data.size(),
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "upload nonmanifest texture",
    });
    recorder.destroy_buffer_deferred(staging_buffer);
    std::memcpy(device.buffer_host_address(staging_buffer).value(), parsed.data.data(), parsed.data.size());

    for (u32 mip = 0; mip < parsed.descriptor.info.mip_level_count; ++mip)
    {
        recorder.copy_buffer_to_image({
            .src_buffer = staging_buffer,
            .buffer_offset = parsed.descriptor.subresources.at(parsed.descriptor.layer_mip_to_subresource_index(0, mip)).offset,
            .dst_image = image,
            .image_slice = {
                .mip_level = mip,
                .base_array_layer = layer,
            },
            .image_offset = {0, 0, 0},
            .image_extent = {
                std::max(1u, parsed.descriptor.info.size.x >> mip),
                std::max(1u, parsed.descriptor.info.size.y >> mip),
                std::max(1u, parsed.descriptor.info.size.z >> mip),
            },
        });
    }

    recorder.pipeline_image_barrier({
        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_access = daxa::AccessConsts::READ,
        .image = image,
    });

    device.wait_on_submit({
        daxa::QUEUE_MAIN,
        device.submit_commands({.command_lists = std::array{recorder.complete_current_commands()}}),
    });
    device.collect_garbage();
}

} // namespace

auto load_nonmanifest_texture(daxa::Device & device, LoadNonManifestTextureInfo const & info) -> std::optional<daxa::ImageId>
{
    std::vector<TidoImageWithData> parsed_layers = {};
    parsed_layers.reserve(info.layers);

    for (u32 layer = 0; layer < info.layers; ++layer)
    {
        std::optional<TidoImageWithData> parsed = parse_png_file(layer_file_path(info.filepath, layer), info.load_as_srgb);
        if (!parsed.has_value()) { return std::nullopt; }

        // Every layer of one array must agree, or the copies below would write past a layer's extent.
        if (layer > 0)
        {
            TidoImageDescriptor::ImageInfo const & layer_info = parsed->descriptor.info;
            TidoImageDescriptor::ImageInfo const & first_info = parsed_layers.front().descriptor.info;
            if (layer_info.size != first_info.size || layer_info.format != first_info.format)
            {
                DEBUG_MSG(fmt::format("[ERROR][load_nonmanifest_texture] layer {} of '{}' does not match layer 0",
                    layer, info.filepath.string()));
                return std::nullopt;
            }
        }
        parsed_layers.push_back(std::move(parsed.value()));
    }

    TidoImageDescriptor::ImageInfo const & first_info = parsed_layers.front().descriptor.info;
    daxa::ImageId image = device.create_image({
        .dimensions = first_info.dimensions,
        .format = first_info.format,
        .size = {first_info.size.x, first_info.size.y, first_info.size.z},
        .mip_level_count = first_info.mip_level_count,
        .array_layer_count = info.layers,
        .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
        .name = info.filepath.filename().string(),
    });

    for (u32 layer = 0; layer < info.layers; ++layer)
    {
        upload_layer(device, image, parsed_layers[layer], layer);
    }
    return image;
}
