#include "asset_processor.hpp"
#include <daxa/types.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <fstream>
#include <cstring>
#include <png.h>
#include <variant>

#include "optimizers/image_processor.hpp"
#include "optimizers/geometry_optimizer.hpp"

#include <ktx.h>

#pragma region IMAGE_RAW_DATA_LOADING_HELPERS
struct ImageFromRawInfo
{
    std::vector<std::byte> raw_data;
    std::filesystem::path image_path;
    fastgltf::MimeType mime_type;
    int ktx_compression;
};

using RawDataRet = std::variant<std::monostate, AssetProcessor::AssetLoadResultCode, ImageFromRawInfo>;

struct RawImageDataFromURIInfo
{
    fastgltf::sources::URI const & uri;
    fastgltf::Asset const & asset;
    // Wihtout the scename.glb part
    std::filesystem::path const scene_dir_path;
};

auto raw_image_data_from_path(std::filesystem::path image_path) -> RawDataRet
{
    std::ifstream ifs{image_path, std::ios::binary};
    if (!ifs)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_COULD_NOT_OPEN_TEXTURE_FILE;
    }
    ifs.seekg(0, ifs.end);
    i64 const filesize = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    std::vector<std::byte> raw(filesize);
    if (!ifs.read(r_cast<char *>(raw.data()), filesize))
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_COULD_NOT_READ_TEXTURE_FILE;
    }
    return ImageFromRawInfo{
        .raw_data = std::move(raw),
        .image_path = image_path,
        .mime_type = {}};
}

static auto raw_image_data_from_URI(RawImageDataFromURIInfo const & info) -> RawDataRet
{
    /// NOTE: Having global paths in your gltf is just wrong. I guess we could later support them by trying to
    //        load the file anyways but cmon what are the chances of that being successful - for now let's just return error
    if (!info.uri.uri.isLocalPath())
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_UNSUPPORTED_ABSOLUTE_PATH;
    }
    /// NOTE: I don't really see how fileoffsets could be valid in a URI gpu_context. Since we have no information about the size
    //        of the data we always just load everything in the file. Having just a single offset thus does not allow to pack
    //        multiple images into a single file so we just error on this for now.
    if (info.uri.fileByteOffset != 0)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_URI_FILE_OFFSET_NOT_SUPPORTED;
    }
    std::filesystem::path const full_image_path = info.scene_dir_path / info.uri.uri.fspath();
    DEBUG_MSG(fmt::format("[AssetProcessor::raw_image_data_from_URI] Loading image {} ...", full_image_path.string()));
    RawDataRet raw_image_data_ret = raw_image_data_from_path(full_image_path);
    if (std::holds_alternative<AssetProcessor::AssetLoadResultCode>(raw_image_data_ret))
    {
        return raw_image_data_ret;
    }
    ImageFromRawInfo & raw_data = std::get<ImageFromRawInfo>(raw_image_data_ret);
    raw_data.mime_type = info.uri.mimeType;
    if (info.uri.uri.string().ends_with(".ktx2"))
    {
        raw_data.mime_type = fastgltf::MimeType::KTX2;
    }

    return raw_data;
}

struct RawImageDataFromBufferViewInfo
{
    fastgltf::sources::BufferView const & buffer_view;
    fastgltf::Asset const & asset;
    // Wihtout the scename.glb part
    std::filesystem::path const scene_dir_path;
};

static auto raw_image_data_from_buffer_view(RawImageDataFromBufferViewInfo const & info) -> RawDataRet
{
    fastgltf::BufferView const & gltf_buffer_view = info.asset.bufferViews.at(info.buffer_view.bufferViewIndex);
    fastgltf::Buffer const & gltf_buffer = info.asset.buffers.at(gltf_buffer_view.bufferIndex);

    if (!std::holds_alternative<fastgltf::sources::URI>(gltf_buffer.data))
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_FAULTY_BUFFER_VIEW;
    }
    fastgltf::sources::URI uri = std::get<fastgltf::sources::URI>(gltf_buffer.data);

    /// NOTE: load the section of the file containing the buffer for the mesh index buffer.
    std::filesystem::path const full_buffer_path = info.scene_dir_path / uri.uri.fspath();
    std::ifstream ifs{full_buffer_path, std::ios::binary};
    if (!ifs)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_COULD_NOT_OPEN_GLTF;
    }
    /// NOTE: Only load the relevant part of the file containing the view of the buffer we actually need.
    ifs.seekg(gltf_buffer_view.byteOffset + uri.fileByteOffset);
    std::vector<std::byte> raw = {};
    raw.resize(gltf_buffer_view.byteLength);
    /// NOTE: Only load the relevant part of the file containing the view of the buffer we actually need.
    if (!ifs.read(r_cast<char *>(raw.data()), gltf_buffer_view.byteLength))
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_COULD_NOT_READ_BUFFER_IN_GLTF;
    }
    return ImageFromRawInfo{
        .raw_data = std::move(raw),
        .image_path = full_buffer_path,
        .mime_type = uri.mimeType};
}
#pragma endregion

#pragma region IMAGE_RAW_DATA_PARSING_HELPERS

struct ParsedImageData
{
    std::vector<std::byte> src_data = {};
    daxa::ImageInfo image_info = {};
    u32 mips_to_copy = {};
    std::array<u64, 16> mip_copy_offsets = {};
    bool compressed_bc5_rg = {};
};

using ParsedImageRet = std::variant<std::monostate, AssetProcessor::AssetLoadResultCode, ParsedImageData>;

enum struct ChannelDataType
{
    SIGNED_INT,
    UNSIGNED_INT,
    FLOATING_POINT
};

struct ChannelInfo
{
    u8 byte_size = {};
    ChannelDataType data_type = {};
};
using ParsedChannel = std::variant<std::monostate, AssetProcessor::AssetLoadResultCode, ChannelInfo>;

struct PixelInfo
{
    u8 channel_count = {};
    u8 channel_byte_size = {};
    ChannelDataType channel_data_type = {};
    bool load_as_srgb = {};
};

constexpr static auto daxa_image_format_from_pixel_info(PixelInfo const & info) -> daxa::Format
{
    std::array<std::array<std::array<daxa::Format, 3>, 4>, 3> translation = {
        // BYTE SIZE 1
        std::array{
            // CHANNEL COUNT 1
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R8_UNORM, daxa::Format::R8_SINT, daxa::Format::UNDEFINED}},
            // CHANNEL COUNT 2
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R8G8_UNORM, daxa::Format::R8G8_SINT, daxa::Format::UNDEFINED}},
            // CHANNEL COUNT 3
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R8G8B8A8_UNORM, daxa::Format::R8G8B8A8_SINT, daxa::Format::UNDEFINED}},
            // CHANNEL COUNT 4
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R8G8B8A8_UNORM, daxa::Format::R8G8B8A8_SINT, daxa::Format::UNDEFINED}},
        },
        // BYTE SIZE 2
        std::array{
            // CHANNEL COUNT 1
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R16_UINT, daxa::Format::R16_SINT, daxa::Format::R16_SFLOAT}},
            // CHANNEL COUNT 2
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R16G16_UINT, daxa::Format::R16G16_SINT, daxa::Format::R16G16_SFLOAT}},
            // CHANNEL COUNT 3
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R16G16B16A16_UINT, daxa::Format::R16G16B16A16_SINT, daxa::Format::R16G16B16A16_SFLOAT}},
            // CHANNEL COUNT 4
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R16G16B16A16_UINT, daxa::Format::R16G16B16A16_SINT, daxa::Format::R16G16B16A16_SFLOAT}},
        },
        // BYTE SIZE 4
        std::array{
            // CHANNEL COUNT 1
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R32_UINT, daxa::Format::R32_SINT, daxa::Format::R32_SFLOAT}},
            // CHANNEL COUNT 2
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R32G32_UINT, daxa::Format::R32G32_SINT, daxa::Format::R32G32_SFLOAT}},
            // CHANNEL COUNT 3
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R32G32B32A32_UINT, daxa::Format::R32G32B32A32_SINT, daxa::Format::R32G32B32A32_SFLOAT}},
            // CHANNEL COUNT 4
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R32G32B32A32_UINT, daxa::Format::R32G32B32A32_SINT, daxa::Format::R32G32B32A32_SFLOAT}},
        },
    };
    u8 channel_byte_size_idx{};
    switch (info.channel_byte_size)
    {
        case 1:
            channel_byte_size_idx = 0u;
            break;
        case 2:
            channel_byte_size_idx = 1u;
            break;
        case 4:
            channel_byte_size_idx = 2u;
            break;
        default:
            return daxa::Format::UNDEFINED;
    }
    u8 const channel_count_idx = info.channel_count - 1;
    u8 channel_format_idx{};
    switch (info.channel_data_type)
    {
        case ChannelDataType::UNSIGNED_INT:
            channel_format_idx = 0u;
            break;
        case ChannelDataType::SIGNED_INT:
            channel_format_idx = 1u;
            break;
        case ChannelDataType::FLOATING_POINT:
            channel_format_idx = 2u;
            break;
        default:
            return daxa::Format::UNDEFINED;
    }
    auto format = translation[channel_byte_size_idx][channel_count_idx][channel_format_idx];
    if (info.load_as_srgb)
    {
        format = format == daxa::Format::R8_UNORM ? daxa::Format::R8_SRGB : format;
        format = format == daxa::Format::R8G8_UNORM ? daxa::Format::R8G8_SRGB : format;
        format = format == daxa::Format::R8G8B8A8_UNORM ? daxa::Format::R8G8B8A8_SRGB : format;
    }
    return format;
};

// NOTE: glTF image -> texture cooking moved to importer (load) + optimizer (compress). The PNG parse
// below remains only for the not-yet-ported non-manifest / cloud-volume paths in this file.
static auto libpng_parse_raw_image_data(ImageFromRawInfo && raw_data, bool allow_srgb = true) -> ParsedImageRet
{
    bool load_as_srgb = allow_srgb;

    if (png_sig_cmp((png_bytep)raw_data.raw_data.data(), 0, 8))
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_UNKNOWN_FILETYPE_FORMAT;
    }

    auto png_alloc = [](png_structp, png_size_t size) -> png_voidp { return (png_voidp *)malloc(size); };
    auto png_free = [](png_structp, png_voidp ptr) { free(ptr); };
    auto error_fn = []([[maybe_unused]] png_structp png_ptr, [[maybe_unused]] png_const_charp error_msg) { DBG_ASSERT_TRUE_M(false, error_msg); };
    auto data_fn = [](png_structp png_ptr, png_bytep data, png_size_t length)
    {
        std::byte *& raw_data_ptr = *(std::byte **)png_get_io_ptr(png_ptr);
        memcpy(data, raw_data_ptr, length);
        raw_data_ptr += length;
    };

    auto png_ptr = png_create_read_struct_2(PNG_LIBPNG_VER_STRING, nullptr, error_fn, NULL, NULL, png_alloc, png_free);
    DBG_ASSERT_TRUE_M(png_ptr != nullptr, "Failed to create PNG load context");
    png_set_error_fn(png_ptr, nullptr, error_fn, NULL);
    png_set_sig_bytes(png_ptr, 8);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    auto raw_data_ptr = raw_data.raw_data.data() + 8;
    png_set_read_fn(png_ptr, &raw_data_ptr, data_fn);
    DBG_ASSERT_TRUE_M(info_ptr != nullptr, "Failed to create PNG info ptr");

    png_uint_32 width, height;
    int bit_depth, color_type, interlace_type;
    png_read_info(png_ptr, info_ptr);
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, &interlace_type, nullptr, nullptr);
    int channel_count = png_get_channels(png_ptr, info_ptr);

    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png_ptr);
    if (interlace_type != PNG_INTERLACE_NONE) png_set_interlace_handling(png_ptr);
    if (channel_count == 1) png_set_gray_to_rgb(png_ptr);
    if (channel_count < 4) png_set_add_alpha(png_ptr, 255, PNG_FILLER_AFTER);

    png_read_update_info(png_ptr, info_ptr);
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, &interlace_type, nullptr, nullptr);
    channel_count = png_get_channels(png_ptr, info_ptr);

    DBG_ASSERT_TRUE_M(channel_count == 3 || channel_count == 4, "bruh");
    DBG_ASSERT_TRUE_M(bit_depth == 16 || bit_depth == 8, "bruh");

    daxa::Format daxa_image_format = daxa_image_format_from_pixel_info({
        .channel_count = s_cast<u8>(channel_count),
        .channel_byte_size = s_cast<u8>(bit_depth / 8),
        .channel_data_type = ChannelDataType::UNSIGNED_INT,
        .load_as_srgb = load_as_srgb,
    });

    ParsedImageData ret = {};
    u32 const total_image_byte_size = width * height * channel_count * (bit_depth / 8);
    ret.src_data.resize(total_image_byte_size);

    std::vector<png_bytep> row_pointers(height);
    for (u32 y = 0; y < height; y++)
    {
        row_pointers[y] = (png_bytep)(ret.src_data.data() + width * y * channel_count * bit_depth / 8);
    }
    png_read_image(png_ptr, row_pointers.data());

    ret.mips_to_copy = 1;
    ret.image_info = {
        .dimensions = 2,
        .format = daxa_image_format,
        .size = {width, height, 1},
        .mip_level_count = 1,
        .array_layer_count = 1,
        .sample_count = 1,
        .usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
        .name = raw_data.image_path.filename().string(),
    };
    return ret;
}

#pragma endregion

AssetProcessor::AssetProcessor(daxa::Device device)
    : _device{std::move(device)}
{
// call this ONLY when linking with FreeImage as a static library
#ifdef FREEIMAGE_LIB
    FreeImage_Initialise();
#endif
}

AssetProcessor::~AssetProcessor()
{
// call this ONLY when linking with FreeImage as a static library
#ifdef FREEIMAGE_LIB
    FreeImage_DeInitialise();
#endif
}

AssetProcessor::ConvertVDBTask::ConvertVDBTask(ConvertVDBTaskInfo const & info)
    : info{info}
{
    chunk_count = 1;
}

void AssetProcessor::ConvertVDBTask::callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index)
{
    LoadVDBTaskInfo task_info{
        .vdb_path = info.vdb_path,
        .grids_to_load = info.grids_to_convert,
    };

    auto load_task = std::make_shared<LoadVDBTask>(task_info);

    if(!load_task->initialize())
    {
        result = false;
        error_message = "Failed to initialize VDB load task: " + load_task->error_message;
        return;
    }

    progress_message = "Loading VDB file...";
    current_subtask = load_task.get();
    info.threadpool->blocking_dispatch(load_task);

    if(load_task->result == false)
    {
        result = false;
        error_message = load_task->error_message;
        return;
    }

    // Write all loaded grids to file
    std::ofstream output_file(info.converted_result_path, std::ios::binary);
    if (!output_file.is_open())
    {
        result = false;
        error_message = fmt::format("Failed to open output file {} for writing\n", info.converted_result_path.string());
        return;
    }

    // Write header information
    u32 grid_count = static_cast<u32>(info.grids_to_convert.size());
        
    TidoVolumetricCloudDataHeader header;
    header.magic = {'T', 'D', 'V', 'C'};
    header.version = TIDO_VOLUMETRIC_CLOUD_FILE_VERSION;
    header.format = info.output_format;
    header.field_extents = load_task->grid_extents;

    std::vector<std::vector<std::byte>> fields_converted_data;

    switch(info.output_format)
    {
        case TidoVolumetricCloudFileFormat::RAW:
        {
            constexpr u32 element_size = sizeof(u16);

            header.field_count = 1;
            u64 const per_grid_entries = s_cast<u64>(header.field_extents.x) * s_cast<u64>(header.field_extents.y) * s_cast<u64>(header.field_extents.z);
            u64 const total_byte_size = grid_count * per_grid_entries * element_size;
            fields_converted_data.emplace_back(total_byte_size);

            for (u32 elem_index = 0; elem_index < per_grid_entries; ++elem_index)
            {
                u64 const element_offset = elem_index * element_size * grid_count;
                std::memcpy(&fields_converted_data[0][element_offset + 0 * element_size], &load_task->grids_data[0][elem_index * element_size], element_size);
                std::memcpy(&fields_converted_data[0][element_offset + 1 * element_size], &load_task->grids_data[1][elem_index * element_size], element_size);
                std::memcpy(&fields_converted_data[0][element_offset + 2 * element_size], &load_task->grids_data[2][elem_index * element_size], element_size);
                std::memcpy(&fields_converted_data[0][element_offset + 3 * element_size], &load_task->grids_data[3][elem_index * element_size], element_size);
            }
            break;
        }
        case TidoVolumetricCloudFileFormat::CLOUD_SDF_BC1_DATA_BC6:
        {
            header.field_count = 2;
            DBG_ASSERT_TRUE_M((header.field_extents.x % 4 == 0) && (header.field_extents.y % 4 == 0), "For now the image must be 4 aligned (block compression...)");

            // ========================== BC6 COMPRESSION ==========================
            {

                constexpr u32 element_size = sizeof(u16);
                constexpr u32 TEXELS_PER_BLOCK = 4 * 4;
                constexpr u32 BYTES_PER_BC6_BLOCK = 16;
                const u32 blocks_per_layer = ((header.field_extents.x * header.field_extents.y) / TEXELS_PER_BLOCK);
                fields_converted_data.emplace_back(header.field_extents.z * blocks_per_layer * BYTES_PER_BC6_BLOCK);

                u64 const per_grid_entries = s_cast<u64>(header.field_extents.x) * s_cast<u64>(header.field_extents.y) * s_cast<u64>(header.field_extents.z);

                u64 const total_byte_size = 3 * per_grid_entries * element_size;
                std::vector<std::byte> repacked_loaded_fields(total_byte_size);

                for (u32 elem_index = 0; elem_index < per_grid_entries; ++elem_index)
                {
                    u64 const element_offset = elem_index * element_size * 3;
                    std::memcpy(&repacked_loaded_fields[element_offset + 0 * element_size], &load_task->grids_data[0][elem_index * element_size], element_size);
                    std::memcpy(&repacked_loaded_fields[element_offset + 1 * element_size], &load_task->grids_data[1][elem_index * element_size], element_size);
                    std::memcpy(&repacked_loaded_fields[element_offset + 2 * element_size], &load_task->grids_data[2][elem_index * element_size], element_size);
                }

                auto compress_bc6_task = compress_image({
                    .src_data = std::span(repacked_loaded_fields.data(), repacked_loaded_fields.size()),
                    .image_dimensions = header.field_extents,
                    .target_format = daxa::Format::BC6H_UFLOAT_BLOCK,
                    .source_format = daxa::Format::R16G16B16_UINT,
                    .dst_data = std::span(fields_converted_data.back().data(), fields_converted_data.back().size()),
                });

                current_subtask = compress_bc6_task.get();
                progress_message = "Compressing cloud data fields (BC6)...";

                info.threadpool->blocking_dispatch(compress_bc6_task);
            }

            // ========================== SDF1 COMPRESSION ==========================
            {
                constexpr u32 TEXELS_PER_BLOCK = 4 * 4;
                constexpr u32 BYTES_PER_BC1_BLOCK = 8;
                const u32 blocks_per_layer = ((header.field_extents.x * header.field_extents.y) / TEXELS_PER_BLOCK);
                fields_converted_data.emplace_back(header.field_extents.z * blocks_per_layer * BYTES_PER_BC1_BLOCK);

                for(u32 elem_index = 0; elem_index < load_task->grids_data[3].size(); elem_index += sizeof(f32))
                {
                    float const original_value = *r_cast<float*>(&load_task->grids_data[3][elem_index]);
                    float const remapped_value = (original_value + 32.0f) / (512.0f + 32.0f);
                    std::memcpy(&load_task->grids_data[3][elem_index], &remapped_value, sizeof(f32));
                }

                auto compress_bc1_sdf_task = compress_image({
                    .src_data = std::span(reinterpret_cast<std::byte*>(load_task->grids_data[3].data()), load_task->grids_data[3].size()),
                    .image_dimensions = header.field_extents,
                    .target_format = daxa::Format::BC1_RGBA_UNORM_BLOCK,
                    .source_format = daxa::Format::R32_SFLOAT,
                    .dst_data = std::span(fields_converted_data.back().data(), fields_converted_data.back().size()),
                });

                current_subtask = compress_bc1_sdf_task.get();
                progress_message = "Compressing cloud data fields (BC1 SDF)...";

                info.threadpool->blocking_dispatch(compress_bc1_sdf_task);
            }
        }
    }

    output_file.write(reinterpret_cast<const char*>(&header), sizeof(TidoVolumetricCloudDataHeader));

    DBG_ASSERT_TRUE_M(header.field_count == fields_converted_data.size(), "Field count in header does not match number of fields converted");

    // Write each grid's data
    for (u32 field_index = 0; field_index < header.field_count; ++field_index)
    {
        output_file.write(reinterpret_cast<const char*>(fields_converted_data[field_index].data()), fields_converted_data[field_index].size());
    }

    output_file.close();

    if (!output_file.good())
    {
        result = false;
        error_message = fmt::format("Error occurred while writing to file {}\n", info.converted_result_path.string());
        return;
    }

    result = true;
}

void upload_texture(daxa::Device & device, ParsedImageData parsed_data, daxa::ImageId & image, u32 layer = 0u)
{
    daxa::ImageViewInfo image_view_info = device.info(image.default_view()).value();

    auto cr = device.create_command_recorder({.name = "upload image"});

    cr.pipeline_image_barrier({
        .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
        .image = image,
        .layout_operation = daxa::ImageLayoutOperation::TO_GENERAL,
    });

    daxa::BufferId staging_buffer = device.create_buffer({
        .size = parsed_data.src_data.size(),
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "upload image",
    });
    cr.destroy_buffer_deferred(staging_buffer);
    std::memcpy(device.buffer_host_address(staging_buffer).value(), parsed_data.src_data.data(), parsed_data.src_data.size());

    // device.transition_image_layout({
    //     .image = image,
    //     .new_image_layout = daxa::ImageLayout::GENERAL,
    //     .image_slice = image_view_info.slice,
    // });
    daxa::ImageInfo image_info = device.image_info(image).value();
    for (u32 mip = 0; mip < parsed_data.mips_to_copy; ++mip)
    {
        u32 width = std::max(1u, image_info.size.x >> mip);
        u32 height = std::max(1u, image_info.size.y >> mip);
        u32 depth = std::max(1u, image_info.size.z >> mip);
        // device.copy_memory_to_image({
        //     .memory_ptr = parsed_data.src_data.data() + parsed_data.mip_copy_offsets[mip],
        //     .image = image,
        //     .image_slice = {
        //         .mip_level = mip,
        //         .base_array_layer = layer,
        //     },
        //     .image_offset = {0, 0, 0},
        //     .image_extent = {width, height, depth},
        // });
        cr.copy_buffer_to_image({
            .src_buffer = staging_buffer,
            .buffer_offset = parsed_data.mip_copy_offsets[mip],
            .dst_image = image,
            .image_slice = {
                .mip_level = mip,
                .base_array_layer = layer,
            },
            .image_offset = {0, 0, 0},
            .image_extent = {width, height, depth},
        });
    }

    cr.pipeline_image_barrier({
        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_access = daxa::AccessConsts::READ,
        .image = image,
    });

    device.wait_on_submit({
        daxa::QUEUE_MAIN,
        device.submit_commands({
            .command_lists = std::array{cr.complete_current_commands()},
        }),
    });
    device.collect_garbage();

    parsed_data.src_data.resize(0);
    parsed_data.src_data.shrink_to_fit();
}

auto AssetProcessor::load_cloud_volumetric_data(LoadCloudVolumetricDataInfo const & info) -> AssetLoadResultCode
{
    // Open the binary file for reading
    std::ifstream input_file(info.volumetric_data_path, std::ios::binary);
    if (!input_file.is_open()) { return AssetLoadResultCode::ERROR_COULD_NOT_OPEN_FILE; }

    // Read the header
    TidoVolumetricCloudDataHeader header;
    input_file.read(reinterpret_cast<char*>(&header), sizeof(TidoVolumetricCloudDataHeader));
    
    if (!input_file.good()) { return AssetLoadResultCode::ERROR_COULD_NOT_READ_FILE; }

    // Validate magic number
    if (header.magic[0] != 'T' || header.magic[1] != 'D' || header.magic[2] != 'V' || header.magic[3] != 'C')
    {
        return AssetLoadResultCode::ERROR_INVALID_HEADER_MAGIC_CONSTANT_IN_FILE;
    }

    DBG_ASSERT_TRUE_M(header.version == TIDO_VOLUMETRIC_CLOUD_FILE_VERSION, "Currently only the current version of the volumetric cloud file is supported");

    daxa::ImageId cloud_data_image = {};
    daxa::ImageId cloud_sdf_image = {};

    struct FieldImageInfo
    {
        daxa::Format format;
        u64 total_byte_size;
        daxa::ImageId * dst_image_id;
        u32 image_manifest_index;
        std::string name;
    };

    std::vector<FieldImageInfo> fields_info;

    switch(header.format)
    {
        case TidoVolumetricCloudFileFormat::RAW:
        {
            if (header.field_count != 1) { return AssetLoadResultCode::ERROR_INVALID_FIELD_COUNT_IN_FILE; }

            // Calculate total size for all fields
            u32vec3 const field_extents = u32vec3(header.field_extents.x, header.field_extents.y, header.field_extents.z);
            u64 const total_texels = static_cast<u64>(field_extents.x) * static_cast<u64>(field_extents.y) * static_cast<u64>(field_extents.z);

            fields_info = std::vector<FieldImageInfo>{
                FieldImageInfo{
                    .format = daxa::Format::R16G16B16A16_SFLOAT,
                    .total_byte_size = total_texels * 4 * sizeof(u16),
                    .dst_image_id = &cloud_data_image,
                    .image_manifest_index = info.cloud_data_image_manifest_index,
                    .name = "Cloud volumetric uncompressed data"},
            };
            break;
        }

        case TidoVolumetricCloudFileFormat::CLOUD_SDF_BC1_DATA_BC6:
        {
            if (header.field_count != 2) { return AssetLoadResultCode::ERROR_INVALID_FIELD_COUNT_IN_FILE; }

            constexpr u32 TEXELS_PER_BLOCK = 4 * 4;
            constexpr u32 BYTES_PER_BC6_BLOCK = 16;
            constexpr u32 BYTES_PER_BC1_BLOCK = 8;

            const u64 total_blocks = header.field_extents.z * ((header.field_extents.x * header.field_extents.y) / TEXELS_PER_BLOCK);

            fields_info = std::vector<FieldImageInfo>{
                FieldImageInfo{
                    .format = daxa::Format::BC6H_UFLOAT_BLOCK,
                    .total_byte_size = total_blocks * BYTES_PER_BC6_BLOCK,
                    .dst_image_id = &cloud_data_image,
                    .image_manifest_index = info.cloud_data_image_manifest_index,
                    .name = "Cloud volumetric BC6 data" 
                },
                FieldImageInfo{
                    .format = daxa::Format::BC1_RGBA_UNORM_BLOCK,
                    .total_byte_size = total_blocks * BYTES_PER_BC1_BLOCK,
                    .dst_image_id = &cloud_sdf_image,
                    .image_manifest_index = info.cloud_sdf_image_manifest_index,
                    .name = "Cloud volumetric SDF data"
                },
            };
            break;
        }
    }

    auto cr = _device.create_command_recorder({.name = "upload cloud volumetric data"});
    ParsedImageData parsed_data = {};
    for(u32 field_index = 0; field_index < header.field_count; ++field_index)
    {
        FieldImageInfo & field_info = fields_info[field_index];
        parsed_data.src_data.resize(field_info.total_byte_size);

        input_file.read(reinterpret_cast<char*>(parsed_data.src_data.data()), field_info.total_byte_size);
        
        if (!input_file.good()) { return AssetLoadResultCode::ERROR_COULD_NOT_READ_FILE; }

        daxa::ImageInfo image_info = {
            .flags = daxa::ImageCreateFlagBits::COMPATIBLE_2D_ARRAY,
            .dimensions = 3,
            .format = field_info.format,
            .size = {s_cast<u32>(header.field_extents.x), s_cast<u32>(header.field_extents.y), s_cast<u32>(header.field_extents.z)},
            .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .memory_flags = {},
            .name = field_info.name,
        };
        *field_info.dst_image_id = _device.create_image(image_info);

        parsed_data.image_info = image_info;
        parsed_data.mips_to_copy = 1;
        upload_texture(_device, parsed_data, *field_info.dst_image_id);
    }

    input_file.close();

    for(FieldImageInfo & field_info : fields_info)
    {
        /// NOTE: Append the processed texture to the upload queue.
        {
            std::lock_guard<std::mutex> lock{*_texture_upload_mutex};
            _upload_texture_queue.push_back(LoadedTextureInfo{
                .image = *field_info.dst_image_id,
                .image_manifest_index = field_info.image_manifest_index,
            });
        }
    }
    return AssetLoadResultCode::SUCCESS;
}

auto AssetProcessor::load_nonmanifest_texture(LoadNonManifestTextureInfo const & info) -> NonmanifestLoadRet
{
    std::vector<ParsedImageData> parsed_images = {};
    parsed_images.reserve(info.layers);

    auto extension = info.filepath.extension();
    auto filename = info.filepath.filename().replace_extension("").string();

    for (u32 l = 0; l < info.layers; ++l)
    {
        auto filepath = info.filepath;
        if (l > 0)
        {
            auto filename_tmp = filename;
            filename_tmp.pop_back();
            filename_tmp.append(std::to_string(l));
            filename_tmp.append(extension.string());
            filepath.replace_filename(filename_tmp);
        }

        RawDataRet raw_data_ret = raw_image_data_from_path(filepath);
        if (std::holds_alternative<AssetProcessor::AssetLoadResultCode>(raw_data_ret))
        {
            return std::get<AssetProcessor::AssetLoadResultCode>(raw_data_ret);
        }
        ImageFromRawInfo & raw_data = std::get<ImageFromRawInfo>(raw_data_ret);
        ParsedImageRet parsed_data_ret = libpng_parse_raw_image_data(std::move(raw_data), info.load_as_srgb);
        if (auto const * error = std::get_if<AssetProcessor::AssetLoadResultCode>(&parsed_data_ret))
        {
            return *error;
        }
        parsed_images.push_back(std::get<ParsedImageData>(parsed_data_ret));

        if (l > 0)
        {
            auto const & image_info = parsed_images.back().image_info;
            auto const & prev_info = parsed_images.at(parsed_images.size() - 2).image_info;

            if (image_info.size != prev_info.size)
            {
                return AssetLoadResultCode::ERROR_LAYER_IMAGES_NOT_IDENTICAL_SIZE;
            }
            if (image_info.format != prev_info.format)
            {
                return AssetLoadResultCode::ERROR_LAYER_IMAGES_NOT_IDENTICAL_FORMAT;
            }
        }
    }

    daxa::ImageInfo image_info = parsed_images[0].image_info;
    image_info.array_layer_count = info.layers;
    daxa::ImageId image = _device.create_image(image_info);

    for (u32 l = 0; l < info.layers; ++l)
    {
        upload_texture(_device, parsed_images[l], image, l);
    }

    return image;
}

auto AssetProcessor::collect_loaded_resources() -> LoadedResources
{
    LoadedResources ret = {};
    {
        std::lock_guard<std::mutex> lock{*_texture_upload_mutex};
        ret.uploaded_textures = std::move(_upload_texture_queue);
        _upload_texture_queue = {};
    }
    return ret;
}

void AssetProcessor::clear()
{
    {
        std::lock_guard<std::mutex> lock{*_texture_upload_mutex};
        _upload_texture_queue.clear();
    }
}

