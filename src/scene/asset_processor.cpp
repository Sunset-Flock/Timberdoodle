#include "asset_processor.hpp"
#include <daxa/types.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <fstream>
#include <cstring>
#include <FreeImage.h>
#include <variant>

#include <ktx.h>

#pragma region IMAGE_RAW_DATA_LOADING_HELPERS
struct ImageFromRawInfo
{
    std::vector<std::byte> raw_data;
    std::filesystem::path image_path;
    fastgltf::MimeType mime_type;
    int ktx_compression = KTX_TTF_BC7_RGBA;
};

using RawDataRet = std::variant<std::monostate, AssetProcessor::AssetLoadResultCode, ImageFromRawInfo>;

struct RawImageDataFromURIInfo
{
    fastgltf::sources::URI const & uri;
    fastgltf::Asset const & asset;
    // Wihtout the scename.glb part
    std::filesystem::path const scene_dir_path;
};

static auto raw_image_data_from_path(std::filesystem::path image_path) -> RawDataRet
{
    std::ifstream ifs{image_path, std::ios::binary};
    if (!ifs)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_COULD_NOT_OPEN_TEXTURE_FILE;
    }
    ifs.seekg(0, ifs.end);
    i32 const filesize = ifs.tellg();
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
    daxa::BufferId src_buffer = {};
    daxa::ImageId dst_image = {};
    u32 mips_to_copy = {};
    std::array<u32, 16> mip_copy_offsets = {};
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

constexpr static auto parse_channel_info(FREE_IMAGE_TYPE image_type) -> ParsedChannel
{
    ChannelInfo ret = {};
    switch (image_type)
    {
        case FREE_IMAGE_TYPE::FIT_BITMAP:
        {
            ret.byte_size = 1u;
            ret.data_type = ChannelDataType::UNSIGNED_INT;
            break;
        }
        case FREE_IMAGE_TYPE::FIT_UINT16:
        {
            ret.byte_size = 2u;
            ret.data_type = ChannelDataType::UNSIGNED_INT;
            break;
        }
        case FREE_IMAGE_TYPE::FIT_INT16:
        {
            ret.byte_size = 2u;
            ret.data_type = ChannelDataType::SIGNED_INT;
            break;
        }
        case FREE_IMAGE_TYPE::FIT_UINT32:
        {
            ret.byte_size = 4u;
            ret.data_type = ChannelDataType::UNSIGNED_INT;
            break;
        }
        case FREE_IMAGE_TYPE::FIT_INT32:
        {
            ret.byte_size = 4u;
            ret.data_type = ChannelDataType::SIGNED_INT;
            break;
        }
        case FREE_IMAGE_TYPE::FIT_FLOAT:
        {
            ret.byte_size = 4u;
            ret.data_type = ChannelDataType::FLOATING_POINT;
            break;
        }
        case FREE_IMAGE_TYPE::FIT_RGB16:
        {
            ret.byte_size = 2u;
            ret.data_type = ChannelDataType::UNSIGNED_INT;
            break;
        }
        case FREE_IMAGE_TYPE::FIT_RGBA16:
        {
            ret.byte_size = 2u;
            ret.data_type = ChannelDataType::UNSIGNED_INT;
            break;
        }
        case FREE_IMAGE_TYPE::FIT_RGBF:
        {
            ret.byte_size = 4u;
            ret.data_type = ChannelDataType::FLOATING_POINT;
            break;
        }
        case FREE_IMAGE_TYPE::FIT_RGBAF:
        {
            ret.byte_size = 4u;
            ret.data_type = ChannelDataType::FLOATING_POINT;
            break;
        }
        default:
            return AssetProcessor::AssetLoadResultCode::ERROR_UNSUPPORTED_TEXTURE_PIXEL_FORMAT;
    }
    return ret;
};

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
            /// NOTE: Free image stores images in BGRA on little endians (Win,Linux) this will break on Mac
            // CHANNEL COUNT 3
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::B8G8R8A8_UNORM, daxa::Format::B8G8R8A8_SINT, daxa::Format::UNDEFINED}},
            // CHANNEL COUNT 4
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::B8G8R8A8_UNORM, daxa::Format::B8G8R8A8_SINT, daxa::Format::UNDEFINED}},
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
            /// TODO: Channel count 3 might not be supported possible just replace with four channel alternatives
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R32G32B32_UINT, daxa::Format::R32G32B32_SINT, daxa::Format::R32G32B32_SFLOAT}},
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
        format = format == daxa::Format::B8G8R8A8_UNORM ? daxa::Format::B8G8R8A8_SRGB : format;
    }
    return format;
};

static auto free_image_parse_raw_image_data(ImageFromRawInfo && raw_data, daxa::Device & device, TextureMaterialType type) -> ParsedImageRet
{
    bool load_as_srgb = type == TextureMaterialType::DIFFUSE;
    /// NOTE: Since we handle the image data loading ourselves we need to wrap the buffer with a FreeImage
    //        wrapper so that it can internally process the data
    FIMEMORY * fif_memory_wrapper = FreeImage_OpenMemory(r_cast<BYTE *>(raw_data.raw_data.data()), raw_data.raw_data.size());
    defer
    {
        FreeImage_CloseMemory(fif_memory_wrapper);
    };
    FREE_IMAGE_FORMAT image_format = FreeImage_GetFileTypeFromMemory(fif_memory_wrapper, 0);
    // could not deduce filetype from metadata in memory try to guess the format from the file extension
    if (image_format == FIF_UNKNOWN)
    {
        image_format = FreeImage_GetFIFFromFilename(raw_data.image_path.string().c_str());
    }
    // could not deduce filetype at all
    if (image_format == FIF_UNKNOWN)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_UNKNOWN_FILETYPE_FORMAT;
    }
    if (!FreeImage_FIFSupportsReading(image_format))
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_UNSUPPORTED_READ_FOR_FILEFORMAT;
    }
    FIBITMAP * image_bitmap = FreeImage_LoadFromMemory(image_format, fif_memory_wrapper);
    defer
    {
        FreeImage_Unload(image_bitmap);
    };
    if (!image_bitmap)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_COULD_NOT_READ_TEXTURE_FILE_FROM_MEMSTREAM;
    }
    u32 bits_per_pixel = FreeImage_GetBPP(image_bitmap);
    if (bits_per_pixel != 32 && bits_per_pixel != 24)
    {
        auto * temp = FreeImage_ConvertTo32Bits(image_bitmap);
        FreeImage_Unload(image_bitmap);
        image_bitmap = temp;
        bits_per_pixel = 32;
    }
    FREE_IMAGE_TYPE const image_type = FreeImage_GetImageType(image_bitmap);
    FREE_IMAGE_COLOR_TYPE const color_type = FreeImage_GetColorType(image_bitmap);
    u32 const width = FreeImage_GetWidth(image_bitmap);
    u32 const height = FreeImage_GetHeight(image_bitmap);
    bool const has_red_channel = FreeImage_GetRedMask(image_bitmap) != 0;
    bool const has_green_channel = FreeImage_GetGreenMask(image_bitmap) != 0;
    bool const has_blue_channel = FreeImage_GetBlueMask(image_bitmap) != 0;

    bool const should_contain_all_color_channels =
        (color_type == FREE_IMAGE_COLOR_TYPE::FIC_RGB) ||
        (color_type == FREE_IMAGE_COLOR_TYPE::FIC_RGBALPHA);
    bool const contains_all_color_channels = has_red_channel && has_green_channel && has_blue_channel;
    DBG_ASSERT_TRUE_M(should_contain_all_color_channels == contains_all_color_channels,
        std::string("[ERROR][free_image_parse_raw_image_data()] Image color type indicates color channels present") +
            std::string(" but not all channels were present accoring to color masks"));

    ParsedChannel parsed_channel = parse_channel_info(image_type);
    if (auto const * err = std::get_if<AssetProcessor::AssetLoadResultCode>(&parsed_channel))
    {
        return *err;
    }

    ChannelInfo const & channel_info = std::get<ChannelInfo>(parsed_channel);
    u32 const channel_count = bits_per_pixel / (channel_info.byte_size * 8u);

    daxa::Format daxa_image_format = daxa_image_format_from_pixel_info({
        .channel_count = s_cast<u8>(channel_count),
        .channel_byte_size = channel_info.byte_size,
        .channel_data_type = channel_info.data_type,
        .load_as_srgb = load_as_srgb,
    });

    /// TODO: Breaks for 32bit 3 channel images (or overallocates idk)
    u32 const rounded_channel_count = channel_count == 3 ? 4 : channel_count;
    FIBITMAP * modified_bitmap;
    if (channel_count == 3)
    {
        modified_bitmap = FreeImage_ConvertTo32Bits(image_bitmap);
    }
    else
    {
        modified_bitmap = image_bitmap;
    }
    defer
    {
        if (channel_count == 3)
            FreeImage_Unload(modified_bitmap);
    };
    FreeImage_FlipVertical(modified_bitmap);
    ParsedImageData ret = {};
    u32 const total_image_byte_size = width * height * rounded_channel_count * channel_info.byte_size;
    ret.src_buffer = device.create_buffer({
        .size = total_image_byte_size,
        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .name = raw_data.image_path.filename().string() + " staging",
    });
    std::byte * staging_dst_ptr = device.buffer_host_address_as<std::byte>(ret.src_buffer).value();
    memcpy(staging_dst_ptr, r_cast<std::byte *>(FreeImage_GetBits(modified_bitmap)), total_image_byte_size);

    ret.mips_to_copy = 1;
    ret.dst_image = device.create_image({
        .dimensions = 2,
        .format = daxa_image_format,
        .size = {width, height, 1},
        /// TODO: Add support for generating mip levels
        .mip_level_count = 1,
        .array_layer_count = 1,
        .sample_count = 1,
        /// TODO: Potentially take more flags from the user here
        .usage =
            daxa::ImageUsageFlagBits::TRANSFER_DST |
            daxa::ImageUsageFlagBits::SHADER_SAMPLED,
        .name = raw_data.image_path.filename().string(),
    });
    return ret;
}

static auto ktx_parse_raw_image_data(ImageFromRawInfo & raw_data, daxa::Device & device, TextureMaterialType type) -> ParsedImageRet
{
    bool const load_as_srgb = (type == TextureMaterialType::DIFFUSE) || (type == TextureMaterialType::DIFFUSE_OPACITY);
    ktx_transcode_fmt_e transcode_format;
    switch (type)
    {
        case TextureMaterialType::NORMAL:          transcode_format = KTX_TTF_BC5_RG; break;
        case TextureMaterialType::DIFFUSE_OPACITY: transcode_format = KTX_TTF_BC4_R; break;
        default:                                   transcode_format = KTX_TTF_BC7_RGBA; break;
    }

    ktxTexture2 * texture;
    KTX_error_code result;
    ktx_size_t offset;

    result = ktxTexture2_CreateFromMemory(
        r_cast<ktx_uint8_t *>(raw_data.raw_data.data()),
        raw_data.raw_data.size(),
        KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
        &texture);
    if (result != KTX_SUCCESS)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_FAILED_TO_PROCESS_KTX;
    }
    defer
    {
        ktxTexture_Destroy(ktxTexture(texture));
    };

    ktx_transcode_flags flags = KTX_TF_HIGH_QUALITY;
    flags |= type == TextureMaterialType::DIFFUSE_OPACITY ? KTX_TF_TRANSCODE_ALPHA_DATA_TO_OPAQUE_FORMATS : 0u;
    result = ktxTexture2_TranscodeBasis(texture, transcode_format, flags);
    if (result != KTX_SUCCESS)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_FAILED_TO_PROCESS_KTX;
    }

    u32 const numLevels = texture->numLevels;
    u32 const numLayers = texture->numLayers;
    u32 const baseWidth = texture->baseWidth;
    u32 const baseHeight = texture->baseHeight;
    u32 const baseDepth = texture->baseDepth;
    u32 const mips = static_cast<u32>(floor(std::log2(std::min(baseWidth, baseHeight)))) + 1;
    bool const isArray = texture->isArray;

    ParsedImageData ret = {};
    daxa::BufferId staging = device.create_buffer({
        .size = texture->dataSize,
        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM, // Host local memory.
        .name = raw_data.image_path.string() + " staging",
    });
    std::byte * staging_ptr = device.buffer_host_address(staging).value();
    ktx_uint8_t * image_ktx_data = ktxTexture_GetData(ktxTexture(texture));
    daxa::Format const format = std::bit_cast<daxa::Format>(texture->vkFormat);
    daxa::ImageId image_id = device.create_image({
        .flags = {},
        .dimensions = 2,
        .format = format,
        .size = {baseWidth, baseHeight, baseDepth},
        .mip_level_count = numLevels,
        .array_layer_count = numLayers,
        .sample_count = 1,
        .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
        .allocate_info = {},
        .name = raw_data.image_path.filename().string(),
    });
    ret.dst_image = image_id;
    ret.src_buffer = staging;
    ret.compressed_bc5_rg = transcode_format == KTX_TTF_BC5_RG;
    ret.mips_to_copy = texture->numLevels;
    for (u32 mip = 0; mip < texture->numLevels; ++mip)
    {
        u32 const layer = 0;
        u32 const faceSlice = 0;
        usize offset = {};
        result = ktxTexture_GetImageOffset(ktxTexture(texture), mip, layer, faceSlice, &offset);
        if (result != KTX_SUCCESS)
        {
            return AssetProcessor::AssetLoadResultCode::ERROR_FAILED_TO_PROCESS_KTX;
        }
        usize size = ktxTexture_GetImageSize(ktxTexture(texture), mip);
        std::memcpy(staging_ptr + offset, image_ktx_data + offset, size);
        ret.mip_copy_offsets[mip] = offset;
    }

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

auto AssetProcessor::load_nonmanifest_texture(std::filesystem::path const & filepath, bool const load_as_srgb) -> NonmanifestLoadRet
{
    RawDataRet raw_data_ret = raw_image_data_from_path(filepath);
    if (std::holds_alternative<AssetProcessor::AssetLoadResultCode>(raw_data_ret))
    {
        return std::get<AssetProcessor::AssetLoadResultCode>(raw_data_ret);
    }
    ImageFromRawInfo & raw_data = std::get<ImageFromRawInfo>(raw_data_ret);
    ParsedImageRet parsed_data_ret = free_image_parse_raw_image_data(std::move(raw_data), _device, TextureMaterialType::DIFFUSE);
    if (auto const * error = std::get_if<AssetProcessor::AssetLoadResultCode>(&parsed_data_ret))
    {
        return *error;
    }
    ParsedImageData const & parsed_data = std::get<ParsedImageData>(parsed_data_ret);

    auto recorder = _device.create_command_recorder({});
    recorder.destroy_buffer_deferred(parsed_data.src_buffer);
    recorder.pipeline_barrier_image_transition({
        .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
        .image_id = parsed_data.dst_image,
    });

    recorder.copy_buffer_to_image({
        .buffer = parsed_data.src_buffer,
        .image = parsed_data.dst_image,
        .image_extent = _device.image_info(parsed_data.dst_image).value().size,
    });

    recorder.pipeline_barrier_image_transition({
        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_access = daxa::AccessConsts::ALL_GRAPHICS_READ,
        .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
        /// TODO: Take the usage from the user for now images only used as attachments
        .dst_layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
        .image_id = parsed_data.dst_image,
    });

    daxa::ExecutableCommandList command_list = recorder.complete_current_commands();
    _device.submit_commands({.command_lists = {&command_list, 1}});
    _device.wait_idle();
    return parsed_data.dst_image;
}

auto AssetProcessor::load_texture(LoadTextureInfo const & info) -> AssetLoadResultCode
{
    fastgltf::Asset const & gltf_asset = *info.asset;
    fastgltf::Image const & image = gltf_asset.images.at(info.gltf_image_index);
    std::vector<std::byte> raw_data = {};

    RawDataRet ret = {};
    if (auto const * uri = std::get_if<fastgltf::sources::URI>(&image.data))
    {
        ret = std::move(raw_image_data_from_URI(RawImageDataFromURIInfo{
            .uri = *uri,
            .asset = gltf_asset,
            .scene_dir_path = std::filesystem::path(info.asset_path).remove_filename(),
        }));
    }
    else if (auto const * buffer_view = std::get_if<fastgltf::sources::BufferView>(&image.data))
    {
        ret = std::move(raw_image_data_from_buffer_view(RawImageDataFromBufferViewInfo{
            .buffer_view = *buffer_view,
            .asset = gltf_asset,
            .scene_dir_path = std::filesystem::path(info.asset_path).remove_filename(),
        }));
    }
    else
    {
        return AssetLoadResultCode::ERROR_FAULTY_BUFFER_VIEW;
    }

    if (auto const * error = std::get_if<AssetLoadResultCode>(&ret))
    {
        return *error;
    }
    ImageFromRawInfo & raw_image_data = std::get<ImageFromRawInfo>(ret);
    ParsedImageRet parsed_data_ret = {};
    ParsedImageRet opaque_data_ret = {std::monostate{}};
    if (raw_image_data.mime_type == fastgltf::MimeType::KTX2)
    {
        parsed_data_ret = ktx_parse_raw_image_data(raw_image_data, _device, info.texture_material_type);
        if (info.texture_material_type == TextureMaterialType::DIFFUSE)
        {
            opaque_data_ret = ktx_parse_raw_image_data(raw_image_data, _device, TextureMaterialType::DIFFUSE_OPACITY);
        }
    }
    else
    {
        parsed_data_ret = free_image_parse_raw_image_data(std::move(raw_image_data), _device, info.texture_material_type);
    }
    if (auto const * error = std::get_if<AssetProcessor::AssetLoadResultCode>(&parsed_data_ret))
    {
        return *error;
    }
    ParsedImageData const & parsed_data = std::get<ParsedImageData>(parsed_data_ret);
    ParsedImageData const * opaque_data = std::get_if<ParsedImageData>(&opaque_data_ret);
    /// NOTE: Append the processed texture to the upload queue.
    {
        std::lock_guard<std::mutex> lock{*_texture_upload_mutex};
        _upload_texture_queue.push_back(LoadedTextureInfo{
            .staging_buffer = parsed_data.src_buffer,
            .dst_image = parsed_data.dst_image,
            .mips_to_copy = parsed_data.mips_to_copy,
            .mip_copy_offsets = parsed_data.mip_copy_offsets,
            .texture_manifest_index = info.texture_manifest_index,
            .compressed_bc5_rg = parsed_data.compressed_bc5_rg,
        });
        if (opaque_data)
        {
            _upload_texture_queue.push_back(LoadedTextureInfo{
                .staging_buffer = opaque_data->src_buffer,
                .dst_image = opaque_data->dst_image,
                .mips_to_copy = opaque_data->mips_to_copy,
                .mip_copy_offsets = opaque_data->mip_copy_offsets,
                .texture_manifest_index = info.texture_manifest_index,
                .secondary_texture = true,
                .compressed_bc5_rg = false,
            });
        }
    }
    return AssetLoadResultCode::SUCCESS;
}

/// NOTE: Overload ElementTraits for glm vec3 for fastgltf to understand the type.
template <>
struct fastgltf::ElementTraits<glm::vec4> : fastgltf::ElementTraitsBase<float, fastgltf::AccessorType::Vec4>
{
};

/// NOTE: Overload ElementTraits for glm vec3 for fastgltf to understand the type.
template <>
struct fastgltf::ElementTraits<glm::vec3> : fastgltf::ElementTraitsBase<float, fastgltf::AccessorType::Vec3>
{
};

template <>
struct fastgltf::ElementTraits<glm::vec2> : fastgltf::ElementTraitsBase<float, fastgltf::AccessorType::Vec2>
{
};

template <typename ElemT, bool IS_INDEX_BUFFER>
auto load_accessor_data_from_file(
    std::filesystem::path const & root_path,
    fastgltf::Asset const & gltf_asset,
    fastgltf::Accessor const & accesor)
    -> std::variant<std::vector<ElemT>, AssetProcessor::AssetLoadResultCode>
{
    static_assert(!IS_INDEX_BUFFER || std::is_same_v<ElemT, u32>, "Index Buffer must be u32");
    fastgltf::BufferView const & gltf_buffer_view = gltf_asset.bufferViews.at(accesor.bufferViewIndex.value());
    fastgltf::Buffer const & gltf_buffer = gltf_asset.buffers.at(gltf_buffer_view.bufferIndex);
    if (!std::holds_alternative<fastgltf::sources::URI>(gltf_buffer.data))
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_FAULTY_BUFFER_VIEW;
    }
    fastgltf::sources::URI uri = std::get<fastgltf::sources::URI>(gltf_buffer.data);

    /// NOTE: load the section of the file containing the buffer for the mesh index buffer.
    std::filesystem::path const full_buffer_path = root_path / uri.uri.fspath();
    std::ifstream ifs{full_buffer_path, std::ios::binary};
    if (!ifs)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_COULD_NOT_OPEN_GLTF;
    }
    /// NOTE: Only load the relevant part of the file containing the view of the buffer we actually need.
    ifs.seekg(gltf_buffer_view.byteOffset + accesor.byteOffset + uri.fileByteOffset);
    std::vector<u16> raw = {};
    auto const elem_byte_size = fastgltf::getElementByteSize(accesor.type, accesor.componentType);
    raw.resize((accesor.count * elem_byte_size) / 2);
    /// NOTE: Only load the relevant part of the file containing the view of the buffer we actually need.
    if (!ifs.read(r_cast<char *>(raw.data()), accesor.count * elem_byte_size))
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_COULD_NOT_READ_BUFFER_IN_GLTF;
    }
    auto buffer_adapter = [&](fastgltf::Buffer const & buffer)
    {
        /// NOTE:   We only have a ptr to the loaded data to the accessors section of the buffer.
        ///         Fastgltf expects a ptr to the begin of the buffer, so we just subtract the offsets.
        ///         Fastgltf adds these on in the accessor tool, so in the end it gets the right ptr.
        auto const fastgltf_reverse_byte_offset = (gltf_buffer_view.byteOffset + accesor.byteOffset);
        return r_cast<std::byte *>(raw.data()) - fastgltf_reverse_byte_offset;
    };

    std::vector<ElemT> ret(accesor.count);
    if constexpr (IS_INDEX_BUFFER)
    {
        /// NOTE: Transform the loaded file section into a 32 bit index buffer.
        if (accesor.componentType == fastgltf::ComponentType::UnsignedShort)
        {
            std::vector<u16> u16_index_buffer(accesor.count);
            fastgltf::copyFromAccessor<u16>(gltf_asset, accesor, u16_index_buffer.data(), buffer_adapter);
            for (size_t i = 0; i < u16_index_buffer.size(); ++i)
            {
                ret[i] = s_cast<u32>(u16_index_buffer[i]);
            }
        }
        else
        {
            fastgltf::copyFromAccessor<u32>(gltf_asset, accesor, ret.data(), buffer_adapter);
        }
    }
    else
    {
        fastgltf::copyFromAccessor<ElemT>(gltf_asset, accesor, ret.data(), buffer_adapter);
    }
    return ret;
}

auto AssetProcessor::load_mesh(LoadMeshLodGroupInfo const & info) -> AssetLoadResultCode
{
    fastgltf::Asset & gltf_asset = *info.asset;
    fastgltf::Mesh & gltf_mesh = gltf_asset.meshes[info.gltf_mesh_index];
    fastgltf::Primitive & gltf_prim = gltf_mesh.primitives[info.gltf_primitive_index];

/// NOTE: Process indices (they are required)
#pragma region INDICES
    if (!gltf_prim.indicesAccessor.has_value())
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_MISSING_INDEX_BUFFER;
    }
    fastgltf::Accessor & index_buffer_gltf_accessor = gltf_asset.accessors.at(gltf_prim.indicesAccessor.value());
    bool const index_buffer_accessor_valid =
        (index_buffer_gltf_accessor.componentType == fastgltf::ComponentType::UnsignedInt ||
         index_buffer_gltf_accessor.componentType == fastgltf::ComponentType::UnsignedShort) &&
         index_buffer_gltf_accessor.type == fastgltf::AccessorType::Scalar &&
         index_buffer_gltf_accessor.bufferViewIndex.has_value();
    if (!index_buffer_accessor_valid)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_FAULTY_INDEX_BUFFER_GLTF_ACCESSOR;
    }
    auto index_buffer_data = load_accessor_data_from_file<u32, true>(std::filesystem::path{info.asset_path}.remove_filename(), gltf_asset, index_buffer_gltf_accessor);
    if (auto const * err = std::get_if<AssetProcessor::AssetLoadResultCode>(&index_buffer_data))
    {
        return *err;
    }
    std::vector<u32> lod0_index_buffer = std::get<std::vector<u32>>(std::move(index_buffer_data));
#pragma endregion

/// NOTE: Load vertex positions
#pragma region VERTICES
    auto vert_attrib_iter = gltf_prim.findAttribute(VERT_ATTRIB_POSITION_NAME);
    if (vert_attrib_iter == gltf_prim.attributes.end())
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_MISSING_VERTEX_POSITIONS;
    }
    fastgltf::Accessor & gltf_vertex_pos_accessor = gltf_asset.accessors.at(vert_attrib_iter->second);
    bool const gltf_vertex_pos_accessor_valid =
        gltf_vertex_pos_accessor.componentType == fastgltf::ComponentType::Float &&
        gltf_vertex_pos_accessor.type == fastgltf::AccessorType::Vec3;
    if (!gltf_vertex_pos_accessor_valid)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_POSITIONS;
    }
    // TODO: we can probably load this directly into the staging buffer.
    auto vertex_pos_result = load_accessor_data_from_file<glm::vec3, false>(std::filesystem::path{info.asset_path}.remove_filename(), gltf_asset, gltf_vertex_pos_accessor);
    if (auto const * err = std::get_if<AssetProcessor::AssetLoadResultCode>(&vertex_pos_result))
    {
        return *err;
    }
    std::vector<glm::vec3> vert_positions = std::get<std::vector<glm::vec3>>(std::move(vertex_pos_result));
    u32 const vertex_count = s_cast<u32>(vert_positions.size());
#pragma endregion

/// NOTE: Load vertex UVs
#pragma region UVS
    auto texcoord0_attrib_iter = gltf_prim.findAttribute(VERT_ATTRIB_TEXCOORD0_NAME);
    bool has_uv = true;
    if (texcoord0_attrib_iter == gltf_prim.attributes.end())
    {
        has_uv = false;
        // return AssetProcessor::AssetLoadResultCode::ERROR_MISSING_VERTEX_TEXCOORD_0;
    }
    fastgltf::Accessor & gltf_vertex_texcoord0_accessor = gltf_asset.accessors.at(texcoord0_attrib_iter->second);
    bool const gltf_vertex_texcoord0_accessor_valid =
        gltf_vertex_texcoord0_accessor.componentType == fastgltf::ComponentType::Float &&
        gltf_vertex_texcoord0_accessor.type == fastgltf::AccessorType::Vec2;
    if (!gltf_vertex_texcoord0_accessor_valid && has_uv)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_TEXCOORD_0;
    }
    std::vector<glm::vec2> vert_texcoord0;
    if (has_uv)
    {
        auto vertex_texcoord0_pos_result = load_accessor_data_from_file<glm::vec2, false>(std::filesystem::path{info.asset_path}.remove_filename(), gltf_asset, gltf_vertex_texcoord0_accessor);
        if (auto const * err = std::get_if<AssetProcessor::AssetLoadResultCode>(&vertex_texcoord0_pos_result))
        {
            return *err;
        }
        vert_texcoord0 = std::get<std::vector<glm::vec2>>(std::move(vertex_texcoord0_pos_result));
    }
    else
    {
        vert_texcoord0 = std::vector<glm::vec2>(vert_positions.size());
    }
    DBG_ASSERT_TRUE_M(vert_texcoord0.size() == vert_positions.size(), "[AssetProcessor::load_mesh()] Mismatched position and uv count");
#pragma endregion

/// NOTE: Load vertex normals
#pragma region NORMALS
    auto normals_attrib_iter = gltf_prim.findAttribute(VERT_ATTRIB_NORMAL_NAME);
    if (normals_attrib_iter == gltf_prim.attributes.end())
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_MISSING_VERTEX_NORMALS;
    }
    fastgltf::Accessor & gltf_vertex_normals_accessor = gltf_asset.accessors.at(normals_attrib_iter->second);
    bool const gltf_vertex_normals_accessor_valid =
        gltf_vertex_normals_accessor.componentType == fastgltf::ComponentType::Float &&
        gltf_vertex_normals_accessor.type == fastgltf::AccessorType::Vec3;
    if (!gltf_vertex_normals_accessor_valid)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_NORMALS;
    }
    auto vertex_normals_pos_result = load_accessor_data_from_file<glm::vec3, false>(std::filesystem::path{info.asset_path}.remove_filename(), gltf_asset, gltf_vertex_normals_accessor);
    if (auto const * err = std::get_if<AssetProcessor::AssetLoadResultCode>(&vertex_normals_pos_result))
    {
        return *err;
    }
    std::vector<glm::vec3> vert_normals = std::get<std::vector<glm::vec3>>(std::move(vertex_normals_pos_result));
    DBG_ASSERT_TRUE_M(vert_normals.size() == vert_positions.size(), "[AssetProcessor::load_mesh()] Mismatched position and uv count");
#pragma endregion

    /// NOTE: Generate meshlets:
    constexpr usize MAX_VERTICES = MAX_VERTICES_PER_MESHLET;
    constexpr usize MAX_TRIANGLES = MAX_TRIANGLES_PER_MESHLET;
    // No clue what cone culling is.
    constexpr float CONE_WEIGHT = 1.0f;
    // TODO: Make this optimization optional!
    {
        std::vector<u32> optimized_indices(lod0_index_buffer.size());
        meshopt_optimizeVertexCache(optimized_indices.data(), lod0_index_buffer.data(), lod0_index_buffer.size(), vertex_count);
        lod0_index_buffer = std::move(optimized_indices);
    }

    std::array<daxa::BufferId, MAX_MESHES_PER_LOD_GROUP> staging_buffers = {};
    std::array<GPUMesh, MAX_MESHES_PER_LOD_GROUP> lods = {};
    u32 lod_count = 0;

    /// ===== Calculate Normalized Vertex Distance =====
    // - When simplifying/ generating lods mesh optimizer takes into account position error AND attribute error (in our case normals)
    // - We need to give the attributes a meaningful weight to mix position and normal errorion
    // - The smaller the triangles, the less the normals matter.
    // - A good estimation for visual impact of normals is their distortion multiplied with the average vertex distance
    // - This is because the normals will not change the visuals past vertex distance, as each vertex normal can only effect the space between two vertices.
    // - We calculate the average vertex distance for lod0 and then estimate the vertex distance for other lods with a heuristic to speed it up.
    f32 modelspace_average_vertex_distance = 0.0f;
    glm::vec3 vertices_max = vert_positions[lod0_index_buffer[0]];
    glm::vec3 vertices_min = vert_positions[lod0_index_buffer[0]];
    for (u32 tri = 0; tri < lod0_index_buffer.size()/3; ++tri)
    {
        glm::vec3 c0 = vert_positions[lod0_index_buffer[tri*3 + 0]];
        glm::vec3 c1 = vert_positions[lod0_index_buffer[tri*3 + 1]];
        glm::vec3 c2 = vert_positions[lod0_index_buffer[tri*3 + 2]];
        vertices_max = glm::max(glm::max(vertices_max, c0), glm::max(c1, c2));
        vertices_min = glm::min(glm::min(vertices_min, c0), glm::min(c1, c2));
        f32 const e0_dst = glm::length(c0 - c1);
        f32 const e1_dst = glm::length(c1 - c2);
        f32 const e2_dst = glm::length(c2 - c0);
        f32 const max_edge = std::max(std::max(e0_dst, e1_dst), e2_dst);
        modelspace_average_vertex_distance += max_edge;
    }
    modelspace_average_vertex_distance /= static_cast<f32>(lod0_index_buffer.size() / 3ull);
    glm::vec3 const vertex_bounds_size = vertices_max - vertices_min;
    f32 const vertex_bounds_scale = std::max(vertex_bounds_size.x, std::max(vertex_bounds_size.y, vertex_bounds_size.z));
    f32 const normalized_average_vertex_distance = modelspace_average_vertex_distance / vertex_bounds_scale;
    f32 const lod0_average_vertex_distance = normalized_average_vertex_distance;
    /// ===== Calculate Normalized Vertex Distance =====

    std::vector<daxa::u32> prev_lod_index_buffer = {};
    for (u32 lod = 0; lod < MAX_MESHES_PER_LOD_GROUP; ++lod)
    {
        std::vector<u32> simplified_indices = {};
        std::vector<u32> * index_buffer = {};
        f32 lod_error = 0.0f;
        if (lod == 0)
        {
            index_buffer = &lod0_index_buffer;
        }
        else
        {

            const u32 lod_index_count = round_up_div(prev_lod_index_buffer.size(), 3 * 2) * 3u;
            simplified_indices.resize(prev_lod_index_buffer.size(), 0u); // Mesh optimizer needs them to be this large for some reason....
            index_buffer = &simplified_indices;
            f32 target_error = std::numeric_limits<f32>::max();
            f32 max_acceptable_error = 0.95f;
            // TODO: Only enable this for meshes that really need it!
            // It completely prevents foliage optimization and we desperately need foliage optimization!
            // It worsenes performance a lot
            // It should only be on for things that need it like street tiles or planes.
            u32 options = meshopt_SimplifyLockBorder;
            f32 result_error = {};

            /// ===== Estimate Average Vertex Distance For LOD ====
            // We assume a simplification rate that halves triangles from lod to lod.
            // In this case, the average vertex distance increases at a rate of sqrt(2) per lod.
            // This gives us this vertex distance estimation function: lod_vertex_distance * sqrt(2)^lod
            // This is intuitive when thinking of merging two equirectangular triangles into one,
            //         x --                              x --  
            //      x  x  x  len: sqrt(2)    ==>      x     x  len: sqrt(2)    
            //   x     x    x --             ==>   x          x --  
            // xxxxxxxxxxxxxxxxx                 xxxxxxxxxxxxxxxxx
            // |    len: 1     |                 |    len: 1     | 
            // In this case the two longest edges before simplification are len sqrt(2)
            // The longest edge of the simplified triangle is len 2.
            f32 const lod_average_normalized_vertex_distance = lod0_average_vertex_distance * std::pow(sqrt(2.0f), lod);
            // - We bias the weight towards the normal a little here with a factor of 2
            // - Typically normals are a little more important for visual error than position as they effect the shading more.
            f32 const MESH_LOD_GEN_NORMAL_IMPORTANCE_FACTOR = 2.0f;
            f32 const lod_normal_weight = lod_average_normalized_vertex_distance * MESH_LOD_GEN_NORMAL_IMPORTANCE_FACTOR;
            /// ===== Estimate Average Vertex Distance For LOD ====

            f32 normal_weights[] = { lod_normal_weight, lod_normal_weight, lod_normal_weight };
            i32 result_index_count = meshopt_simplifyWithAttributes(
                index_buffer->data(), prev_lod_index_buffer.data(), prev_lod_index_buffer.size(), 
                &vert_positions.data()->x, vert_positions.size(), sizeof(glm::vec3), 
                &vert_normals.data()->x, sizeof(glm::vec3), normal_weights, 3, 
                lod_index_count, target_error, options, &result_error);
            lod_error = lods[lod-1].lod_error + result_error;
            if (result_index_count > lod_index_count || result_index_count < 12 || result_error > max_acceptable_error)
            {
                break;
            }
            index_buffer->resize(result_index_count);
        }
        prev_lod_index_buffer = *index_buffer;

        size_t max_meshlets = meshopt_buildMeshletsBound(index_buffer->size(), MAX_VERTICES, MAX_TRIANGLES);
        std::vector<meshopt_Meshlet> meshlets(max_meshlets);
        std::vector<u32> meshlet_indirect_vertices(max_meshlets * MAX_VERTICES);
        std::vector<u8> meshlet_micro_indices(max_meshlets * MAX_TRIANGLES * 3);
        size_t meshlet_count = meshopt_buildMeshlets(
            meshlets.data(),
            meshlet_indirect_vertices.data(),
            meshlet_micro_indices.data(),
            index_buffer->data(),
            index_buffer->size(),
            r_cast<float *>(vert_positions.data()),
            s_cast<usize>(vertex_count),
            sizeof(glm::vec3),
            MAX_VERTICES,
            MAX_TRIANGLES,
            CONE_WEIGHT);
        // TODO: Compute OBBs
        std::vector<BoundingSphere> meshlet_bounds(meshlet_count);
        std::vector<AABB> meshlet_aabbs(meshlet_count);
        glm::vec3 mesh_min_pos;
        glm::vec3 mesh_max_pos;
        for (size_t meshlet_i = 0; meshlet_i < meshlet_count; ++meshlet_i)
        {
            meshopt_Bounds raw_bounds = meshopt_computeMeshletBounds(
                &meshlet_indirect_vertices[meshlets[meshlet_i].vertex_offset],
                &meshlet_micro_indices[meshlets[meshlet_i].triangle_offset],
                meshlets[meshlet_i].triangle_count,
                r_cast<float *>(vert_positions.data()),
                s_cast<usize>(vertex_count),
                sizeof(glm::vec3));
            meshlet_bounds[meshlet_i].center.x = raw_bounds.center[0];
            meshlet_bounds[meshlet_i].center.y = raw_bounds.center[1];
            meshlet_bounds[meshlet_i].center.z = raw_bounds.center[2];
            meshlet_bounds[meshlet_i].radius = raw_bounds.radius;

            glm::vec3 min_pos = vert_positions[meshlet_indirect_vertices[meshlets[meshlet_i].vertex_offset]];
            glm::vec3 max_pos = vert_positions[meshlet_indirect_vertices[meshlets[meshlet_i].vertex_offset]];

            if (meshlet_i == 0)
            {
                mesh_min_pos = vert_positions[meshlet_indirect_vertices[meshlets[0].vertex_offset]];
                mesh_max_pos = vert_positions[meshlet_indirect_vertices[meshlets[0].vertex_offset]];
            }

            for (int vert_i = 1; vert_i < meshlets[meshlet_i].vertex_count; ++vert_i)
            {
                glm::vec3 pos = vert_positions[meshlet_indirect_vertices[meshlets[meshlet_i].vertex_offset + vert_i]];
                min_pos = glm::min(min_pos, pos);
                max_pos = glm::max(max_pos, pos);
            }
            mesh_min_pos = glm::min(mesh_min_pos, min_pos);
            mesh_max_pos = glm::max(mesh_max_pos, max_pos);

            meshlet_aabbs[meshlet_i].center = std::bit_cast<daxa_f32vec3>((max_pos + min_pos) * 0.5f);
            meshlet_aabbs[meshlet_i].size = std::bit_cast<daxa_f32vec3>(max_pos - min_pos);
        }
        AABB mesh_aabb;
        mesh_aabb.center = std::bit_cast<daxa_f32vec3>((mesh_max_pos + mesh_min_pos) * 0.5f);
        mesh_aabb.size = std::bit_cast<daxa_f32vec3>(mesh_max_pos - mesh_min_pos);
        // Trimm array sizes.
        meshopt_Meshlet const & last = meshlets[meshlet_count - 1];
        meshlet_indirect_vertices.resize(last.vertex_offset + last.vertex_count);
        meshlet_micro_indices.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
        meshlets.resize(meshlet_count);

        u32 const total_mesh_buffer_size =
            sizeof(Meshlet) * meshlet_count +
            sizeof(BoundingSphere) * meshlet_count +
            sizeof(AABB) * meshlet_count +
            sizeof(u8) * meshlet_micro_indices.size() +
            sizeof(u32) * meshlet_indirect_vertices.size() +
            sizeof(u32) * index_buffer->size() +
            sizeof(daxa_f32vec3) * vert_positions.size() +
            sizeof(daxa_f32vec2) * vert_texcoord0.size() +
            sizeof(daxa_f32vec3) * vert_normals.size();

        /// NOTE: Fill GPUMesh runtime data
        GPUMesh mesh = {};
        mesh.lod_error = lod_error;

        mesh.aabb = mesh_aabb;
        daxa::DeviceAddress mesh_bda = {};
        daxa::BufferId staging_buffer = {};
        {
            mesh.mesh_buffer = _device.create_buffer({
                .size = s_cast<daxa::usize>(total_mesh_buffer_size),
                .name = std::string(gltf_mesh.name.c_str()) + "." + std::to_string(info.gltf_primitive_index),
            });
            mesh_bda = _device.buffer_device_address(std::bit_cast<daxa::BufferId>(mesh.mesh_buffer)).value();

            staging_buffer = _device.create_buffer({
                .size = s_cast<daxa::usize>(total_mesh_buffer_size),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = std::string(gltf_mesh.name.c_str()) + "." + std::to_string(info.gltf_primitive_index) + " staging",
            });
        }
        auto staging_ptr = _device.buffer_host_address(staging_buffer).value();

        u32 accumulated_offset = 0;
        // ---
        mesh.meshlets = mesh_bda + accumulated_offset;
        std::memcpy(
            staging_ptr + accumulated_offset,
            meshlets.data(),
            meshlets.size() * sizeof(Meshlet));
        accumulated_offset += sizeof(Meshlet) * meshlet_count;
        // ---
        mesh.meshlet_bounds = mesh_bda + accumulated_offset;
        std::memcpy(
            staging_ptr + accumulated_offset,
            meshlet_bounds.data(),
            meshlet_bounds.size() * sizeof(BoundingSphere));
        accumulated_offset += sizeof(BoundingSphere) * meshlet_count;
        // ---
        mesh.meshlet_aabbs = mesh_bda + accumulated_offset;
        std::memcpy(
            staging_ptr + accumulated_offset,
            meshlet_aabbs.data(),
            meshlet_aabbs.size() * sizeof(AABB));
        accumulated_offset += sizeof(AABB) * meshlet_count;
        // ---
        DBG_ASSERT_TRUE_M(meshlet_micro_indices.size() % 4 == 0, "Thats crazy");
        mesh.micro_indices = mesh_bda + accumulated_offset;
        std::memcpy(
            staging_ptr + accumulated_offset,
            meshlet_micro_indices.data(),
            meshlet_micro_indices.size() * sizeof(u8));
        accumulated_offset += sizeof(u8) * meshlet_micro_indices.size();
        // ---
        mesh.indirect_vertices = mesh_bda + accumulated_offset;
        std::memcpy(
            staging_ptr + accumulated_offset,
            meshlet_indirect_vertices.data(),
            meshlet_indirect_vertices.size() * sizeof(u32));
        accumulated_offset += sizeof(u32) * meshlet_indirect_vertices.size();
        // ---
        mesh.primitive_indices = mesh_bda + accumulated_offset;
        std::memcpy(
            staging_ptr + accumulated_offset,
            index_buffer->data(),
            index_buffer->size() * sizeof(daxa_u32));
        accumulated_offset += sizeof(daxa_u32) * index_buffer->size();
        // ---
        mesh.vertex_positions = mesh_bda + accumulated_offset;
        std::memcpy(
            staging_ptr + accumulated_offset,
            vert_positions.data(),
            vert_positions.size() * sizeof(daxa_f32vec3));
        accumulated_offset += sizeof(daxa_f32vec3) * vert_positions.size();
        // ---
        mesh.vertex_uvs = mesh_bda + accumulated_offset;
        std::memcpy(
            staging_ptr + accumulated_offset,
            vert_texcoord0.data(),
            vert_texcoord0.size() * sizeof(daxa_f32vec2));
        accumulated_offset += sizeof(daxa_f32vec2) * vert_texcoord0.size();
        // ---
        mesh.vertex_normals = mesh_bda + accumulated_offset;
        std::memcpy(
            staging_ptr + accumulated_offset,
            vert_normals.data(),
            vert_normals.size() * sizeof(daxa_f32vec3));
        accumulated_offset += sizeof(daxa_f32vec3) * vert_normals.size();
        // ---
        mesh.material_index = info.material_manifest_index;
        mesh.meshlet_count = meshlet_count;
        mesh.vertex_count = vertex_count;
        mesh.primitive_count = index_buffer->size() / 3;

        lods[lod] = mesh;
        staging_buffers[lod] = staging_buffer;
        lod_count += 1;
    }

    /// NOTE: Append the processed mesh to the upload queue.
    {
        std::lock_guard<std::mutex> lock{*_mesh_upload_mutex};
        _upload_mesh_queue.push_back(MeshLodGroupUploadInfo{
            .staging_buffers = staging_buffers,
            .lods = lods,
            .lod_count = lod_count,
            .mesh_lod_manifest_index = info.mesh_lod_manifest_index});
    }
    return AssetProcessor::AssetLoadResultCode::SUCCESS;
}

auto AssetProcessor::record_gpu_load_processing_commands() -> RecordCommandsRet
{
    RecordCommandsRet ret = {};
    {
        std::lock_guard<std::mutex> lock{*_mesh_upload_mutex};
        ret.uploaded_meshes = std::move(_upload_mesh_queue);
        _upload_mesh_queue = {};
    }
    auto recorder = _device.create_command_recorder({});
#pragma region RECORD_MESH_UPLOAD_COMMANDS
    for (MeshLodGroupUploadInfo & mesh_upload : ret.uploaded_meshes)
    {
        for (u32 lod = 0; lod < mesh_upload.lod_count; ++lod)
        {
            recorder.copy_buffer_to_buffer({
                .src_buffer = mesh_upload.staging_buffers[lod],
                .dst_buffer = std::bit_cast<daxa::BufferId>(mesh_upload.lods[lod].mesh_buffer),
                .size = _device.buffer_info(std::bit_cast<daxa::BufferId>(mesh_upload.lods[lod].mesh_buffer)).value().size,
            });
            recorder.destroy_buffer_deferred(mesh_upload.staging_buffers[lod]);
        }
    }
    recorder.pipeline_barrier({
        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_access = daxa::AccessConsts::READ_WRITE,
    });
#pragma endregion

#pragma region RECORD_TEXTURE_UPLOAD_COMMANDS
    {
        std::lock_guard<std::mutex> lock{*_texture_upload_mutex};
        ret.uploaded_textures = std::move(_upload_texture_queue);
        _upload_texture_queue = {};
    }
    for (LoadedTextureInfo const & texture_upload : ret.uploaded_textures)
    {
        daxa::ImageViewInfo image_view_info = _device.image_view_info(texture_upload.dst_image.default_view()).value();
        /// TODO: If we are generating mips this will need to change
        recorder.pipeline_barrier_image_transition({
            .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .image_slice = image_view_info.slice,
            .image_id = texture_upload.dst_image,
        });
    }
    for (LoadedTextureInfo const & texture_upload : ret.uploaded_textures)
    {
        daxa::ImageInfo image_info = _device.image_info(texture_upload.dst_image).value();
        for (u32 mip = 0; mip < texture_upload.mips_to_copy; ++mip)
        {
            u32 width = std::max(1u, image_info.size.x >> mip);
            u32 height = std::max(1u, image_info.size.y >> mip);
            u32 depth = std::max(1u, image_info.size.z >> mip);
            recorder.copy_buffer_to_image({
                .buffer = texture_upload.staging_buffer,
                .buffer_offset = texture_upload.mip_copy_offsets[mip],
                .image = texture_upload.dst_image,
                .image_slice = {
                    .mip_level = mip,
                },
                .image_offset = {0, 0, 0},
                .image_extent = {width, height, depth},
            });
        }
        recorder.destroy_buffer_deferred(texture_upload.staging_buffer);
    }
    for (LoadedTextureInfo const & texture_upload : ret.uploaded_textures)
    {
        recorder.pipeline_barrier_image_transition({
            .src_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_access = daxa::AccessConsts::TOP_OF_PIPE_READ_WRITE,
            .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .dst_layout = daxa::ImageLayout::READ_ONLY_OPTIMAL,
            .image_id = texture_upload.dst_image,
        });
    }
#pragma endregion
    ret.upload_commands = recorder.complete_current_commands();
    return ret;
}