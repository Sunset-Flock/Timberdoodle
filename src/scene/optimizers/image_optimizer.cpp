#include "image_optimizer.hpp"

#include <cstring>
#include <bit>
#include <span>
#include <optional>

#include <ktx.h>
#include <png.h>

namespace
{
enum struct ChannelDataType
{
    SIGNED_INT,
    UNSIGNED_INT,
    FLOATING_POINT
};

struct PixelInfo
{
    u8 channel_count = {};
    u8 channel_byte_size = {};
    ChannelDataType channel_data_type = {};
    bool load_as_srgb = {};
};

constexpr auto daxa_image_format_from_pixel_info(PixelInfo const & info) -> daxa::Format
{
    std::array<std::array<std::array<daxa::Format, 3>, 4>, 3> translation = {
        // BYTE SIZE 1
        std::array{
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R8_UNORM, daxa::Format::R8_SINT, daxa::Format::UNDEFINED}},
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R8G8_UNORM, daxa::Format::R8G8_SINT, daxa::Format::UNDEFINED}},
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R8G8B8A8_UNORM, daxa::Format::R8G8B8A8_SINT, daxa::Format::UNDEFINED}},
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R8G8B8A8_UNORM, daxa::Format::R8G8B8A8_SINT, daxa::Format::UNDEFINED}},
        },
        // BYTE SIZE 2
        std::array{
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R16_UINT, daxa::Format::R16_SINT, daxa::Format::R16_SFLOAT}},
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R16G16_UINT, daxa::Format::R16G16_SINT, daxa::Format::R16G16_SFLOAT}},
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R16G16B16A16_UINT, daxa::Format::R16G16B16A16_SINT, daxa::Format::R16G16B16A16_SFLOAT}},
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R16G16B16A16_UINT, daxa::Format::R16G16B16A16_SINT, daxa::Format::R16G16B16A16_SFLOAT}},
        },
        // BYTE SIZE 4
        std::array{
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R32_UINT, daxa::Format::R32_SINT, daxa::Format::R32_SFLOAT}},
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R32G32_UINT, daxa::Format::R32G32_SINT, daxa::Format::R32G32_SFLOAT}},
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R32G32B32A32_UINT, daxa::Format::R32G32B32A32_SINT, daxa::Format::R32G32B32A32_SFLOAT}},
            std::array{/* CHANNEL FORMAT */ std::array{daxa::Format::R32G32B32A32_UINT, daxa::Format::R32G32B32A32_SINT, daxa::Format::R32G32B32A32_SFLOAT}},
        },
    };
    u8 channel_byte_size_idx{};
    switch (info.channel_byte_size)
    {
        case 1: channel_byte_size_idx = 0u; break;
        case 2: channel_byte_size_idx = 1u; break;
        case 4: channel_byte_size_idx = 2u; break;
        default: return daxa::Format::UNDEFINED;
    }
    u8 const channel_count_idx = info.channel_count - 1;
    u8 channel_format_idx{};
    switch (info.channel_data_type)
    {
        case ChannelDataType::UNSIGNED_INT: channel_format_idx = 0u; break;
        case ChannelDataType::SIGNED_INT: channel_format_idx = 1u; break;
        case ChannelDataType::FLOATING_POINT: channel_format_idx = 2u; break;
        default: return daxa::Format::UNDEFINED;
    }
    auto format = translation[channel_byte_size_idx][channel_count_idx][channel_format_idx];
    if (info.load_as_srgb)
    {
        format = format == daxa::Format::R8_UNORM ? daxa::Format::R8_SRGB : format;
        format = format == daxa::Format::R8G8_UNORM ? daxa::Format::R8G8_SRGB : format;
        format = format == daxa::Format::R8G8B8A8_UNORM ? daxa::Format::R8G8B8A8_SRGB : format;
    }
    return format;
}

// Decoded raw pixels (intermediate result of decoding a PNG before wrapping into ProcessedImage).
struct DecodedPixels
{
    std::vector<std::byte> data = {};
    u32 width = {};
    u32 height = {};
    u32 channel_count = {};
    u32 channel_byte_size = {};
    bool srgb = {};
};

// Decode PNG file bytes into raw RGB(A) pixels.
auto decode_png(std::span<std::byte const> png_bytes, bool load_as_srgb) -> std::optional<DecodedPixels>
{
    if (png_sig_cmp((png_bytep)png_bytes.data(), 0, 8))
    {
        return std::nullopt;
    }
    auto png_alloc = [](png_structp, png_size_t size) -> png_voidp { return (png_voidp *)malloc(size); };
    auto png_free = [](png_structp, png_voidp ptr) { free(ptr); };
    auto error_fn = []([[maybe_unused]] png_structp png_ptr, [[maybe_unused]] png_const_charp error_msg) { DBG_ASSERT_TRUE_M(false, error_msg); };
    auto data_fn = [](png_structp png_ptr, png_bytep data, png_size_t length)
    {
        std::byte const *& raw_data_ptr = *(std::byte const **)png_get_io_ptr(png_ptr);
        memcpy(data, raw_data_ptr, length);
        raw_data_ptr += length;
    };

    auto png_ptr = png_create_read_struct_2(PNG_LIBPNG_VER_STRING, nullptr, error_fn, NULL, NULL, png_alloc, png_free);
    DBG_ASSERT_TRUE_M(png_ptr != nullptr, "Failed to create PNG load context");
    png_set_error_fn(png_ptr, nullptr, error_fn, NULL);
    png_set_sig_bytes(png_ptr, 8);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    std::byte const * raw_data_ptr = png_bytes.data() + 8;
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

    DecodedPixels ret = {};
    ret.width = width;
    ret.height = height;
    ret.channel_count = s_cast<u32>(channel_count);
    ret.channel_byte_size = s_cast<u32>(bit_depth / 8);
    ret.srgb = load_as_srgb;
    ret.data.resize(width * height * channel_count * (bit_depth / 8));

    std::vector<png_bytep> row_pointers(height);
    for (u32 y = 0; y < height; y++)
    {
        row_pointers[y] = (png_bytep)(ret.data.data() + width * y * channel_count * bit_depth / 8);
    }
    png_read_image(png_ptr, row_pointers.data());
    return ret;
}

// Wrap decoded pixels into upload-ready processed memory.
// TODO: BC-compress here (using tex_compression); for now pixels are passed through uncompressed.
auto pixels_to_processed(DecodedPixels & pixels, std::string name) -> ProcessedImage
{
    daxa::Format const daxa_image_format = daxa_image_format_from_pixel_info({
        .channel_count = s_cast<u8>(pixels.channel_count),
        .channel_byte_size = s_cast<u8>(pixels.channel_byte_size),
        .channel_data_type = ChannelDataType::UNSIGNED_INT,
        .load_as_srgb = pixels.srgb,
    });

    ProcessedImage ret = {};
    ret.src_data = std::move(pixels.data);
    ret.mips_to_copy = 1;
    ret.image_info = {
        .dimensions = 2,
        .format = daxa_image_format,
        .size = {pixels.width, pixels.height, 1},
        /// TODO: Add support for generating mip levels
        .mip_level_count = 1,
        .array_layer_count = 1,
        .sample_count = 1,
        .usage =
            daxa::ImageUsageFlagBits::TRANSFER_DST |
            daxa::ImageUsageFlagBits::SHADER_SAMPLED,
        .name = std::move(name),
    };
    return ret;
}

// KTX2 -> basis transcode to BCn.
auto ktx_transcode(std::span<std::byte const> ktx2_bytes, TextureMaterialType type, std::string name) -> std::variant<ImageOptimizeError, ProcessedImage>
{
    ktx_transcode_fmt_e transcode_format;
    switch (type)
    {
        case TextureMaterialType::NORMAL:          transcode_format = KTX_TTF_BC5_RG; break;
        case TextureMaterialType::DIFFUSE_OPACITY: transcode_format = KTX_TTF_BC4_R; break;
        default:                                   transcode_format = KTX_TTF_BC7_RGBA; break;
    }

    ktxTexture2 * texture;
    KTX_error_code result;

    result = ktxTexture2_CreateFromMemory(
        r_cast<ktx_uint8_t const *>(ktx2_bytes.data()),
        ktx2_bytes.size(),
        KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
        &texture);
    if (result != KTX_SUCCESS)
    {
        return ImageOptimizeError::FAILED_TO_PROCESS_KTX;
    }

    ktx_transcode_flags flags = KTX_TF_HIGH_QUALITY;
    flags |= type == TextureMaterialType::DIFFUSE_OPACITY ? KTX_TF_TRANSCODE_ALPHA_DATA_TO_OPAQUE_FORMATS : 0u;
    result = ktxTexture2_TranscodeBasis(texture, transcode_format, flags);
    if (result != KTX_SUCCESS)
    {
        return ImageOptimizeError::FAILED_TO_PROCESS_KTX;
    }

    u32 const numLevels = texture->numLevels;
    u32 const numLayers = texture->numLayers;
    u32 const baseWidth = texture->baseWidth;
    u32 const baseHeight = texture->baseHeight;
    u32 const baseDepth = texture->baseDepth;

    ProcessedImage ret = {};
    ret.src_data.resize(texture->dataSize);
    ktx_uint8_t * image_ktx_data = ktxTexture_GetData(ktxTexture(texture));
    daxa::Format const format = std::bit_cast<daxa::Format>(texture->vkFormat);
    ret.image_info = {
        .flags = {},
        .dimensions = 2,
        .format = format,
        .size = {baseWidth, baseHeight, baseDepth},
        .mip_level_count = numLevels,
        .array_layer_count = numLayers,
        .sample_count = 1,
        .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED |
                 daxa::ImageUsageFlagBits::TRANSFER_DST,
        .memory_flags = {},
        .name = std::move(name),
    };
    ret.mips_to_copy = texture->numLevels;
    for (u32 mip = 0; mip < texture->numLevels; ++mip)
    {
        u32 const layer = 0;
        u32 const faceSlice = 0;
        usize offset = {};
        result = ktxTexture_GetImageOffset(ktxTexture(texture), mip, layer, faceSlice, &offset);
        if (result != KTX_SUCCESS)
        {
            ktxTexture_Destroy(ktxTexture(texture));
            return ImageOptimizeError::FAILED_TO_PROCESS_KTX;
        }
        usize size = ktxTexture_GetImageSize(ktxTexture(texture), mip);
        std::memcpy(ret.src_data.data() + offset, image_ktx_data + offset, size);
        ret.mip_copy_offsets[mip] = offset;
    }

    ktxTexture_Destroy(ktxTexture(texture));
    return ret;
}
} // namespace

auto process_image(OptimizeImageInfo const & info) -> std::variant<ImageOptimizeError, ProcessedImage>
{
    // Decode/transcode into GPU-ready cooked CPU memory and return it. Writing the .tido is the caller's
    // job (write_texture_tido), mirroring optimize_mesh -> write_mesh_tido.
    switch (info.format)
    {
        case ImageFileFormat::KTX2:
        {
            return ktx_transcode(info.data, info.type, info.name);
        }
        case ImageFileFormat::PNG:
        {
            bool const load_as_srgb = info.type == TextureMaterialType::DIFFUSE;
            auto decoded = decode_png(info.data, load_as_srgb);
            if (!decoded.has_value())
            {
                return ImageOptimizeError::FAILED_TO_DECODE_PNG;
            }
            return pixels_to_processed(decoded.value(), info.name);
        }
    }
    return ImageOptimizeError::FAILED_TO_DECODE_PNG;
}
