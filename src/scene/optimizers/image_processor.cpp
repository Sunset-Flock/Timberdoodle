#include "image_processor.hpp"

#include <cstring>
#include <bit>
#include <span>
#include <optional>
#include <cmath>
#include <algorithm>

#include <ktx.h>
#include <png.h>

#include "../../multithreading/thread_pool.hpp"
#include "../../shader_shared/shared.inl" // round_up_div

// ================================ Image parsing ===================================
namespace
{
auto image_parse_ktx(std::span<std::byte const> data) -> std::variant<ImageProcessResult, TidoImageWithData>
{
    ktxTexture2 * texture;

    // Explicit NO_FLAGS so that ktx does not load and de-compress the texture initially.
    KTX_error_code create_from_memory_result = ktxTexture2_CreateFromMemory(
        r_cast<ktx_uint8_t const *>(data.data()),
        data.size(),
        KTX_TEXTURE_CREATE_NO_FLAGS,
        &texture);

    if(create_from_memory_result != KTX_SUCCESS) { return ImageProcessResult::FAILED_TO_PARSE_KTX; }

    if(ktxTexture2_NeedsTranscoding(texture))
    {
        ktxTexture2_Destroy(texture);
        DBG_ASSERT_TRUE_M(false, "image_parse_ktx: this image needs transcoding (it uses a compression scheme that is not directly parseable)");
        return ImageProcessResult::FAILED_KTX_NEEDS_TRANSCODE;
    }

    TidoImageWithData image_with_data = {};
    image_with_data.data = std::vector<std::byte>(ktxTexture_GetDataSizeUncompressed(ktxTexture(texture)));

    KTX_error_code load_image_data_result = ktxTexture2_LoadImageData(texture, r_cast<ktx_uint8_t*>(image_with_data.data.data()), image_with_data.data.size());
    if(load_image_data_result != KTX_SUCCESS) { ktxTexture2_Destroy(texture); return ImageProcessResult::FAILED_TO_LOAD_KTX; }

    image_with_data.descriptor.info.format = std::bit_cast<daxa::Format>(texture->vkFormat);
    image_with_data.descriptor.info.dimensions = texture->numDimensions;
    image_with_data.descriptor.info.size = {texture->baseWidth, texture->baseHeight, texture->baseDepth};
    image_with_data.descriptor.info.mip_level_count = texture->numLevels;
    image_with_data.descriptor.info.array_layer_count = texture->numLayers;

    image_with_data.descriptor.subresources.resize(texture->numLevels * texture->numLayers);

    DBG_ASSERT_TRUE_M(texture->baseDepth == 1, "image_parse_ktx: 3D textures are not yet supported properly");
    DBG_ASSERT_TRUE_M(texture->numFaces == 1, "image_parse_ktx: Cubemap textures are not yet supported properly");

    for(u32 mip = 0; mip < texture->numLevels; ++mip)
    {
        for(u32 layer = 0; layer < texture->numLayers; ++layer)
        {
            u64 offset = {};
            KTX_error_code get_image_offset_result = ktxTexture_GetImageOffset(ktxTexture(texture), mip, layer, 0, &offset);
            if(get_image_offset_result != KTX_SUCCESS) { ktxTexture2_Destroy(texture); return ImageProcessResult::FAILED_TO_PARSE_KTX; }

            u32 const byte_size = s_cast<u32>(ktxTexture_GetImageSize(ktxTexture(texture), mip));

            u32 const subresource_index = image_with_data.descriptor.layer_mip_to_subresource_index(layer, mip);
            image_with_data.descriptor.subresources[subresource_index] = {.offset = offset, .byte_size = byte_size};
        }
    }
    ktxTexture2_Destroy(texture);

    return image_with_data;
}

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
    bool is_srgb = {};
};

constexpr auto image_format_from_pixel_info(PixelInfo const & info) -> daxa::Format
{
    DBG_ASSERT_TRUE_M(info.channel_count >= 1 && info.channel_count <= 4, "image_format_from_pixel_info: channel count must be between 1 and 4");
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
        default:
            DBG_ASSERT_TRUE_M(false, "Unhandled ChannelDataType");
            return daxa::Format::UNDEFINED;
    }
    auto format = translation[channel_byte_size_idx][channel_count_idx][channel_format_idx];
    if (info.is_srgb)
    {
        format = format == daxa::Format::R8_UNORM ? daxa::Format::R8_SRGB : format;
        format = format == daxa::Format::R8G8_UNORM ? daxa::Format::R8G8_SRGB : format;
        format = format == daxa::Format::R8G8B8A8_UNORM ? daxa::Format::R8G8B8A8_SRGB : format;
    }
    return format;
}

auto image_parse_png(std::span<std::byte const>  png_bytes) -> std::variant<ImageProcessResult, TidoImageWithData>
{
    if (png_sig_cmp((png_bytep)png_bytes.data(), 0, 8)) { return ImageProcessResult::FAILED_TO_PARSE_PNG; }

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
    if (interlace_type != PNG_INTERLACE_NONE) png_set_interlace_handling(png_ptr);

    png_read_update_info(png_ptr, info_ptr);
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, &interlace_type, nullptr, nullptr);
    channel_count = png_get_channels(png_ptr, info_ptr);

    DBG_ASSERT_TRUE_M(bit_depth == 16 || bit_depth == 8, "Unexpected PNG bit depth: " + std::to_string(bit_depth) + " (expected 8 or 16)");

    PixelInfo pixel_info = {};
    pixel_info.channel_count = s_cast<u8>(channel_count);
    pixel_info.channel_byte_size = s_cast<u8>(bit_depth / 8);
    pixel_info.channel_data_type = ChannelDataType::UNSIGNED_INT;
    pixel_info.is_srgb = false;

    TidoImageWithData ret = {};
    ret.descriptor.info.format = image_format_from_pixel_info(pixel_info);
    ret.descriptor.info.dimensions = 2;
    ret.descriptor.info.size = {width, height, 1};
    ret.descriptor.info.mip_level_count = 1;
    ret.descriptor.info.array_layer_count = 1;
    ret.data.resize(width * height * channel_count * (bit_depth / 8));
    ret.descriptor.subresources = {{.offset = 0, .byte_size = s_cast<u32>(ret.data.size())}};

    std::vector<png_bytep> row_pointers(height);
    for (u32 row_index = 0; row_index < height; row_index++)
    {
        row_pointers[row_index] = (png_bytep)(ret.data.data() + width * row_index * channel_count * bit_depth / 8);
    }
    png_read_image(png_ptr, row_pointers.data());

    return ImageProcessResult::SUCCESS;
}
}

auto image_parse(ImageParseInfo const & parse_info) -> std::variant<ImageProcessResult, TidoImageWithData>
{
    switch(parse_info.source_format)
    {
        case ImageFileFormat::KTX2:
        {
            return image_parse_ktx(parse_info.src_data);
        }
        case ImageFileFormat::PNG:
        {
            return image_parse_png(parse_info.src_data);
        }
        default:
        {
            DBG_ASSERT_TRUE_M(false, "image_parse: unhandled ImageFileFormat");
            return ImageProcessResult::FAILED_UNHANDLED_FILE_FORMAT;
        }
    }
}

// ================================ Mipmap generation ===============================
namespace
{
inline auto srgb_to_linear(f32 value) -> f32
{
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}
inline auto linear_to_srgb(f32 value) -> f32
{
    return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

// Read/write one channel sample as a normalized [0,1] float, for 8- or 16-bit unsigned channels.
inline auto read_normalized(std::byte const * sample_ptr, u32 channel_byte_size) -> f32
{
    if (channel_byte_size == 2)
    {
        u16 sample = {};
        std::memcpy(&sample, sample_ptr, 2);
        return s_cast<f32>(sample) / 65535.0f;
    }
    return s_cast<f32>(s_cast<u8>(*sample_ptr)) / 255.0f;
}
inline void write_normalized(std::byte * sample_ptr, u32 channel_byte_size, f32 value)
{
    value = std::clamp(value, 0.0f, 1.0f);
    if (channel_byte_size == 2)
    {
        u16 const quantized = s_cast<u16>(std::lround(value * 65535.0f));
        std::memcpy(sample_ptr, &quantized, 2);
    }
    else
    {
        *sample_ptr = s_cast<std::byte>(s_cast<u8>(std::lround(value * 255.0f)));
    }
}

struct FormatInfo
{
    u32 channel_count = {};
    u32 channel_byte_size = {};
    bool is_srgb = {};
    u32 block_width = {};
    u32 block_height = {};

    // 1x1 block (uncompressed) or 4x4 block (BC compressed) byte size.
    u32 block_byte_size = {};
};

auto get_format_info(daxa::Format format) -> FormatInfo
{
    switch (format)
    {
        case daxa::Format::R8_UNORM:           return {.channel_count = 1, .channel_byte_size = 1, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 1};
        case daxa::Format::R8_SRGB:            return {.channel_count = 1, .channel_byte_size = 1, .is_srgb = true,  .block_width = 1, .block_height = 1, .block_byte_size = 1};
        case daxa::Format::R8G8_UNORM:         return {.channel_count = 2, .channel_byte_size = 1, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R8G8_SRGB:          return {.channel_count = 2, .channel_byte_size = 1, .is_srgb = true,  .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R8G8B8A8_UNORM:     return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R8G8B8A8_SRGB:      return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = true,  .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R16_UINT:           return {.channel_count = 1, .channel_byte_size = 2, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R16G16_UINT:        return {.channel_count = 2, .channel_byte_size = 2, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R16G16B16A16_UINT:  return {.channel_count = 4, .channel_byte_size = 2, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 8};
        case daxa::Format::BC1_RGB_UNORM_BLOCK:
        case daxa::Format::BC1_RGB_SRGB_BLOCK:
        case daxa::Format::BC1_RGBA_UNORM_BLOCK:
        case daxa::Format::BC1_RGBA_SRGB_BLOCK:
        case daxa::Format::BC4_UNORM_BLOCK:
        case daxa::Format::BC4_SNORM_BLOCK:    return {.channel_count = 0, .channel_byte_size = 0, .is_srgb = false, .block_width = 4, .block_height = 4, .block_byte_size = 8};
        case daxa::Format::BC2_UNORM_BLOCK:
        case daxa::Format::BC2_SRGB_BLOCK:
        case daxa::Format::BC3_UNORM_BLOCK:
        case daxa::Format::BC3_SRGB_BLOCK:
        case daxa::Format::BC5_UNORM_BLOCK:
        case daxa::Format::BC5_SNORM_BLOCK:
        case daxa::Format::BC6H_UFLOAT_BLOCK:
        case daxa::Format::BC6H_SFLOAT_BLOCK:
        case daxa::Format::BC7_UNORM_BLOCK:
        case daxa::Format::BC7_SRGB_BLOCK:     return {.channel_count = 0, .channel_byte_size = 0, .is_srgb = false, .block_width = 4, .block_height = 4, .block_byte_size = 16};
        default:
            DBG_ASSERT_TRUE_M(false, "get_format_info: Unhandled format");
            return {};
    }
}

struct DownsampleImageTask final : Task
{
    static constexpr u32 TARGET_TEXELS_PER_CHUNK = 8192;

    DownsampleImageInfo info = {};
    FormatInfo format_info = {};
    u32 dst_width = {};
    u32 dst_height = {};
    u32 rows_per_chunk = {};

    DownsampleImageTask(DownsampleImageInfo const & info)
        : info{info}, format_info{get_format_info(info.format)}
    {
        if(format_info.channel_count == 0 || format_info.channel_byte_size == 0)
        {
            DBG_ASSERT_TRUE_M(false, "DownsampleImageTask: format is not a supported mip-generation source");
        }
        dst_width = std::max(1u, info.src_dimensions.x >> 1);
        dst_height = std::max(1u, info.src_dimensions.y >> 1);
        rows_per_chunk = std::max(1u, TARGET_TEXELS_PER_CHUNK / std::max(1u, dst_width));
        chunk_count = round_up_div(dst_height, rows_per_chunk);
    }

    virtual void callback(u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        u32 const channel_count = format_info.channel_count;
        u32 const channel_byte_size = format_info.channel_byte_size;
        bool const srgb = format_info.is_srgb;
        u32 const src_width = info.src_dimensions.x;
        u32 const src_height = info.src_dimensions.y;
        u32 const src_row_stride = src_width * channel_count * channel_byte_size;
        u32 const dst_row_stride = dst_width * channel_count * channel_byte_size;

        std::byte const * src_data = info.src_data.data();
        std::byte * dst_data = info.dst_data.data();

        u32 const dst_y_begin = chunk_index * rows_per_chunk;
        u32 const dst_y_end = std::min(dst_y_begin + rows_per_chunk, dst_height);
        for (u32 dst_y = dst_y_begin; dst_y < dst_y_end; ++dst_y)
        {
            for (u32 dst_x = 0; dst_x < dst_width; ++dst_x)
            {
                for (u32 channel = 0; channel < channel_count; ++channel)
                {
                    bool const is_srgb_color = srgb && channel < 3u;
                    f32 channel_sum = 0.0f;
                    for (u32 tap_y = 0; tap_y < 2u; ++tap_y)
                    {
                        for (u32 tap_x = 0; tap_x < 2u; ++tap_x)
                        {
                            u32 const src_x = std::min(dst_x * 2u + tap_x, src_width - 1u);
                            u32 const src_y = std::min(dst_y * 2u + tap_y, src_height - 1u);
                            std::byte const * src_sample_ptr = &src_data[src_y * src_row_stride + (src_x * channel_count + channel) * channel_byte_size];
                            f32 sample = read_normalized(src_sample_ptr, channel_byte_size);
                            if (is_srgb_color) { sample = srgb_to_linear(sample); }
                            channel_sum += sample;
                        }
                    }
                    f32 average = channel_sum * 0.25f;
                    if (is_srgb_color) { average = linear_to_srgb(average); }
                    std::byte * dst_sample_ptr = &dst_data[dst_y * dst_row_stride + (dst_x * channel_count + channel) * channel_byte_size];
                    write_normalized(dst_sample_ptr, channel_byte_size, average);
                }
            }
        }
    }
};
}

auto downsample_image(DownsampleImageInfo const & info) -> std::shared_ptr<Task>
{
    FormatInfo const format_info = get_format_info(info.format);
    DBG_ASSERT_TRUE_M(format_info.channel_count != 0 && format_info.channel_byte_size != 0, "downsample_image: format is not a supported mip-generation source");

    u32 const texel_byte_size = format_info.block_byte_size;
    u32 const dst_width = std::max(1u, info.src_dimensions.x >> 1);
    u32 const dst_height = std::max(1u, info.src_dimensions.y >> 1);

    DBG_ASSERT_TRUE_M(info.src_data.size() == s_cast<usize>(info.src_dimensions.x) * info.src_dimensions.y * texel_byte_size,
        "downsample_image: src_data size does not match src_dimensions/format");
    DBG_ASSERT_TRUE_M(info.dst_data.size() == s_cast<usize>(dst_width) * dst_height * texel_byte_size,
        "downsample_image: dst_data size does not match the halved src_dimensions/format");

    return std::make_shared<DownsampleImageTask>(info);
}

void image_resize_for_mipmaps(TidoImageWithData & image, u32 mip_count)
{
    DBG_ASSERT_TRUE_M(image.descriptor.info.array_layer_count <= 1, "image_resize_for_mipmaps: array/layered images are not yet supported");

    FormatInfo const layout = get_format_info(image.descriptor.info.format);
    u32 const base_width = image.descriptor.info.size.x;
    u32 const base_height = image.descriptor.info.size.y;

    // Lay out every level contiguously up front so the buffer never reallocates while a running task holds
    // pointers into it. One block formula covers both layouts: uncompressed is a 1x1 block of texel bytes, a
    // BC format a 4x4 block of 8/16 bytes.
    std::vector<TidoImageDescriptor::SubresourceEntry> subresources(mip_count);
    u64 total_size = 0;
    for (u32 mip = 0; mip < mip_count; ++mip)
    {
        u32 const mip_width = std::max(1u, base_width >> mip);
        u32 const mip_height = std::max(1u, base_height >> mip);
        u64 const byte_size = s_cast<u64>(round_up_div(mip_width, layout.block_width)) * round_up_div(mip_height, layout.block_height) * layout.block_byte_size;
        subresources[mip] = {.offset = total_size, .byte_size = s_cast<u32>(byte_size)};
        total_size += byte_size;
    }
    // The existing buffer must be a valid prefix of the new chain: either empty (an output image, its levels
    // filled later) or exactly level 0 (an already-decoded source), which the resize preserves at offset 0.
    DBG_ASSERT_TRUE_M(image.data.empty() || image.data.size() == subresources[0].byte_size,
        "image_resize_for_mipmaps: existing data is neither empty nor exactly level 0 for its format/size");

    TidoImageWithData ret = std::move(image);
    ret.data.resize(total_size);
    ret.descriptor.subresources = std::move(subresources);
    ret.descriptor.info.mip_level_count = mip_count;
}

namespace
{
struct RemapChannelsTask final : Task
{
    static constexpr u32 TARGET_TEXELS_PER_CHUNK = 8192;

    RemapChannelsInfo info = {};
    FormatInfo format_info = {};
    u32 rows_per_chunk = {};

    RemapChannelsTask(RemapChannelsInfo const & info)
        : info{info}, format_info{get_format_info(info.format)}
    {
        if(format_info.channel_count == 0 || format_info.channel_byte_size == 0)
        {
            DBG_ASSERT_TRUE_M(false, "RemapChannelsTask: format is not a supported remap source");
        }
        rows_per_chunk = std::max(1u, TARGET_TEXELS_PER_CHUNK / std::max(1u, info.dimensions.x));
        chunk_count = round_up_div(info.dimensions.y, rows_per_chunk);
    }

    virtual void callback(u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        u32 const src_channel_count = format_info.channel_count;
        u32 const channel_byte_size = format_info.channel_byte_size;
        u32 const dst_channel_count = s_cast<u32>(info.channel_mapping.size());
        u32 const width = info.dimensions.x;

        std::byte const * src_data = info.src_data.data();
        std::byte * dst_data = info.dst_data.data();

        u32 const y_begin = chunk_index * rows_per_chunk;
        u32 const y_end = std::min(y_begin + rows_per_chunk, info.dimensions.y);
        for (u32 texel_y = y_begin; texel_y < y_end; ++texel_y)
        {
            for (u32 texel_x = 0; texel_x < width; ++texel_x)
            {
                usize const texel_index = s_cast<usize>(texel_y) * width + texel_x;
                for (u32 dst_channel = 0; dst_channel < dst_channel_count; ++dst_channel)
                {
                    u32 const src_channel = info.channel_mapping[dst_channel];
                    std::byte const * src_sample_ptr = src_data + (texel_index * src_channel_count + src_channel) * channel_byte_size;
                    std::byte * dst_sample_ptr = dst_data + (texel_index * dst_channel_count + dst_channel) * channel_byte_size;
                    std::memcpy(dst_sample_ptr, src_sample_ptr, channel_byte_size);
                }
            }
        }
    }
};
}

auto remap_channels(RemapChannelsInfo const & info) -> std::shared_ptr<Task>
{
    FormatInfo const format_info = get_format_info(info.format);
    DBG_ASSERT_TRUE_M(format_info.channel_count != 0 && format_info.channel_byte_size != 0, "remap_channels: format is not a supported remap source");

    for (u8 const src_channel : info.channel_mapping)
    {
        DBG_ASSERT_TRUE_M(src_channel < format_info.channel_count, "remap_channels: channel_mapping indexes a channel the source format does not have");
    }
    u32 const dst_channel_count = s_cast<u32>(info.channel_mapping.size());
    DBG_ASSERT_TRUE_M(info.src_data.size() == s_cast<usize>(info.dimensions.x) * info.dimensions.y * format_info.channel_count * format_info.channel_byte_size,
        "remap_channels: src_data size does not match dimensions/format");
    DBG_ASSERT_TRUE_M(info.dst_data.size() == s_cast<usize>(info.dimensions.x) * info.dimensions.y * dst_channel_count * format_info.channel_byte_size,
        "remap_channels: dst_data size does not match dimensions and remapped channel count");
    return std::make_shared<RemapChannelsTask>(info);
}

namespace
{
    auto image_transcode_ktx(std::span<std::byte const> src_data, ktx_transcode_fmt_e transcode_format, bool transcode_alpha_to_opaque) -> std::variant<ImageProcessResult, TidoImageWithData>
    {
        ktxTexture2 * texture;
        KTX_error_code result = ktxTexture2_CreateFromMemory(
            r_cast<ktx_uint8_t const *>(src_data.data()),
            src_data.size(),
            KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
            &texture);
        if (result != KTX_SUCCESS) { return ImageProcessResult::FAILED_TO_PARSE_KTX; }

        if (!ktxTexture2_NeedsTranscoding(texture))
        {
            ktxTexture2_Destroy(texture);
            DBG_ASSERT_TRUE_M(false, "image_transcode_ktx: source image is not Basis-compressed and cannot be transcoded");
            return ImageProcessResult::FAILED_TO_PARSE_KTX;
        }

        ktx_transcode_flags flags = KTX_TF_HIGH_QUALITY;
        if (transcode_alpha_to_opaque) { flags |= KTX_TF_TRANSCODE_ALPHA_DATA_TO_OPAQUE_FORMATS; }

        result = ktxTexture2_TranscodeBasis(texture, transcode_format, flags);
        if (result != KTX_SUCCESS) { ktxTexture2_Destroy(texture); return ImageProcessResult::FAILED_TO_LOAD_KTX; }

        DBG_ASSERT_TRUE_M(texture->baseDepth == 1, "image_transcode_ktx: 3D textures are not yet supported properly");
        DBG_ASSERT_TRUE_M(texture->numFaces == 1, "image_transcode_ktx: Cubemap textures are not yet supported properly");

        TidoImageWithData image_with_data = {};
        image_with_data.data = std::vector<std::byte>(ktxTexture_GetDataSize(ktxTexture(texture)));

        image_with_data.descriptor.info.format = std::bit_cast<daxa::Format>(texture->vkFormat);
        image_with_data.descriptor.info.dimensions = texture->numDimensions;
        image_with_data.descriptor.info.size = {texture->baseWidth, texture->baseHeight, texture->baseDepth};
        image_with_data.descriptor.info.mip_level_count = texture->numLevels;
        image_with_data.descriptor.info.array_layer_count = texture->numLayers;
        image_with_data.descriptor.subresources.resize(texture->numLevels * texture->numLayers);

        ktx_uint8_t const * texture_data = ktxTexture_GetData(ktxTexture(texture));
        for (u32 mip = 0; mip < texture->numLevels; ++mip)
        {
            for (u32 layer = 0; layer < texture->numLayers; ++layer)
            {
                u64 offset = {};
                result = ktxTexture_GetImageOffset(ktxTexture(texture), mip, layer, 0, &offset);
                if (result != KTX_SUCCESS) { ktxTexture2_Destroy(texture); return ImageProcessResult::FAILED_TO_PARSE_KTX; }

                u32 const byte_size = s_cast<u32>(ktxTexture_GetImageSize(ktxTexture(texture), mip));
                std::memcpy(image_with_data.data.data() + offset, texture_data + offset, byte_size);

                u32 const subresource_index = image_with_data.descriptor.layer_mip_to_subresource_index(layer, mip);
                image_with_data.descriptor.subresources[subresource_index] = {.offset = offset, .byte_size = byte_size};
            }
        }

        ktxTexture2_Destroy(texture);
        return image_with_data;
    }
}

auto image_transcode(ImageTranscodeInfo const & info) -> std::variant<ImageProcessResult, TidoImageWithData>
{
    DBG_ASSERT_TRUE_M(info.source_format == ImageFileFormat::KTX2, "image_transcode: only KTX2 source format is supported");

    auto validate_channel_mapping = [&](std::initializer_list<u8> const expected, char const * const message)
    {
        bool const valid = info.channel_mapping.size() == expected.size() &&
            std::equal(info.channel_mapping.begin(), info.channel_mapping.end(), expected.begin());
        DBG_ASSERT_TRUE_M(valid, message);
    };

    ktx_transcode_fmt_e transcode_format;
    switch (info.target_format)
    {
        case daxa::Format::BC1_RGB_UNORM_BLOCK:
        case daxa::Format::BC1_RGB_SRGB_BLOCK:
            validate_channel_mapping({0, 1, 2}, "ktx_transcode: BC1 target format requires channel mapping to RGB channels");
            transcode_format = KTX_TTF_BC1_RGB;
            break;
        case daxa::Format::BC3_UNORM_BLOCK:
        case daxa::Format::BC3_SRGB_BLOCK:
            validate_channel_mapping({0, 1, 2, 3}, "ktx_transcode: BC3 target format requires channel mapping to RGBA channels");
            transcode_format = KTX_TTF_BC3_RGBA;
            break;
        case daxa::Format::BC4_UNORM_BLOCK:
        case daxa::Format::BC4_SNORM_BLOCK:
            validate_channel_mapping({3}, "ktx_transcode: BC4 target format requires channel mapping to the alpha channel");
            transcode_format = KTX_TTF_BC4_R;
            break;
        case daxa::Format::BC5_UNORM_BLOCK:
        case daxa::Format::BC5_SNORM_BLOCK:
            validate_channel_mapping({0, 1}, "ktx_transcode: BC5 target format requires channel mapping to RGB channels");
            transcode_format = KTX_TTF_BC5_RG;
            break;
        case daxa::Format::BC7_SRGB_BLOCK:
            validate_channel_mapping({0, 1, 2}, "ktx_transcode: BC7 SRGB target format requires channel mapping to RGB channels");
            transcode_format = KTX_TTF_BC7_RGBA;
            break;
        case daxa::Format::BC7_UNORM_BLOCK:
            validate_channel_mapping({0, 1, 2, 3}, "ktx_transcode: BC7 UNORM target format requires channel mapping to RGBA channels");
            transcode_format = KTX_TTF_BC7_RGBA;
            break;
        case daxa::Format::R8G8B8A8_UINT:
            validate_channel_mapping({0, 1, 2, 3}, "ktx_transcode: RGBA32 target format requires channel mapping to RGBA channels");
            transcode_format = KTX_TTF_RGBA32;
            break;
        case daxa::Format::B5G6R5_UNORM_PACK16:
            validate_channel_mapping({0, 1, 2}, "ktx_transcode: BGR565 target format requires channel mapping to RGB channels");
            transcode_format = KTX_TTF_BGR565;
            break;
        case daxa::Format::R5G6B5_UNORM_PACK16:
            validate_channel_mapping({0, 1, 2}, "ktx_transcode: RGB565 target format requires channel mapping to RGB channels");
            transcode_format = KTX_TTF_RGB565;
            break;
        default:
            DBG_ASSERT_TRUE_M(false, "ktx_transcode: target_format is not a KTX-transcodable BCn format");
            transcode_format = s_cast<ktx_transcode_fmt_e>(~0u);
            break;
    }

    // An alpha-only mapping ({3}) targets BC4_R, so KTX must route the source alpha into the single output channel.
    bool const transcode_alpha_to_opaque = info.channel_mapping.size() == 1 && info.channel_mapping[0] == 3;
    return image_transcode_ktx(info.src_data, transcode_format, transcode_alpha_to_opaque);
}

// // Interleave N fp16 grids into one channel-interleaved buffer, channel order = grids_data order. Mirrors
// // the repack loops LoadVDBTask's callers used to do by hand for the RAW (4-grid) and BC6 (3-grid) cases.
// auto interleave_fp16_grids(std::vector<std::vector<std::byte>> const & grids_data, i32vec3 grid_extents) -> std::vector<std::byte>
// {
//     constexpr u32 element_size = sizeof(u16);
//     u32 const channel_count = s_cast<u32>(grids_data.size());
//     u64 const entry_count = s_cast<u64>(grid_extents.x) * grid_extents.y * grid_extents.z;
//     std::vector<std::byte> interleaved(entry_count * channel_count * element_size);
//     for (u64 entry_index = 0; entry_index < entry_count; ++entry_index)
//     {
//         u64 const dst_offset = entry_index * channel_count * element_size;
//         for (u32 channel = 0; channel < channel_count; ++channel)
//         {
//             std::memcpy(&interleaved[dst_offset + channel * element_size], &grids_data[channel][entry_index * element_size], element_size);
//         }
//     }
//     return interleaved;
// }

// // Remap one fp32 grid's (already value_range-clamped, see VDBGridInfo) samples into [0,1] -
// // CompressBlockBC1SDF asserts its input lies in that range.
// auto normalize_f32_grid(std::vector<std::byte> const & grid_data, f32vec2 value_range) -> std::vector<std::byte>
// {
//     std::vector<std::byte> normalized(grid_data.size());
//     usize const entry_count = grid_data.size() / sizeof(f32);
//     f32 const range = value_range.y - value_range.x;
//     for (usize entry_index = 0; entry_index < entry_count; ++entry_index)
//     {
//         f32 value = {};
//         std::memcpy(&value, &grid_data[entry_index * sizeof(f32)], sizeof(f32));
//         f32 const remapped = (value - value_range.x) / range;
//         std::memcpy(&normalized[entry_index * sizeof(f32)], &remapped, sizeof(f32));
//     }
//     return normalized;
// }

// auto process_volume(OptimizeVolumeInfo const & info, ThreadPool * threadpool) -> ProcessedImage
// {
//     DBG_ASSERT_TRUE_M(info.grids_data.size() == info.grids.size(), "process_volume: grids_data must have one entry per recipe grid");
//     u32vec3 const volume_dimensions = {s_cast<u32>(info.grid_extents.x), s_cast<u32>(info.grid_extents.y), s_cast<u32>(info.grid_extents.z)};

//     ProcessedImage ret = {};
//     ret.mips_to_copy = 1;
//     ret.mip_copy_offsets[0] = 0;
//     ret.image_info = {
//         .dimensions = 3,
//         .size = {volume_dimensions.x, volume_dimensions.y, volume_dimensions.z},
//         .mip_level_count = 1,
//         .array_layer_count = 1,
//         .sample_count = 1,
//         .usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
//         .name = info.name,
//     };

//     switch (info.target)
//     {
//         case Compression::BC1_SDF:
//         {
//             DBG_ASSERT_TRUE_M(info.grids.size() == 1, "process_volume: BC1_SDF compresses exactly one grid");
//             DBG_ASSERT_TRUE_M(!info.grids[0].convert_to_fp16, "process_volume: BC1_SDF requires its grid decoded as fp32");
//             std::vector<std::byte> const normalized = normalize_f32_grid(info.grids_data[0], info.grids[0].value_range);

//             u64 const block_count = s_cast<u64>(round_up_div(volume_dimensions.x, 4u)) * round_up_div(volume_dimensions.y, 4u) * volume_dimensions.z;
//             ret.src_data.resize(block_count * bc_block_bytes(Compression::BC1_SDF));
//             auto compress_task = compress_image({
//                 .in_data = normalized,
//                 .out_data = ret.src_data,
//                 .image_dimensions = volume_dimensions,
//                 .compression = Compression::BC1_SDF,
//             });
//             threadpool->blocking_dispatch(compress_task);
//             ret.image_info.format = daxa::Format::BC1_RGBA_UNORM_BLOCK;
//             break;
//         }
//         case Compression::BC6:
//         {
//             DBG_ASSERT_TRUE_M(info.grids.size() == 3, "process_volume: BC6 interleaves exactly three grids (RGB16F)");
//             for (VDBGridInfo const & grid : info.grids)
//             {
//                 DBG_ASSERT_TRUE_M(grid.convert_to_fp16, "process_volume: BC6 requires its grids decoded as fp16");
//             }
//             std::vector<std::byte> const interleaved = interleave_fp16_grids(info.grids_data, info.grid_extents);

//             u64 const block_count = s_cast<u64>(round_up_div(volume_dimensions.x, 4u)) * round_up_div(volume_dimensions.y, 4u) * volume_dimensions.z;
//             ret.src_data.resize(block_count * bc_block_bytes(Compression::BC6));
//             auto compress_task = compress_image({
//                 .in_data = interleaved,
//                 .out_data = ret.src_data,
//                 .image_dimensions = volume_dimensions,
//                 .compression = Compression::BC6,
//             });
//             threadpool->blocking_dispatch(compress_task);
//             ret.image_info.format = daxa::Format::BC6H_UFLOAT_BLOCK;
//             break;
//         }
//         case Compression::UNDEFINED:
//         {
//             // "none" means uncompressed RGBA16F, matching the legacy RAW cloud format - exactly 4 grids.
//             DBG_ASSERT_TRUE_M(info.grids.size() == 4, "process_volume: uncompressed volumes need exactly four grids (RGBA16F)");
//             for (VDBGridInfo const & grid : info.grids)
//             {
//                 DBG_ASSERT_TRUE_M(grid.convert_to_fp16, "process_volume: uncompressed volumes require their grids decoded as fp16");
//             }
//             ret.src_data = interleave_fp16_grids(info.grids_data, info.grid_extents);
//             ret.image_info.format = daxa::Format::R16G16B16A16_SFLOAT;
//             break;
//         }
//         case Compression::BC1:
//         case Compression::BC4:
//         case Compression::BC5:
//         case Compression::BC7:
//         default:
//             DBG_ASSERT_TRUE_M(false, "process_volume: unsupported volume compression target");
//             break;
//     }
//     return ret;
// }
