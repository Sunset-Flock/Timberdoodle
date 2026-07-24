#include "image_processor.hpp"

#include <cstring>
#include <bit>
#include <span>
#include <optional>
#include <cmath>
#include <algorithm>
#include <limits>

#include <ktx.h>
#include <png.h>
#include <glm/detail/type_half.hpp>

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

auto image_parse_png(std::span<std::byte const> png_bytes, bool is_srgb) -> std::variant<ImageProcessResult, TidoImageWithData>
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

    // Always decode to 4-channel RGBA so the cook can index any recipe channel (including alpha) uniformly:
    // expand palette/gray to RGB, promote tRNS to a real alpha channel, and fill an opaque alpha where the
    // source has none (0xFFFF -> full alpha at both 8- and 16-bit).
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png_ptr);
    if (interlace_type != PNG_INTERLACE_NONE) png_set_interlace_handling(png_ptr);
    if (channel_count < 4) png_set_add_alpha(png_ptr, 0xFFFF, PNG_FILLER_AFTER);

    png_read_update_info(png_ptr, info_ptr);
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, &interlace_type, nullptr, nullptr);
    channel_count = png_get_channels(png_ptr, info_ptr);

    DBG_ASSERT_TRUE_M(channel_count == 4, "image_parse_png: expected the decode transforms to yield 4-channel RGBA");
    DBG_ASSERT_TRUE_M(bit_depth == 16 || bit_depth == 8, "Unexpected PNG bit depth: " + std::to_string(bit_depth) + " (expected 8 or 16)");

    PixelInfo pixel_info = {};
    pixel_info.channel_count = s_cast<u8>(channel_count);
    pixel_info.channel_byte_size = s_cast<u8>(bit_depth / 8);
    pixel_info.channel_data_type = ChannelDataType::UNSIGNED_INT;
    pixel_info.is_srgb = is_srgb;

    TidoImageWithData ret = {};
    ret.descriptor.info.format = image_format_from_pixel_info(pixel_info);
    ret.descriptor.info.dimensions = 2;
    ret.descriptor.info.size = {width, height, 1};
    ret.descriptor.info.mip_level_count = 1;
    ret.descriptor.info.array_layer_count = 1;
    ret.data.resize(s_cast<usize>(width) * height * channel_count * (bit_depth / 8));
    ret.descriptor.subresources = {{.offset = 0, .byte_size = s_cast<u32>(ret.data.size())}};

    std::vector<png_bytep> row_pointers(height);
    for (u32 row_index = 0; row_index < height; row_index++)
    {
        row_pointers[row_index] = (png_bytep)(ret.data.data() + s_cast<usize>(width) * row_index * channel_count * (bit_depth / 8));
    }
    png_read_image(png_ptr, row_pointers.data());

    return ret;
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
            return image_parse_png(parse_info.src_data, parse_info.is_srgb);
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
inline auto srgb_to_linear(f64 value) -> f64
{
    return value <= 0.04045 ? value / 12.92 : std::pow((value + 0.055) / 1.055, 2.4);
}
inline auto linear_to_srgb(f64 value) -> f64
{
    return value <= 0.0031308 ? value * 12.92 : 1.055 * std::pow(value, 1.0 / 2.4) - 0.055;
}

// Largest integer value a channel of this width holds - numeric_limits::max() of its underlying integer type.
auto numeric_magnitude(FormatNumericType numeric_type, u32 channel_byte_size) -> f64
{
    DBG_ASSERT_TRUE_M(numeric_type != FormatNumericType::SFLOAT, "numeric_magnitude: float has no integer magnitude");
    bool const is_signed = numeric_type == FormatNumericType::SNORM || numeric_type == FormatNumericType::SINT;
    switch (channel_byte_size)
    {
        case 1: return is_signed ? s_cast<f64>(std::numeric_limits<i8>::max())  : s_cast<f64>(std::numeric_limits<u8>::max());
        case 2: return is_signed ? s_cast<f64>(std::numeric_limits<i16>::max()) : s_cast<f64>(std::numeric_limits<u16>::max());
        case 4: return is_signed ? s_cast<f64>(std::numeric_limits<i32>::max()) : s_cast<f64>(std::numeric_limits<u32>::max());
        default: DBG_ASSERT_TRUE_M(false, "numeric_magnitude: unsupported channel byte size"); return 1.0;
    }
}

// Load a narrow signed channel as i64. memcpy into the exact-width signed type so the standard integral
// conversion to i64 sign-extends it - no bit manipulation, no implementation-defined behavior.
auto sign_extend(std::byte const * src_ptr, u32 byte_size) -> i64
{
    switch (byte_size)
    {
        case 1: { i8  value; std::memcpy(&value, src_ptr, 1); return value; }
        case 2: { i16 value; std::memcpy(&value, src_ptr, 2); return value; }
        case 4: { i32 value; std::memcpy(&value, src_ptr, 4); return value; }
        default: DBG_ASSERT_TRUE_M(false, "sign_extend: unsupported channel byte size"); return 0;
    }
}

// Decode one channel sample into a kind-canonical fp64 value (which ensures we lose no precision):
//   NORM formats -> a value in the range [0,1]/[-1,1]
//   INT formats -> the integer value cast to fp64
//   FLOAT formats -> the float value (half-float widened) cast to fp64
auto decode_channel_sample(std::byte const * src_ptr, FormatNumericType numeric_type, u32 channel_byte_size) -> f64
{
    u64 src_bits = 0;
    std::memcpy(&src_bits, src_ptr, channel_byte_size);
    switch (numeric_type)
    {
        case FormatNumericType::UNORM:  return s_cast<f64>(src_bits) / numeric_magnitude(numeric_type, channel_byte_size);
        case FormatNumericType::SNORM:  return std::max(s_cast<f64>(sign_extend(src_ptr, channel_byte_size)) / numeric_magnitude(numeric_type, channel_byte_size), -1.0);
        case FormatNumericType::UINT:   return s_cast<f64>(src_bits);
        case FormatNumericType::SINT:   return s_cast<f64>(sign_extend(src_ptr, channel_byte_size));
        case FormatNumericType::SFLOAT: return channel_byte_size == 2 ? s_cast<f64>(glm::detail::toFloat32(s_cast<glm::detail::hdata>(src_bits))) : s_cast<f64>(std::bit_cast<f32>(s_cast<u32>(src_bits)));
    }
    DBG_ASSERT_TRUE_M(false, "decode_channel_sample: unhandled numeric type");
    return 0.0;
}

// Encode a kind-canonical fp64 value into one destination channel sample, clamping/rounding into its
// representable range. Signed rounds land in the low bytes of the u64 as two's complement, so the same
// little-endian memcpy writes them back.
void encode_channel_sample(std::byte * dst_ptr, FormatNumericType numeric_type, u32 channel_byte_size, f64 value)
{
    u64 dst_bits = 0;
    switch (numeric_type)
    {
        case FormatNumericType::UNORM:  dst_bits = s_cast<u64>(std::llround(std::clamp(value, 0.0, 1.0) * numeric_magnitude(numeric_type, channel_byte_size))); break;
        case FormatNumericType::SNORM:  dst_bits = s_cast<u64>(std::llround(std::clamp(value, -1.0, 1.0) * numeric_magnitude(numeric_type, channel_byte_size))); break;
        case FormatNumericType::UINT:   dst_bits = s_cast<u64>(std::llround(std::clamp(value, 0.0, numeric_magnitude(numeric_type, channel_byte_size)))); break;
        case FormatNumericType::SINT:   dst_bits = s_cast<u64>(std::llround(std::clamp(value, -numeric_magnitude(numeric_type, channel_byte_size) - 1.0, numeric_magnitude(numeric_type, channel_byte_size)))); break;
        case FormatNumericType::SFLOAT: dst_bits = channel_byte_size == 2 ? s_cast<u64>(s_cast<u16>(glm::detail::toFloat16(s_cast<f32>(value)))) : s_cast<u64>(std::bit_cast<u32>(s_cast<f32>(value))); break;
    }
    std::memcpy(dst_ptr, &dst_bits, channel_byte_size);
}

// Byte offset of a channel sample in a tightly packed image, texels in z-major order (slice, then row, then
// column). Reduces to the row-major 2D offset when coord.z is 0.
auto texel_channel_offset(u32vec3 dimensions, u32vec3 coord, u32 channel, u32 channel_count, u32 channel_byte_size) -> u64
{
    u64 const texel_index = (s_cast<u64>(coord.z) * dimensions.y + coord.y) * dimensions.x + coord.x;
    return (texel_index * channel_count + channel) * s_cast<u64>(channel_byte_size);
}

struct DownsampleImageTask final : Task
{
    static constexpr u32 TARGET_TEXELS_PER_CHUNK = 8192;

    DownsampleImageInfo info = {};
    FormatInfo format_info = {};
    u32vec3 dst_dimensions = {};
    u32 rows_per_chunk = {};   // destination rows across the whole volume (dst_z * dst_height + dst_y)

    DownsampleImageTask(DownsampleImageInfo const & info)
        : info{info}, format_info{get_format_info(info.format)}
    {
        if(format_info.channel_count == 0 || format_info.channel_byte_size == 0)
        {
            DBG_ASSERT_TRUE_M(false, "DownsampleImageTask: format is not a supported mip-generation source");
        }
        dst_dimensions = {
            std::max(1u, info.src_dimensions.x >> 1),
            std::max(1u, info.src_dimensions.y >> 1),
            std::max(1u, info.src_dimensions.z >> 1),
        };
        rows_per_chunk = std::max(1u, TARGET_TEXELS_PER_CHUNK / dst_dimensions.x);
        u32 const total_dst_rows = dst_dimensions.y * dst_dimensions.z;
        chunk_count = round_up_div(total_dst_rows, rows_per_chunk);
    }

    // Box-filter one destination texel by averaging its source neighborhood: two taps along each source axis
    // of extent > 1, one tap along an axis of extent 1. So a 2D source averages 2x2 and a volume 2x2x2. sRGB
    // color channels are averaged in linear space.
    void filter_texel(u32vec3 dst_coord)
    {
        u32 const channel_count = format_info.channel_count;
        u32 const channel_byte_size = format_info.channel_byte_size;
        FormatNumericType const numeric_type = format_info.numeric_type;
        bool const srgb = format_info.is_srgb;
        u32vec3 const src_dimensions = info.src_dimensions;

        u32vec3 const tap_counts = {
            src_dimensions.x > 1u ? 2u : 1u,
            src_dimensions.y > 1u ? 2u : 1u,
            src_dimensions.z > 1u ? 2u : 1u,
        };
        f64 const inverse_tap_total = 1.0 / s_cast<f64>(tap_counts.x * tap_counts.y * tap_counts.z);

        std::byte const * src_data = info.src_data.data();
        std::byte * dst_data = info.dst_data.data();

        for (u32 channel = 0; channel < channel_count; ++channel)
        {
            bool const is_srgb_color = srgb && channel < 3u;
            f64 channel_sum = 0.0;
            for (u32 tap_z = 0; tap_z < tap_counts.z; ++tap_z)
            {
                for (u32 tap_y = 0; tap_y < tap_counts.y; ++tap_y)
                {
                    for (u32 tap_x = 0; tap_x < tap_counts.x; ++tap_x)
                    {
                        u32vec3 const src_coord = {
                            std::min(dst_coord.x * 2u + tap_x, src_dimensions.x - 1u),
                            std::min(dst_coord.y * 2u + tap_y, src_dimensions.y - 1u),
                            std::min(dst_coord.z * 2u + tap_z, src_dimensions.z - 1u),
                        };
                        std::byte const * src_sample_ptr = &src_data[texel_channel_offset(src_dimensions, src_coord, channel, channel_count, channel_byte_size)];
                        f64 sample = decode_channel_sample(src_sample_ptr, numeric_type, channel_byte_size);
                        if (is_srgb_color) { sample = srgb_to_linear(sample); }
                        channel_sum += sample;
                    }
                }
            }
            f64 average = channel_sum * inverse_tap_total;
            if (is_srgb_color) { average = linear_to_srgb(average); }
            std::byte * dst_sample_ptr = &dst_data[texel_channel_offset(dst_dimensions, dst_coord, channel, channel_count, channel_byte_size)];
            encode_channel_sample(dst_sample_ptr, numeric_type, channel_byte_size, average);
        }
    }

    virtual void callback(u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        u32 const total_dst_rows = dst_dimensions.y * dst_dimensions.z;
        u32 const row_begin = chunk_index * rows_per_chunk;
        u32 const row_end = std::min(row_begin + rows_per_chunk, total_dst_rows);
        for (u32 row = row_begin; row < row_end; ++row)
        {
            u32 const dst_z = row / dst_dimensions.y;
            u32 const dst_y = row % dst_dimensions.y;
            for (u32 dst_x = 0; dst_x < dst_dimensions.x; ++dst_x)
            {
                filter_texel({dst_x, dst_y, dst_z});
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
    u32 const dst_depth = std::max(1u, info.src_dimensions.z >> 1);

    DBG_ASSERT_TRUE_M(info.src_data.size() == s_cast<usize>(info.src_dimensions.x) * info.src_dimensions.y * info.src_dimensions.z * texel_byte_size,
        "downsample_image: src_data size does not match src_dimensions/format");
    DBG_ASSERT_TRUE_M(info.dst_data.size() == s_cast<usize>(dst_width) * dst_height * dst_depth * texel_byte_size,
        "downsample_image: dst_data size does not match the halved src_dimensions/format");

    return std::make_shared<DownsampleImageTask>(info);
}

void image_resize_for_mipmaps(TidoImageWithData & image, u32 mip_count)
{
    DBG_ASSERT_TRUE_M(image.descriptor.info.array_layer_count <= 1, "image_resize_for_mipmaps: array/layered images are not yet supported");

    u32 const existing_mip_count = image.descriptor.info.mip_level_count;
    if (existing_mip_count >= mip_count) { return; }
    DBG_ASSERT_TRUE_M(image.descriptor.subresources.size() == existing_mip_count, "image_resize_for_mipmaps: subresource count does not match mip_level_count");

    FormatInfo const layout = get_format_info(image.descriptor.info.format);
    u32 const base_width = image.descriptor.info.size.x;
    u32 const base_height = image.descriptor.info.size.y;

    // Append storage + descriptors for mip levels that are not yet present.
    // Existing levels keep their data and offsets, the new levels are laid out contiguously after the current buffer end.
    image.descriptor.subresources.reserve(mip_count);
    u64 offset = image.data.size();
    for (u32 mip = existing_mip_count; mip < mip_count; ++mip)
    {
        u32 const mip_width = std::max(1u, base_width >> mip);
        u32 const mip_height = std::max(1u, base_height >> mip);
        u64 const byte_size = s_cast<u64>(round_up_div(mip_width, layout.block_width)) * round_up_div(mip_height, layout.block_height) * layout.block_byte_size;
        image.descriptor.subresources.push_back({.offset = offset, .byte_size = s_cast<u32>(byte_size)});
        offset += byte_size;
    }
    image.data.resize(offset);
    image.descriptor.info.mip_level_count = mip_count;
}

namespace
{
// This table describes the operations that are needed to convert a sample from one numeric type to another.
// ASSIGN - The source value can be assigned directly to the destination without any scaling or normalization.
// NORMALIZE_BY_SRC - The source value needs to be divided by its maximum representable value to convert it into the range of [0, 1] or [-1, 1].
// DENORMALIZE_TO_DST - The source value (which is normalized) needs to be multiplied by the maximum representable value of the destination type.
enum struct SampleScale { ASSIGN, NORMALIZE_BY_SRC, DENORMALIZE_TO_DST };
constexpr SampleScale SAMPLE_SCALE_MATRIX[5][5] = {
    /* src \ dst      UNORM                           SNORM                           UINT                             SINT                             SFLOAT */
    /* UNORM  */ {SampleScale::ASSIGN,            SampleScale::ASSIGN,            SampleScale::DENORMALIZE_TO_DST, SampleScale::DENORMALIZE_TO_DST, SampleScale::ASSIGN},
    /* SNORM  */ {SampleScale::ASSIGN,            SampleScale::ASSIGN,            SampleScale::DENORMALIZE_TO_DST, SampleScale::DENORMALIZE_TO_DST, SampleScale::ASSIGN},
    /* UINT   */ {SampleScale::NORMALIZE_BY_SRC,  SampleScale::NORMALIZE_BY_SRC,  SampleScale::ASSIGN,             SampleScale::ASSIGN,             SampleScale::ASSIGN},
    /* SINT   */ {SampleScale::NORMALIZE_BY_SRC,  SampleScale::NORMALIZE_BY_SRC,  SampleScale::ASSIGN,             SampleScale::ASSIGN,             SampleScale::ASSIGN},
    /* SFLOAT */ {SampleScale::ASSIGN,            SampleScale::ASSIGN,            SampleScale::ASSIGN,             SampleScale::ASSIGN,             SampleScale::ASSIGN},
};

// Convert one source channel sample into one destination channel sample:
//   1) decode the source channel into a kind-canonical fp64 value,
//   2) scale it according to the source/destination numeric types (using the conversion table above),
//   3) encode the scaled value into the destination channel.
void convert_channel_sample(
    std::byte const * src_ptr, FormatNumericType src_numeric_type, u32 src_channel_byte_size,
    std::byte * dst_ptr, FormatNumericType dst_numeric_type, u32 dst_channel_byte_size)
{
    f64 value = decode_channel_sample(src_ptr, src_numeric_type, src_channel_byte_size);

    switch (SAMPLE_SCALE_MATRIX[s_cast<u32>(src_numeric_type)][s_cast<u32>(dst_numeric_type)])
    {
        case SampleScale::ASSIGN:             break;
        case SampleScale::NORMALIZE_BY_SRC:   value /= numeric_magnitude(src_numeric_type, src_channel_byte_size); break;
        case SampleScale::DENORMALIZE_TO_DST: value *= numeric_magnitude(dst_numeric_type, dst_channel_byte_size); break;
    }

    encode_channel_sample(dst_ptr, dst_numeric_type, dst_channel_byte_size, value);
}

struct RemapChannelsTask final : Task
{
    static constexpr u32 TARGET_TEXELS_PER_CHUNK = 8192;

    RemapChannelsInfo info = {};
    FormatInfo format_info = {};
    FormatInfo dst_format_info = {};

    RemapChannelsTask(RemapChannelsInfo const & info)
        : info{info}, format_info{get_format_info(info.format)}, dst_format_info{get_format_info(info.dst_format)}
    {
        if(format_info.channel_count == 0 || format_info.channel_byte_size == 0)
        {
            DBG_ASSERT_TRUE_M(false, "RemapChannelsTask: format is not a supported remap source");
        }
        chunk_count = round_up_div(info.texel_count, TARGET_TEXELS_PER_CHUNK);
    }

    virtual void callback(u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        u32 const src_channel_count = format_info.channel_count;
        u32 const src_channel_byte_size = format_info.channel_byte_size;
        u32 const dst_channel_count = dst_format_info.channel_count;
        u32 const dst_channel_byte_size = dst_format_info.channel_byte_size;

        std::byte const * src_data = info.src_data.data();
        std::byte * dst_data = info.dst_data.data();

        u32 const texel_begin = chunk_index * TARGET_TEXELS_PER_CHUNK;
        u32 const texel_end = std::min(texel_begin + TARGET_TEXELS_PER_CHUNK, info.texel_count);
        for (u32 texel_index = texel_begin; texel_index < texel_end; ++texel_index)
        {
            for (u32 dst_channel = 0; dst_channel < dst_channel_count; ++dst_channel)
            {
                u32 const src_channel = info.channel_mapping[dst_channel];
                std::byte const * src_sample_ptr = src_data + (s_cast<usize>(texel_index) * src_channel_count + src_channel) * src_channel_byte_size;
                std::byte * dst_sample_ptr = dst_data + (s_cast<usize>(texel_index) * dst_channel_count + dst_channel) * dst_channel_byte_size;
                convert_channel_sample(src_sample_ptr, format_info.numeric_type, src_channel_byte_size,
                    dst_sample_ptr, dst_format_info.numeric_type, dst_channel_byte_size);
            }
        }
    }
};
}

auto remap_channels(RemapChannelsInfo const & info) -> std::shared_ptr<Task>
{
    FormatInfo const format_info = get_format_info(info.format);
    FormatInfo const dst_format_info = get_format_info(info.dst_format);
    DBG_ASSERT_TRUE_M(format_info.channel_count != 0 && format_info.channel_byte_size != 0, "remap_channels: format is not a supported remap source");
    DBG_ASSERT_TRUE_M(dst_format_info.channel_count != 0 && dst_format_info.channel_byte_size != 0, "remap_channels: dst_format is not a supported remap destination");
    DBG_ASSERT_TRUE_M(dst_format_info.channel_count == info.channel_mapping.size(), "remap_channels: dst_format channel count must equal the channel_mapping size");

    for (u8 const src_channel : info.channel_mapping)
    {
        DBG_ASSERT_TRUE_M(src_channel < format_info.channel_count, "remap_channels: channel_mapping indexes a channel the source format does not have");
    }
    DBG_ASSERT_TRUE_M(info.src_data.size() == s_cast<usize>(info.texel_count) * format_info.channel_count * format_info.channel_byte_size,
        "remap_channels: src_data size does not match texel_count/format");
    DBG_ASSERT_TRUE_M(info.dst_data.size() == s_cast<usize>(info.texel_count) * dst_format_info.channel_count * dst_format_info.channel_byte_size,
        "remap_channels: dst_data size does not match texel_count and dst_format");
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
        // A non-array KTX2 reports numLayers == 0; treat it as a single layer so the subresource table and
        // the copy loop below are non-empty.
        u32 const layer_count = std::max(1u, texture->numLayers);
        image_with_data.descriptor.info.mip_level_count = texture->numLevels;
        image_with_data.descriptor.info.array_layer_count = layer_count;
        image_with_data.descriptor.subresources.resize(texture->numLevels * layer_count);

        ktx_uint8_t const * texture_data = ktxTexture_GetData(ktxTexture(texture));
        for (u32 mip = 0; mip < texture->numLevels; ++mip)
        {
            for (u32 layer = 0; layer < layer_count; ++layer)
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