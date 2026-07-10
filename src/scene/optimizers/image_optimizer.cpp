#include "image_optimizer.hpp"

#include <cstring>
#include <bit>
#include <span>
#include <optional>
#include <cmath>
#include <algorithm>

#include <ktx.h>
#include <png.h>

#include "tex_compression.hpp"
#include "../../shader_shared/shared.inl" // round_up_div

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

// sRGB <-> linear transfer on a normalized [0,1] scalar (mip averaging is done in linear light).
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

// The channel layout shared by every level of a decoded image's mip chain.
inline auto pixel_info_of(DecodedPixels const & pixels) -> PixelInfo
{
    return PixelInfo{
        .channel_count = s_cast<u8>(pixels.channel_count),
        .channel_byte_size = s_cast<u8>(pixels.channel_byte_size),
        .channel_data_type = ChannelDataType::UNSIGNED_INT,
        .load_as_srgb = pixels.srgb,
    };
}

// One box-filter downsample level (previous mip -> this mip).
// Edge-clamps the 2x2 taps on odd extents.
struct MipDownsampleTask final : Task
{
    // Target output texels per chunk (row-aligned), for roughly constant work per chunk.
    static constexpr u32 TARGET_TEXELS_PER_CHUNK = 8192;

    PixelInfo pixel_info = {};
    std::span<std::byte const> src_data = {};
    u32 src_width = {};
    u32 src_height = {};
    std::span<std::byte> dst_data = {};
    u32 dst_width = {};
    u32 dst_height = {};
    u32 rows_per_chunk = {};

    MipDownsampleTask(
        PixelInfo const & pixel_info,
        std::span<std::byte const> src, u32 src_width, u32 src_height,
        std::span<std::byte> dst, u32 dst_width, u32 dst_height)
        : pixel_info{pixel_info}
        , src_data{src}, src_width{src_width}, src_height{src_height}
        , dst_data{dst}, dst_width{dst_width}, dst_height{dst_height}
    {
        rows_per_chunk = std::max(1u, TARGET_TEXELS_PER_CHUNK / std::max(1u, dst_width));
        chunk_count = round_up_div(dst_height, rows_per_chunk);
    }

    virtual void callback(u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        u32 const channel_count = pixel_info.channel_count;
        u32 const channel_byte_size = pixel_info.channel_byte_size;
        bool const srgb = pixel_info.load_as_srgb;
        u32 const src_row_stride = src_width * channel_count * channel_byte_size;
        u32 const dst_row_stride = dst_width * channel_count * channel_byte_size;

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

// Run every chunk of a one-shot task on the calling thread. The bulk texture cook already parallelizes
// across images (one pool task per image), so mip generation and block compression run inline here rather
// than dispatching nested pool tasks: a nested dispatch waits by pulling from the shared queue, which lets
// the worker start yet another image mid-cook, multiplying peak host memory by the nesting depth.
inline void run_task_inline(Task & task)
{
    for (u32 chunk_index = 0; chunk_index < task.chunk_count; ++chunk_index)
    {
        task.callback(chunk_index, 0);
    }
}

// Full box-filtered mip chain in one contiguous buffer: floor(log2(max(w,h)))+1 levels, level 0 the input
// down to 1x1, mip 0 first. Each level reads the previous one, so they are generated in order.
auto generate_mip_chain(DecodedPixels const & base) -> ProcessedImage
{
    PixelInfo const pixel_info = pixel_info_of(base);
    u32 const texel_byte_size = base.channel_count * base.channel_byte_size;

    u32 const max_dim = std::max(base.width, base.height);
    u32 const mip_count = 1u + s_cast<u32>(std::floor(std::log2(s_cast<f32>(max_dim))));

    ProcessedImage chain = {};
    DBG_ASSERT_TRUE_M(mip_count <= chain.mip_copy_offsets.size(), "generate_mip_chain: mip count exceeds mip_copy_offsets capacity");

    // Lay out every level contiguously up front so the buffer never reallocates while the running task holds pointers into it.
    u64 total_size = 0;
    std::array<u32, 16> mip_widths = {};
    std::array<u32, 16> mip_heights = {};
    for (u32 mip = 0; mip < mip_count; ++mip)
    {
        mip_widths[mip] = std::max(1u, base.width >> mip);
        mip_heights[mip] = std::max(1u, base.height >> mip);
        chain.mip_copy_offsets[mip] = total_size;
        total_size += s_cast<u64>(mip_widths[mip]) * mip_heights[mip] * texel_byte_size;
    }
    chain.src_data.resize(total_size);

    // Level 0 is the input and remains unchanged.
    std::memcpy(chain.src_data.data(), base.data.data(), base.data.size());

    for (u32 mip = 1; mip < mip_count; ++mip)
    {
        std::span<std::byte const> const src = std::span<std::byte const>(chain.src_data).subspan(chain.mip_copy_offsets[mip - 1]);
        std::span<std::byte> const dst = std::span<std::byte>(chain.src_data).subspan(chain.mip_copy_offsets[mip]);
        MipDownsampleTask downsample{
            pixel_info,
            src, mip_widths[mip - 1], mip_heights[mip - 1],
            dst, mip_widths[mip], mip_heights[mip]};
        run_task_inline(downsample);
    }

    chain.mips_to_copy = mip_count;
    chain.image_info = {
        .dimensions = 2,
        .format = daxa_image_format_from_pixel_info(pixel_info),
        .size = {base.width, base.height, 1},
        .mip_level_count = mip_count,
        .array_layer_count = 1,
        .sample_count = 1,
    };
    return chain;
}

// Bytes per 4x4 block for a BC format.
inline auto bc_block_bytes(Compression compression) -> u32
{
    switch (compression)
    {
        case Compression::BC1:     return 8u;
        case Compression::BC1_SDF: return 8u;
        case Compression::BC4:     return 8u;
        case Compression::BC5:     return 16u;
        case Compression::BC6:     return 16u;
        case Compression::BC7:     return 16u;
        case Compression::UNDEFINED:
        default:                   DBG_ASSERT_TRUE_M(false, "bc_block_bytes: undefined block compression format"); return 0u;
    }
}

// The cooked BC format, compressor, and interleaved input channel count for a texture usage. Diffuse uses
// an sRGB format only when the source was sRGB; BC5 normals store RG and the shader reconstructs Z
// (tido_format_is_bc5_rg).
struct CompressionTarget
{
    daxa::Format format;
    Compression compression;
    u32 in_channels; // interleaved bytes-per-texel the CompressTask expects (RGBA=4 / RG=2 / R=1), 8-bit
};

auto compression_target_for(TextureMaterialType type, bool srgb) -> CompressionTarget
{
    switch (type)
    {
        case TextureMaterialType::NORMAL:              return {daxa::Format::BC5_UNORM_BLOCK, Compression::BC5, 2u};
        case TextureMaterialType::ROUGHNESS_METALNESS: return {daxa::Format::BC7_UNORM_BLOCK, Compression::BC7, 4u};
        case TextureMaterialType::OPACITY:             return {daxa::Format::BC4_UNORM_BLOCK, Compression::BC4, 1u};
        case TextureMaterialType::DIFFUSE:
        case TextureMaterialType::DIFFUSE_OPACITY:
        default:                                       return {srgb ? daxa::Format::BC7_SRGB_BLOCK : daxa::Format::BC7_UNORM_BLOCK, Compression::BC7, 4u};
    }
}

// Repack a mip's interleaved pixels into the tightly packed 8-bit, out_channels-interleaved layout
// compress_image expects. 16-bit samples are narrowed to 8-bit; requesting more channels than the source
// has clamps to its last channel.
auto repack_interleaved_8bit(std::span<std::byte const> pixels, u32 width, u32 height, PixelInfo const & pixel_info, u32 out_channels) -> std::vector<std::byte>
{
    u32 const channel_count = pixel_info.channel_count;
    u32 const channel_byte_size = pixel_info.channel_byte_size;
    usize const texel_count = s_cast<usize>(width) * height;
    std::vector<std::byte> packed(texel_count * out_channels);
    for (usize texel_index = 0; texel_index < texel_count; ++texel_index)
    {
        for (u32 channel = 0; channel < out_channels; ++channel)
        {
            u32 const src_channel = std::min(channel, channel_count - 1u);
            std::byte const * sample_ptr = pixels.data() + (texel_index * channel_count + src_channel) * channel_byte_size;
            f32 const sample = std::clamp(read_normalized(sample_ptr, channel_byte_size), 0.0f, 1.0f);
            packed[texel_index * out_channels + channel] = s_cast<std::byte>(s_cast<u8>(sample * 255.0f + 0.5f));
        }
    }
    return packed;
}

// Cook decoded pixels into GPU-ready BC memory: box-filter a mip chain, then BC-compress each level into
// src_data (mip 0 = finest, contiguous), format chosen per texture usage.
auto pixels_to_processed(DecodedPixels const & pixels, TextureMaterialType type, std::string name) -> ProcessedImage
{
    CompressionTarget const target = compression_target_for(type, pixels.srgb);
    u32 const block_bytes = bc_block_bytes(target.compression);
    PixelInfo const pixel_info = pixel_info_of(pixels);

    ProcessedImage const mip_chain = generate_mip_chain(pixels);
    u32 const mip_count = mip_chain.mips_to_copy;

    ProcessedImage ret = {};
    DBG_ASSERT_TRUE_M(mip_count <= ret.mip_copy_offsets.size(), "pixels_to_processed: mip count exceeds mip_copy_offsets capacity");

    // Pass 1: size the compressed payload and record each mip's offset. The ceil-based block count makes
    // each mip's slice exactly the byte size write_texture_tido reads back for that subresource.
    u64 total_size = 0;
    std::array<u64, 16> mip_sizes = {};
    for (u32 mip = 0; mip < mip_count; ++mip)
    {
        u32 const mip_width = std::max(1u, pixels.width >> mip);
        u32 const mip_height = std::max(1u, pixels.height >> mip);
        u64 const block_count = s_cast<u64>(round_up_div(mip_width, 4u)) * round_up_div(mip_height, 4u);
        ret.mip_copy_offsets[mip] = total_size;
        mip_sizes[mip] = block_count * block_bytes;
        total_size += mip_sizes[mip];
    }
    ret.src_data.resize(total_size);

    // Pass 2: compress each mip into its slice of src_data. mip_input stays alive while the compress task
    // (which spans into it) runs.
    for (u32 mip = 0; mip < mip_count; ++mip)
    {
        u32 const mip_width = std::max(1u, pixels.width >> mip);
        u32 const mip_height = std::max(1u, pixels.height >> mip);
        std::span<std::byte const> const mip_pixels = std::span<std::byte const>(mip_chain.src_data).subspan(mip_chain.mip_copy_offsets[mip]);
        std::vector<std::byte> const mip_input = repack_interleaved_8bit(mip_pixels, mip_width, mip_height, pixel_info, target.in_channels);
        CreateCompressedImageInfo const compress_info = {
            .in_data = mip_input,
            .out_data = std::span<std::byte>(ret.src_data).subspan(ret.mip_copy_offsets[mip], mip_sizes[mip]),
            .image_dimensions = {mip_width, mip_height, 1u},
            .compression = target.compression,
        };
        run_task_inline(*compress_image(compress_info));
    }

    ret.mips_to_copy = mip_count;
    ret.image_info = {
        .dimensions = 2,
        .format = target.format,
        .size = {pixels.width, pixels.height, 1},
        .mip_level_count = mip_count,
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
    // Decode/transcode into GPU-ready cooked CPU memory and return it. The PNG decode path box-filters a
    // mip chain and BC-compresses each level; KTX2 already carries BCn + its mips.
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
            return pixels_to_processed(decoded.value(), info.type, info.name);
        }
    }
    return ImageOptimizeError::FAILED_TO_DECODE_PNG;
}
