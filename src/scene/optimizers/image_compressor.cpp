#include "image_processor.hpp"
#include "sdf_bc1_compressor.hpp"
#include <CMP_Core.h>
#include <algorithm>
#include <array>
#include <iostream>

#include "../../shader_shared/shared.inl" // round_up_div

static constexpr u32 DEFAULT_BLOCKS_PER_CHUNK = 128;

template <u32 PixelByteCount>
struct CompressTask : Task
{
    CreateCompressedImageInfo info;
    u32 blocks_per_chunk;

    private:
        typedef std::array<std::byte, PixelByteCount> Pixel;

        u32 blocks_total;
        u32 blocks_per_layer;
        u32 blocks_per_row;
        u32 pixels_per_layer;

    public:

    CompressTask(CreateCompressedImageInfo const & info, u32 const blocks_per_chunk = DEFAULT_BLOCKS_PER_CHUNK)
        :  info{info}
         , blocks_per_chunk{DEFAULT_BLOCKS_PER_CHUNK}
    {
        // Ceil-based block counts so non-4-aligned extents (and sub-4x4 mip levels) still produce a full
        // set of blocks - the partial edge blocks are padded by clamping in the gather loop below. This
        // matches write_texture_tido, which sizes every mip with ceil(dim/4) blocks.
        blocks_per_row = round_up_div(info.image_dimensions.x, 4u);
        u32 const blocks_per_col = round_up_div(info.image_dimensions.y, 4u);
        blocks_per_layer = blocks_per_row * blocks_per_col;
        blocks_total = blocks_per_layer * info.image_dimensions.z;
        pixels_per_layer = info.image_dimensions.x * info.image_dimensions.y;

        chunk_count = round_up_div(blocks_total, blocks_per_chunk);
    }

    virtual void callback(u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        auto const block_index_to_image_coords = [&](u32 block_index) -> u32vec3 {
            u32 const image_z = block_index / blocks_per_layer;

            u32 const in_layer_block_index = (block_index - (blocks_per_layer * image_z));
            u32 const image_y = in_layer_block_index / blocks_per_row;

            u32 const image_x = in_layer_block_index - (image_y * blocks_per_row);

            return u32vec3(image_x * 4, image_y * 4, image_z);
        };

        u32 const start_block_index = chunk_index * blocks_per_chunk;
        u32 const end_block_index = std::min((chunk_index + 1) * blocks_per_chunk, blocks_total);
        std::array<Pixel, 16> data_block_to_compress = {};

        for (u32 block_index = start_block_index; block_index < end_block_index; ++block_index)
        {
            u32vec3 const block_start_image_coords = block_index_to_image_coords(block_index);

            DBG_ASSERT_TRUE_M(
                block_start_image_coords.x < info.image_dimensions.x &&
                block_start_image_coords.y < info.image_dimensions.y &&
                block_start_image_coords.z < info.image_dimensions.z,
                "Calculated coordinates outside of image bounds");

            // Gather the 4x4 source block one texel at a time, clamping to the image bounds.
            for (u32 block_y = 0; block_y < 4; ++block_y)
            {
                for (u32 block_x = 0; block_x < 4; ++block_x)
                {
                    u32 const src_x = std::min(block_start_image_coords.x + block_x, info.image_dimensions.x - 1);
                    u32 const src_y = std::min(block_start_image_coords.y + block_y, info.image_dimensions.y - 1);
                    u32 const src_z = block_start_image_coords.z;

                    u32 const linear_src_pixel_index = src_x + (src_y * info.image_dimensions.x) + (src_z * pixels_per_layer);
                    u32 const linear_src_data_index = linear_src_pixel_index * sizeof(Pixel);
                    DBG_ASSERT_TRUE_M(linear_src_data_index < info.src_data.size(), "Calculated linear source data index outside of image bounds");

                    u32 const block_linear_index = (block_y * 4) + block_x;
                    std::memcpy(&data_block_to_compress[block_linear_index], &info.src_data[linear_src_data_index], sizeof(Pixel));
                }
            }

            u32 const stride_in_bytes = 4 * sizeof(Pixel);
            switch(info.target_format)
            {
                case daxa::Format::BC1_RGB_UNORM_BLOCK:
                case daxa::Format::BC1_RGB_SRGB_BLOCK:
                case daxa::Format::BC1_RGBA_UNORM_BLOCK:
                case daxa::Format::BC1_RGBA_SRGB_BLOCK:
                {
                    // BC1 stores 8 byes per block. A single-channel fp32 source is normalized SDF data
                    // compressed with the custom encoder; any other source is a plain BC1 colour block.
                    if (info.source_format == daxa::Format::R32_SFLOAT)
                    {
                        u64 * const destination = reinterpret_cast<u64 *>(&info.dst_data[block_index * 8]);
                        CompressBlockBC1SDF(destination, std::span<float>(reinterpret_cast<float*>(data_block_to_compress.data()), 16));
                    }
                    else
                    {
                        unsigned char * const destination = reinterpret_cast<unsigned char *>(&info.dst_data[block_index * 8]);
                        CompressBlockBC1(reinterpret_cast<unsigned char const* const>(data_block_to_compress.data()), stride_in_bytes, destination);
                    }
                    break;
                }
                case daxa::Format::BC4_UNORM_BLOCK:
                case daxa::Format::BC4_SNORM_BLOCK:
                {
                    // BC4 stores 8 byes per block.
                    unsigned char * const destination = reinterpret_cast<unsigned char *>(&info.dst_data[block_index * 8]);
                    CompressBlockBC4(reinterpret_cast<unsigned char const* const>(data_block_to_compress.data()), stride_in_bytes, destination);
                    break;
                }
                case daxa::Format::BC5_UNORM_BLOCK:
                case daxa::Format::BC5_SNORM_BLOCK:
                {
                    // For some reason the BC5 commpress function wants the two channels not interleaved.
                    std::array<unsigned char, 16> red_block = {};
                    std::array<unsigned char, 16> green_block = {};
                    auto const * const interleaved = reinterpret_cast<unsigned char const *>(data_block_to_compress.data());
                    for (u32 texel_in_block = 0; texel_in_block < 16; ++texel_in_block)
                    {
                        red_block[texel_in_block] = interleaved[texel_in_block * 2 + 0];
                        green_block[texel_in_block] = interleaved[texel_in_block * 2 + 1];
                    }
                    // BC5 stores 16 byes per block.
                    unsigned char * const destination = reinterpret_cast<unsigned char *>(&info.dst_data[block_index * 16]);
                    // BC5 takes stride per channel not per pixel, so the stride is half the interleaved stride since there are two channels.
                    CompressBlockBC5(red_block.data(), stride_in_bytes / 2, green_block.data(), stride_in_bytes / 2, destination);
                    break;
                }
                case daxa::Format::BC6H_UFLOAT_BLOCK:
                case daxa::Format::BC6H_SFLOAT_BLOCK:
                {
                    // BC6 stores 16 byes per block.
                    unsigned char * const destination = reinterpret_cast<unsigned char *>(&info.dst_data[block_index * 16]);
                    // BC6 takes stride in shorts, not bytes.
                    CompressBlockBC6(reinterpret_cast<unsigned short const* const>(data_block_to_compress.data()), stride_in_bytes / 2, destination);
                    break;
                }
                case daxa::Format::BC7_UNORM_BLOCK:
                case daxa::Format::BC7_SRGB_BLOCK:
                {
                    // BC1 stores 16 byes per block.
                    unsigned char * const destination = reinterpret_cast<unsigned char *>(&info.dst_data[block_index * 16]);
                    // BC6 takes stride in shorts, not bytes.
                    CompressBlockBC7(reinterpret_cast<unsigned char const* const>(data_block_to_compress.data()), stride_in_bytes, destination);
                    break;
                }
                default:
                {
                    DBG_ASSERT_TRUE_M(false, "Undefined block compression format!");
                    return;
                }
            }
        }

    };
};

auto compress_image(CreateCompressedImageInfo const & info) -> std::shared_ptr<Task>
{
    FormatInfo const source_format_info = get_format_info(info.source_format);
    u32 const texel_size_in_bytes = source_format_info.block_byte_size;
    DBG_ASSERT_TRUE_M(source_format_info.channel_count != 0 && texel_size_in_bytes != 0,
        "compress_image: source_format is not a supported uncompressed compression source");

    [[maybe_unused]] u32 const texels_requested_for_compression = info.image_dimensions.x * info.image_dimensions.y * info.image_dimensions.z;
    DBG_ASSERT_TRUE_M(info.src_data.size() / texel_size_in_bytes >= texels_requested_for_compression,
                      "Mismatch between image dimensions and data provided for compression");

    switch(info.target_format)
    {
        case daxa::Format::BC1_RGB_UNORM_BLOCK:
        case daxa::Format::BC1_RGB_SRGB_BLOCK:
        case daxa::Format::BC1_RGBA_UNORM_BLOCK:
        case daxa::Format::BC1_RGBA_SRGB_BLOCK:
        {
            return std::make_shared<CompressTask<4>>(info);
        }
        case daxa::Format::BC4_UNORM_BLOCK:
        case daxa::Format::BC4_SNORM_BLOCK:
        {
            return std::make_shared<CompressTask<1>>(info);
        }
        case daxa::Format::BC5_UNORM_BLOCK:
        case daxa::Format::BC5_SNORM_BLOCK:
        {
            return std::make_shared<CompressTask<2>>(info);
        }
        case daxa::Format::BC6H_UFLOAT_BLOCK:
        case daxa::Format::BC6H_SFLOAT_BLOCK:
        {
            return std::make_shared<CompressTask<6>>(info);
        }
        case daxa::Format::BC7_UNORM_BLOCK:
        case daxa::Format::BC7_SRGB_BLOCK:
        {
            return std::make_shared<CompressTask<4>>(info);
        }
        default:
        {
            DBG_ASSERT_TRUE_M(false, "compress_image: target_format is not a supported BC block format");
            return nullptr;
        }
    }
}
