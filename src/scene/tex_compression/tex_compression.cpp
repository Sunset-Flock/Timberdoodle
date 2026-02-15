#include "tex_compression.hpp"
#include "sdf_bc1_compressor.hpp"
#include <CMP_Core.h>
#include <iostream>

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
        blocks_per_layer = (info.image_dimensions.x / 4) * (info.image_dimensions.y / 4);
        blocks_per_row = (info.image_dimensions.x / 4);
        blocks_total = blocks_per_layer * info.image_dimensions.z;
        pixels_per_layer = info.image_dimensions.x * info.image_dimensions.y;

        chunk_count = (blocks_total + blocks_per_chunk - 1) / blocks_per_chunk;
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

            // Inner loop that writes the data for the conversion
            for (u32 block_y = 0; block_y < 4; ++block_y)
            {
                u32 const linear_src_pixel_index = 
                    (block_start_image_coords.x) +
                    ((block_start_image_coords.y + block_y) * info.image_dimensions.x) +
                    (block_start_image_coords.z * pixels_per_layer);

                DBG_ASSERT_TRUE_M(
                    linear_src_pixel_index < info.image_dimensions.x * info.image_dimensions.y * info.image_dimensions.z,
                    "Calculated linear source pixel index outside of image bounds");

                u32 const linear_src_data_index = linear_src_pixel_index * sizeof(Pixel);
                DBG_ASSERT_TRUE_M(linear_src_data_index < info.in_data.size(), "Calculate linear source data index outside of image bounds");

                u32 const block_linear_index = (block_y * 4);
                std::memcpy(&data_block_to_compress[block_linear_index], &info.in_data[linear_src_data_index], 4 * sizeof(Pixel));
            }

            u32 const stride_in_bytes = 4 * sizeof(Pixel);
            switch(info.compression)
            {
                case Compression::BC1: 
                { 
                    // BC1 stores 8 byes per block.
                    unsigned char * const destination = reinterpret_cast<unsigned char *>(&info.out_data[block_index * 8]);
                    CompressBlockBC1(reinterpret_cast<unsigned char const* const>(data_block_to_compress.data()), stride_in_bytes, destination);
                    break;
                }
                case Compression::BC1_SDF: 
                { 
                    // BC1 SDF stores 8 byes per block.
                    u64 * const destination = reinterpret_cast<u64 *>(&info.out_data[block_index * 8]);
                    CompressBlockBC1SDF(destination, std::span<float>(reinterpret_cast<float*>(data_block_to_compress.data()), 16));
                    break;
                }
                case Compression::BC4: 
                { 
                    // BC4 stores 8 byes per block.
                    unsigned char * const destination = reinterpret_cast<unsigned char *>(&info.out_data[block_index * 8]);
                    CompressBlockBC4(reinterpret_cast<unsigned char const* const>(data_block_to_compress.data()), stride_in_bytes, destination);
                    break;
                }
                case Compression::BC6: 
                { 
                    // BC6 stores 16 byes per block.
                    unsigned char * const destination = reinterpret_cast<unsigned char *>(&info.out_data[block_index * 16]);
                    // BC6 takes stride in shorts, not bytes.
                    CompressBlockBC6(reinterpret_cast<unsigned short const* const>(data_block_to_compress.data()), stride_in_bytes / 2, destination);
                    break;
                }
                case Compression::BC7: 
                { 
                    // BC1 stores 16 byes per block.
                    unsigned char * const destination = reinterpret_cast<unsigned char *>(&info.out_data[block_index * 16]);
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
    DBG_ASSERT_TRUE_M(info.compression != Compression::UNDEFINED, "Undefined block compression format!");
    DBG_ASSERT_TRUE_M((info.image_dimensions.x % 4 == 0) && (info.image_dimensions.y % 4 == 0),
                      "Dimensions of block compressed images must be 4 aligned");


    u32 texel_size_in_bytes = 0;
    switch(info.compression)
    {
        case Compression::BC1    : { texel_size_in_bytes = 4u; break; }
        case Compression::BC1_SDF: { texel_size_in_bytes = 4u; break; }
        case Compression::BC4    : { texel_size_in_bytes = 1u; break; }
        case Compression::BC6    : { texel_size_in_bytes = 6u; break; }
        case Compression::BC7    : { texel_size_in_bytes = 4u; break; }
        default:
        {
            DBG_ASSERT_TRUE_M(false, "Undefined block compression format!");
            return nullptr;
        }
    }

    [[maybe_unused]] u32 const texels_requested_for_compression = info.image_dimensions.x * info.image_dimensions.y * info.image_dimensions.z;
    DBG_ASSERT_TRUE_M(info.in_data.size() / texel_size_in_bytes >= texels_requested_for_compression,
                      "Mismatch between image dimensions and data provided for compression");

    switch(info.compression)
    {
        case Compression::BC1: 
        { 
            return std::make_shared<CompressTask<4>>(info);
        }
        case Compression::BC1_SDF: 
        { 
            return std::make_shared<CompressTask<4>>(info);
        }
        case Compression::BC4: 
        {
            return std::make_shared<CompressTask<1>>(info);
        }
        case Compression::BC6: 
        { 
            return std::make_shared<CompressTask<6>>(info);
        }
        case Compression::BC7: 
        { 
            return std::make_shared<CompressTask<4>>(info);
        }
        default:
        {
            DBG_ASSERT_TRUE_M(false, "Undefined block compression format!");
            return nullptr;
        }
    }
}