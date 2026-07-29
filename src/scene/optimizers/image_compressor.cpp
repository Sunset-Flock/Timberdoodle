#include "image_processor.hpp"
#include "sdf_bc1_compressor.hpp"
#include <CMP_Core.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>

#include "../../shader_shared/shared.inl" // round_up_div

static constexpr u32 DEFAULT_BLOCKS_PER_CHUNK = 128;

template <u32 TexelByteCount>
struct CompressTask : Task
{
    private:
        typedef std::array<std::byte, TexelByteCount> BlockTexel;

        // Which codec's CreateOptions produced codec_options, and so which DestroyOptions must free it.
        enum struct CodecOptionsOwner
        {
            NONE,
            BC6,
            BC7,
        };

        // One row of a 4x4 block, which is the source stride the codec entry points take.
        static constexpr u32 BLOCK_ROW_STRIDE_IN_BYTES = 4 * TexelByteCount;

        CreateCompressedImageInfo info;
        u32 const blocks_per_chunk = DEFAULT_BLOCKS_PER_CHUNK;

        u32 blocks_total;
        u32 blocks_per_layer;
        u32 blocks_per_row;
        u32 texels_per_layer;

        u32 source_texel_bytes;
        u32 target_block_bytes;

        void * codec_options = nullptr;
        CodecOptionsOwner codec_options_owner = CodecOptionsOwner::NONE;

        auto block_index_to_image_coords(u32 block_index) const -> u32vec3
        {
            u32 const image_z = block_index / blocks_per_layer;

            u32 const in_layer_block_index = (block_index - (blocks_per_layer * image_z));
            u32 const image_y = in_layer_block_index / blocks_per_row;

            u32 const image_x = in_layer_block_index - (image_y * blocks_per_row);

            return u32vec3(image_x * 4, image_y * 4, image_z);
        }

    public:

    CompressTask(CreateCompressedImageInfo const & info)
        : info{info}
    {
        FormatInfo const source_format_info = get_info_from_format(info.source_format);
        source_texel_bytes = source_format_info.block_byte_size;
        DBG_ASSERT_TRUE_M(source_format_info.block_width == 1 && source_format_info.block_height == 1 && source_texel_bytes != 0,
            "CompressTask: source_format is not a supported uncompressed compression source");
        // The gather memcpys source_texel_bytes into a TexelByteCount texel, so a wider source overruns the block.
        DBG_ASSERT_TRUE_M(source_texel_bytes <= TexelByteCount,
            "CompressTask: source texel is wider than the block texel the codec is fed");

        target_block_bytes = get_info_from_format(info.target_format).block_byte_size;

        if (info.target_format == daxa::Format::BC7_UNORM_BLOCK || info.target_format == daxa::Format::BC7_SRGB_BLOCK)
        {
            CreateOptionsBC7(&codec_options);
            codec_options_owner = CodecOptionsOwner::BC7;
            // A source with fewer than 4 channels carries no alpha, so it is opaque-padded into the block;
            // imageNeedsAlpha=false then frees BC7 to use its opaque-only modes for better colour quality.
            bool const image_needs_alpha = source_format_info.channel_count >= 4;
            SetAlphaOptionsBC7(codec_options, image_needs_alpha, false, false);
        }
        if (info.target_format == daxa::Format::BC6H_UFLOAT_BLOCK || info.target_format == daxa::Format::BC6H_SFLOAT_BLOCK)
        {
            CreateOptionsBC6(&codec_options);
            codec_options_owner = CodecOptionsOwner::BC6;
            // BC6H block texels are always fp16; this only picks the UF16 vs SF16 clamping of the encoded values.
            SetSignedBC6(codec_options, info.target_format == daxa::Format::BC6H_SFLOAT_BLOCK);
        }
        // Ceil-based block counts so non-4-aligned extents (and sub-4x4 mip levels) still produce a full
        // set of blocks - the partial edge blocks are padded by clamping in the gather loop below. This
        // matches write_texture_tido, which sizes every mip with ceil(dim/4) blocks.
        blocks_per_row = round_up_div(info.image_dimensions.x, 4u);
        u32 const blocks_per_col = round_up_div(info.image_dimensions.y, 4u);
        blocks_per_layer = blocks_per_row * blocks_per_col;
        blocks_total = blocks_per_layer * info.image_dimensions.z;
        texels_per_layer = info.image_dimensions.x * info.image_dimensions.y;

        // Stated as a multiplication rather than a division so an unsupported source cannot fault here.
        [[maybe_unused]] u64 const texels_requested_for_compression = s_cast<u64>(texels_per_layer) * info.image_dimensions.z;
        DBG_ASSERT_TRUE_M(info.src_data.size() >= texels_requested_for_compression * source_texel_bytes,
            "Mismatch between image dimensions and data provided for compression");
        DBG_ASSERT_TRUE_M(info.dst_data.size() >= s_cast<u64>(blocks_total) * target_block_bytes,
            "Destination smaller than the ceil-based block count the image dimensions require");

        chunk_count = round_up_div(blocks_total, blocks_per_chunk);
    }

    ~CompressTask() override
    {
        switch (codec_options_owner)
        {
            case CodecOptionsOwner::NONE: break;
            case CodecOptionsOwner::BC6:  DestroyOptionsBC6(codec_options); break;
            case CodecOptionsOwner::BC7:  DestroyOptionsBC7(codec_options); break;
        }
    }

    virtual void callback(u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        u32 const start_block_index = chunk_index * blocks_per_chunk;
        u32 const end_block_index = std::min((chunk_index + 1) * blocks_per_chunk, blocks_total);

        std::array<BlockTexel, 16> data_block_to_compress = {};
        // Opaque-pad the channels the source lacks (e.g. an RGB source into a 4-channel BC7 block). The gather
        // overwrites only the leading source_texel_bytes of a texel, so this holds for every block in the chunk.
        for (BlockTexel & block_texel : data_block_to_compress) { block_texel.fill(std::byte{0xFF}); }

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

                    // 64 bit index math: a large 3D source overflows a 32 bit byte offset well before it overflows the texel count.
                    u64 const linear_src_texel_index = src_x + (s_cast<u64>(src_y) * info.image_dimensions.x) + (s_cast<u64>(src_z) * texels_per_layer);
                    u64 const linear_src_data_index = linear_src_texel_index * source_texel_bytes;
                    DBG_ASSERT_TRUE_M(linear_src_data_index + source_texel_bytes <= info.src_data.size(), "Calculated linear source data index outside of image bounds");

                    u32 const block_linear_index = (block_y * 4) + block_x;
                    BlockTexel & block_texel = data_block_to_compress[block_linear_index];
                    std::memcpy(block_texel.data(), &info.src_data[linear_src_data_index], source_texel_bytes);
                }
            }

            // Blocks are stored back to back at the target's block byte size - 8 for BC1/BC4, 16 for BC5/BC6/BC7.
            unsigned char * const destination = reinterpret_cast<unsigned char *>(&info.dst_data[s_cast<u64>(block_index) * target_block_bytes]);

            switch(info.target_format)
            {
                case daxa::Format::BC1_RGB_UNORM_BLOCK:
                case daxa::Format::BC1_RGB_SRGB_BLOCK:
                case daxa::Format::BC1_RGBA_UNORM_BLOCK:
                case daxa::Format::BC1_RGBA_SRGB_BLOCK:
                {
                    // A single-channel fp32 source is normalized SDF data compressed with the custom encoder;
                    // any other source is a plain BC1 colour block.
                    if (info.source_format == daxa::Format::R32_SFLOAT)
                    {
                        CompressBlockBC1SDF(reinterpret_cast<u64 *>(destination), std::span<float>(reinterpret_cast<float*>(data_block_to_compress.data()), 16));
                    }
                    else
                    {
                        CompressBlockBC1(reinterpret_cast<unsigned char const* const>(data_block_to_compress.data()), BLOCK_ROW_STRIDE_IN_BYTES, destination);
                    }
                    break;
                }
                case daxa::Format::BC4_UNORM_BLOCK:
                case daxa::Format::BC4_SNORM_BLOCK:
                {
                    // The SNORM target is a separate codec entry point taking a signed source, not an option.
                    if (info.target_format == daxa::Format::BC4_SNORM_BLOCK)
                    {
                        CompressBlockBC4S(reinterpret_cast<char const* const>(data_block_to_compress.data()), BLOCK_ROW_STRIDE_IN_BYTES, destination);
                    }
                    else
                    {
                        CompressBlockBC4(reinterpret_cast<unsigned char const* const>(data_block_to_compress.data()), BLOCK_ROW_STRIDE_IN_BYTES, destination);
                    }
                    break;
                }
                case daxa::Format::BC5_UNORM_BLOCK:
                case daxa::Format::BC5_SNORM_BLOCK:
                {
                    // For some reason the BC5 commpress function wants the two channels not interleaved.
                    std::array<char, 16> red_block = {};
                    std::array<char, 16> green_block = {};
                    auto const * const interleaved = reinterpret_cast<char const *>(data_block_to_compress.data());
                    for (u32 texel_in_block = 0; texel_in_block < 16; ++texel_in_block)
                    {
                        red_block[texel_in_block] = interleaved[texel_in_block * 2 + 0];
                        green_block[texel_in_block] = interleaved[texel_in_block * 2 + 1];
                    }
                    // BC5 takes stride per channel not per texel, so the stride is half the interleaved stride since there are two channels.
                    // The SNORM target is a separate codec entry point taking a signed source, as with BC4.
                    if (info.target_format == daxa::Format::BC5_SNORM_BLOCK)
                    {
                        CompressBlockBC5S(red_block.data(), BLOCK_ROW_STRIDE_IN_BYTES / 2, green_block.data(), BLOCK_ROW_STRIDE_IN_BYTES / 2, destination);
                    }
                    else
                    {
                        CompressBlockBC5(
                            reinterpret_cast<unsigned char const *>(red_block.data()), BLOCK_ROW_STRIDE_IN_BYTES / 2,
                            reinterpret_cast<unsigned char const *>(green_block.data()), BLOCK_ROW_STRIDE_IN_BYTES / 2, destination);
                    }
                    break;
                }
                case daxa::Format::BC6H_UFLOAT_BLOCK:
                case daxa::Format::BC6H_SFLOAT_BLOCK:
                {
                    // codec_options carries the UF16/SF16 clamping the target asks for (see CompressTask ctor);
                    // null would select the codec's unsigned default.
                    // BC6 takes stride in shorts, not bytes.
                    CompressBlockBC6(reinterpret_cast<unsigned short const* const>(data_block_to_compress.data()), BLOCK_ROW_STRIDE_IN_BYTES / 2, destination, codec_options);
                    break;
                }
                case daxa::Format::BC7_UNORM_BLOCK:
                case daxa::Format::BC7_SRGB_BLOCK:
                {
                    // codec_options restricts the encoder to opaque modes when the source carries no alpha
                    // (see CompressTask ctor); null selects the codec defaults.
                    CompressBlockBC7(reinterpret_cast<unsigned char const* const>(data_block_to_compress.data()), BLOCK_ROW_STRIDE_IN_BYTES, destination, codec_options);
                    break;
                }
                default:
                {
                    DBG_ASSERT_TRUE_M(false, "Undefined block compression format!");
                    return;
                }
            }
        }
    }
};

auto compress_image(CreateCompressedImageInfo const & info) -> std::shared_ptr<Task>
{
    // CMP_Core's init_BC7ramps() marks its ramp table initialized before it finishes filling it, so two
    // threads entering CreateOptionsBC7 at once can encode against a half-built table and emit different
    // blocks run to run. Warming it up once here - the single entry point - keeps cooked output deterministic.
    static std::once_flag bc7_ramp_warmup;
    std::call_once(bc7_ramp_warmup, []{
        void * warmup_options = nullptr;
        CreateOptionsBC7(&warmup_options);
        DestroyOptionsBC7(warmup_options);
    });

    // The template parameter is the byte width of the block texel each codec entry point wants to be fed;
    // CompressTask validates the source against it and opaque-pads any channels the source is missing.
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
