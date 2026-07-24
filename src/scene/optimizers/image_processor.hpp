#pragma once

#include <vector>
#include <array>
#include <string>
#include <span>
#include <memory>
#include <utility>
#include <variant>
#include <cstddef>

#include <daxa/daxa.hpp>

#include "../../timberdoodle.hpp"
#include "../../multithreading/thread_pool.hpp"
#include "../importers/openvdb_importer.hpp"
#include "../tido_format/tido_format.hpp"
using namespace tido::types;

enum struct ImageFileFormat
{
    PNG,
    KTX2,
};

struct TidoImageWithData
{
    TidoImageDescriptor descriptor = {};
    std::vector<std::byte> data = {};
};

enum struct ImageProcessResult
{
    SUCCESS,
    FAILED_UNHANDLED_FILE_FORMAT,
    FAILED_TO_PARSE_KTX,
    FAILED_TO_PARSE_PNG,
    FAILED_KTX_NEEDS_TRANSCODE,
    FAILED_TO_LOAD_KTX,
    FAILED_TO_DECODE_PNG,
    SOURCE_HAS_NO_ALPHA,
};

struct ImageParseInfo
{
    std::span<std::byte const> src_data = {};
    ImageFileFormat source_format = {};
    // Tag the decoded format sRGB when the caller's target format is sRGB, so downstream mip filtering
    // (which operates on the decoded format) happens in gamma-correct space.
    bool is_srgb = false;
};
auto image_parse(ImageParseInfo const & info) -> std::variant<ImageProcessResult, TidoImageWithData>;

struct ImageTranscodeInfo
{
    std::span<std::byte const> src_data = {};
    ImageFileFormat source_format = {};
    daxa::Format target_format = {};
    std::vector<u8> channel_mapping = {};
};
auto image_transcode(ImageTranscodeInfo const & info) -> std::variant<ImageProcessResult, TidoImageWithData>;

void image_resize_for_mipmaps(TidoImageWithData & image, u32 mip_count);

struct DownsampleImageInfo
{
    std::span<std::byte const> src_data;
    std::span<std::byte> dst_data;
    u32vec3 src_dimensions;               // z == 1 for a 2D source; the box filter is 2x2 then, 2x2x2 in 3D
    daxa::Format format;
};
auto downsample_image(DownsampleImageInfo const & info) -> std::shared_ptr<Task>;

struct RemapChannelsInfo
{
    std::span<std::byte const> src_data;
    u32 texel_count;                      // flat count (2D or 3D); remap is purely per-texel
    daxa::Format format;                  // source layout: channel count + numeric type + bit depth of src_data
    std::span<u8 const> channel_mapping;  // dst channel d <- src channel channel_mapping[d]
    daxa::Format dst_format;              // output layout: channel_count must equal channel_mapping.size()

    std::span<std::byte> dst_data;
};
// Select/reorder channels from src into dst per channel_mapping, converting each sample by the well-defined
// (source numeric type -> destination numeric type) matrix: NORM<->NORM requantizes bit depth, FLOAT<->FLOAT
// passes through, and cross-domain rescales (NORM<->INT), assigns (NORM/INT<->FLOAT) or clamps/rounds. No
// opaque padding: the compressor expands a narrower source into its block layout.
auto remap_channels(RemapChannelsInfo const & info) -> std::shared_ptr<Task>;

struct CreateCompressedImageInfo
{
    std::span<std::byte const> src_data;
    u32vec3 image_dimensions;
    daxa::Format target_format;
    // Uncompressed source layout. Its channel count drives how each source texel expands into the codec's
    // block: a 3-channel source feeding a 4-channel BC7 block is opaque-padded, and BC7 is told the image
    // has no alpha so it may use its opaque-only modes.
    daxa::Format source_format;

    std::span<std::byte> dst_data;
};

auto compress_image(CreateCompressedImageInfo const & info) -> std::shared_ptr<Task>;
