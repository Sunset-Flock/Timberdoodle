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
    u32vec2 src_dimensions;
    daxa::Format format;
};
auto downsample_image(DownsampleImageInfo const & info) -> std::shared_ptr<Task>;

struct RemapChannelsInfo
{
    std::span<std::byte const> src_data;
    u32vec2 dimensions;
    daxa::Format format;
    std::span<u8 const> channel_mapping;

    std::span<std::byte> dst_data;
};
auto remap_channels(RemapChannelsInfo const & info) -> std::shared_ptr<Task>;

struct CreateCompressedImageInfo
{
    std::span<std::byte const> src_data;
    u32vec3 image_dimensions;
    daxa::Format target_format;
    daxa::Format source_format;

    std::span<std::byte> dst_data;
};

auto compress_image(CreateCompressedImageInfo const & info) -> std::shared_ptr<Task>;
