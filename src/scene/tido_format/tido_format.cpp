#include "tido_format.hpp"
#include "../streamer.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <string_view>
#include <vector>

#include <chrono>
#include <thread>

#include <fmt/format.h>

#include "tido_util.hpp"
#include "../../io/file_io.hpp"
#include "../../json_utils/tido_format.hpp"

namespace
{
// Retry a single-attempt write while the file is held open by someone else; give up on any other error.
template <typename WriteOp>
auto retry_while_locked(WriteOp && write_op) -> bool
{
    static constexpr u32 WRITE_RETRY_DELAY_MS = 1;
    while (true)
    {
        FileIoResult const result = write_op();
        if (result == FileIoResult::SUCCESS) { return true; }
        if (result != FileIoResult::LOCKED) { return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(WRITE_RETRY_DELAY_MS));
    }
}

auto write_file_exclusive_retry(std::filesystem::path const & path, void const * data, usize size) -> bool
{
    return retry_while_locked([&]() { return write_file(path, data, size); });
}

// The header region is a single JSON array [metadata_hash, descriptor]. Each element is pretty-printed
// independently, and these are just the literal wrapper bytes stitched around them (not a re-serialization
// of the combined structure).
constexpr std::string_view TIDO_HEADER_ARRAY_PREFIX = "[\n";
constexpr std::string_view TIDO_HEADER_ARRAY_SEPARATOR = ",\n";
constexpr std::string_view TIDO_HEADER_ARRAY_SUFFIX = "\n]";

struct WriteTidoFileResult
{
    std::filesystem::path bin_source = {};
    u64 file_data_offset = {};
};

// Assembles [preamble][JSON array: metadata_hash, descriptor][payload] and writes it to the artifact's
// keyed path in the store; callers pass their already-serialized descriptor and built payload.
auto write_tido_file(WriteTidoFileInfo const & info, std::string const & serialized_descriptor, std::span<std::byte const> data_payload) -> std::optional<WriteTidoFileResult>
{
    std::filesystem::path const tido_path = tido_artifact_path(info.store_dir, info.metadata_hash.cache_key);

    std::error_code create_destination_folder_error = {};
    std::filesystem::create_directories(tido_path.parent_path(), create_destination_folder_error);

    DBG_ASSERT_TRUE_M(
        !create_destination_folder_error || create_destination_folder_error == std::errc::file_exists,
        "write_tido_file: failed to create destination folder");

    std::vector<std::byte> header_payload = {};
    header_payload.resize(TIDO_FILE_PREAMBLE_SIZE);

    std::string const serialized_tido_metadata_hash = serialize_tido_metadata_hash(info.metadata_hash);
    tido_append_bytes(header_payload, TIDO_HEADER_ARRAY_PREFIX.data(), TIDO_HEADER_ARRAY_PREFIX.size());
    tido_append_bytes(header_payload, serialized_tido_metadata_hash.data(), serialized_tido_metadata_hash.size());
    tido_append_bytes(header_payload, TIDO_HEADER_ARRAY_SEPARATOR.data(), TIDO_HEADER_ARRAY_SEPARATOR.size());
    tido_append_bytes(header_payload, serialized_descriptor.data(), serialized_descriptor.size());
    tido_append_bytes(header_payload, TIDO_HEADER_ARRAY_SUFFIX.data(), TIDO_HEADER_ARRAY_SUFFIX.size());

    // A fixed preamble precedes the json metadata headers, which precede the raw payload.
    u64 const file_data_offset = header_payload.size();
    TidoFilePreamble preamble = {};
    preamble.header_byte_length = header_payload.size() - TIDO_FILE_PREAMBLE_SIZE;
    std::copy(r_cast<std::byte *>(&preamble), r_cast<std::byte *>(&preamble) + TIDO_FILE_PREAMBLE_SIZE, header_payload.begin());

    // TODO(saky): still one copy of data_payload here; a scatter/gather write_file_exclusive would avoid it.
    header_payload.reserve(header_payload.size() + data_payload.size());
    header_payload.insert(header_payload.end(), data_payload.begin(), data_payload.end());

    // Write to a private temp file and rename it onto the key: a torn artifact whose header still parses would
    // be sticky forever, since a probe only checks that a file exists at the key. The counter keeps two threads
    // cooking the same key off each other's temp file; the rename replaces whatever is already there.
    static std::atomic<u64> temp_file_counter = 0;
    std::filesystem::path const temp_path = tido_path.parent_path() /
        fmt::format("{:016x}.{}.tmp", info.metadata_hash.cache_key, temp_file_counter.fetch_add(1));
    if (!write_file_exclusive_retry(temp_path, header_payload.data(), header_payload.size())) { return std::nullopt; }

    if (!retry_while_locked([&]() { return rename_file(temp_path, tido_path); }))
    {
        std::error_code remove_error = {};
        std::filesystem::remove(temp_path, remove_error);
        return std::nullopt;
    }

    return WriteTidoFileResult{.bin_source = tido_path, .file_data_offset = file_data_offset};
}
}

auto get_info_from_format(daxa::Format format) -> FormatInfo
{
    switch (format)
    {
        // 8-bit
        case daxa::Format::R8_UNORM:             return {.channel_count = 1, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::UNORM, .block_width = 1, .block_height = 1, .block_byte_size = 1};
        case daxa::Format::R8_SNORM:             return {.channel_count = 1, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::SNORM, .block_width = 1, .block_height = 1, .block_byte_size = 1};
        case daxa::Format::R8_SRGB:              return {.channel_count = 1, .channel_byte_size = 1, .is_srgb = true,  .numeric_type = FormatNumericType::UNORM, .block_width = 1, .block_height = 1, .block_byte_size = 1};
        case daxa::Format::R8_SINT:              return {.channel_count = 1, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::SINT,  .block_width = 1, .block_height = 1, .block_byte_size = 1};
        case daxa::Format::R8G8_UNORM:           return {.channel_count = 2, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::UNORM, .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R8G8_SNORM:           return {.channel_count = 2, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::SNORM, .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R8G8_SRGB:            return {.channel_count = 2, .channel_byte_size = 1, .is_srgb = true,  .numeric_type = FormatNumericType::UNORM, .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R8G8_SINT:            return {.channel_count = 2, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::SINT,  .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R8G8B8_UNORM:         return {.channel_count = 3, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::UNORM, .block_width = 1, .block_height = 1, .block_byte_size = 3};
        case daxa::Format::R8G8B8_SNORM:         return {.channel_count = 3, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::SNORM, .block_width = 1, .block_height = 1, .block_byte_size = 3};
        case daxa::Format::R8G8B8_SRGB:          return {.channel_count = 3, .channel_byte_size = 1, .is_srgb = true,  .numeric_type = FormatNumericType::UNORM, .block_width = 1, .block_height = 1, .block_byte_size = 3};
        case daxa::Format::R8G8B8A8_UNORM:       return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::UNORM, .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R8G8B8A8_SNORM:       return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::SNORM, .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R8G8B8A8_SRGB:        return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = true,  .numeric_type = FormatNumericType::UNORM, .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R8G8B8A8_SINT:        return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::SINT,  .block_width = 1, .block_height = 1, .block_byte_size = 4};
        // 16-bit
        case daxa::Format::R16_UNORM:            return {.channel_count = 1, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::UNORM,  .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R16G16_UNORM:         return {.channel_count = 2, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::UNORM,  .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R16G16B16_UNORM:      return {.channel_count = 3, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::UNORM,  .block_width = 1, .block_height = 1, .block_byte_size = 6};
        case daxa::Format::R16G16B16A16_UNORM:   return {.channel_count = 4, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::UNORM,  .block_width = 1, .block_height = 1, .block_byte_size = 8};
        case daxa::Format::R16_UINT:             return {.channel_count = 1, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::UINT,   .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R16_SINT:             return {.channel_count = 1, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::SINT,   .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R16_SFLOAT:           return {.channel_count = 1, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::SFLOAT, .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R16G16_UINT:          return {.channel_count = 2, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::UINT,   .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R16G16_SINT:          return {.channel_count = 2, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::SINT,   .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R16G16_SFLOAT:        return {.channel_count = 2, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::SFLOAT, .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R16G16B16_UINT:       return {.channel_count = 3, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::UINT,   .block_width = 1, .block_height = 1, .block_byte_size = 6};
        case daxa::Format::R16G16B16_SINT:       return {.channel_count = 3, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::SINT,   .block_width = 1, .block_height = 1, .block_byte_size = 6};
        case daxa::Format::R16G16B16_SFLOAT:     return {.channel_count = 3, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::SFLOAT, .block_width = 1, .block_height = 1, .block_byte_size = 6};
        case daxa::Format::R16G16B16A16_UINT:    return {.channel_count = 4, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::UINT,   .block_width = 1, .block_height = 1, .block_byte_size = 8};
        case daxa::Format::R16G16B16A16_SINT:    return {.channel_count = 4, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::SINT,   .block_width = 1, .block_height = 1, .block_byte_size = 8};
        case daxa::Format::R16G16B16A16_SFLOAT:  return {.channel_count = 4, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::SFLOAT, .block_width = 1, .block_height = 1, .block_byte_size = 8};
        // 32-bit
        case daxa::Format::R32_UINT:             return {.channel_count = 1, .channel_byte_size = 4, .is_srgb = false, .numeric_type = FormatNumericType::UINT,   .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R32_SINT:             return {.channel_count = 1, .channel_byte_size = 4, .is_srgb = false, .numeric_type = FormatNumericType::SINT,   .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R32_SFLOAT:           return {.channel_count = 1, .channel_byte_size = 4, .is_srgb = false, .numeric_type = FormatNumericType::SFLOAT, .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R32G32_UINT:          return {.channel_count = 2, .channel_byte_size = 4, .is_srgb = false, .numeric_type = FormatNumericType::UINT,   .block_width = 1, .block_height = 1, .block_byte_size = 8};
        case daxa::Format::R32G32_SINT:          return {.channel_count = 2, .channel_byte_size = 4, .is_srgb = false, .numeric_type = FormatNumericType::SINT,   .block_width = 1, .block_height = 1, .block_byte_size = 8};
        case daxa::Format::R32G32_SFLOAT:        return {.channel_count = 2, .channel_byte_size = 4, .is_srgb = false, .numeric_type = FormatNumericType::SFLOAT, .block_width = 1, .block_height = 1, .block_byte_size = 8};
        case daxa::Format::R32G32B32_SFLOAT:     return {.channel_count = 3, .channel_byte_size = 4, .is_srgb = false, .numeric_type = FormatNumericType::SFLOAT, .block_width = 1, .block_height = 1, .block_byte_size = 12};
        case daxa::Format::R32G32B32A32_UINT:    return {.channel_count = 4, .channel_byte_size = 4, .is_srgb = false, .numeric_type = FormatNumericType::UINT,   .block_width = 1, .block_height = 1, .block_byte_size = 16};
        case daxa::Format::R32G32B32A32_SINT:    return {.channel_count = 4, .channel_byte_size = 4, .is_srgb = false, .numeric_type = FormatNumericType::SINT,   .block_width = 1, .block_height = 1, .block_byte_size = 16};
        case daxa::Format::R32G32B32A32_SFLOAT:  return {.channel_count = 4, .channel_byte_size = 4, .is_srgb = false, .numeric_type = FormatNumericType::SFLOAT, .block_width = 1, .block_height = 1, .block_byte_size = 16};
        // Block compressed (8 bytes / 4x4 block). For block formats the channel fields describe the codec's
        // texel - what it encodes from and decodes to - which is what get_format_from_info turns back into the
        // uncompressed source format a compress pass must be fed. block_* describes the storage.
        case daxa::Format::BC1_RGB_UNORM_BLOCK:  return {.channel_count = 3, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::UNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 8};
        case daxa::Format::BC1_RGB_SRGB_BLOCK:   return {.channel_count = 3, .channel_byte_size = 1, .is_srgb = true,  .numeric_type = FormatNumericType::UNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 8};
        case daxa::Format::BC1_RGBA_UNORM_BLOCK: return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::UNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 8};
        case daxa::Format::BC1_RGBA_SRGB_BLOCK:  return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = true,  .numeric_type = FormatNumericType::UNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 8};
        case daxa::Format::BC4_UNORM_BLOCK:      return {.channel_count = 1, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::UNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 8};
        case daxa::Format::BC4_SNORM_BLOCK:      return {.channel_count = 1, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::SNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 8};
        // Block compressed (16 bytes / 4x4 block)
        case daxa::Format::BC2_UNORM_BLOCK:      return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::UNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 16};
        case daxa::Format::BC2_SRGB_BLOCK:       return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = true,  .numeric_type = FormatNumericType::UNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 16};
        case daxa::Format::BC3_UNORM_BLOCK:      return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::UNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 16};
        case daxa::Format::BC3_SRGB_BLOCK:       return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = true,  .numeric_type = FormatNumericType::UNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 16};
        case daxa::Format::BC5_UNORM_BLOCK:      return {.channel_count = 2, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::UNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 16};
        case daxa::Format::BC5_SNORM_BLOCK:      return {.channel_count = 2, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::SNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 16};
        // BC6H texels are fp16 RGB; UFLOAT vs SFLOAT changes only how the codec clamps, not the source layout.
        case daxa::Format::BC6H_UFLOAT_BLOCK:
        case daxa::Format::BC6H_SFLOAT_BLOCK:    return {.channel_count = 3, .channel_byte_size = 2, .is_srgb = false, .numeric_type = FormatNumericType::SFLOAT, .block_width = 4, .block_height = 4, .block_byte_size = 16};
        case daxa::Format::BC7_UNORM_BLOCK:      return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = false, .numeric_type = FormatNumericType::UNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 16};
        case daxa::Format::BC7_SRGB_BLOCK:       return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = true,  .numeric_type = FormatNumericType::UNORM,  .block_width = 4, .block_height = 4, .block_byte_size = 16};
        default:
            DBG_ASSERT_TRUE_M(false, "get_info_from_format: unhandled format");
            return {};
    }
}

auto get_format_from_info(FormatInfo const & info) -> daxa::Format
{
    DBG_ASSERT_TRUE_M(info.channel_count >= 1 && info.channel_count <= 4, "get_format_from_info: channel count must be 1-4");

    constexpr daxa::Format X = daxa::Format::UNDEFINED; // combination get_info_from_format does not register
    // [numeric_type][byte_size: 1/2/4][channel_count: 1..4]. sRGB is applied afterwards (8-bit UNORM only).
    constexpr daxa::Format TABLE[5][3][4] = {
        /* UNORM  */ {
            {daxa::Format::R8_UNORM, daxa::Format::R8G8_UNORM, daxa::Format::R8G8B8_UNORM, daxa::Format::R8G8B8A8_UNORM},
            {daxa::Format::R16_UNORM, daxa::Format::R16G16_UNORM, daxa::Format::R16G16B16_UNORM, daxa::Format::R16G16B16A16_UNORM},
            {X, X, X, X},
        },
        /* SNORM  */ {
            {daxa::Format::R8_SNORM, daxa::Format::R8G8_SNORM, daxa::Format::R8G8B8_SNORM, daxa::Format::R8G8B8A8_SNORM},
            {X, X, X, X},
            {X, X, X, X},
        },
        /* UINT   */ {
            {X, X, X, X},
            {daxa::Format::R16_UINT, daxa::Format::R16G16_UINT, daxa::Format::R16G16B16_UINT, daxa::Format::R16G16B16A16_UINT},
            {daxa::Format::R32_UINT, daxa::Format::R32G32_UINT, X, daxa::Format::R32G32B32A32_UINT},
        },
        /* SINT   */ {
            {daxa::Format::R8_SINT, daxa::Format::R8G8_SINT, X, daxa::Format::R8G8B8A8_SINT},
            {daxa::Format::R16_SINT, daxa::Format::R16G16_SINT, daxa::Format::R16G16B16_SINT, daxa::Format::R16G16B16A16_SINT},
            {daxa::Format::R32_SINT, daxa::Format::R32G32_SINT, X, daxa::Format::R32G32B32A32_SINT},
        },
        /* SFLOAT */ {
            {X, X, X, X},
            {daxa::Format::R16_SFLOAT, daxa::Format::R16G16_SFLOAT, daxa::Format::R16G16B16_SFLOAT, daxa::Format::R16G16B16A16_SFLOAT},
            {daxa::Format::R32_SFLOAT, daxa::Format::R32G32_SFLOAT, daxa::Format::R32G32B32_SFLOAT, daxa::Format::R32G32B32A32_SFLOAT},
        },
    };

    u32 byte_size_index = 0;
    switch (info.channel_byte_size)
    {
        case 1: byte_size_index = 0; break;
        case 2: byte_size_index = 1; break;
        case 4: byte_size_index = 2; break;
        default: DBG_ASSERT_TRUE_M(false, "get_format_from_info: channel byte size must be 1, 2 or 4"); return daxa::Format::UNDEFINED;
    }

    daxa::Format format = TABLE[s_cast<u32>(info.numeric_type)][byte_size_index][info.channel_count - 1];
    if (info.is_srgb)
    {
        // sRGB is an 8-bit UNORM-only variant of the same channel layout.
        switch (format)
        {
            case daxa::Format::R8_UNORM:       format = daxa::Format::R8_SRGB; break;
            case daxa::Format::R8G8_UNORM:     format = daxa::Format::R8G8_SRGB; break;
            case daxa::Format::R8G8B8_UNORM:   format = daxa::Format::R8G8B8_SRGB; break;
            case daxa::Format::R8G8B8A8_UNORM: format = daxa::Format::R8G8B8A8_SRGB; break;
            default: DBG_ASSERT_TRUE_M(false, "get_format_from_info: sRGB only applies to 8-bit UNORM"); return daxa::Format::UNDEFINED;
        }
    }
    DBG_ASSERT_TRUE_M(format != daxa::Format::UNDEFINED, "get_format_from_info: unsupported channel/byte-size/numeric-type combination");
    return format;
}

namespace
{
// Byte size of one (mip, single layer) subresource at the given mip 0 extents.
auto subresource_byte_size(FormatInfo const & block, u32 width, u32 height, u32 depth, u32 mip) -> u64
{
    u32 const mip_w = std::max(1u, width >> mip);
    u32 const mip_h = std::max(1u, height >> mip);
    u32 const mip_d = std::max(1u, depth >> mip);
    u64 const blocks_x = (mip_w + block.block_width - 1) / block.block_width;
    u64 const blocks_y = (mip_h + block.block_height - 1) / block.block_height;
    return blocks_x * blocks_y * mip_d * block.block_byte_size;
}

} // namespace

auto write_tido_image(WriteTidoFileInfo const & info, TidoImageDescriptor const & descriptor) -> std::optional<ImageStreamerData>
{
    FormatInfo const block = get_info_from_format(descriptor.info.format);
    DBG_ASSERT_TRUE_M(block.block_byte_size != 0, "write_tido_image: unsupported texture format");

    std::vector<std::byte> data_payload = {};

    // The destination desrciptor might be different from the source descriptor (the subresource offsets can be different).
    TidoImageDescriptor dst_descriptor = { .info = descriptor.info, .subresources = {} };
    dst_descriptor.subresources.resize(descriptor.info.mip_level_count * descriptor.info.array_layer_count);

    data_payload.reserve(info.data.size());

    for (i32 mip = s_cast<i32>(descriptor.info.mip_level_count) - 1; mip >= 0; --mip)
    {
        for (u32 layer = 0; layer < descriptor.info.array_layer_count; ++layer)
        {
            u32 const subresource_index = descriptor.layer_mip_to_subresource_index(layer, s_cast<u32>(mip));
            TidoImageDescriptor::SubresourceEntry const & entry = descriptor.subresources.at(subresource_index);

            u64 const calculated_size = subresource_byte_size(block, descriptor.info.size.x, descriptor.info.size.y, descriptor.info.size.z, s_cast<u32>(mip));
            DBG_ASSERT_TRUE_M(entry.byte_size == calculated_size, "write_tido_image: size in the subresource entry does not match the calculated size for the mip level");
            DBG_ASSERT_TRUE_M(entry.offset + calculated_size <= info.data.size(), "write_tido_image: subresource out of source data bounds");

            u32 const dst_subresource_index = dst_descriptor.layer_mip_to_subresource_index(layer, s_cast<u32>(mip));
            dst_descriptor.subresources[dst_subresource_index] = {.offset = data_payload.size(), .byte_size = calculated_size};
            data_payload.insert(data_payload.end(), info.data.begin() + entry.offset, info.data.begin() + entry.offset + entry.byte_size);
        }
    }

    std::optional<WriteTidoFileResult> const write_result = write_tido_file(info, serialize_tido_image_descriptor(dst_descriptor), data_payload);
    if (!write_result.has_value()) { return std::nullopt; }

    ImageStreamerData result = {};
    result.descriptor = dst_descriptor;
    result.bin_source = write_result->bin_source;
    result.file_data_offset = write_result->file_data_offset;
    return result;
}

auto write_tido_mesh(WriteTidoFileInfo const & info, TidoMeshDescriptor const & descriptor) -> std::optional<MeshStreamerData>
{
    std::vector<std::byte> data_payload = {};
    data_payload.reserve(info.data.size());

    // The destination descriptor might be different from the source descriptor (the LOD blob offsets can be different).
    TidoMeshDescriptor dst_descriptor = {};
    dst_descriptor.lods.resize(descriptor.lods.size());

    for (u64 lod = 0; lod < descriptor.lods.size(); ++lod)
    {
        TidoMeshDescriptor::LodDescriptor const & entry = descriptor.lods.at(lod);
        DBG_ASSERT_TRUE_M(entry.offset + entry.byte_size <= info.data.size(), "write_tido_mesh: lod out of source data bounds");

        dst_descriptor.lods[lod] = entry;
        dst_descriptor.lods[lod].offset = data_payload.size();
        data_payload.insert(data_payload.end(), info.data.begin() + entry.offset, info.data.begin() + entry.offset + entry.byte_size);
    }

    std::optional<WriteTidoFileResult> const write_result = write_tido_file(info, serialize_tido_mesh_descriptor(dst_descriptor), data_payload);
    if (!write_result.has_value()) { return std::nullopt; }

    MeshStreamerData result = {};
    result.descriptor = dst_descriptor;
    result.bin_source = write_result->bin_source;
    result.file_data_offset = write_result->file_data_offset;
    return result;
}

auto tido_parse_preamble(std::span<std::byte const> preamble_data) -> std::optional<TidoFilePreamble>
{
    if (preamble_data.size() < sizeof(TidoFilePreamble)) { return std::nullopt; }
    TidoFilePreamble preamble = {};
    std::memcpy(&preamble, preamble_data.data(), sizeof(TidoFilePreamble));
    TidoFilePreamble const expected = {};
    if (preamble.magic != expected.magic || preamble.version != TidoFilePreamble::CURRENT_VERSION) { return std::nullopt; }

    return preamble;
}