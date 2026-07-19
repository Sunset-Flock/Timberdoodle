#include "tido_format.hpp"
#include "../streamer.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include <chrono>
#include <thread>

#include "tido_util.hpp"
#include "../../io/file_io.hpp"
#include "../../json_utils/tido_format.hpp"

namespace
{
auto write_file_exclusive_retry(std::filesystem::path const & path, void const * data, usize size) -> bool
{
    static constexpr u32 WRITE_RETRY_DELAY_MS = 1;
    while (true)
    {
        std::variant<std::monostate, FileIoResult> const result = write_file_exclusive(path, data, size);
        if (std::holds_alternative<std::monostate>(result)) { return true; }
        if (std::get<FileIoResult>(result) != FileIoResult::LOCKED) { return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(WRITE_RETRY_DELAY_MS));
    }
}

void append_string_bytes(std::vector<std::byte> & dst, std::string const & src)
{
    std::byte const * const begin = r_cast<std::byte const *>(src.data());
    dst.insert(dst.end(), begin, begin + src.size());
}
}

auto get_format_info(daxa::Format format) -> FormatInfo
{
    switch (format)
    {
        // 8-bit
        case daxa::Format::R8_UNORM:             return {.channel_count = 1, .channel_byte_size = 1, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 1};
        case daxa::Format::R8_SRGB:              return {.channel_count = 1, .channel_byte_size = 1, .is_srgb = true,  .block_width = 1, .block_height = 1, .block_byte_size = 1};
        case daxa::Format::R8_SINT:              return {.channel_count = 1, .channel_byte_size = 1, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 1};
        case daxa::Format::R8G8_UNORM:           return {.channel_count = 2, .channel_byte_size = 1, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R8G8_SRGB:            return {.channel_count = 2, .channel_byte_size = 1, .is_srgb = true,  .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R8G8_SINT:            return {.channel_count = 2, .channel_byte_size = 1, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R8G8B8A8_UNORM:       return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R8G8B8A8_SRGB:        return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = true,  .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R8G8B8A8_SINT:        return {.channel_count = 4, .channel_byte_size = 1, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 4};
        // 16-bit
        case daxa::Format::R16_UINT:
        case daxa::Format::R16_SINT:
        case daxa::Format::R16_SFLOAT:           return {.channel_count = 1, .channel_byte_size = 2, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 2};
        case daxa::Format::R16G16_UINT:
        case daxa::Format::R16G16_SINT:
        case daxa::Format::R16G16_SFLOAT:        return {.channel_count = 2, .channel_byte_size = 2, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R16G16B16_UINT:
        case daxa::Format::R16G16B16_SINT:
        case daxa::Format::R16G16B16_SFLOAT:     return {.channel_count = 3, .channel_byte_size = 2, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 6};
        case daxa::Format::R16G16B16A16_UINT:
        case daxa::Format::R16G16B16A16_SINT:
        case daxa::Format::R16G16B16A16_SFLOAT:  return {.channel_count = 4, .channel_byte_size = 2, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 8};
        // 32-bit
        case daxa::Format::R32_UINT:
        case daxa::Format::R32_SINT:
        case daxa::Format::R32_SFLOAT:           return {.channel_count = 1, .channel_byte_size = 4, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 4};
        case daxa::Format::R32G32_UINT:
        case daxa::Format::R32G32_SINT:
        case daxa::Format::R32G32_SFLOAT:        return {.channel_count = 2, .channel_byte_size = 4, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 8};
        case daxa::Format::R32G32B32A32_UINT:
        case daxa::Format::R32G32B32A32_SINT:
        case daxa::Format::R32G32B32A32_SFLOAT:  return {.channel_count = 4, .channel_byte_size = 4, .is_srgb = false, .block_width = 1, .block_height = 1, .block_byte_size = 16};
        // Block compressed (8 bytes / 4x4 block)
        case daxa::Format::BC1_RGB_UNORM_BLOCK:
        case daxa::Format::BC1_RGB_SRGB_BLOCK:
        case daxa::Format::BC1_RGBA_UNORM_BLOCK:
        case daxa::Format::BC1_RGBA_SRGB_BLOCK:
        case daxa::Format::BC4_UNORM_BLOCK:
        case daxa::Format::BC4_SNORM_BLOCK:      return {.channel_count = 0, .channel_byte_size = 0, .is_srgb = false, .block_width = 4, .block_height = 4, .block_byte_size = 8};
        // Block compressed (16 bytes / 4x4 block)
        case daxa::Format::BC2_UNORM_BLOCK:
        case daxa::Format::BC2_SRGB_BLOCK:
        case daxa::Format::BC3_UNORM_BLOCK:
        case daxa::Format::BC3_SRGB_BLOCK:
        case daxa::Format::BC5_UNORM_BLOCK:
        case daxa::Format::BC5_SNORM_BLOCK:
        case daxa::Format::BC6H_UFLOAT_BLOCK:
        case daxa::Format::BC6H_SFLOAT_BLOCK:
        case daxa::Format::BC7_UNORM_BLOCK:
        case daxa::Format::BC7_SRGB_BLOCK:       return {.channel_count = 0, .channel_byte_size = 0, .is_srgb = false, .block_width = 4, .block_height = 4, .block_byte_size = 16};
        default:
            DBG_ASSERT_TRUE_M(false, "get_format_info: unhandled format");
            return {};
    }
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
    FormatInfo const block = get_format_info(descriptor.info.format);
    DBG_ASSERT_TRUE_M(block.block_byte_size != 0, "write_tido_image: unsupported texture format");

    std::error_code create_destination_folder_error = {};
    std::filesystem::create_directories(info.destination_folder, create_destination_folder_error);

    std::string const stem = tido_stem(info.name, info.metadata_hash.cache_key);
    std::filesystem::path const tido_path = info.destination_folder / (stem + ".tido_bin");


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
            dst_descriptor.subresources[dst_subresource_index] = {.offset = data_payload.size(), .byte_size = s_cast<u32>(calculated_size)};
            data_payload.insert(data_payload.end(), info.data.begin() + entry.offset, info.data.begin() + entry.offset + entry.byte_size);
        }
    }

    std::vector<std::byte> header_payload = {};
    header_payload.resize(TIDO_FILE_PREAMBLE_SIZE);

    std::string const serialized_tido_metadata_hash = serialize_tido_metadata_hash(info.metadata_hash);
    std::string const serialized_tido_image_descriptor = serialize_tido_image_descriptor(dst_descriptor);
    std::string const separator = "\n";
    append_string_bytes(header_payload, serialized_tido_metadata_hash);
    append_string_bytes(header_payload, separator);
    append_string_bytes(header_payload, serialized_tido_image_descriptor);
    append_string_bytes(header_payload, separator);

    // A fixed preamble precedes the json metadata headers, which precede the raw image data.
    u64 const file_data_offset = header_payload.size();
    TidoFilePreamble preamble = {};
    preamble.header_byte_length = header_payload.size() - TIDO_FILE_PREAMBLE_SIZE;
    std::copy(r_cast<std::byte *>(&preamble), r_cast<std::byte *>(&preamble) + TIDO_FILE_PREAMBLE_SIZE, header_payload.begin());

    // TODO(saky): still one copy of data_payload here; a scatter/gather write_file_exclusive would avoid it.
    header_payload.reserve(header_payload.size() + data_payload.size());
    header_payload.insert(header_payload.end(), data_payload.begin(), data_payload.end());

    if (!write_file_exclusive_retry(tido_path, header_payload.data(), header_payload.size())) { return std::nullopt; }

    ImageStreamerData result = {};
    result.descriptor = dst_descriptor;
    result.bin_source = tido_path;
    result.file_data_offset = file_data_offset;
    return result;
}

auto write_tido_mesh(WriteTidoFileInfo const & info, TidoMeshDescriptor const & descriptor) -> std::optional<MeshStreamerData>
{
    std::error_code create_destination_folder_error = {};
    std::filesystem::create_directories(info.destination_folder, create_destination_folder_error);

    DBG_ASSERT_TRUE_M(
        !create_destination_folder_error || create_destination_folder_error == std::errc::file_exists,
        "write_tido_mesh: failed to create destination folder");

    std::string const stem = tido_stem(info.name, info.metadata_hash.cache_key);
    std::filesystem::path const tido_path = info.destination_folder / (stem + ".tido_bin");

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

    std::vector<std::byte> header_payload = {};
    header_payload.resize(TIDO_FILE_PREAMBLE_SIZE);

    std::string const serialized_tido_metadata_hash = serialize_tido_metadata_hash(info.metadata_hash);
    std::string const serialized_tido_mesh_descriptor = serialize_tido_mesh_descriptor(dst_descriptor);

    std::string const separator = "\n";
    append_string_bytes(header_payload, serialized_tido_metadata_hash);
    append_string_bytes(header_payload, separator);
    append_string_bytes(header_payload, serialized_tido_mesh_descriptor);
    append_string_bytes(header_payload, separator);

    // A fixed preamble precedes the json metadata headers, which precede the raw mesh data.
    u64 const file_data_offset = header_payload.size();
    TidoFilePreamble preamble = {};
    preamble.header_byte_length = header_payload.size() - TIDO_FILE_PREAMBLE_SIZE;
    std::copy(r_cast<std::byte *>(&preamble), r_cast<std::byte *>(&preamble) + TIDO_FILE_PREAMBLE_SIZE, header_payload.begin());

    // TODO(saky): still one copy of data_payload here; a scatter/gather write_file_exclusive would avoid it.
    header_payload.reserve(header_payload.size() + data_payload.size());
    header_payload.insert(header_payload.end(), data_payload.begin(), data_payload.end());

    if (!write_file_exclusive_retry(tido_path, header_payload.data(), header_payload.size())) { return std::nullopt; }

    MeshStreamerData result = {};
    result.descriptor = dst_descriptor;
    result.bin_source = tido_path;
    result.file_data_offset = file_data_offset;
    return result;
}

auto tido_extract_and_validate_header_preamble(std::span<std::byte const> file_data) -> std::optional<TidoFilePreamble>
{
    if (file_data.size() < sizeof(TidoFilePreamble)) { return std::nullopt; }
    TidoFilePreamble preamble = {};
    std::memcpy(&preamble, file_data.data(), sizeof(TidoFilePreamble));
    TidoFilePreamble const expected = {};
    if (preamble.magic != expected.magic || preamble.version != TidoFilePreamble::CURRENT_VERSION) { return std::nullopt; }

    return preamble;
}