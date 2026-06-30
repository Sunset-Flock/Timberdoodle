#include "tido_texture.hpp"

#include <fstream>
#include <algorithm>
#include <cctype>

#include "../optimizers/image_optimizer.hpp" // full CookedImageData (forward-declared in the header)

namespace
{
// Block dimensions + byte size for the formats the image cook can produce. Uncompressed formats are
// 1x1 blocks of bytes-per-texel; BCn formats are 4x4 blocks of 8 or 16 bytes.
struct FormatBlockInfo
{
    u32 block_width = 1;
    u32 block_height = 1;
    u32 bytes_per_block = 0; // 0 => unsupported / unknown
};

auto format_block_info(daxa::Format format) -> FormatBlockInfo
{
    switch (format)
    {
        // 8-bit
        case daxa::Format::R8_UNORM:
        case daxa::Format::R8_SRGB:
        case daxa::Format::R8_SINT:           return {1, 1, 1};
        case daxa::Format::R8G8_UNORM:
        case daxa::Format::R8G8_SRGB:
        case daxa::Format::R8G8_SINT:         return {1, 1, 2};
        case daxa::Format::R8G8B8A8_UNORM:
        case daxa::Format::R8G8B8A8_SRGB:
        case daxa::Format::R8G8B8A8_SINT:     return {1, 1, 4};
        // 16-bit
        case daxa::Format::R16_UINT:
        case daxa::Format::R16_SINT:
        case daxa::Format::R16_SFLOAT:        return {1, 1, 2};
        case daxa::Format::R16G16_UINT:
        case daxa::Format::R16G16_SINT:
        case daxa::Format::R16G16_SFLOAT:     return {1, 1, 4};
        case daxa::Format::R16G16B16A16_UINT:
        case daxa::Format::R16G16B16A16_SINT:
        case daxa::Format::R16G16B16A16_SFLOAT: return {1, 1, 8};
        // 32-bit
        case daxa::Format::R32_UINT:
        case daxa::Format::R32_SINT:
        case daxa::Format::R32_SFLOAT:        return {1, 1, 4};
        case daxa::Format::R32G32_UINT:
        case daxa::Format::R32G32_SINT:
        case daxa::Format::R32G32_SFLOAT:     return {1, 1, 8};
        case daxa::Format::R32G32B32A32_UINT:
        case daxa::Format::R32G32B32A32_SINT:
        case daxa::Format::R32G32B32A32_SFLOAT: return {1, 1, 16};
        // Block compressed (8 bytes / 4x4 block)
        case daxa::Format::BC1_RGB_UNORM_BLOCK:
        case daxa::Format::BC1_RGB_SRGB_BLOCK:
        case daxa::Format::BC1_RGBA_UNORM_BLOCK:
        case daxa::Format::BC1_RGBA_SRGB_BLOCK:
        case daxa::Format::BC4_UNORM_BLOCK:
        case daxa::Format::BC4_SNORM_BLOCK:   return {4, 4, 8};
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
        case daxa::Format::BC7_SRGB_BLOCK:    return {4, 4, 16};
        default:                              return {1, 1, 0};
    }
}

// Byte size of one (mip, single layer) subresource at the given mip 0 extents.
auto subresource_byte_size(FormatBlockInfo const & block, u32 width, u32 height, u32 depth, u32 mip) -> u64
{
    u32 const mip_w = std::max(1u, width >> mip);
    u32 const mip_h = std::max(1u, height >> mip);
    u32 const mip_d = std::max(1u, depth >> mip);
    u64 const blocks_x = (mip_w + block.block_width - 1) / block.block_width;
    u64 const blocks_y = (mip_h + block.block_height - 1) / block.block_height;
    return blocks_x * blocks_y * mip_d * block.bytes_per_block;
}

// Turn an arbitrary asset name into a safe file stem (gltf image names can be empty or contain
// characters that are not valid in a path).
auto sanitize_stem(std::string const & name) -> std::string
{
    std::string out;
    out.reserve(name.size());
    for (char const c : name)
    {
        bool const ok = std::isalnum(s_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.';
        out.push_back(ok ? c : '_');
    }
    // Drop a trailing extension (e.g. ".png") so the stem is clean.
    auto const dot = out.find_last_of('.');
    if (dot != std::string::npos) { out.erase(dot); }
    if (out.empty()) { out = "unnamed"; }
    return out;
}

// 64-bit FNV-1a over the cooked bytes. Disambiguates same-named (or unnamed) images and makes the
// write idempotent for identical content.
auto content_hash(std::span<std::byte const> data) -> u64
{
    u64 hash = 0xcbf29ce484222325ull;
    for (std::byte const b : data)
    {
        hash ^= s_cast<u64>(s_cast<u8>(b));
        hash *= 0x100000001b3ull;
    }
    return hash;
}
} // namespace

auto write_texture_tido(CookedImageData const & cooked, std::filesystem::path const & cache_dir, std::string const & name) -> std::optional<TidoTextureCookResult>
{
    auto const & image_info = cooked.image_info;
    u32 const mip_count = cooked.mips_to_copy;
    u32 const array_layers = image_info.array_layer_count;
    u32 const width = image_info.size.x;
    u32 const height = image_info.size.y;
    u32 const depth = image_info.size.z;

    // The cook currently only fills layer-0 data (mip_copy_offsets is per-mip, layer 0). Multi-layer /
    // cubemap textures are not produced yet; assert so this is revisited when they are.
    DBG_ASSERT_TRUE_M(array_layers == 1, "write_texture_tido: only single-layer textures are supported for now");

    FormatBlockInfo const block = format_block_info(image_info.format);
    DBG_ASSERT_TRUE_M(block.bytes_per_block != 0, "write_texture_tido: unsupported texture format");

    std::error_code ec = {};
    std::filesystem::create_directories(cache_dir, ec); // ignore "already exists"; the open below reports real failures

    std::string const stem = fmt::format("{}_{:016x}", sanitize_stem(name), content_hash(cooked.src_data));
    std::filesystem::path const tido_path = cache_dir / (stem + ".tido");

    // Build the .tido payload mip-major, coarse-first; all (single) layers of a mip are contiguous.
    // The subresource table is in this same physical order, so the entry for a (mip, layer) lives at
    // ((mip_count - 1 - mip) * array_layers + layer) - the mip flip maps the coarsest mip to block 0.
    std::vector<TidoSubresourceEntry> subresources(s_cast<usize>(array_layers) * mip_count);
    std::vector<std::byte> payload = {};
    payload.reserve(cooked.src_data.size());

    for (i64 mip = s_cast<i64>(mip_count) - 1; mip >= 0; --mip)
    {
        for (u32 layer = 0; layer < array_layers; ++layer)
        {
            u64 const size = subresource_byte_size(block, width, height, depth, s_cast<u32>(mip));
            u64 const src_offset = cooked.mip_copy_offsets[mip];
            DBG_ASSERT_TRUE_M(src_offset + size <= cooked.src_data.size(), "write_texture_tido: subresource out of source bounds");

            u64 const dst_offset = payload.size();
            payload.insert(payload.end(), cooked.src_data.begin() + src_offset, cooked.src_data.begin() + src_offset + size);

            u32 const subresource_index = (mip_count - 1u - s_cast<u32>(mip)) * array_layers + layer;
            subresources[subresource_index] = {.offset = dst_offset, .byte_size = s_cast<u32>(size)};
        }
    }

    std::ofstream ofs{tido_path, std::ios::binary | std::ios::trunc};
    if (!ofs) { return std::nullopt; }
    ofs.write(r_cast<char const *>(payload.data()), s_cast<std::streamsize>(payload.size()));
    if (!ofs.good()) { return std::nullopt; }

    TidoTextureCookResult result = {};
    result.descriptor = {
        .format = s_cast<u32>(image_info.format),
        .width = width,
        .height = height,
        .depth = depth,
        .array_layers = array_layers,
        .mip_count = mip_count,
    };
    result.subresources = std::move(subresources);
    result.tido_path = tido_path;
    return result;
}
