#pragma once

#include <filesystem>
#include <cstddef>
#include <vector>
#include <optional>

#include <daxa/daxa.hpp>

#include "../../timberdoodle.hpp"
using namespace tido::types;

struct ProcessedImage;

struct TidoTextureDescriptor
{
    u32 format = {};

    // Mip 0 dimensions
    u32 width = {};
    u32 height = {};

    u32 depth = {};
    u32 array_layers = {};
    u32 mip_count = {};
};

struct TidoSubresourceEntry
{
    u64 offset = {};    // byte offset from the start of the .tido data file
    u32 byte_size = {}; // byte size of this subresource
};

// True if a descriptor's stored format is the two-channel BC5 normal-map encoding (the shader
// reconstructs Z). Deduced from the cooked format rather than tracked separately through the import.
inline auto tido_format_is_bc5_rg(u32 format) -> bool
{
    daxa::Format const f = std::bit_cast<daxa::Format>(format);
    return f == daxa::Format::BC5_UNORM_BLOCK || f == daxa::Format::BC5_SNORM_BLOCK;
}

// The cooked metadata produced alongside the .tido data file. Persisted into the .tido_cache.
struct TidoTextureStreamerData
{
    std::filesystem::path bin_source = {};
    std::vector<TidoSubresourceEntry> subresources = {};
    TidoTextureDescriptor info = {};
};
struct TidoTextureCookResult
{
    // Texture hash identifyinng the resulting .tido file.
    u64 cache_key = {};
    i64 source_modified = {};
    u64 content_hash = {};

    TidoTextureStreamerData streamer_data = {};
};


inline std::filesystem::path const TIDO_ASSET_CACHE_DIR = "tido_asset_cache";

auto write_texture_tido(ProcessedImage const & processed, std::filesystem::path const & cache_dir, std::string const & name, u64 cache_key) -> std::optional<TidoTextureCookResult>;
