#pragma once

#include <filesystem>
#include <cstddef>
#include <array>
#include <vector>
#include <optional>

#include <daxa/daxa.hpp>

#include "../../timberdoodle.hpp"
#include "../../shader_shared/shared.inl"
#include "../../shader_shared/geometry.inl"

using namespace tido::types;

struct TidoMetadataHash
{
    u64 cache_key = {};
    i64 source_mtime_at_bake = {};
    u64 content_hash = {};
    u32 version = {};
};

// Fixed-size binary preamble at byte 0 of every .tido_bin, before the JSON header region.
struct TidoFilePreamble
{
    static constexpr u32 CURRENT_VERSION = 1;
    std::array<char, 8> magic = {'T', 'I', 'D', 'O', 'B', 'I', 'N', '\0'};
    u32 version = CURRENT_VERSION;
    u32 _reserved = 0;
    u64 header_byte_length = {}; // byte length of the JSON header region immediately following this preamble
};

constexpr static u32 TIDO_FILE_PREAMBLE_SIZE = sizeof(TidoFilePreamble);
static_assert(TIDO_FILE_PREAMBLE_SIZE == 24, "TidoFilePreamble must stay tightly packed with no padding");

// How a channel's stored bits are interpreted numerically. Distinguishes normalized (UNORM/SNORM, a
// bounded real) from raw integer (UINT/SINT) - a distinction signed/unsigned/float alone can't make, but
// which channel conversion (remap) needs to pick between rescaling and value-preserving.
enum struct FormatNumericType
{
    UNORM,
    SNORM,
    UINT,
    SINT,
    SFLOAT,
};

struct FormatInfo
{
    static constexpr u32 UNSUPPORTED_FORMAT = 0;

    u32 channel_count = {};
    u32 channel_byte_size = {};
    bool is_srgb = {};
    FormatNumericType numeric_type = {};
    u32 block_width = 1;
    u32 block_height = 1;

    // 1x1 block (uncompressed) or 4x4 block (BC compressed) byte size.
    u32 block_byte_size = UNSUPPORTED_FORMAT;
};

auto get_format_info(daxa::Format format) -> FormatInfo;

// ================================= TIDO IMAGE =================================

struct TidoImageDescriptor
{
    struct ImageInfo
    {
        daxa::Format format = {};

        u32 dimensions = {};
        u32vec3 size = {};
        u32 mip_level_count = {};
        u32 array_layer_count = {};
    };

    struct SubresourceEntry
    {
        u64 offset = {};
        u32 byte_size = {};
    };

    ImageInfo info = {};
    std::vector<SubresourceEntry> subresources = {};

    inline auto layer_mip_to_subresource_index(u32 layer, u32 mip) const -> u32
    {
        return layer * info.mip_level_count + mip;
    }
};

// ================================= TIDO MESH =================================

struct TidoMeshDescriptor
{
    struct LodDescriptor
    {
        u64 offset = {};
        u64 byte_size = {};

        AABB aabb = {};
        BoundingSphere bounding_sphere = {};
        f32 lod_error = {};
        u32 vertex_count = {};
        u32 primitive_count = {};
        u32 meshlet_count = {};
        u32 micro_indices_count = {};
        u32 indirect_vertices_count = {};
        u32 primitive_indices_count = {};
        u32 has_uv = {};
    };

    std::vector<LodDescriptor> lods = {};
};

inline std::filesystem::path const TIDO_ASSET_CACHE_DIR = "tido_asset_cache";

struct WriteTidoFileInfo
{
    std::filesystem::path destination_folder = {};
    std::string name = {};
    TidoMetadataHash metadata_hash = {};
    std::span<std::byte const> data = {};
};

struct ImageStreamerData;
struct MeshStreamerData;

auto write_tido_image(WriteTidoFileInfo const & info, TidoImageDescriptor const & descriptor) -> std::optional<ImageStreamerData>;
auto write_tido_mesh(WriteTidoFileInfo const & info, TidoMeshDescriptor const & descriptor) -> std::optional<MeshStreamerData>;

// Validates the fixed preamble at the start of a .tido_bin and returns just the JSON header region that
// follows it. nullopt if the file is too small or the preamble's magic/version does not match.
auto tido_header_region(std::span<std::byte const> file_data) -> std::optional<std::span<std::byte const>>;

// Patch only the metadata-hash header of an existing .tido_bin in place (descriptor + payload untouched).
// The metadata fields are fixed-width, so old_hash and new_hash serialize to the same length and the patch
// can't move the payload offset. Returns false on a write failure, or as a safety net if that length
// invariant is ever broken (lengths differ).
auto try_patch_tido_metadata_hash(std::filesystem::path const & path, TidoMetadataHash const & old_hash, TidoMetadataHash const & new_hash) -> bool;