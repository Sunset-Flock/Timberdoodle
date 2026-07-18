#pragma once

#include <filesystem>
#include <cstddef>
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