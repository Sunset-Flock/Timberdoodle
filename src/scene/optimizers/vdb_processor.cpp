#include "vdb_processor.hpp"

#include <algorithm>
#include <limits>
#include <string>

#include <fmt/format.h>

#if TIDO_BUILT_WITH_UTILS_VDB_LOADER
#include <array>
#include <istream>
#include <streambuf>
#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <openvdb/tools/Dense.h>

namespace
{

// Read-only stream over an already-loaded buffer, so feeding OpenVDB the .vdb bytes costs no second copy.
struct LoadedBytesStreamBuf : std::streambuf
{
    explicit LoadedBytesStreamBuf(std::span<std::byte const> data)
    {
        // setg needs a mutable char range; nothing here ever writes through it.
        char * const begin = const_cast<char *>(reinterpret_cast<char const *>(data.data()));
        setg(begin, begin, begin + data.size());
    }

    auto seekoff(off_type offset, std::ios_base::seekdir direction, std::ios_base::openmode mode) -> pos_type override
    {
        if ((mode & std::ios_base::in) == 0) { return pos_type(off_type(-1)); }
        off_type position = offset;
        if (direction == std::ios_base::cur) { position += gptr() - eback(); }
        else if (direction == std::ios_base::end) { position += egptr() - eback(); }
        else if (direction != std::ios_base::beg) { return pos_type(off_type(-1)); }
        if (position < 0 || position > egptr() - eback()) { return pos_type(off_type(-1)); }
        setg(eback(), eback() + position, egptr());
        return pos_type(position);
    }

    auto seekpos(pos_type position, std::ios_base::openmode mode) -> pos_type override
    {
        return seekoff(off_type(position), std::ios_base::beg, mode);
    }
};

} // namespace

auto vdb_parse(VdbParseInfo const & info) -> std::optional<TidoImageWithData>
{
    u32 const channel_count = s_cast<u32>(info.grid_names.size());
    // The recipe is producer data, so a grid count no image channel layout can hold is rejected, not asserted.
    if (channel_count < 1 || channel_count > 4)
    {
        DEBUG_MSG(fmt::format("[ERROR][vdb_parse] only 1-4 grids (channels) are supported, got {}", channel_count));
        return std::nullopt;
    }

    openvdb::initialize();

    // OpenVDB reads only from a std::istream, so wrap the loaded bytes in one. delayLoad=false forces a fully
    // eager read of every grid into memory - no temp-file copy and no seekable-file requirement - which is
    // exactly what densifying every grid needs.
    LoadedBytesStreamBuf source_buffer(info.src_data);
    std::istream stream(&source_buffer);
    openvdb::io::Stream vdb_stream(stream, /*delayLoad=*/false);
    openvdb::GridPtrVecPtr all_grids = vdb_stream.getGrids();

    // Resolve each requested grid name to its FloatGrid, preserving the requested channel order.
    std::vector<openvdb::FloatGrid::Ptr> selected_grids(channel_count);
    for (u32 channel = 0; channel < channel_count; ++channel)
    {
        std::string const & wanted_name = info.grid_names[channel];
        openvdb::FloatGrid::Ptr found = {};
        for (auto const & base_grid : *all_grids)
        {
            if (base_grid->getName() == wanted_name)
            {
                found = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid);
                break;
            }
        }
        if (!found)
        {
            DEBUG_MSG(fmt::format("[ERROR][vdb_parse] .vdb has no FloatGrid named '{}'", wanted_name));
            return std::nullopt;
        }
        selected_grids[channel] = found;
    }

    // Combined active-voxel bounds across all selected grids, in VDB space. Each grid's own bounds are kept
    // for the densify pass - evaluating them walks the whole tree.
    std::vector<openvdb::CoordBBox> grid_bounds_per_channel(channel_count);
    i32vec3 min_extents = i32vec3(std::numeric_limits<i32>::max());
    i32vec3 max_extents = i32vec3(std::numeric_limits<i32>::lowest());
    for (u32 channel = 0; channel < channel_count; ++channel)
    {
        openvdb::FloatGrid::Ptr const & grid = selected_grids[channel];
        openvdb::CoordBBox const active_bounds = grid->evalActiveVoxelBoundingBox();
        // An empty bbox holds min > max, which would wrap the combined extents into a bogus volume size.
        if (active_bounds.empty())
        {
            DEBUG_MSG(fmt::format("[ERROR][vdb_parse] grid '{}' has no active voxels", grid->getName()));
            return std::nullopt;
        }
        grid_bounds_per_channel[channel] = active_bounds;
        auto const grid_min = active_bounds.min().asVec3i();
        auto const grid_max = active_bounds.max().asVec3i();
        min_extents = i32vec3(std::min(min_extents.x, grid_min[0]), std::min(min_extents.y, grid_min[1]), std::min(min_extents.z, grid_min[2]));
        max_extents = i32vec3(std::max(max_extents.x, grid_max[0]), std::max(max_extents.y, grid_max[1]), std::max(max_extents.z, grid_max[2]));
    }

    i32vec3 const vdb_extents = (max_extents - min_extents) + i32vec3(1);
    // VDB has Y up - TIDO has Z up.
    i32vec3 const tido_extents = i32vec3(vdb_extents.x, vdb_extents.z, vdb_extents.y);
    if (tido_extents.x <= 0 || tido_extents.y <= 0 || tido_extents.z <= 0)
    {
        DEBUG_MSG(fmt::format("[ERROR][vdb_parse] invalid grid extents {} {} {}", tido_extents.x, tido_extents.y, tido_extents.z));
        return std::nullopt;
    }

    u32 const width = s_cast<u32>(tido_extents.x);
    u32 const height = s_cast<u32>(tido_extents.y);
    u32 const depth = s_cast<u32>(tido_extents.z);
    u64 const total_texels = s_cast<u64>(width) * height * depth;
    u64 const payload_byte_size = total_texels * channel_count * sizeof(f32);

    TidoImageWithData result = {};
    result.descriptor.info = {
        .format = get_format_from_info(FormatInfo{.channel_count = channel_count, .channel_byte_size = s_cast<u32>(sizeof(f32)), .numeric_type = FormatNumericType::SFLOAT}),
        .dimensions = 3,
        .size = {width, height, depth},
        .mip_level_count = 1,
        .array_layer_count = 1,
    };
    result.descriptor.subresources.push_back({.offset = 0, .byte_size = payload_byte_size});
    result.data.resize(payload_byte_size);

    // Interleaved N-channel fp32 volume: channel c of texel t lives at (t * channel_count + c).
    f32 * const voxels = reinterpret_cast<f32 *>(result.data.data());

    // Prefill every channel with its grid's background so voxels outside that grid's own bounds read a
    // defined value; one sequential pass over the interleaved volume.
    std::array<f32, 4> channel_backgrounds = {};
    for (u32 channel = 0; channel < channel_count; ++channel)
    {
        channel_backgrounds[channel] = selected_grids[channel]->background();
    }
    for (u64 texel = 0; texel < total_texels; ++texel)
    {
        for (u32 channel = 0; channel < channel_count; ++channel)
        {
            voxels[texel * channel_count + channel] = channel_backgrounds[channel];
        }
    }

    // Densify each grid over its own active bounds, then swizzle that block into the grid's channel.
    for (u32 channel = 0; channel < channel_count; ++channel)
    {
        openvdb::CoordBBox const & grid_bounds = grid_bounds_per_channel[channel];
        openvdb::Coord const block_dimensions = grid_bounds.dim();

        // copyToDense overwrites every voxel of its bbox, inactive ones included, so the block needs no prefill.
        openvdb::tools::Dense<f32, openvdb::tools::LayoutXYZ> dense_block(grid_bounds);
        openvdb::tools::copyToDense(*selected_grids[channel], dense_block, /*serial=*/false);
        f32 const * const dense_voxels = dense_block.data();

        // Where the block starts inside the combined volume; never negative, as min_extents is the minimum
        // across every grid.
        i32vec3 const block_origin = i32vec3(grid_bounds.min().x(), grid_bounds.min().y(), grid_bounds.min().z()) - min_extents;

        // The dense block is x-major and TIDO's x stride is one texel, so both sides run x innermost and the
        // destination is written sequentially.
        for (i32 block_y = 0; block_y < block_dimensions.y(); ++block_y)
        {
            for (i32 block_z = 0; block_z < block_dimensions.z(); ++block_z)
            {
                u64 const source_row = s_cast<u64>(block_y) * block_dimensions.x() +
                                       s_cast<u64>(block_z) * block_dimensions.x() * block_dimensions.y();
                // z-major TIDO layout with the VDB Y-up -> Z-up axis swap folded into the strides.
                u64 const destination_row = (s_cast<u64>(block_origin.x) +
                                             s_cast<u64>(block_origin.z + block_z) * width +
                                             s_cast<u64>(block_origin.y + block_y) * width * height) * channel_count + channel;
                for (i32 block_x = 0; block_x < block_dimensions.x(); ++block_x)
                {
                    voxels[destination_row + s_cast<u64>(block_x) * channel_count] = dense_voxels[source_row + s_cast<u64>(block_x)];
                }
            }
        }
    }

    return result;
}

#else // TIDO_BUILT_WITH_UTILS_VDB_LOADER

auto vdb_parse(VdbParseInfo const &) -> std::optional<TidoImageWithData>
{
    DEBUG_MSG("[ERROR][vdb_parse] VDB loading utility not enabled during build");
    return std::nullopt;
}

#endif // TIDO_BUILT_WITH_UTILS_VDB_LOADER
