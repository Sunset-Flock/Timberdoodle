#pragma once

#include <optional>
#include <span>
#include <vector>
#include <cstddef>
#include <string>

#include "../../timberdoodle.hpp"
#include "../tido_format/tido_format.hpp"
using namespace tido::types;

struct VdbParseInfo
{
    std::span<std::byte const> src_data = {};
    std::vector<std::string> grid_names = {};
};

// Densify the requested VDB grids from an in-memory .vdb into one interleaved multi-channel 3D volume, in the
// grids' natural fp32 precision. Mirrors image_parse: it applies no recipe format conversion - the cook
// tail's remap_channels handles channel selection and precision conversion to the target format. One channel
// per grid, in the given order. nullopt on failure (grid count outside 1-4, missing grid, a grid with no
// active voxels, invalid extents, VDB loader disabled).
auto vdb_parse(VdbParseInfo const & info) -> std::optional<TidoImageWithData>;
