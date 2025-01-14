#include "spatial_grid.hpp"

#include <algorithm>

auto SpatialGrid::position_to_cell_coordinates(f32vec3 const & position) const -> i32vec3
{
    return s_cast<i32vec3>(position / f32vec3(cell_size));
}

auto SpatialGrid::hash_cell_coordinates(i32vec3 const & coordinates) const -> u32
{
    return s_cast<u32>(coordinates.x * HASH_KEY_1 + coordinates.y * HASH_KEY_2 + coordinates.z * HASH_KEY_3);
}

auto SpatialGrid::get_key_from_hash(u32 const hash) const -> u32
{
    return hash % spatial_lookup.size();
}

SpatialGrid::SpatialGrid(std::vector<f64vec3> const & positions, f32 const cell_size) :
    cell_size{cell_size}
{
    DBG_ASSERT_TRUE_M(positions.size() < std::numeric_limits<u16>::max(), 
        "Currently only using 16 bit uints to represent index");

    spatial_lookup.resize(positions.size());
    for(i32 position_index = 0; position_index < positions.size(); ++position_index)
    {
        auto const & position = positions.at(position_index);

        u32vec3 const cell_coordinates = position_to_cell_coordinates(position);
        u32 const cell_key = get_key_from_hash(hash_cell_coordinates(cell_coordinates));
        spatial_lookup.at(position_index) = SpatialEntry{
            .index = static_cast<u16>(position_index),
            .cell_key = cell_key
        };
    }

    std::sort(spatial_lookup.begin(), spatial_lookup.end(), 
        [](SpatialEntry const & first, SpatialEntry const & second) -> bool
        {
            return first.cell_key < second.cell_key;
        }
    );

    cell_start_indices.resize(positions.size());
    std::fill(cell_start_indices.begin(), cell_start_indices.end(), std::numeric_limits<u32>::max());

    for(i32 spatial_lookup_index = 0; spatial_lookup_index < spatial_lookup.size(); ++spatial_lookup_index)
    {
        u32 const current_key = spatial_lookup.at(spatial_lookup_index).cell_key;
        u32 const previous_key = spatial_lookup_index == 0 ?
                std::numeric_limits<u32>::max() :
                spatial_lookup.at(spatial_lookup_index - 1).cell_key;

        if(current_key != previous_key)
        {
            cell_start_indices.at(current_key) = spatial_lookup_index; 
        }
    }
}

auto SpatialGrid::get_neighbor_candidate_indices(f64vec3 const & position, f32 const radius) const -> std::vector<u16>
{
    std::vector<u16> neighbor_candidate = {};
    i32 const search_radius_in_cells = s_cast<i32>(std::ceil(radius / cell_size));

    i32vec3 const start_cell_coordinates = position_to_cell_coordinates(position);

    for(i32 cell_x_offset = -search_radius_in_cells; cell_x_offset <= search_radius_in_cells; ++cell_x_offset)
    {
        for(i32 cell_y_offset = -search_radius_in_cells; cell_y_offset <= search_radius_in_cells; ++cell_y_offset)
        {
            for(i32 cell_z_offset = -search_radius_in_cells; cell_z_offset <= search_radius_in_cells; ++cell_z_offset)
            {
                i32vec3 const cell_offset = i32vec3(cell_x_offset, cell_y_offset, cell_z_offset);
                i32vec3 const offset_cell_coordinates = start_cell_coordinates + cell_offset;

                u32 const key = get_key_from_hash(hash_cell_coordinates(offset_cell_coordinates));
                u32 const cell_start_index = cell_start_indices.at(key);

                for (i32 index = cell_start_index; index < spatial_lookup.size(); ++index)
                {
                    if(spatial_lookup.at(index).cell_key != key) { break; }

                    neighbor_candidate.push_back(spatial_lookup.at(index).index);
                }
            }
        }
    }

    return neighbor_candidate;
}
