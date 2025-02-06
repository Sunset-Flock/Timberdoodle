#pragma once

#include "../timberdoodle.hpp"

using namespace tido::types;

struct SpatialEntry
{
    u16 index;
    u32 cell_key;
};

struct SpatialGrid
{
    SpatialGrid(std::vector<f32vec3> const & positions, f32 const cell_size);

    auto get_neighbor_candidate_indices(f32vec3 const & position, f32 const radius) const -> std::vector<u16>;

    private:
        static constexpr u32 HASH_KEY_1 = 15823;
        static constexpr u32 HASH_KEY_2 = 9737333;
        static constexpr u32 HASH_KEY_3 = 440817757;

        f32 cell_size;

        std::vector<SpatialEntry> spatial_lookup;
        std::vector<u32> cell_start_indices;

        auto position_to_cell_coordinates(f32vec3 const & position) const -> i32vec3;
        auto hash_cell_coordinates(i32vec3 const & coordinates) const -> u32;
        auto get_key_from_hash(u32 const hash) const -> u32;
};