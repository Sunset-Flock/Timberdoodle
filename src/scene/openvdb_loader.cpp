#include "openvdb_loader.hpp"

#include <fstream>
#if TIDO_BUILT_WITH_UTILS_VDB_LOADER
#include <openvdb/openvdb.h>

struct LoadVDBTaskImpl
{
    std::optional<openvdb::io::File> file;
    std::vector<openvdb::FloatGrid::Ptr> grids;
};

LoadVDBTask::LoadVDBTask(LoadVDBTaskInfo const & info)
    :  info{info},
       impl{new LoadVDBTaskImpl}
{
    chunk_count = 1;
}

LoadVDBTask::~LoadVDBTask()
{
    delete impl;
}

auto LoadVDBTask::initialize() -> bool
{
    openvdb::initialize();

    impl->file = openvdb::io::File(info.vdb_path.string());
    impl->file->open();

    // Validate that all requested grids exist.
    for(auto const & grid : info.grids_to_load)
    {
        if(!impl->file->hasGrid(grid.name))
        {
            result = false;
            error_message = fmt::format("VDB file {} does not have grid named {}\n", info.vdb_path.string(), grid.name);
            return false;
        }
    }

    min_grid_extents = i32vec3( std::numeric_limits<i32>::infinity());
    max_grid_extents = i32vec3(-std::numeric_limits<i32>::infinity());

    // Once we know that that all requested grids exist we can find their extents.
    for(auto const & grid : info.grids_to_load)
    {
        openvdb::GridBase::Ptr base_grid = impl->file->readGrid(grid.name);

        impl->grids.push_back(openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid));

        auto const curr_grid_min_extents = base_grid->evalActiveVoxelBoundingBox().min().asVec3i();
        auto const curr_grid_max_extents = base_grid->evalActiveVoxelBoundingBox().max().asVec3i();

        min_grid_extents = f32vec3(
            std::min(min_grid_extents[0], curr_grid_min_extents[0]),
            std::min(min_grid_extents[1], curr_grid_min_extents[1]),
            std::min(min_grid_extents[2], curr_grid_min_extents[2])
        );
        max_grid_extents = f32vec3(
            std::max(max_grid_extents[0], curr_grid_max_extents[0]),
            std::max(max_grid_extents[1], curr_grid_max_extents[1]),
            std::max(max_grid_extents[2], curr_grid_max_extents[2])
        );
    }

    grid_extents = i32vec3(max_grid_extents - min_grid_extents) + i32vec3(1);
    // VDB has Y up - TIDO has Z up.
    grid_extents = i32vec3(grid_extents.x, grid_extents.z, grid_extents.y);
    if(grid_extents.x < 0 || grid_extents.y < 0 || grid_extents.z < 0) 
    {
        result = false;
        error_message = fmt::format(
            "Invalid grid extents calculated from VDB file {} - \n min {} {} {} max {} {} {} results in {} {} {}\n",
            info.vdb_path.string(),
            min_grid_extents.x, min_grid_extents.y, min_grid_extents.z,
            max_grid_extents.x, max_grid_extents.y, max_grid_extents.z,
            grid_extents.x, grid_extents.y, grid_extents.z
        );
        return false;
    }

    grids_data = std::vector(info.grids_to_load.size(), std::vector<std::byte>{});

    for(u32 grid_to_load_index = 0; grid_to_load_index < info.grids_to_load.size(); ++grid_to_load_index)
    {
        auto const grid_user_data = info.grids_to_load[grid_to_load_index];
        u64 const element_size = grid_user_data.convert_to_fp16 ? sizeof(u16) : sizeof(f32);
        u64 const total_byte_size = grid_extents.x * grid_extents.y * grid_extents.z * element_size;

        auto & grid_data = grids_data[grid_to_load_index];
        grid_data.resize(total_byte_size);

        if (grid_user_data.convert_to_fp16)
        {
            u16 const bcg_value = glm::detail::toFloat16(grid_user_data.background_value.value_or(impl->grids.at(grid_to_load_index)->background()));
            std::span<u16> grid_data_as_halves = std::span<u16>(reinterpret_cast<u16*>(grid_data.data()), grid_data.size() / sizeof(u16));
            std::fill(grid_data_as_halves.begin(), grid_data_as_halves.end(), bcg_value);
        }
        else 
        {
            float const bcg_value = grid_user_data.background_value.value_or(impl->grids.at(grid_to_load_index)->background());
            std::span<float> grid_data_as_floats = std::span<float>(reinterpret_cast<float*>(grid_data.data()), grid_data.size() / sizeof(float));
            std::fill(grid_data_as_floats.begin(), grid_data_as_floats.end(), bcg_value);
        }
    }

    chunk_count = s_cast<u32>(info.grids_to_load.size() * grid_extents.z);
    initialized = true;

    return true;
}

void LoadVDBTask::callback(u32 chunk_index, [[maybe_unused]] u32 thread_index)
{
    if(!initialized)
    {
        result = false;
        error_message = "Task not initialized or initalization failed!";
        return;
    }

    u32 grid_to_load_index = chunk_index / grid_extents.z;
    u32 y_slice_to_load_index = chunk_index % grid_extents.z;

    auto const grid_user_data = info.grids_to_load[grid_to_load_index];
    auto & grid_data = grids_data[grid_to_load_index];
    u64 const element_size = grid_user_data.convert_to_fp16 ? sizeof(u16) : sizeof(f32);

    auto const grid = impl->grids[grid_to_load_index];

    openvdb::tree::ValueAccessor<const openvdb::FloatTree> accessor = grid->getConstAccessor();

    auto const curr_grid_min_extents = grid->evalActiveVoxelBoundingBox().min().asVec3i();
    auto const curr_grid_max_extents = grid->evalActiveVoxelBoundingBox().max().asVec3i();

    const int y = min_grid_extents[1] + y_slice_to_load_index;
    if(y > curr_grid_max_extents[1])
    {
        // This slice is outside the active voxel bounds of the current grid.
        // We can skip it since we already filled the output with the background value.
        return;
    }

    for (int z = curr_grid_min_extents[2]; z <= curr_grid_max_extents[2]; ++z)
    {
        for (int x = curr_grid_min_extents[0]; x <= curr_grid_max_extents[0]; ++x)
        {
            const i32vec3 zero_based_coord = i32vec3(x, y, z) - i32vec3(min_grid_extents[0], min_grid_extents[1], min_grid_extents[2]);
            const openvdb::Coord coord(x, y, z);

            const u64 index = (zero_based_coord[0] + (zero_based_coord[2] * grid_extents[0]) + (zero_based_coord[1] * grid_extents[0] * grid_extents[1])) * element_size;

            const float value = accessor.getValue(coord);
            const float clamped_value = std::clamp(value, grid_user_data.value_range.x, grid_user_data.value_range.y);
            if (grid_user_data.convert_to_fp16)
            {
                u16 const clamped_value_fp16 = glm::detail::toFloat16(clamped_value);
                std::memcpy(&grid_data[index], &clamped_value_fp16, sizeof(u16));
            }
            else
            {
                std::memcpy(&grid_data[index], &clamped_value, sizeof(f32));
            }
        }
    }
    result = true;
}

auto read_vdb_header(std::filesystem::path const & vdb_path) -> std::vector<VDBGridMetaData>
{
    openvdb::initialize();

    if(!std::filesystem::exists(vdb_path))
    {
        fmt::print("VDB file {} does not exist!\n", vdb_path.string());
        return {};
    }

    openvdb::io::File file(vdb_path.string());
    file.open();

    std::vector<VDBGridMetaData> grids_metadata;

    for (openvdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter)
    {
        openvdb::GridBase::Ptr base_grid = file.readGrid(nameIter.gridName());

        auto const grid_min_extents = base_grid->evalActiveVoxelBoundingBox().min().asVec3i();
        auto const grid_max_extents = base_grid->evalActiveVoxelBoundingBox().max().asVec3i();

        grids_metadata.push_back(VDBGridMetaData{
            .name = nameIter.gridName(),
            .min_extents = i32vec3(grid_min_extents[0], grid_min_extents[1], grid_min_extents[2]),
            .max_extents = i32vec3(grid_max_extents[0], grid_max_extents[1], grid_max_extents[2])
        });
    }

    return grids_metadata;
}

#else // #if TIDO_BUILT_WITH_UTILS_VDB_LOADER

struct LoadVDBTaskImpl { };

LoadVDBTask::LoadVDBTask(LoadVDBTaskInfo const & info)
    :  info{info},
       impl{new LoadVDBTaskImpl}
{
       chunk_count = 1;
}

LoadVDBTask::~LoadVDBTask() {};

bool LoadVDBTask::initialize()
{
    result = false;
    error_message = "VDB loading utility not enabled during build!";
    return false;
}

void LoadVDBTask::callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index)
{
    result = false;
    error_message = "VDB loading utility not enabled during build!";
}

auto read_vdb_header(std::filesystem::path const & vdb_path) -> std::vector<VDBGridMetaData>
{
    return {};
}

#endif // #if TIDO_BUILT_WITH_UTILS_VDB_LOADER