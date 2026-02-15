#pragma once
#include <filesystem>

#include "../timberdoodle.hpp"
#include "../multithreading/thread_pool.hpp"
using namespace tido::types;

struct VDBGridInfo
{
    std::string name;
    // Valud to which the grid will be prefilled outside active voxels.
    std::optional<f32> background_value;
    // Each value in the grid will be clamped to this range during conversion.
    f32vec2 value_range = f32vec2(std::numeric_limits<f32>::lowest(), std::numeric_limits<f32>::max()) ;
    bool convert_to_fp16;
};
struct LoadVDBTaskInfo
{
    std::filesystem::path vdb_path;

    std::vector<VDBGridInfo> const grids_to_load;
};

struct LoadVDBTaskImpl;

struct LoadVDBTask : Task
{
    // ================== Inputs =====================
    LoadVDBTaskInfo info;

    // ================== Outputs ====================
    bool result;
    std::string error_message;
    std::vector<std::vector<std::byte>> grids_data;
    i32vec3 grid_extents;

private:
    bool initialized = {};

    i32vec3 min_grid_extents = {};
    i32vec3 max_grid_extents = {};
    LoadVDBTaskImpl * impl;

public:
    LoadVDBTask(LoadVDBTaskInfo const & info);
    ~LoadVDBTask();

    // Load the VDB header without reading the contents, setup the chunk count etc...
    // If initalization fails return false.
    bool initialize();
    virtual void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override;
};

auto load_vdb(LoadVDBTaskInfo const & info, ThreadPool * threadpool) -> std::shared_ptr<LoadVDBTask>;

struct VDBGridMetaData
{
    std::string name;
    i32vec3 min_extents;
    i32vec3 max_extents;
};

auto read_vdb_header(std::filesystem::path const & vdb_path) -> std::vector<VDBGridMetaData>;