#include "openvdb_loader.hpp"

#include <openvdb/openvdb.h>

daxa::ImageId load_vdb(std::filesystem::path const & path, daxa::Device & device, bool normalize_sdf, bool remap_channels)
{
    openvdb::initialize();

    openvdb::io::File file(path.string());
    file.open();

    f32vec4 bcg_value = f32vec4(0.0f, 0.0f, 1.0f, 0.01f);
    
    int mapping[4] = {0, 1, 2, 3};

    if(remap_channels) {
        mapping[0] = 2;
        mapping[1] = 1;
        mapping[2] = 0;
        mapping[3] = 3;
    }

    i32vec3 min_grid_extents = i32vec3( std::numeric_limits<i32>::infinity());
    i32vec3 max_grid_extents = i32vec3(-std::numeric_limits<i32>::infinity());

    int i = 0;
    for (openvdb::io::File::NameIterator name_iterator = file.beginName(); name_iterator != file.endName(); ++name_iterator, ++i)
    {
        openvdb::GridBase::Ptr base_grid = file.readGrid(name_iterator.gridName());
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

        openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid);
        bcg_value[mapping[i]] = grid->background();
        grid->print();
    }

    i32vec3 const grid_extents = i32vec3(max_grid_extents - min_grid_extents);
    DBG_ASSERT_TRUE_M(grid_extents.x > 0 && grid_extents.y > 0 && grid_extents.z > 0, "Extents must be positive");

    auto const image_size = u32vec3(s_cast<u32>(grid_extents[0]) + 1u, s_cast<u32>(grid_extents[2]) + 1u, s_cast<u32>(grid_extents[1]) + 1);
    daxa::ImageId cloud_data_image = device.create_image({
        .flags = daxa::ImageCreateFlagBits::COMPATIBLE_2D_ARRAY,
        .dimensions = 3,
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size =  {image_size.x, image_size.y, image_size.z},
        .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::HOST_TRANSFER,
        // Ignored when allocating with a memory block.
        .allocate_info = {},
        .name = {"Clouds voxel image"},
    });

    std::vector<f32vec4> raw_data = {};
    raw_data.resize(image_size[0] * image_size[1] * image_size[2]);
    std::fill(raw_data.begin(), raw_data.end(), bcg_value);

    {
        int j = 0;
        for (openvdb::io::File::NameIterator name_iterator = file.beginName(); name_iterator != file.endName(); ++name_iterator, ++j)
        {
            openvdb::GridBase::Ptr base_grid = file.readGrid(name_iterator.gridName());
            openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid);
            openvdb::tree::ValueAccessor<const openvdb::FloatTree> accessor = grid->getConstAccessor();

            auto const curr_grid_min_extents = base_grid->evalActiveVoxelBoundingBox().min().asVec3i();
            auto const curr_grid_max_extents = base_grid->evalActiveVoxelBoundingBox().max().asVec3i();

            for (int z = curr_grid_min_extents[2]; z <= curr_grid_max_extents[2]; ++z)
            {
                for (int y = curr_grid_min_extents[1]; y <= curr_grid_max_extents[1]; ++y)
                {
                    for (int x = curr_grid_min_extents[0]; x <= curr_grid_max_extents[0]; ++x)
                    {
                        const i32vec3 zero_based_coord = i32vec3(x, y, z) - i32vec3(min_grid_extents[0], min_grid_extents[1], min_grid_extents[2]);
                        openvdb::Coord coord(x, y, z);
                        const i32 index = zero_based_coord[0] + (zero_based_coord[2] * image_size[0]) + (zero_based_coord[1] * image_size[0] * image_size[1]);
                        const float value = accessor.getValue(coord);
                        raw_data.at(index)[mapping[j]] = value;
                    }
                }
            }
        }
    }

    // Normalize the SDF values so that they are in the range of [-0.0625; 1.0] instead of the original range [-256;4096]
    if(normalize_sdf)
    {
        for (auto & field_value : raw_data)
        {
            field_value.a = field_value.a / 4096.0f;
        }
    }

    file.close();

    device.transition_image_layout({
        .image = cloud_data_image,
        .new_image_layout = daxa::ImageLayout::READ_ONLY_OPTIMAL,
        .image_slice = {
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        }
    });

    daxa::ImageInfo image_info = device.image_info(cloud_data_image).value();
    device.copy_memory_to_image({
        .memory_ptr = r_cast<std::byte const*>(raw_data.data()),
        .image = cloud_data_image,
        .image_layout = daxa::ImageLayout::READ_ONLY_OPTIMAL,
        .image_offset = {0, 0, 0},
        .image_extent = {image_info.size.x, image_info.size.y, image_info.size.z}
    });
    return cloud_data_image;
}
