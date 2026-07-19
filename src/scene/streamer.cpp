#include "streamer.hpp"

#include <cstring>
#include <utility>
#include <vector>

#include "../io/file_io.hpp"

void ImageStreamTask::callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index)
{
    TidoImageDescriptor const & desc = artifact.descriptor;

    std::pair<FileIoResult, std::vector<std::byte>> file_data_result = read_file_shared(artifact.bin_source);
    DBG_ASSERT_TRUE_M(file_data_result.first == FileIoResult::SUCCESS, fmt::format("make_resident_image: failed to open .tido_bin '{}'", artifact.bin_source.string()).c_str());
    std::vector<std::byte> const & file_data = file_data_result.second;
    std::streamsize const file_size = s_cast<std::streamsize>(file_data.size());

    bool const is_3d = desc.info.size.z > 1;
    bool const is_cube = !is_3d && desc.info.array_layer_count != 0 && (desc.info.array_layer_count % 6 == 0);

    daxa::ImageInfo const image_info = {
        .flags = is_cube ? daxa::ImageCreateFlagBits::COMPATIBLE_CUBE : daxa::ImageCreateFlagBits::NONE,
        .dimensions = is_3d ? 3u : 2u,
        .format = desc.info.format,
        .size = {desc.info.size.x, desc.info.size.y, desc.info.size.z},
        .mip_level_count = desc.info.mip_level_count,
        .array_layer_count = desc.info.array_layer_count,
        .sample_count = 1,
        .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
        .name = name,
    };
    daxa::ImageId image = device.create_image(image_info);

    auto cr = device.create_command_recorder({.name = "upload image"});
    cr.pipeline_image_barrier({
        .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
        .image = image,
        .layout_operation = daxa::ImageLayoutOperation::TO_GENERAL,
    });

    daxa::BufferId staging_buffer = device.create_buffer({
        .size = s_cast<daxa::usize>(file_size),
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "upload image",
    });
    cr.destroy_buffer_deferred(staging_buffer);
    std::memcpy(device.buffer_host_address(staging_buffer).value(), file_data.data(), s_cast<usize>(file_size));

    u64 const base_offset = artifact.file_data_offset;
    for (u32 mip = 0; mip < desc.info.mip_level_count; ++mip)
    {
        u32 const width = std::max(1u, desc.info.size.x >> mip);
        u32 const height = std::max(1u, desc.info.size.y >> mip);
        u32 const depth = std::max(1u, desc.info.size.z >> mip);
        for (u32 layer = 0; layer < desc.info.array_layer_count; ++layer)
        {
            // Table is in storage order (coarse-first); see TidoSubresourceEntry indexing.
            u32 const subresource_index = desc.layer_mip_to_subresource_index(layer, s_cast<u32>(mip));
            TidoImageDescriptor::SubresourceEntry const & entry = desc.subresources.at(subresource_index);
            DBG_ASSERT_TRUE_M(base_offset + entry.offset + entry.byte_size <= file_data.size(), "make_resident_image: subresource out of .tido_bin bounds");
            cr.copy_buffer_to_image({
                .src_buffer = staging_buffer,
                .buffer_offset = base_offset + entry.offset,
                .dst_image = image,
                .image_slice = {
                    .mip_level = mip,
                    .base_array_layer = layer,
                },
                .image_offset = {0, 0, 0},
                .image_extent = {width, height, depth},
            });
        }
    }

    cr.pipeline_image_barrier({
        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_access = daxa::AccessConsts::READ,
        .image = image,
    });

    device.wait_on_submit({
        daxa::QUEUE_MAIN,
        device.submit_commands({
            .command_lists = std::array{cr.complete_current_commands()},
        }),
    });
    device.collect_garbage();

    result = image;
    finished.store(true, std::memory_order_release);
}

void MeshStreamTask::callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index)
{
    std::pair<FileIoResult, std::vector<std::byte>> file_data_result = read_file_shared(artifact.bin_source);
    DBG_ASSERT_TRUE_M(file_data_result.first == FileIoResult::SUCCESS, fmt::format("make_resident_mesh: failed to open .tido_bin '{}'", artifact.bin_source.string()).c_str());
    std::vector<std::byte> const & file_data = file_data_result.second;

    for (u32 lod = 0; lod < artifact.descriptor.lods.size(); ++lod)
    {
        TidoMeshDescriptor::LodDescriptor const & desc = artifact.descriptor.lods[lod];
        bool const lod_has_uv = desc.has_uv != 0;

        GPUMesh mesh = {};
        mesh.lod_error = desc.lod_error;
        mesh.aabb = desc.aabb;
        mesh.bounding_sphere = desc.bounding_sphere;

        mesh.mesh_buffer = device.create_buffer({
            .size = s_cast<daxa::usize>(desc.byte_size),
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
            .name = name + "." + std::to_string(lod),
        });
        daxa::DeviceAddress const mesh_bda = device.buffer_device_address(std::bit_cast<daxa::BufferId>(mesh.mesh_buffer)).value();
        auto mesh_gpu_mem_ptr = device.buffer_host_address(std::bit_cast<daxa::BufferId>(mesh.mesh_buffer)).value();

        DBG_ASSERT_TRUE_M(artifact.file_data_offset + desc.offset + desc.byte_size <= file_data.size(), "make_resident_mesh: LOD blob out of .tido_bin bounds");
        std::memcpy(mesh_gpu_mem_ptr, file_data.data() + artifact.file_data_offset + desc.offset, s_cast<usize>(desc.byte_size));

        u64 accumulated_offset = 0;
        auto sub_ptr = [&](u64 byte_size) -> daxa::DeviceAddress
        {
            daxa::DeviceAddress const address = mesh_bda + accumulated_offset;
            accumulated_offset += byte_size;
            return address;
        };
        mesh.meshlets = sub_ptr(sizeof(Meshlet) * desc.meshlet_count);
        mesh.meshlet_bounds = sub_ptr(sizeof(BoundingSphere) * desc.meshlet_count);
        mesh.meshlet_aabbs = sub_ptr(sizeof(AABB) * desc.meshlet_count);
        mesh.micro_indices = sub_ptr(sizeof(u8) * desc.micro_indices_count);
        mesh.indirect_vertices = sub_ptr(sizeof(u32) * desc.indirect_vertices_count);
        mesh.primitive_indices = sub_ptr(sizeof(u32) * desc.primitive_indices_count);
        mesh.vertex_positions = sub_ptr(sizeof(daxa_f32vec3) * desc.vertex_count);
        if (lod_has_uv)
        {
            mesh.vertex_uvs = sub_ptr(sizeof(daxa_f32vec2) * desc.vertex_count);
        }
        mesh.vertex_normals = sub_ptr(sizeof(daxa_f32vec3) * desc.vertex_count);
        DBG_ASSERT_TRUE_M(accumulated_offset == desc.byte_size, "make_resident_mesh: LOD sub-pointer walk did not consume the whole blob");

        mesh.material_index = material_manifest_index;
        mesh.meshlet_count = desc.meshlet_count;
        mesh.vertex_count = desc.vertex_count;
        mesh.primitive_count = desc.primitive_count;

        result.push_back(mesh);
    }
    finished.store(true, std::memory_order_release);
}
