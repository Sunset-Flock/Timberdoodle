#include "streamer.hpp"

#include <cstring>

void upload_texture(daxa::Device & device, CookedImageData const & cooked, daxa::ImageId image, u32 layer)
{
    auto cr = device.create_command_recorder({.name = "upload image"});

    cr.pipeline_image_barrier({
        .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
        .image = image,
        .layout_operation = daxa::ImageLayoutOperation::TO_GENERAL,
    });

    daxa::BufferId staging_buffer = device.create_buffer({
        .size = cooked.src_data.size(),
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "upload image",
    });
    cr.destroy_buffer_deferred(staging_buffer);
    std::memcpy(device.buffer_host_address(staging_buffer).value(), cooked.src_data.data(), cooked.src_data.size());

    daxa::ImageInfo image_info = device.image_info(image).value();
    for (u32 mip = 0; mip < cooked.mips_to_copy; ++mip)
    {
        u32 width = std::max(1u, image_info.size.x >> mip);
        u32 height = std::max(1u, image_info.size.y >> mip);
        u32 depth = std::max(1u, image_info.size.z >> mip);
        cr.copy_buffer_to_image({
            .src_buffer = staging_buffer,
            .buffer_offset = cooked.mip_copy_offsets[mip],
            .dst_image = image,
            .image_slice = {
                .mip_level = mip,
                .base_array_layer = layer,
            },
            .image_offset = {0, 0, 0},
            .image_extent = {width, height, depth},
        });
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
}

auto make_resident_image(daxa::Device & device, CookedImageData const & cooked) -> daxa::ImageId
{
    daxa::ImageId image = device.create_image(cooked.image_info);
    upload_texture(device, cooked, image);
    return image;
}

auto make_resident_mesh(daxa::Device & device, MakeResidentMeshInfo const & info) -> MeshLodGroupUploadInfo
{
    ProcessedMesh const & processed = info.processed;
    MeshLodGroupUploadInfo ret = {};
    ret.lod_count = processed.lod_count;
    ret.mesh_lod_manifest_index = info.mesh_lod_manifest_index;

    // Pack each cooked LOD into a single GPU mesh buffer (mirrors the GPUMesh BDA layout).
    for (u32 lod = 0; lod < processed.lod_count; ++lod)
    {
        ProcessedMeshLod const & cooked = processed.lods[lod];
        bool const lod_has_uv = !cooked.vertex_uvs.empty();

        u64 total_mesh_buffer_size =
            sizeof(Meshlet) * cooked.meshlets.size() +
            sizeof(BoundingSphere) * cooked.meshlet_bounds.size() +
            sizeof(AABB) * cooked.meshlet_aabbs.size() +
            sizeof(u8) * cooked.micro_indices.size() +
            sizeof(u32) * cooked.indirect_vertices.size() +
            sizeof(u32) * cooked.primitive_indices.size() +
            sizeof(daxa_f32vec3) * cooked.vertex_positions.size() +
            sizeof(daxa_f32vec3) * cooked.vertex_normals.size();
        if (lod_has_uv)
        {
            total_mesh_buffer_size += sizeof(daxa_f32vec2) * cooked.vertex_uvs.size();
        }

        GPUMesh mesh = {};
        mesh.lod_error = cooked.lod_error;
        mesh.aabb = cooked.aabb;
        mesh.bounding_sphere = cooked.bounding_sphere;

        mesh.mesh_buffer = device.create_buffer({
            .size = s_cast<daxa::usize>(total_mesh_buffer_size),
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
            .name = info.name + "." + std::to_string(lod),
        });
        daxa::DeviceAddress const mesh_bda = device.buffer_device_address(std::bit_cast<daxa::BufferId>(mesh.mesh_buffer)).value();
        auto mesh_gpu_mem_ptr = device.buffer_host_address(std::bit_cast<daxa::BufferId>(mesh.mesh_buffer)).value();

        u64 accumulated_offset = 0;
        auto pack = [&](auto & dst_ptr_field, void const * src, u64 byte_size)
        {
            dst_ptr_field = mesh_bda + accumulated_offset;
            std::memcpy(mesh_gpu_mem_ptr + accumulated_offset, src, byte_size);
            accumulated_offset += byte_size;
        };
        pack(mesh.meshlets, cooked.meshlets.data(), sizeof(Meshlet) * cooked.meshlets.size());
        pack(mesh.meshlet_bounds, cooked.meshlet_bounds.data(), sizeof(BoundingSphere) * cooked.meshlet_bounds.size());
        pack(mesh.meshlet_aabbs, cooked.meshlet_aabbs.data(), sizeof(AABB) * cooked.meshlet_aabbs.size());
        pack(mesh.micro_indices, cooked.micro_indices.data(), sizeof(u8) * cooked.micro_indices.size());
        pack(mesh.indirect_vertices, cooked.indirect_vertices.data(), sizeof(u32) * cooked.indirect_vertices.size());
        pack(mesh.primitive_indices, cooked.primitive_indices.data(), sizeof(u32) * cooked.primitive_indices.size());
        pack(mesh.vertex_positions, cooked.vertex_positions.data(), sizeof(daxa_f32vec3) * cooked.vertex_positions.size());
        if (lod_has_uv)
        {
            pack(mesh.vertex_uvs, cooked.vertex_uvs.data(), sizeof(daxa_f32vec2) * cooked.vertex_uvs.size());
        }
        pack(mesh.vertex_normals, cooked.vertex_normals.data(), sizeof(daxa_f32vec3) * cooked.vertex_normals.size());

        mesh.material_index = info.material_manifest_index;
        mesh.meshlet_count = s_cast<u32>(cooked.meshlets.size());
        mesh.vertex_count = cooked.vertex_count;
        mesh.primitive_count = cooked.primitive_count;

        ret.lods[lod] = mesh;
    }
    return ret;
}
