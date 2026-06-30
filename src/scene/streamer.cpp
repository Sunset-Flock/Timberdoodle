#include "streamer.hpp"

#include <cstring>
#include <fstream>
#include <vector>

auto make_resident_image(daxa::Device & device, TidoTextureCookResult const & artifact) -> daxa::ImageId
{
    TidoTextureDescriptor const & desc = artifact.descriptor;

    // Read the whole .tido data file. The subresource offsets in the artifact are absolute from byte 0
    // of this file, so it maps directly onto the staging buffer with no rebasing.
    std::ifstream ifs{artifact.tido_path, std::ios::binary | std::ios::ate};
    DBG_ASSERT_TRUE_M(ifs.good(), fmt::format("make_resident_image: failed to open .tido '{}'", artifact.tido_path.string()).c_str());
    std::streamsize const file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<std::byte> file_data(s_cast<usize>(file_size));
    ifs.read(r_cast<char *>(file_data.data()), file_size);
    DBG_ASSERT_TRUE_M(ifs.good(), fmt::format("make_resident_image: failed to read .tido '{}'", artifact.tido_path.string()).c_str());

    // 3D vs 2D/array/cube is deduced from the extents/layer count, not stored: a depth > 1 means a 3D
    // texture; an array_layers that is a multiple of 6 is a cubemap (gets the cube-compatible flag).
    bool const is_3d = desc.depth > 1;
    bool const is_cube = !is_3d && desc.array_layers != 0 && (desc.array_layers % 6 == 0);

    daxa::ImageInfo const image_info = {
        .flags = is_cube ? daxa::ImageCreateFlagBits::COMPATIBLE_CUBE : daxa::ImageCreateFlagBits::NONE,
        .dimensions = is_3d ? 3u : 2u,
        .format = std::bit_cast<daxa::Format>(desc.format),
        .size = {desc.width, desc.height, desc.depth},
        .mip_level_count = desc.mip_count,
        .array_layer_count = desc.array_layers,
        .sample_count = 1,
        .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
        .name = artifact.tido_path.filename().string(),
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

    // One copy per subresource. The buffer_offset is the subresource's offset within the .tido (==
    // within the staging buffer). image_extent is in texels even for block-compressed formats.
    for (u32 mip = 0; mip < desc.mip_count; ++mip)
    {
        u32 const width = std::max(1u, desc.width >> mip);
        u32 const height = std::max(1u, desc.height >> mip);
        u32 const depth = std::max(1u, desc.depth >> mip);
        for (u32 layer = 0; layer < desc.array_layers; ++layer)
        {
            // Table is in storage order (coarse-first); see TidoSubresourceEntry indexing.
            u32 const subresource_index = (desc.mip_count - 1u - mip) * desc.array_layers + layer;
            TidoSubresourceEntry const & entry = artifact.subresources.at(subresource_index);
            cr.copy_buffer_to_image({
                .src_buffer = staging_buffer,
                .buffer_offset = entry.offset,
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
