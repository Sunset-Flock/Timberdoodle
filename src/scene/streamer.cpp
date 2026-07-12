#include "streamer.hpp"

#include <cstring>
#include <fstream>
#include <vector>

auto make_resident_image(daxa::Device & device, TidoTextureStreamerData const & artifact) -> daxa::ImageId
{
    TidoTextureDescriptor const & desc = artifact.info;

    // Read the whole .tido data file. The subresource offsets in the artifact are absolute from byte 0
    // of this file, so it maps directly onto the staging buffer with no rebasing.
    std::ifstream ifs{artifact.bin_source, std::ios::binary | std::ios::ate};
    DBG_ASSERT_TRUE_M(ifs.good(), fmt::format("make_resident_image: failed to open .tido '{}'", artifact.bin_source.string()).c_str());
    std::streamsize const file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<std::byte> file_data(s_cast<usize>(file_size));
    ifs.read(r_cast<char *>(file_data.data()), file_size);
    DBG_ASSERT_TRUE_M(ifs.good(), fmt::format("make_resident_image: failed to read .tido '{}'", artifact.bin_source.string()).c_str());

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
        .name = artifact.bin_source.filename().string(),
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
            DBG_ASSERT_TRUE_M(entry.offset + entry.byte_size <= file_data.size(), "make_resident_image: subresource out of .tido bounds");
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
    TidoMeshStreamerData const & artifact = info.artifact;
    MeshLodGroupUploadInfo ret = {};
    ret.lod_count = artifact.descriptor.lod_count;
    ret.mesh_lod_manifest_index = info.mesh_lod_manifest_index;

    // Read the whole .tido data file. Each LOD's blob offset/size in the descriptor is absolute from
    // byte 0 of this file, so a LOD blob maps directly onto the staging upload with no rebasing.
    std::ifstream ifs{artifact.bin_source, std::ios::binary | std::ios::ate};
    DBG_ASSERT_TRUE_M(ifs.good(), fmt::format("make_resident_mesh: failed to open .tido '{}'", artifact.bin_source.string()).c_str());
    std::streamsize const file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<std::byte> file_data(s_cast<usize>(file_size));
    ifs.read(r_cast<char *>(file_data.data()), file_size);
    DBG_ASSERT_TRUE_M(ifs.good(), fmt::format("make_resident_mesh: failed to read .tido '{}'", artifact.bin_source.string()).c_str());

    // Upload each LOD into its own GPU mesh buffer. The .tido blob is already laid out in GPUMesh BDA
    // order, so the whole blob is copied in verbatim and the per-array sub-pointers are wired from the
    // descriptor's element counts (same order write_mesh_tido packed them).
    for (u32 lod = 0; lod < artifact.descriptor.lod_count; ++lod)
    {
        TidoMeshLodDescriptor const & desc = artifact.lods[lod];
        bool const lod_has_uv = desc.has_uv != 0;

        GPUMesh mesh = {};
        mesh.lod_error = desc.lod_error;
        mesh.aabb = desc.aabb;
        mesh.bounding_sphere = desc.bounding_sphere;

        mesh.mesh_buffer = device.create_buffer({
            .size = s_cast<daxa::usize>(desc.blob_byte_size),
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
            .name = info.name + "." + std::to_string(lod),
        });
        daxa::DeviceAddress const mesh_bda = device.buffer_device_address(std::bit_cast<daxa::BufferId>(mesh.mesh_buffer)).value();
        auto mesh_gpu_mem_ptr = device.buffer_host_address(std::bit_cast<daxa::BufferId>(mesh.mesh_buffer)).value();

        // The blob is contiguous and already in BDA order; copy it in one shot.
        DBG_ASSERT_TRUE_M(desc.blob_offset + desc.blob_byte_size <= file_data.size(), "make_resident_mesh: LOD blob out of .tido bounds");
        std::memcpy(mesh_gpu_mem_ptr, file_data.data() + desc.blob_offset, s_cast<usize>(desc.blob_byte_size));

        // Carve out the per-array BDA sub-pointers by walking the blob in pack order. meshlet_bounds /
        // meshlet_aabbs share meshlet_count; vertex_positions / vertex_normals (and vertex_uvs when
        // present) share vertex_count.
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
        DBG_ASSERT_TRUE_M(accumulated_offset == desc.blob_byte_size, "make_resident_mesh: LOD sub-pointer walk did not consume the whole blob");

        mesh.material_index = info.material_manifest_index;
        mesh.meshlet_count = desc.meshlet_count;
        mesh.vertex_count = desc.vertex_count;
        mesh.primitive_count = desc.primitive_count;

        ret.lods[lod] = mesh;
    }
    return ret;
}
