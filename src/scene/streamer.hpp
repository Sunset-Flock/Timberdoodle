#pragma once

#include <array>
#include <string>

#include <daxa/daxa.hpp>

#include "../timberdoodle.hpp"

#include "../multithreading/thread_pool.hpp"

#include "../shader_shared/geometry.inl"
#include "tido_format/tido_format.hpp"
using namespace tido::types;

struct ImageStreamerData
{
    TidoImageDescriptor descriptor = {};

    std::filesystem::path bin_source = {};
    u64 file_data_offset = {};
};

struct ImageStreamTask : Task
{
    daxa::Device device = {};

    ImageStreamerData artifact = {};
    u32 image_manifest_index = {};
    std::string name = {};

    daxa::ImageId result = {};

    void callback(u32 chunk_index, u32 thread_index) override;
};

struct MeshStreamerData
{
    TidoMeshDescriptor descriptor = {};

    std::filesystem::path bin_source = {};
    u64 file_data_offset = {};
};
struct MeshStreamTask : Task
{
    daxa::Device device = {};

    MeshStreamerData artifact = {};
    u32 mesh_lod_manifest_index = {};
    u32 material_manifest_index = {};
    std::string name = {};

    std::vector<GPUMesh> result = {};

    void callback(u32 chunk_index, u32 thread_index) override;
};