#pragma once

#include <array>
#include <string>

#include <daxa/daxa.hpp>

#include "../timberdoodle.hpp"

#include "../multithreading/thread_pool.hpp"

#include "../shader_shared/geometry.inl"
#include "tido_format/tido_texture.hpp"
#include "tido_format/tido_mesh.hpp"
using namespace tido::types;

struct TextureStreamTask : Task
{
    daxa::Device device = {};
    // Copied (not referenced) so it stays valid if _texture_manifest reallocates mid-stream.
    TidoTextureStreamerData artifact = {};
    u32 texture_manifest_index = {};
    daxa::ImageId result = {};
    std::atomic<bool> finished = false;

    void callback(u32 chunk_index, u32 thread_index) override;
};

struct MeshLodGroupUploadInfo
{
    std::array<GPUMesh, MAX_MESHES_PER_LOD_GROUP> lods = {};
    u32 lod_count = {};
    u32 mesh_lod_manifest_index = {};
};
struct MeshStreamTask : Task
{
    daxa::Device device = {};
    // Copied (not referenced) so it stays valid if _mesh_lod_group_manifest reallocates mid-stream.
    TidoMeshStreamerData artifact = {};
    u32 mesh_lod_manifest_index = {};
    u32 material_manifest_index = {};
    std::string name = {};
    MeshLodGroupUploadInfo result = {};
    std::atomic<bool> finished = false;

    void callback(u32 chunk_index, u32 thread_index) override;
};

/// --- Streamer ---
/// Makes cooked artifacts resident on the GPU. Both textures and meshes are streamed in from their
/// cooked .tido_bin data file on disk (the streamer no longer receives the cooked bytes in memory). This
/// grows into the full .tido_bin streaming back-end later.

// Creates a resident daxa image described by the streamer data and uploads its texel data, read back
// from the .tido_bin file on disk. Reads every subresource (full residency for now).
auto make_resident_image(daxa::Device & device, TidoTextureStreamerData const & artifact) -> daxa::ImageId;

struct MakeResidentMeshInfo
{
    TidoMeshStreamerData const & artifact;
    u32 mesh_lod_manifest_index = {};
    u32 material_manifest_index = {};
    std::string name = {};
};
// Uploads each LOD into a single per-LOD GPU buffer, read back from the artifact's .tido_bin file on disk.
// The .tido_bin LOD blob is already packed in GPUMesh BDA order, so the blob is copied in verbatim and the
// per-array BDA sub-pointers are wired from the descriptor's element counts. Mirrors
// make_resident_image: cooked .tido_bin on disk -> GPU mesh out.
auto make_resident_mesh(daxa::Device & device, MakeResidentMeshInfo const & info) -> MeshLodGroupUploadInfo;
