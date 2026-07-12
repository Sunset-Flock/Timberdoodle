#pragma once

#include <array>
#include <string>

#include <daxa/daxa.hpp>

#include "../timberdoodle.hpp"
#include "../shader_shared/geometry.inl"
#include "tido_format/tido_texture.hpp"
#include "tido_format/tido_mesh.hpp"
using namespace tido::types;

/// --- Streamer ---
/// Makes cooked artifacts resident on the GPU. Both textures and meshes are streamed in from their
/// cooked .tido data file on disk (the streamer no longer receives the cooked bytes in memory). This
/// grows into the full .tido streaming back-end later.

// Creates a resident daxa image described by the streamer data and uploads its texel data, read back
// from the .tido file on disk. Reads every subresource (full residency for now).
auto make_resident_image(daxa::Device & device, TidoTextureStreamerData const & artifact) -> daxa::ImageId;

// The GPU-resident result of a cooked mesh: the per-LOD GPUMesh array (each packed into its own BDA
// buffer) plus the manifest slot it belongs to. Consumed by SceneRuntime::update, which
// copies it into the GPU mesh manifest and tracks mesh-group completeness.
struct MeshLodGroupUploadInfo
{
    std::array<GPUMesh, MAX_MESHES_PER_LOD_GROUP> lods = {};
    u32 lod_count = {};
    u32 mesh_lod_manifest_index = {};
};

struct MakeResidentMeshInfo
{
    TidoMeshStreamerData const & artifact;
    u32 mesh_lod_manifest_index = {};
    u32 material_manifest_index = {};
    std::string name = {};
};
// Uploads each LOD into a single per-LOD GPU buffer, read back from the artifact's .tido file on disk.
// The .tido LOD blob is already packed in GPUMesh BDA order, so the blob is copied in verbatim and the
// per-array BDA sub-pointers are wired from the descriptor's element counts. Mirrors
// make_resident_image: cooked .tido on disk -> GPU mesh out.
auto make_resident_mesh(daxa::Device & device, MakeResidentMeshInfo const & info) -> MeshLodGroupUploadInfo;
