#pragma once

#include <array>
#include <string>

#include <daxa/daxa.hpp>

#include "../timberdoodle.hpp"
#include "../shader_shared/geometry.inl"
#include "optimizers/geometry_optimizer.hpp"
#include "tido_format/tido_texture.hpp"
using namespace tido::types;

/// --- Streamer ---
/// Makes cooked artifacts resident on the GPU. Textures are streamed in from their cooked .tido data
/// file (it no longer receives the raw pixel bytes in memory); meshes are still uploaded from the
/// in-memory cooked form. This grows into the full .tido streaming back-end later.

// Creates a resident daxa image described by the cooked artifact and uploads its texel data, read
// back from the artifact's .tido file on disk. Reads every subresource (full residency for now).
auto make_resident_image(daxa::Device & device, TidoTextureCookResult const & artifact) -> daxa::ImageId;

// The GPU-resident result of a cooked mesh: the per-LOD GPUMesh array (each packed into its own BDA
// buffer) plus the manifest slot it belongs to. Consumed by Scene::record_gpu_manifest_update, which
// copies it into the GPU mesh manifest and tracks mesh-group completeness.
struct MeshLodGroupUploadInfo
{
    std::array<GPUMesh, MAX_MESHES_PER_LOD_GROUP> lods = {};
    u32 lod_count = {};
    u32 mesh_lod_manifest_index = {};
};

struct MakeResidentMeshInfo
{
    ProcessedMesh const & processed;
    u32 mesh_lod_manifest_index = {};
    u32 material_manifest_index = {};
    std::string name = {};
};
// Packs each cooked LOD into a single per-LOD GPU buffer (mirroring the GPUMesh BDA layout) and
// returns the GPU-resident handles. Mirrors make_resident_image: cooked CPU mesh in -> GPU mesh out.
auto make_resident_mesh(daxa::Device & device, MakeResidentMeshInfo const & info) -> MeshLodGroupUploadInfo;
