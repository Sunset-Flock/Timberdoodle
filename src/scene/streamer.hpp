#pragma once

#include <array>
#include <string>

#include <daxa/daxa.hpp>

#include "../timberdoodle.hpp"
#include "../shader_shared/geometry.inl"
#include "optimizers/image_optimizer.hpp"
#include "optimizers/geometry_optimizer.hpp"
using namespace tido::types;

/// --- Streamer ---
/// Makes cooked artifacts resident on the GPU. For now this is just the image + mesh upload paths
/// lifted out of AssetProcessor; it grows into the full .tido streaming back-end later.

// Uploads cooked image memory into one array layer of an existing daxa image.
void upload_texture(daxa::Device & device, CookedImageData const & cooked, daxa::ImageId image, u32 layer = 0u);

// Creates a resident daxa image from cooked image memory and uploads it.
auto make_resident_image(daxa::Device & device, CookedImageData const & cooked) -> daxa::ImageId;

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
