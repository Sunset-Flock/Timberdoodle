#pragma once

#include <array>
#include <vector>
#include <glm/glm.hpp>

#include "../../timberdoodle.hpp"
#include "../../shader_shared/shared.inl"
#include "../../shader_shared/geometry.inl"
using namespace tido::types;

/// --- Geometry Optimizer ---
/// Generic, format-agnostic geometry cook. Takes raw vertex/index streams (extracted by an importer)
/// and produces the runtime-ready, CPU-side cooked form: per-LOD meshlets, optimized index buffers,
/// bounds. Knows nothing about glTF or the GPU - the scene/streamer packs the cooked arrays into a
/// GPU buffer when uploading. Mirrors the texture optimizer (tex_compression): raw CPU in -> cooked
/// CPU out.

// Raw, format-neutral input for a single mesh (one material). `uvs` empty == mesh has no uvs.
struct RawMesh
{
    std::vector<u32> indices = {};
    std::vector<glm::vec3> positions = {};
    std::vector<glm::vec3> normals = {};
    std::vector<glm::vec2> uvs = {};
};

// One cooked LOD held entirely in CPU memory. The scalar fields mirror GPUMesh; the vectors are the
// arrays that get packed into the single GPU mesh buffer at upload time.
struct ProcessedMeshLod
{
    AABB aabb = {};
    BoundingSphere bounding_sphere = {};
    f32 lod_error = {};
    u32 vertex_count = {};
    u32 primitive_count = {};
    std::vector<Meshlet> meshlets = {};
    std::vector<BoundingSphere> meshlet_bounds = {};
    std::vector<AABB> meshlet_aabbs = {};
    std::vector<u8> micro_indices = {};       // packed to a multiple of 4 bytes
    std::vector<u32> indirect_vertices = {};
    std::vector<u32> primitive_indices = {};
    std::vector<glm::vec3> vertex_positions = {};
    std::vector<glm::vec2> vertex_uvs = {};   // empty if the mesh has no uvs
    std::vector<glm::vec3> vertex_normals = {};
};

// A mesh's fixed array of LODs. Only the first `lod_count` entries are valid.
struct ProcessedMesh
{
    std::array<ProcessedMeshLod, MAX_MESHES_PER_LOD_GROUP> lods = {};
    u32 lod_count = {};
};

auto optimize_mesh(RawMesh const & raw) -> ProcessedMesh;
