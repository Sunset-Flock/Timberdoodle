#pragma once

#include <filesystem>
#include <cstddef>
#include <array>
#include <optional>

#include "../../timberdoodle.hpp"
#include "../../shader_shared/shared.inl"   // MAX_MESHES_PER_LOD_GROUP
#include "../../shader_shared/geometry.inl" // AABB, BoundingSphere
using namespace tido::types;

// Defined in optimizers/geometry_optimizer.hpp. Forward-declared here (instead of included) so that
// geometry_optimizer.hpp can include this header to use TidoMeshCookResult as a cook return type
// without an include cycle. tido_mesh.cpp includes geometry_optimizer.hpp for the full type.
struct ProcessedMesh;

/// --- .tido mesh format (writer) ---
/// The cooked, runtime-ready on-disk form of a mesh (one MeshLodGroup: a single material's primitive
/// together with its LOD chain). Mirrors the texture .tido (see "Texture .tido format.md" /
/// tido_texture.hpp). A cooked mesh is two files sharing one stem:
///   <stem>.tido        - raw packed geometry data only, no header. One contiguous blob per LOD; within
///                        a blob the arrays are packed back-to-back in the SAME order
///                        make_resident_mesh packs the GPU mesh buffer (meshlets, meshlet_bounds,
///                        meshlet_aabbs, micro_indices, indirect_vertices, primitive_indices,
///                        vertex_positions, [vertex_uvs], vertex_normals). The streamer can therefore
///                        read a LOD blob, memcpy it straight into a BDA buffer, and wire the
///                        sub-pointers from the stored element counts.
///   <name>.tido_cache - per-imported-file manifest recording the cook key + every cooked artifact.
///
/// LODs are written finest-first (LOD 0 first), matching the ProcessedMesh order. Per-LOD residency
/// streaming is future work; the streamer makes every LOD resident for now.

// Per-LOD description: the scalar GPUMesh fields plus the array element counts needed to reconstruct
// the BDA sub-pointers from the packed blob, and the blob's location in the .tido file. Counts that
// are derivable from these (meshlet_bounds / meshlet_aabbs == meshlet_count; vertex_positions /
// vertex_normals == vertex_count; vertex_uvs == vertex_count when has_uv) are not stored; only the
// independent ones are.
struct TidoMeshLodDescriptor
{
    u64 blob_offset = {};             // byte offset of this LOD's packed blob in the .tido file
    u64 blob_byte_size = {};          // total byte size of the LOD blob
    AABB aabb = {};
    BoundingSphere bounding_sphere = {};
    f32 lod_error = {};
    u32 vertex_count = {};            // == vertex_positions == vertex_normals (== vertex_uvs if has_uv)
    u32 primitive_count = {};
    u32 meshlet_count = {};           // == meshlets == meshlet_bounds == meshlet_aabbs
    u32 micro_indices_count = {};     // u8 count (already padded to a multiple of 4)
    u32 indirect_vertices_count = {}; // u32 count
    u32 primitive_indices_count = {}; // u32 count
    u32 has_uv = {};                  // 1 if vertex_uvs is present in the blob, else 0
};

// Fixed mesh description: how many of the LOD slots below are valid.
struct TidoMeshDescriptor
{
    u32 lod_count = {};
};

// The cooked metadata produced alongside the .tido data file. Persisted into the .tido_cache.
struct TidoMeshCookResult
{
    // Stable per-artifact lookup key (hash of the mesh's source identity). Used as the key of the
    // .tido_cache index so an importer can find this entry by recomputing the key from the source.
    u64 cache_key = {};
    TidoMeshDescriptor descriptor = {};
    std::array<TidoMeshLodDescriptor, MAX_MESHES_PER_LOD_GROUP> lods = {}; // first descriptor.lod_count valid
    std::filesystem::path tido_path = {};                                 // the written .tido data file
};

// Writes <cache_dir>/<name>.tido (raw packed per-LOD geometry blobs) from an already-cooked mesh and
// returns its descriptor + per-LOD blob table. `cache_key` is the mesh's stable source-identity hash
// (from the unique gltf mesh/primitive index); it is recorded on the result AND used as the .tido file
// stem's disambiguator, so two distinct primitives never resolve to the same path (content-identical
// meshes used to collide and race on one file). Does NOT write the .tido_cache manifest yet (that is
// aggregated per imported file in a later step). Returns std::nullopt on an IO failure.
auto write_mesh_tido(ProcessedMesh const & processed, std::filesystem::path const & cache_dir, std::string const & name, u64 cache_key) -> std::optional<TidoMeshCookResult>;
