#pragma once

// INFO:
// This file should contain general information for geometry rendering/reading in shading
// Ideally this file only contains structs used for shading
// There will always be a lot of overlap with drawing and culling of the geometry pipeline.
// If possible split rasterization/culling only data into geometry_pipeline.inl

#include <daxa/daxa.inl>
#include "shared.inl"
#include "visbuffer.inl"

#if defined(__cplusplus)
#include <algorithm>
#include <cstring>
#endif

#define INVALID_MESHLET_INDEX (~(0u))

#define MAX_MESHLET_INSTANCES TRIANGLE_ID_MAX_MESHLET_INSTANCES
// MUST never be greater then 124!
#define MAX_TRIANGLES_PER_MESHLET (64)
// MUST never be greater then MAX_TRIANGLES_PER_MESHLET * 3!
#define MAX_VERTICES_PER_MESHLET (64)

#define ENTITY_MESHLET_VISIBILITY_ARENA_SIZE (1<<20)
#define ENTITY_MESHLET_VISIBILITY_ARENA_BIT_SIZE (ENTITY_MESHLET_VISIBILITY_ARENA_SIZE * 8)
#define ENTITY_MESHLET_VISIBILITY_ARENA_UINT_SIZE (ENTITY_MESHLET_VISIBILITY_ARENA_SIZE / 4)

// TODO: split this struct into multiple arrays!
// !!NEEDS TO BE ABI COMPATIBLE WITH meshopt_Meshlet!!
struct Meshlet
{
    // Offset into the meshs vertex index array.
    daxa_u32 indirect_vertex_offset;
    // Equivalent to meshoptimizers triangle_offset.
    // Renamed the field for more clarity.
    daxa_u32 micro_indices_offset;
    daxa_u32 vertex_count;
    daxa_u32 triangle_count;
};
DAXA_DECL_BUFFER_PTR(Meshlet)

struct MeshletInstance
{
    daxa_u32 entity_index;
    daxa_u32 meshlet_index;
    daxa_u32 mesh_index;
    daxa_u32 material_index;
    daxa_u32 in_mesh_group_index;
    daxa_u32 mesh_instance_index;
};
DAXA_DECL_BUFFER_PTR(MeshletInstance)

struct BoundingSphere
{
    daxa_f32vec3 center;
    daxa_f32 radius;
};
DAXA_DECL_BUFFER_PTR(BoundingSphere)

struct AABB
{
    daxa_f32vec3 center;
    daxa_f32vec3 size;
};
DAXA_DECL_BUFFER_PTR(AABB)

struct GPUMesh
{
    daxa_BufferId mesh_buffer;
    daxa_u32 material_index;
    daxa_u32 meshlet_count;
    daxa_u32 vertex_count;
    daxa_u32 primitive_count;
    daxa_f32 lod_error;
    daxa_u32 padd;
    AABB aabb;
    BoundingSphere bounding_sphere;
    daxa_BufferPtr(Meshlet) meshlets;
    daxa_BufferPtr(BoundingSphere) meshlet_bounds;
    daxa_BufferPtr(AABB) meshlet_aabbs;
    daxa_BufferPtr(daxa_u32) micro_indices;         // Index into indirect vertices, usually multiple micro indices are packed into a single uint
    daxa_BufferPtr(daxa_u32) indirect_vertices;     // Lists of unique vertices per meshlet, indexes vertex arrays. 
    daxa_BufferPtr(daxa_u32) primitive_indices;     // List of triangles, every three uints form a triangle, each value is a vertex index.
    daxa_BufferPtr(daxa_f32vec3) vertex_positions;
    daxa_BufferPtr(daxa_f32vec2) vertex_uvs;        // NOTE: Possibly nullptr
    daxa_BufferPtr(daxa_f32vec3) vertex_normals;
};
DAXA_DECL_BUFFER_PTR_ALIGN(GPUMesh, 8)

struct GPUMeshLodGroup
{
    daxa_u32 lod_count;
};
DAXA_DECL_BUFFER_PTR_ALIGN(GPUMeshLodGroup, 4)

#define MATERIAL_TYPE_SURFACE 0

// GLSL shares this struct and has no sizeof, so the payload cannot be a computed array bound. The assert
// below pins this to the largest material type, so it fails the build rather than silently drifting.
#define GPU_MATERIAL_PAYLOAD_U64S 9

struct SurfaceMaterial
{
    daxa_ImageViewId diffuse_texture_id;
    daxa_ImageViewId opacity_texture_id;
    daxa_ImageViewId normal_texture_id;
    daxa_ImageViewId roughnes_metalness_id;
    daxa_b32 alpha_discard_enabled;
    daxa_b32 double_sided_enabled;
    daxa_b32 blend_enabled;
    daxa_b32 normal_compressed_bc5_rg;
    daxa_f32vec3 base_color;
    daxa_f32vec3 emissive_color;
};

// One entry per material of any type: a tag naming the type, and a payload holding exactly that type.
// The payload is u64 so it stays 8 byte aligned for the image ids the material types put in it.
struct GPUMaterial
{
    daxa_u32 material_type;
    daxa_u64 payload[GPU_MATERIAL_PAYLOAD_U64S];
};
DAXA_DECL_BUFFER_PTR_ALIGN(GPUMaterial, 8)

#if DAXA_LANGUAGE != DAXA_LANGUAGE_GLSL
static const SurfaceMaterial SURFACE_MATERIAL_FALLBACK = SurfaceMaterial(
    daxa_ImageViewId(0),
    daxa_ImageViewId(0),
    daxa_ImageViewId(0),
    daxa_ImageViewId(0),
    false,
    false,
    false,
    false,
    daxa_f32vec3(1,1,1),
    daxa_f32vec3(0,0,0)
);
#endif

#if DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG
// Unchecked: the caller must already know the material's type. Passes that can shade any material switch on
// material_type first; the raster passes only ever reach surface materials and cast straight to one.
SurfaceMaterial * as_surface_material(GPUMaterial * material)
{
    return (SurfaceMaterial *)(&material.payload);
}
#endif

#if defined(__cplusplus)
// Every material type belongs in this list.
inline constexpr daxa_u32 GPU_MATERIAL_PAYLOAD_REQUIRED_U64S = (std::max({sizeof(SurfaceMaterial)}) + 7u) / 8u;
static_assert(GPU_MATERIAL_PAYLOAD_U64S == GPU_MATERIAL_PAYLOAD_REQUIRED_U64S,
    "set GPU_MATERIAL_PAYLOAD_U64S to GPU_MATERIAL_PAYLOAD_REQUIRED_U64S - the largest material type in whole u64s");

template <typename MaterialT>
inline auto pack_material(daxa_u32 material_type, MaterialT const & material) -> GPUMaterial
{
    static_assert(sizeof(MaterialT) <= sizeof(GPUMaterial::payload), "material type outgrew GPU_MATERIAL_PAYLOAD_U64S");
    static_assert(alignof(MaterialT) <= alignof(daxa_u64), "material type needs more alignment than the payload has");
    GPUMaterial ret = {};
    ret.material_type = material_type;
    std::memcpy(&ret.payload, &material, sizeof(MaterialT));
    return ret;
}
#endif

#if DAXA_SHADER
uint get_micro_index(daxa_BufferPtr(daxa_u32) micro_indices, daxa_u32 index_offset)
{
    uint pack_index = index_offset / 4;
    uint index_pack = deref_i(micro_indices, pack_index);
    uint in_pack_offset = index_offset % 4;
    uint in_pack_shift = in_pack_offset * 8;
    return (index_pack >> in_pack_shift) & 0xFF;
}
#endif // #if defined(DAXA_SHADER)

struct MeshletDrawList2
{
    daxa_u32 pass_counts[2];
    daxa_RWBufferPtr(daxa_u32) instances;
};

struct MeshletInstancesBufferHead
{
    daxa_u32 pass_counts[2];
    daxa_RWBufferPtr(MeshletInstance) meshlets;
    MeshletDrawList2 prepass_draw_lists[2];
};
DAXA_DECL_BUFFER_PTR_ALIGN(MeshletInstancesBufferHead, 8)

#if defined(__cplusplus)
inline auto make_meshlet_instance_buffer_head(daxa::DeviceAddress address) -> MeshletInstancesBufferHead
{
    MeshletInstancesBufferHead ret = {};
    address = ret.meshlets = address + sizeof(MeshletInstancesBufferHead);
    address = ret.prepass_draw_lists[0].instances = address + sizeof(MeshletInstance) * MAX_MESHLET_INSTANCES;
    address = ret.prepass_draw_lists[1].instances = address + sizeof(daxa_u32) * MAX_MESHLET_INSTANCES;
    return ret;
}
inline auto size_of_meshlet_instance_buffer() -> daxa::usize
{
    return  sizeof(MeshletInstancesBufferHead) + 
            sizeof(MeshletInstance) * MAX_MESHLET_INSTANCES + 
            sizeof(daxa_u32) * MAX_MESHLET_INSTANCES * 2;
}
#endif

#define INVALID_MANIFEST_INDEX (~0u)

struct GPUMeshGroup
{
    daxa_u32 mesh_lod_group_count;
    daxa_u32 padd;
};
DAXA_DECL_BUFFER_PTR_ALIGN(GPUMeshGroup, 8)