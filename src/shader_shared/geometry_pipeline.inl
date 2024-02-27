#pragma once

/// --- Goal of file ---
/// - should contain anything needed only for forward drawing
/// - should NOT contain things required for shading or post processing

#include <daxa/daxa.inl>

#include "shared.inl"
#include "asset.inl"

#define OPAQUE_DRAW_LIST_SOLID 0
#define OPAQUE_DRAW_LIST_MASKED 1

struct MeshDrawTuple
{
    daxa_u32 entity_index;
    daxa_u32 mesh_index;
    daxa_u32 in_mesh_group_index;
};
DAXA_DECL_BUFFER_PTR_ALIGN(MeshDrawTuple, 4);

struct OpaqueMeshDrawListBufferHead
{
    daxa_u32 list_sizes[2];
    daxa_BufferPtr(MeshDrawTuple) mesh_draw_tuples[2];
}; 
DAXA_DECL_BUFFER_PTR_ALIGN(OpaqueMeshDrawListBufferHead, 8)

#if defined(__cplusplus)
#include <span>
inline auto make_opaque_draw_list_buffer_head(daxa::DeviceAddress address, std::array<std::span<MeshDrawTuple>, 2> draw_lists) -> OpaqueMeshDrawListBufferHead
{
    OpaqueMeshDrawListBufferHead ret = {};
    ret.list_sizes[0] = static_cast<daxa::u32>(draw_lists[0].size());
    ret.list_sizes[1] = static_cast<daxa::u32>(draw_lists[1].size());
    ret.mesh_draw_tuples[0] = address + sizeof(OpaqueMeshDrawListBufferHead);
    ret.mesh_draw_tuples[1] = address + sizeof(OpaqueMeshDrawListBufferHead) + sizeof(MeshDrawTuple) * draw_lists[0].size();
    return ret;
}
inline auto get_opaque_draw_list_buffer_size() -> daxa::usize
{
    return sizeof(OpaqueMeshDrawListBufferHead) + sizeof(MeshDrawTuple) * MAX_MESH_INSTANCES * 2;
}
#endif // #if defined(__cplusplus)

/// NOTE: In the future we want a TransparentMeshDrawListBufferHead, that has a much larger array for custom material permutations.

#if !defined(__cplusplus)
DAXA_DECL_BUFFER_REFERENCE_ALIGN(4) U32ArenaBufferRef
{
    daxa_u32 offsets_section_size;
    daxa_u32 bitfield_section_size;
    daxa_u32 uints[];
};
#endif // #if !defined(__cplusplus)

#define FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID (~0u)
#define FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED (~0u ^ 1u)
#define FIRST_PASS_MESHLET_BITFIELD_OFFSET_DEBUG (~0u ^ 2u)

#define FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_SIZE (1u<<20u)
