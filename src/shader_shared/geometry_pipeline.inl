#pragma once

/// --- Goal of file ---
/// - should contain anything needed only for forward drawing
/// - should NOT contain things required for shading or post processing

#include "daxa/daxa.inl"

#include "shared.inl"
#include "geometry.inl"

/// --- Mesh Instance Draw List Begin ---

#define PREPASS_DRAW_LIST_OPAQUE 0
#define PREPASS_DRAW_LIST_MASKED 1
#define PREPASS_DRAW_LIST_TYPE_COUNT 2

struct MeshDrawList
{
    daxa_u32 count;
    daxa_RWBufferPtr(daxa_u32) instances;
};

#define MESH_INSTANCE_FLAG_OPAQUE (1 << 0)
#define MESH_INSTANCE_FLAG_MASKED (1 << 1)

struct MeshInstance
{
    daxa_u32 entity_index;
    daxa_u32 mesh_index;
    daxa_u32 in_mesh_group_index;
    daxa_u32 mesh_group_index;
    daxa_u32 flags;
};
DAXA_DECL_BUFFER_PTR_ALIGN(MeshInstance, 4);


// The engine has one GPUMeshInstances buffer.
// This buffer is written from scratch every frame from cpu.
// It contains all meshes that are desired to be drawn that frame.
// If a mesh is not present here, it should not be drawn by any pass.
//
// This way we gain much of the convenience of an immediate mode renderer on the cpu.
// Code becomes simpler, retained mode would require more book keeping.
//
// We still keep all other things that may change slowly retained, such as scene manifests for meshes textures etc.
// We also keep the entity manifest persistent. This is important, as most entities will be static, uploading them every frame,
// Esp their matrices would be too expensive. It also allows us to have stable entity ids over multiple frames ON the gpu.
// This is very important for temporal techniques such as the visibility culling using last frames meshes or advanced taa subsample rejection.
// 
// We want the drawlists to already be massaged for easy consumption on the gpu.
// That means that we for example pre-sort mesh draws per pipeline permutation.
// This way we spend less time on the gpu doing complex sorting/ compacting.
// It also costs us nothing, as cpus are already very good for such workloads, it makes little sense to make the gpu sort when we can do it on the cpu easily.
// 
// Typically we want an array for a usecase/pass. For example the prepass is an array with one element per pipeline permutation.
// The vsm invalidate draw list for example does not need any permutations so it is just one drawlist.
// 
// If a usecase/ pass has multiple permutations, keep them as multiple drawlists in an array.
//
// The instances should be unique and the draw lists should index into the instance list.
// This way we can keep adding lots of data to MeshInstance and have instances in many draw lists without wasting too much memory.
struct MeshInstancesBufferHead
{   
    daxa_u32 count;
    daxa_BufferPtr(MeshInstance) instances;

    // For each pipeline permutation (eg. opaque vs masked) and for each persepctive/usecase (such as vsm vs prepass vs trans ...)
    // we have draw lists.
    // Draw lists may overlap. This makes code in each path simpler.
    // For example in the pre-pass we only need to have a list per pipeline permutation, no need to care about all the different lists, 
    // we simply pick the pre-pass lists and carry on.
    // This generally also helps keeping the cose less coupled.
    MeshDrawList prepass_draw_lists[PREPASS_DRAW_LIST_TYPE_COUNT];
    // Contains all meshes that changed over frames, eg streamed in this frame or moved.
    MeshDrawList vsm_invalidate_draw_list;
}; 
DAXA_DECL_BUFFER_PTR_ALIGN(MeshInstancesBufferHead, 8)

#if defined(__cplusplus)
#include <span>
inline void fill_draw_list_buffer_head(daxa::DeviceAddress address, uint8_t* host_address, std::array<std::span<MeshInstance const>, 2> prepass_draw_lists)
{
    MeshInstancesBufferHead ret = {};
    auto device_address_back_offset = address + sizeof(MeshInstancesBufferHead);
    ret.instances = device_address_back_offset;
    device_address_back_offset += sizeof(MeshInstance) * MAX_MESH_INSTANCES;
    for (uint32_t draw_list = 0; draw_list < PREPASS_DRAW_LIST_TYPE_COUNT; ++draw_list)
    {
        ret.prepass_draw_lists[draw_list].instances = device_address_back_offset;
        device_address_back_offset += sizeof(daxa_u32) * MAX_MESH_INSTANCES;
        for (uint32_t element = 0; element < prepass_draw_lists[draw_list].size(); ++element)
        {
            uint32_t mesh_instance_index = ret.count++;
            uint32_t draw_list_element_index = ret.prepass_draw_lists[draw_list].count++;
            MeshInstance mesh_instance = (prepass_draw_lists[draw_list])[element];
            mesh_instance.flags = mesh_instance.flags | (draw_list == PREPASS_DRAW_LIST_OPAQUE ? MESH_INSTANCE_FLAG_OPAQUE : 0);
            reinterpret_cast<MeshInstance*>(host_address + sizeof(MeshInstancesBufferHead))[mesh_instance_index] = mesh_instance;
            reinterpret_cast<uint32_t*>(
                host_address + sizeof(MeshInstancesBufferHead) + 
                sizeof(MeshInstance) * MAX_MESH_INSTANCES + 
                sizeof(uint32_t) * MAX_MESH_INSTANCES * draw_list)[element] = mesh_instance_index;
        }
    }
    *reinterpret_cast<MeshInstancesBufferHead*>(host_address) = ret;
}
inline auto get_opaque_draw_list_buffer_size() -> daxa::usize
{
    return 
        sizeof(MeshInstancesBufferHead) + 
        sizeof(MeshInstance) * MAX_MESH_INSTANCES +
        sizeof(uint32_t) * MAX_MESH_INSTANCES * 2;
}
#endif // #if defined(__cplusplus)

/// NOTE: In the future we want a TransparentMeshDrawListBufferHead, that has a much larger array for custom material permutations.

#if defined(DAXA_SHADER)
#if (DAXA_LANGUAGE == DAXA_LANGUAGE_GLSL)

DAXA_DECL_BUFFER_REFERENCE_ALIGN(4) SFPMBitfieldRef
{
    daxa_u32 entity_to_meshlist_offsets[MAX_ENTITIES];
    daxa_u32 dynamic_offset;
    daxa_u32 dynamic_section[];
};

#else

#define SFPMBitfieldRef SFPMMeshletBitfieldBuffer*

struct SFPMMeshletBitfieldBuffer
{
    daxa_u32 entity_to_meshlist_offsets[MAX_ENTITIES];
    daxa_u32 dynamic_offset;
    daxa_u32 dynamic_section[1];
};

#endif 
#endif // #if !defined(__cplusplus)


#define FIRST_PASS_MESHLET_BITFIELD_OFFSET_SECTION_START (MAX_ENTITIES + 1)
#define FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID (0u)
#define FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED (~0u ^ 1u)
#define FIRST_PASS_MESHLET_BITFIELD_OFFSET_DEBUG (~0u ^ 2u)

#define FIRST_OPAQUE_PASS_BITFIELD_ARENA_BASE_OFFSET MAX_ENTITIES
#define FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_SIZE (1u<<21u)
#define FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_DYNAMIC_SIZE (FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_SIZE - FIRST_OPAQUE_PASS_BITFIELD_ARENA_BASE_OFFSET)

/// --- Mesh Instance Draw List End ---

/// --- Analyze Visbuffer Results Begin ---

// TODO: Convert into buffer head.
#if !defined(__cplusplus)
struct VisibleMeshletList
{
    daxa_u32 count;
    daxa_u32vec3 padd;
    daxa_u32 meshlet_ids[1];
};
DAXA_DECL_BUFFER_PTR(VisibleMeshletList)
#endif

struct VisibleMeshesList
{
    daxa_u32 count;
    daxa_u32vec3 padd;
    daxa_u32 mesh_ids[MAX_MESH_INSTANCES];
};
DAXA_DECL_BUFFER_PTR(VisibleMeshesList)

/// --- Analyze Visbuffer Results End ---