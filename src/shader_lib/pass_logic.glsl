#pragma once

#include "daxa/daxa.inl"
#include "../shader_shared/shared.inl"
#include "../shader_shared/globals.inl"
#include "../shader_shared/geometry.inl"

uint get_meshlet_draw_count(
    daxa_BufferPtr(RenderGlobalData) globals,
    daxa_BufferPtr(MeshletInstancesBufferHead) meshlet_instances, 
    uint pass, 
    uint opaque_or_discard)
{
    return deref(meshlet_instances).prepass_draw_lists[opaque_or_discard].pass_counts[pass];
}

uint get_meshlet_instance_index(
    daxa_BufferPtr(RenderGlobalData) globals,
    daxa_BufferPtr(MeshletInstancesBufferHead) meshlet_instances, 
    uint pass, 
    uint draw_list_type, 
    uint draw_instance_index)
{
    const uint draw_list_offset = pass == VISBUF_SECOND_PASS ? deref(meshlet_instances).prepass_draw_lists[draw_list_type].pass_counts[0] : 0;
    
    const uint draw_list_index = draw_list_offset + draw_instance_index;    
    // Due to how mesh and task shaders are dispatched (in blocks of 32 to 32 * 1024 threads),
    // all overhanding threads must be detected and discarded.
    const uint draw_list_instance_count = deref(meshlet_instances).prepass_draw_lists[draw_list_type].pass_counts[0] + deref(meshlet_instances).prepass_draw_lists[draw_list_type].pass_counts[1];
    if (draw_list_index >= draw_list_instance_count)
    {
        return (~0u);
    }
    const uint meshlet_instance_index = deref_i(deref(meshlet_instances).prepass_draw_lists[draw_list_type].instances, draw_list_index);
    return meshlet_instance_index;
}