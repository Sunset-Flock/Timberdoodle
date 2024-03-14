#pragma once

#include "daxa/daxa.inl"
#include "../shader_shared/shared.inl"
#include "../shader_shared/globals.inl"
#include "../shader_shared/geometry.inl"

uint get_meshlet_draw_count(
    daxa_BufferPtr(ShaderGlobals) globals,
    daxa_BufferPtr(MeshletInstancesBufferHead) meshlet_instances, 
    uint pass, 
    uint opaque_or_discard)
{
    switch (pass)
    {
        case PASS0_DRAW_VISIBLE_LAST_FRAME: 
            return deref(meshlet_instances).draw_lists[opaque_or_discard].first_count;
        case PASS1_DRAW_POST_CULL: 
            return deref(meshlet_instances).draw_lists[opaque_or_discard].second_count;
        case PASS2_OBSERVER_DRAW_VISIBLE_LAST_FRAME: 
            return deref(meshlet_instances).draw_lists[opaque_or_discard].first_count;
        case PASS3_OBSERVER_DRAW_POST_CULLED: 
            return deref(meshlet_instances).draw_lists[opaque_or_discard].second_count;
        case PASS4_OBSERVER_DRAW_ALL: 
            return deref(meshlet_instances).draw_lists[opaque_or_discard].first_count + 
            deref(meshlet_instances).draw_lists[opaque_or_discard].second_count;
        default: return 0;
    }
}

uint get_meshlet_instance_index(
    daxa_BufferPtr(ShaderGlobals) globals,
    daxa_BufferPtr(MeshletInstancesBufferHead) meshlet_instances, 
    uint pass, 
    uint draw_list_type, 
    uint draw_instance_index)
{
    uint draw_list_offset = 0;
    switch (pass)
    {
        case PASS0_DRAW_VISIBLE_LAST_FRAME: 
            draw_list_offset = 0;
            break;
        case PASS1_DRAW_POST_CULL: 
            draw_list_offset = deref(meshlet_instances).draw_lists[draw_list_type].first_count;
            break;
        case PASS2_OBSERVER_DRAW_VISIBLE_LAST_FRAME: 
            draw_list_offset = 0;
            break;
        case PASS3_OBSERVER_DRAW_POST_CULLED: 
            draw_list_offset = deref(meshlet_instances).draw_lists[draw_list_type].first_count;
            break;
        case PASS4_OBSERVER_DRAW_ALL: 
            draw_list_offset = 0;
            break;
        default: return 0;
    }
    
    const uint draw_list_index = draw_list_offset + draw_instance_index;
    const uint meshlet_instance_index = deref_i(deref(meshlet_instances).draw_lists[draw_list_type].instances, draw_list_index);
    return meshlet_instance_index;
}