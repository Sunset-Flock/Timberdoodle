#pragma once

#include <daxa/daxa.inl>
#include "../shader_shared/shared.inl"
#include "../shader_shared/asset.inl"

uint observer_get_meshlet_instance_draw_count(daxa_BufferPtr(MeshletInstances) meshlet_instances)
{
    switch (globals.settings.observer_show_pass)
    {
        case 0: return deref(meshlet_instances).first_count;
        case 1: return deref(meshlet_instances).second_count;
        case 2: return deref(meshlet_instances).first_count + deref(meshlet_instances).second_count;
        default: return 0;
    }
}

uint observer_get_meshlet_instance_draw_offset(daxa_BufferPtr(MeshletInstances) meshlet_instances)
{
    switch (globals.settings.observer_show_pass)
    {
        case 0: return 0;
        case 1: return deref(meshlet_instances).first_count;
        case 2: return 0;
        default: return 0;
    }
}