#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_lib/misc.hlsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

[[vk::push_constant]] DebugVirtualPageTableH::AttachmentShaderBlob debug_virtual_page_push;
[[vk::push_constant]] DebugMetaMemoryTableH::AttachmentShaderBlob debug_meta_page_push;

[numthreads(DEBUG_PAGE_TABLE_X_DISPATCH, DEBUG_PAGE_TABLE_Y_DISPATCH, 1)]
[shader("compute")]
void debug_virtual_main(uint3 svdtid : SV_DispatchThreadID)
{
    let push = debug_virtual_page_push;
    if(all(lessThan(svdtid.xy, int2(VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION))))
    {
        const bool level_forced = push.globals.vsm_settings.force_clip_level != 0;
        const int clip_level = level_forced ? push.globals.vsm_settings.force_clip_level : 0;

        const int3 page_entry_coords = int3(svdtid.xy, clip_level);
        const uint page_entry = push.vsm_page_table.get_formatted()[page_entry_coords];
        float4 color = float4(0.0f, 0.0f, 0.0f, 1.0f);

        const float hue = pow(float(page_entry_coords.z) / float(VSM_CLIP_LEVELS - 1), 0.5f);
        if(get_is_allocated(page_entry))        { color.rgb = hsv2rgb(float3(hue, 0.8f, 0.2f)); }
        if(get_is_dirty(page_entry))            { color.rgb = float3(1.0f); }
        if(get_allocation_failed(page_entry))   { color.rgb = float3(1.0f, 0.0f, 0.0f); }
        if(get_is_visited_marked(page_entry))   { color.rgb = hsv2rgb(float3(hue, 1.0f, 0.8f)); }

        push.vsm_debug_page_table.get()[svdtid.xy] = color;
    }
}

[numthreads(DEBUG_META_MEMORY_TABLE_X_DISPATCH, DEBUG_META_MEMORY_TABLE_Y_DISPATCH, 1)]
[shader("compute")]
void debug_meta_main(uint3 svdtid : SV_DispatchThreadID)
{
    let push = debug_meta_page_push;
    if(all(lessThan(svdtid.xy, int2(VSM_META_MEMORY_TABLE_RESOLUTION))))
    {
        const uint64_t meta_entry = push.vsm_meta_memory_table.get_formatted()[svdtid.xy];

        float4 color = float4(0.0f, 0.0f, 0.0f, 1.0f);
        if (get_meta_memory_is_allocated(meta_entry))
        {
            // point spot light
            if(get_meta_memory_is_point_spot_light(meta_entry))
            {
                const PointSpotLightCoords coords = get_vsm_point_spot_light_coords_from_meta_entry(meta_entry);
                if(push.globals.vsm_settings.enable_point_caching == 0)
                {
                    push.vsm_point_spot_page_table[coords.mip_level].get_formatted()[int3(coords.texel_coords, coords.array_layer_index)] = 0;
                    push.vsm_meta_memory_table.get_formatted()[svdtid.xy] = 0;
                }
                let entry = push.vsm_point_spot_page_table[coords.mip_level].get_formatted()[int3(coords.texel_coords, coords.array_layer_index)];
                let reset_entry = entry & (~dirty_mask());
                push.vsm_point_spot_page_table[coords.mip_level].get_formatted()[int3(coords.texel_coords, coords.array_layer_index)] = reset_entry;

                const bool is_point_light = coords.array_layer_index < VSM_SPOT_LIGHT_OFFSET;
                const uint point_light_index = coords.array_layer_index / 6;
                const uint spot_light_index = coords.array_layer_index - VSM_SPOT_LIGHT_OFFSET;

                const bool is_forced_point_light = is_point_light && (point_light_index == push.globals.vsm_settings.force_point_light_idx);
                const bool is_forced_spot_light = !is_point_light && (spot_light_index == push.globals.vsm_settings.force_spot_light_idx);
                if(is_forced_point_light || is_forced_spot_light)
                {
                    color.rgb = float3(0.8f, 0.8f, 1.0f) * 0.4f;
                }
                else 
                {
                    color.rgb = float3(0.03f);
                }
            }
            // Directional
            else
            {
                color.rgb = float3(0.4f);
                const bool point_light_forced = (push.globals.vsm_settings.force_point_light_idx != -1);
                const bool spot_light_forced = (push.globals.vsm_settings.force_spot_light_idx != -1);
                if(point_light_forced || spot_light_forced) 
                {
                    color.rgb = 0.03;
                }
            }
        }

        if(get_meta_memory_is_visited(meta_entry))
        {
            const uint64_t is_visited_erased_entry = meta_entry & (~meta_memory_visited_mask());
            push.vsm_meta_memory_table.get_formatted()[svdtid.xy] = is_visited_erased_entry;

            if(get_meta_memory_is_point_spot_light(meta_entry))
            {
                const PointSpotLightCoords coords = get_vsm_point_spot_light_coords_from_meta_entry(meta_entry);

                const bool is_point_light = coords.array_layer_index < VSM_SPOT_LIGHT_OFFSET;
                const uint point_light_index = coords.array_layer_index / 6;
                const uint spot_light_index = coords.array_layer_index - VSM_SPOT_LIGHT_OFFSET;

                const bool is_forced_point_light = is_point_light && (point_light_index == push.globals.vsm_settings.force_point_light_idx);
                const bool is_forced_spot_light = !is_point_light && (spot_light_index == push.globals.vsm_settings.force_spot_light_idx);
                if(is_forced_point_light || is_forced_spot_light)
                {
                    color.rgb = float3(0.8f, 0.8f, 1.0f);
                }

                const uint virtual_entry = push.vsm_point_spot_page_table[coords.mip_level].get_formatted()[int3(coords.texel_coords, coords.array_layer_index)];

                const uint reset_virtual_entry = virtual_entry & (~(visited_marked_mask() | dirty_mask() | requests_allocation_mask()));
                push.vsm_point_spot_page_table[coords.mip_level].get_formatted()[int3(coords.texel_coords, coords.array_layer_index)] = reset_virtual_entry;
            }
            else
            {
                color.rgb = float3(0.8f, 0.8f, 1.0f);
                const int3 virtual_page_table_coords = get_vsm_coords_from_meta_entry(meta_entry);
                const bool point_light_forced = (push.globals.vsm_settings.force_point_light_idx != -1);
                const bool spot_light_forced = (push.globals.vsm_settings.force_spot_light_idx != -1);
                if(point_light_forced || spot_light_forced) 
                {
                    color.rgb = 0.03;
                }

                const uint virtual_entry = push.vsm_page_table.get_formatted()[virtual_page_table_coords];
                const uint reset_virtual_entry = virtual_entry & (~(visited_marked_mask() | dirty_mask() | requests_allocation_mask()));
                push.vsm_page_table.get_formatted()[virtual_page_table_coords] = reset_virtual_entry;
            }
        }
        push.vsm_debug_meta_memory_table.get()[svdtid.xy] = color;
    }
}