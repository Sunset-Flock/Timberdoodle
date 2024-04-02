#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

#if defined(DEBUG_PAGE_TABLE)
DAXA_DECL_PUSH_CONSTANT(DebugVirtualPageTableH, push)
layout(local_size_x = DEBUG_PAGE_TABLE_X_DISPATCH, local_size_y = DEBUG_PAGE_TABLE_Y_DISPATCH) in;
void main()
{
    if(all(lessThan(gl_GlobalInvocationID.xy, ivec2(VSM_PAGE_TABLE_RESOLUTION, VSM_PAGE_TABLE_RESOLUTION))))
    {
        const bool level_forced = deref(push.globals).vsm_settings.force_clip_level != 0;
        const int force_clip_level = level_forced ? deref(push.globals).vsm_settings.forced_clip_level : 0;

        const ivec3 page_entry_coords = ivec3(gl_GlobalInvocationID.xy, level_forced ? force_clip_level : 0);
        const uint page_entry = imageLoad(daxa_uimage2DArray(push.vsm_page_table), page_entry_coords).r;
        vec4 color = vec4(0.0, 0.0, 0.0, 1.0);

        if      (get_requests_allocation(page_entry)) { color = vec4(0.0, 0.0, 1.0, 1.0); }
        else if (get_is_allocated(page_entry))        { color = vec4(0.0, 1.0, 0.0, 1.0); }
        else if (get_allocation_failed(page_entry))   { color = vec4(1.0, 0.0, 0.0, 1.0); }
        else if (get_is_dirty(page_entry))            { color = vec4(0.0, 0.0, 1.0, 1.0); }

        if (get_is_visited_marked(page_entry))
        { 
            color.xyz = vec3(1.0, 1.0, 0.0);
        }

        if(color.x == 0 && color.y == 0 && color.z == 0) { return; }

        const ivec2 base_pix_pos = ivec2(gl_GlobalInvocationID.xy);
        imageStore(daxa_image2D(push.vsm_debug_page_table), base_pix_pos, color);
    }
}
#endif //DEBUG_PAGE_TABLE

#if defined(DEBUG_META_MEMORY_TABLE)
DAXA_DECL_PUSH_CONSTANT(DebugMetaMemoryTableH, push)
layout(local_size_x = DEBUG_META_MEMORY_TABLE_X_DISPATCH, local_size_y = DEBUG_META_MEMORY_TABLE_Y_DISPATCH) in;
void main()
{
    if(all(lessThan(gl_GlobalInvocationID.xy, ivec2(VSM_META_MEMORY_TABLE_RESOLUTION, VSM_META_MEMORY_TABLE_RESOLUTION))))
    {
        const uint meta_entry = imageLoad(daxa_uimage2D(push.vsm_meta_memory_table), ivec2(gl_GlobalInvocationID.xy)).r;
        vec4 color = vec4(0.0, 0.0, 0.0, 1.0);

        if (get_meta_memory_is_allocated(meta_entry)) { color = vec4(0.0, 1.0, 0.0, 1.0); }

        if (get_meta_memory_is_visited(meta_entry))
        { 
            color = vec4(1.0, 1.0, 0.0, 1.0);
            const uint is_visited_erased_entry = meta_entry & (~meta_memory_visited_mask()); 
            imageStore(daxa_uimage2D(push.vsm_meta_memory_table), ivec2(gl_GlobalInvocationID.xy), uvec4(is_visited_erased_entry));

            const ivec3 vsm_coords = get_vsm_coords_from_meta_entry(meta_entry);
            const uint vsm_entry = imageLoad(daxa_uimage2DArray(push.vsm_page_table), vsm_coords).r;
            const uint visited_reset_vsm_entry = vsm_entry & (~(visited_marked_mask()));
            imageStore(daxa_uimage2DArray(push.vsm_page_table), vsm_coords, uvec4(visited_reset_vsm_entry));
        }
        if(get_meta_memory_needs_clear(meta_entry)) { color += vec4(0.0, 0.0, 1.0, 1.0); }

        if(color.w != 0.0)
        {
            const daxa_i32vec2 shaded_pix_pos = ivec2(gl_GlobalInvocationID.xy);
            imageStore(daxa_image2D(push.vsm_debug_meta_memory_table), shaded_pix_pos, color);
        }
    }
}
#endif //DEBUG_META_MEMORY_TABLE
