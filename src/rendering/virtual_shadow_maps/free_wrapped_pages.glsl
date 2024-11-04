#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

#extension GL_EXT_debug_printf : enable

DAXA_DECL_PUSH_CONSTANT(FreeWrappedPagesH, push)
layout (local_size_x = VSM_PAGE_TABLE_RESOLUTION) in;
void main()
{
    const ivec2 clear_offset = deref_i(push.free_wrapped_pages_info, gl_GlobalInvocationID.z).clear_offset;
    const ivec3 vsm_page_coords = ivec3(gl_LocalInvocationID.x, gl_GlobalInvocationID.y, gl_GlobalInvocationID.z);
    if(vsm_page_coords.x > VSM_PAGE_TABLE_RESOLUTION) { return; }

    const bool should_clear_wrapped = 
        (clear_offset.x > 0 && vsm_page_coords.x <  clear_offset.x) || 
        (clear_offset.x < 0 && vsm_page_coords.x >  VSM_PAGE_TABLE_RESOLUTION + (clear_offset.x - 1)) || 
        (clear_offset.y > 0 && vsm_page_coords.y <  clear_offset.y) || 
        (clear_offset.y < 0 && vsm_page_coords.y >  VSM_PAGE_TABLE_RESOLUTION + (clear_offset.y - 1));
    
    daxa_BufferPtr(FreeWrappedPagesInfo) info = push.free_wrapped_pages_info;

    const uint enable_caching = deref(push.globals).vsm_settings.enable_caching;
    const uint sun_moved = deref(push.globals).vsm_settings.sun_moved;

    const ivec3 vsm_wrapped_page_coords = vsm_page_coords_to_wrapped_coords(vsm_page_coords, push.vsm_clip_projections);

    if(should_clear_wrapped || enable_caching == 0u || sun_moved != 0u)
    {
        const uint vsm_page_entry = imageLoad(daxa_uimage2DArray(push.vsm_page_table), vsm_wrapped_page_coords).r;
        if(get_is_allocated(vsm_page_entry))
        {
            const ivec2 meta_memory_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);
            // debugPrintfEXT("freeing page %d %d %d\n", vsm_wrapped_page_coords.x, vsm_wrapped_page_coords.y, vsm_wrapped_page_coords.z);
            imageStore(daxa_uimage2D(push.vsm_meta_memory_table), meta_memory_coords, uvec4(0));
            imageStore(daxa_uimage2DArray(push.vsm_page_table), vsm_wrapped_page_coords, uvec4(0));
        } 
    }
}