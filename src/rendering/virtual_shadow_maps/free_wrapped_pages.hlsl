#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

[[vk::push_constant]] FreeWrappedPagesH::AttachmentShaderBlob free_wrapped_pages_push;

[numthreads(VSM_PAGE_TABLE_RESOLUTION, 1, 1)]
[shader("compute")]
void main(uint3 svdtid : SV_DispatchThreadID)
{
    const int2 clear_offset = free_wrapped_pages_push.free_wrapped_pages_info[svdtid.z].clear_offset;
    const int3 vsm_page_coords = svdtid.xyz;
    if(vsm_page_coords.x > VSM_PAGE_TABLE_RESOLUTION) { return; }

    const bool should_clear_wrapped = 
        ((clear_offset.x > 0) && (vsm_page_coords.x < clear_offset.x)) || 
        ((clear_offset.x < 0) && (vsm_page_coords.x > VSM_PAGE_TABLE_RESOLUTION + (clear_offset.x - 1))) || 
        ((clear_offset.y > 0) && (vsm_page_coords.y < clear_offset.y)) || 
        ((clear_offset.y < 0) && (vsm_page_coords.y > VSM_PAGE_TABLE_RESOLUTION + (clear_offset.y - 1)));

    const bool enable_caching = free_wrapped_pages_push.globals.vsm_settings.enable_caching != 0u;
    const bool sun_moved = free_wrapped_pages_push.globals.vsm_settings.sun_moved != 0u;

    const int3 vsm_wrapped_page_coords = vsm_page_coords_to_wrapped_coords(vsm_page_coords, free_wrapped_pages_push.vsm_clip_projections);
    if(should_clear_wrapped || !enable_caching || sun_moved)
    {
        const uint vsm_page_entry = free_wrapped_pages_push.vsm_page_table.get()[vsm_wrapped_page_coords];
        if(get_is_allocated(vsm_page_entry))
        {
            const int2 meta_memory_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);
            free_wrapped_pages_push.vsm_meta_memory_table.get()[meta_memory_coords] = 0u;
            free_wrapped_pages_push.vsm_page_table.get()[vsm_wrapped_page_coords] = 0u;
        }
    }
}