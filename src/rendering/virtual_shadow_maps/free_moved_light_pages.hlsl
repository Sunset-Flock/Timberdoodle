#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

[[vk::push_constant]] FreeMovedLightPagesPush free_moved_light_pages_push;

// Each thread owns one mip 0 page coordinate and frees that coordinate in every mip it still fits into.
[numthreads(FREE_MOVED_LIGHT_PAGES_X_DISPATCH, FREE_MOVED_LIGHT_PAGES_Y_DISPATCH, 1)]
[shader("compute")]
void main(uint3 svdtid : SV_DispatchThreadID)
{
    let push = free_moved_light_pages_push;
    const uint array_layer = push.first_array_layer + svdtid.z;

    for (uint mip = 0; mip < push.mip_count; ++mip)
    {
        const uint mip_resolution = VSM_POINT_SPOT_PAGE_TABLE_RESOLUTION >> mip;
        if (any(svdtid.xy >= mip_resolution)) { return; }

        const uint3 page_coords = uint3(svdtid.xy, array_layer);
        const uint vsm_page_entry = push.attachments.vsm_point_spot_page_table[mip].get_formatted()[page_coords];
        if (get_is_allocated(vsm_page_entry))
        {
            const int2 meta_memory_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);
            push.attachments.vsm_meta_memory_table.get_formatted()[meta_memory_coords] = 0u;
            push.attachments.vsm_point_spot_page_table[mip].get_formatted()[page_coords] = 0u;
        }
    }
}
