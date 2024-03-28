#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

DAXA_DECL_PUSH_CONSTANT(ClearDirtyBitH, push)
layout (local_size_x = CLEAR_DIRTY_BIT_X_DISPATCH) in;
void main()
{
    const int id = daxa_i32((gl_GlobalInvocationID.z * CLEAR_DIRTY_BIT_X_DISPATCH) + gl_LocalInvocationID.x);
    if(id >= deref(push.vsm_allocation_count).count) {return;}

    const ivec3 alloc_request_page_coords = deref_i(push.vsm_allocation_requests, id).coords;
    const uint vsm_page_entry = imageLoad(daxa_uimage2DArray(push.vsm_page_table), alloc_request_page_coords).r;
    const uint dirty_bit_reset_page_entry = vsm_page_entry & ~(dirty_mask());
    imageStore(daxa_uimage2DArray(push.vsm_page_table), alloc_request_page_coords, daxa_u32vec4(dirty_bit_reset_page_entry));
}