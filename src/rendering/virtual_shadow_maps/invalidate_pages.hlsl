#include "daxa/daxa.inl"

#include "vsm.inl"
#include "shader_lib/vsm_util.glsl"
#include "shader_lib/cull_util.hlsl"

[[vk::push_constant]] InvalidatePagesH::AttachmentShaderBlob invalidate_pages_push;

[numthreads(INVALIDATE_PAGES_X_DISPATCH, 1, 1)]
[shader("compute")]
void main(uint3 svdtid : SV_DispatchThreadID)
{
    let push = invalidate_pages_push;
    if(svdtid.x < push.mesh_instances.vsm_invalidate_draw_list.count) 
    {
        const uint mesh_instance_idx = push.mesh_instances.vsm_invalidate_draw_list.instances[svdtid.x];
        let mesh_instance = push.mesh_instances.instances[mesh_instance_idx];
        let mesh = push.meshes[mesh_instance.mesh_index];

        let cascade_camera = push.vsm_clip_projections[svdtid.z].camera;
        let model_matrix = mat_4x3_to_4x4(push.entity_combined_transforms[mesh_instance.entity_index]);
        let ndcAABB = calculate_ndc_aabb(cascade_camera, model_matrix, mesh.aabb);

        const int2 pix_space_start = int2(floor(((ndcAABB.ndc_min.xy + 1.0f) * 0.5f) * VSM_PAGE_TABLE_RESOLUTION));
        const int2 pix_space_end = int2(ceil(((ndcAABB.ndc_max.xy + 1.0f) * 0.5f) * VSM_PAGE_TABLE_RESOLUTION));

        const int2 clamped_pix_space_start = clamp(pix_space_start, 0, VSM_PAGE_TABLE_RESOLUTION);
        const int2 clamped_pix_space_end = clamp(pix_space_end, 0, VSM_PAGE_TABLE_RESOLUTION);

        for(int y = clamped_pix_space_start.y; y < clamped_pix_space_end.y; ++y)
        {
            for(int x = clamped_pix_space_start.x; x < clamped_pix_space_end.x; ++x)
            {
                const int3 vsm_wrapped_page_coords = vsm_page_coords_to_wrapped_coords(int3(x, y, svdtid.z), push.vsm_clip_projections);
                const uint vsm_page_entry = push.vsm_page_table.get()[vsm_wrapped_page_coords].r;
                if(get_is_allocated(vsm_page_entry))
                {
                    const int linear_index = y * VSM_PAGE_TABLE_RESOLUTION + x;
                    const int index_offset = linear_index / 32;
                    const int in_uint_offset = linear_index % 32;
                    InterlockedOr(push.free_wrapped_pages_info[svdtid.z].mask[index_offset], 1u << in_uint_offset);

                    const int2 meta_memory_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);
                    InterlockedExchange(push.vsm_page_table.get()[vsm_wrapped_page_coords], uint(0));
                    InterlockedExchange(push.vsm_meta_memory_table.get()[meta_memory_coords], uint(0));
                }
            }
        }
    }
}