#include <daxa/daxa.inl>

#include "analyze_visbuffer.inl"
#include "shader_shared/shared.inl"
#include "shader_shared/globals.inl"

#include "shader_lib/visbuffer.glsl"
#include "shader_lib/misc.hlsl"

func visible_triangle_mask_bit_from_tri_id(uint meshlet_triangle_index) -> uint
{
    #if MAX_TRIANGLES_PER_MESHLET > 64
        return 1u << (meshlet_triangle_index >> 2u);
    #elif MAX_TRIANGLES_PER_MESHLET > 32
        return 1u << (meshlet_triangle_index >> 1u);
    #else
        return 1u << meshlet_triangle_index;
    #endif
}

func update_tri_mask_and_write_visible_meshlet(uint meshlet_instance_index, uint meshlet_triangle_mask)
{
    uint prev_tri_mask;
    InterlockedOr(push.attach.meshlet_visibility_bitfield[meshlet_instance_index], meshlet_triangle_mask, prev_tri_mask);
    if (prev_tri_mask == 0)
    {
        // When the prev tri mask was zero, this is the first thread to see this meshlet.
        // The first thread to see a meshlet writes it to the visible meshlet list.
        uint visible_meshlets_index;
        InterlockedAdd(push.attach.visible_meshlets.count, 1, visible_meshlets_index);
        push.attach.visible_meshlets.meshlet_ids[visible_meshlets_index] = meshlet_instance_index;

        // Too slow, probably need extra compute pass scanning visible meshlets.
        // MeshletInstance meshlet_instance = push.attach.meshlet_instances.meshlets[meshlet_instance_index];
        // uint prev_mesh_mark;
        // InterlockedOr(push.attach.mesh_visibility_bitfield[meshlet_instance.mesh_instance_index], 1, prev_mesh_mark);
        // if (prev_mesh_mark == 0)
        // {
        //     uint visible_meshes_index;
        //     InterlockedAdd(push.attach.visible_meshes.count, 1, visible_meshes_index);
        //     push.attach.visible_meshes.mesh_ids[visible_meshes_index] = meshlet_instance.mesh_instance_index;
        // }
    }
}

[[vk::push_constant]] AnalyzeVisbufferPush2 push;
[numthreads(64,1,1)]
void main(uint2 group_id : SV_GroupID, uint in_group_id : SV_GroupThreadID)
{
    // Reswizzle threads into 8x8 blocks:
    let thread_index = group_id * 8 + uint2(in_group_id % 8, in_group_id / 8);
    let sampleIndex = thread_index << 1u;

    // Load list of 4 visbuffer ids. Filter out invalid ids. Duplicate Ids are filtered out later.
    let gather_uv = (float2(sampleIndex) + 1.0f) * push.inv_size;
    let tex = RWTexture2D<uint>::get(push.attach.visbuffer);                    // TODO: task attachment access generic to allow for sampled usage to allow for gather
    // let smpler = SamplerState::get(push.attach.globals.samplers.linear_clamp);
    uint4 vis_ids;
    if (push.attach.globals.settings.enable_atomic_visbuffer)
    {
        vis_ids[0] = uint(RWTexture2D<daxa::u64>::get(push.attach.visbuffer)[sampleIndex + uint2(0,0)]);
        vis_ids[1] = uint(RWTexture2D<daxa::u64>::get(push.attach.visbuffer)[sampleIndex + uint2(0,1)]);
        vis_ids[2] = uint(RWTexture2D<daxa::u64>::get(push.attach.visbuffer)[sampleIndex + uint2(1,0)]);
        vis_ids[3] = uint(RWTexture2D<daxa::u64>::get(push.attach.visbuffer)[sampleIndex + uint2(1,1)]);
    }
    else
    {
        vis_ids[0] = tex[sampleIndex + uint2(0,0)];
        vis_ids[1] = tex[sampleIndex + uint2(0,1)];
        vis_ids[2] = tex[sampleIndex + uint2(1,0)];
        vis_ids[3] = tex[sampleIndex + uint2(1,1)];
        // vis_ids = tex.GatherRed(smplr, gather_uv);   // TODO: task attachment access generic to allow for sampled usage to allow for gather
    }
    uint list_mask = 0;
    uint4 meshlet_triangle_masks = {0,0,0,0};
    uint4 meshlet_instance_indices = {0,0,0,0};
    [[unroll]] for (uint i = 0; i < 4; ++i)
    {
        meshlet_instance_indices[i] = TRIANGLE_ID_GET_MESHLET_INSTANCE_INDEX(vis_ids[i]);
        meshlet_triangle_masks[i] = visible_triangle_mask_bit_from_tri_id(TRIANGLE_ID_GET_MESHLET_TRIANGLE_INDEX(vis_ids[i]));
        bool list_entry_valid = true;
        list_entry_valid = list_entry_valid && (vis_ids[i] != INVALID_TRIANGLE_ID);
        list_entry_valid = list_entry_valid && (meshlet_instance_indices[i] < MAX_MESHLET_INSTANCES);
        list_mask = list_mask | (list_entry_valid ? (1u << i) : 0u);
    }

    // In most cases, there will be less then Wave_SIZE unique meshlets per Wave.
    // To take advantage of this, we make a unique list of ids, assigning a unique id to each thread in the Wave.
    // Later we perform a single atomic op in which only unique ids are used. This greatly reduces contention.
    
    uint assigned_meshlet_index = ~0;
    uint assigned_triangle_mask = 0;
    while (WaveActiveAnyTrue(assigned_meshlet_index == ~0) && WaveActiveAnyTrue(list_mask != 0))
    {
        // Each iteration, all the threads pick a candidate they want to vote for,
        // A single meshlet is chosen every iteration.
        let thread_active = list_mask != 0;
        const uint voting_candidate = thread_active ? meshlet_instance_indices[firstbitlow(list_mask)] : 0;
        let elected_meshlet_instance_index = WaveShuffle(voting_candidate, firstbitlow_uint4(WaveActiveBallot(thread_active)));

        // Now that a meshlet is voted for, we collect the triangles from all threads for that meshlet.
        // We also remove meshlets from each threads lists here.
        uint triangle_mask_contribution = 0;
        [[unroll]] for (uint i = 0; i < 4; ++i)
        {
            if (meshlet_instance_indices[i] == elected_meshlet_instance_index)
            {
                triangle_mask_contribution |= meshlet_triangle_masks[i];
                list_mask &= ~(1u << i);
            }
        }
        triangle_mask_contribution = WaveActiveBitOr(triangle_mask_contribution);

        // Now a single thread is choosen to pick this voted meshlet and its triangle mask to write out later.
        if ((assigned_meshlet_index == ~0) && WaveIsFirstLane())
        {
            assigned_meshlet_index = elected_meshlet_instance_index;
            assigned_triangle_mask = triangle_mask_contribution;
        }
    }    

    // Mark and write assigned meshlet.
    if (assigned_meshlet_index != ~0)
    {   
        update_tri_mask_and_write_visible_meshlet(assigned_meshlet_index, assigned_triangle_mask);
    }

    // When there are more unique meshlets per warp then warpsize, 
    // the thread needs to write out its remaining meshlet instance indices.
    // This is done more efficiently with a non scalarized loop.
    [[loop]] while (list_mask != 0)
    {
        let lsb = firstbitlow(list_mask);
        let meshlet_instance_index_index = meshlet_instance_indices[lsb];
        let meshlet_triangle_mask = meshlet_triangle_masks[lsb];
        list_mask &= ~(1 << lsb);
        update_tri_mask_and_write_visible_meshlet(meshlet_instance_index_index, meshlet_triangle_mask);
    }
}
