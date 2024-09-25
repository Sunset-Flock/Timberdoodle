#include <daxa/daxa.inl>
#include "gen_hiz.inl"

static let TILE_SIZE = int2(64,64);

[[vk::push_constant]] GenHizPush2 push;

func make_gather_uv(float2 inv_size, uint2 top_left_index) -> float2
{
    return (float2(top_left_index) + 1.0f) * inv_size;
}

groupshared float s_mins[2][GEN_HIZ_Y][GEN_HIZ_X];

// Expects a workgroup of 16x16 threads.
// Each thread will initially downsample a 4x4 area by itself.
// After, all threads will use workgroup memory to downsample the rest.
func downsample_tile(int start_mip, int2 tile, int2 in_tile_thread_index)
{
    float2 read_mip_inf_size = float2(0.0f, 0.0f);
    int2 read_mip_max_index = int2(0, 0);
    int2 read_mip_tile_corner_index = in_tile_thread_index * TILE_SIZE;
    int2 dst_mip0_size = {512,512};
    if (start_mip == -1)
    {
        const uint2 src_size = dst_mip0_size * 2;
        read_mip_inf_size = rcp(float2(src_size));
        read_mip_max_index = src_size - 1;
    }
    else
    {
        read_mip_inf_size = rcp(float2(dst_mip0_size >> start_mip));
        read_mip_max_index = (dst_mip0_size >> start_mip) - 1;
    }

    // Downsample 4x4 section of src.
    {
        float4 quad_values = {0.0f,0.0f,0.0f,0.0f};
        [[unroll]]
        for (uint quad_i = 0; quad_i < 4; ++quad_i)
        {
            const int2 sub_i = int2(quad_i >> 1, quad_i & 1);
            const int2 dst_index = int2((tile * 16 + in_tile_thread_index) * 2 + sub_i);
            const int dst_mip = start_mip + 1;
            const int2 src_index = dst_index * 2;
            const int src_mip = start_mip;
            float4 fetch;
            if (src_mip == -1)
            {
                push.attach.src.get().GatherRed(push.attach.globals.samplers.linear_clamp.get(), make_gather_uv(read_mip_inf_size, src_index), 0);
            }
            else
            {
                fetch.x = push.attach.hiz[src_mip].get_coherent().Load(min(src_index + int2(0,0), read_mip_max_index));
                fetch.y = push.attach.hiz[src_mip].get_coherent().Load(min(src_index + int2(0,1), read_mip_max_index));
                fetch.z = push.attach.hiz[src_mip].get_coherent().Load(min(src_index + int2(1,0), read_mip_max_index));
                fetch.w = push.attach.hiz[src_mip].get_coherent().Load(min(src_index + int2(1,1), read_mip_max_index));
            }
            const float min_v = min(min(fetch.x, fetch.y), min(fetch.z, fetch.w));
            push.attach.hiz[dst_mip].get()[dst_index] = min_v;
            quad_values[quad_i] = min_v;
        }
        const int dst_mip = start_mip + 2;
        // Uniform return.
        if (dst_mip >= push.mip_count) return;
        const float min_v = min(min(quad_values.x, quad_values.y), min(quad_values.z, quad_values.w));
        const int2 dst_index = int2(tile * 16 + in_tile_thread_index);
        push.attach.hiz[dst_mip].get()[dst_index] = min_v;

        // At this point each thread has downsampled a 4x4 area.
        // min_v contains the result of this downsampling.
        // Now each thread has to share its result with the rest of the workgroup.
        s_mins[0][in_tile_thread_index.y][in_tile_thread_index.x] = min_v;
    }

    for (int mip_i = 0; mip_i < 5; ++mip_i)
    {
        // In order to avoid having to barrier twice for 1. writing and 2. reading the shared values,
        // We use two shared memory arrays and ping pong them.
        // This means we read array A, read array B in one loop and then switch/"ping pong" them the next iteration.
        const uint ping_pong_src_index = (mip_i & 1u);
        const uint ping_pong_dst_index = ((mip_i+1) & 1u);

        const int dst_mip = mip_i + 2 + start_mip;
        // Uniform return.
        if (dst_mip >= push.mip_count) return;

        GroupMemoryBarrierWithGroupSync();

        // As the loop progesses, the processed mip size is shrinking.
        // We need to deactivate threads each loop as the tile is getting smaller.
        // NOTE: WE CAN NOT RETURN HERE AS THAT WOULD BE NON UNIFORM!
        //       NON UNIFORM CONTROL FLOW IS ILLEGAL WHEN IT CONTAINS MEMORY BARRIERS LIKE THIS LOOP!
        const uint2 remaining_tile_size = uint2(GEN_HIZ_X >> mip_i, GEN_HIZ_Y >> mip_i);
        const bool thread_active = all(lessThan(in_tile_thread_index, remaining_tile_size));
        if (thread_active)
        {
            const int2 dst_size = read_mip_max_index >> dst_mip;
            const int2 dst_index = in_tile_thread_index + (read_mip_tile_corner_index >> dst_mip);
            const int2 src_index = dst_index * 2;
            const float4 fetch = float4(
                s_mins[ping_pong_src_index][src_index.y+0][src_index.x+0],
                s_mins[ping_pong_src_index][src_index.y+0][src_index.x+1],
                s_mins[ping_pong_src_index][src_index.y+1][src_index.x+0],
                s_mins[ping_pong_src_index][src_index.y+1][src_index.x+1]
            );
            const float min_v = min(min(fetch.x,fetch.y), min(fetch.z,fetch.w));
            s_mins[ping_pong_dst_index][in_tile_thread_index.y][in_tile_thread_index.x] = min_v;

            // The 6th mip is a special case as this mip level will be read by following workgroups IN THE SAME DISPATCH.
            // This means we need coherent reads, as otherwise the writes would not be visible between invocations.
            if (dst_mip == 6)
            {
                push.attach.hiz[dst_mip].get_coherent()[dst_index] = min_v;
            }
            else
            {
                push.attach.hiz[dst_mip].get()[dst_index] = min_v;
            }
        }
    }
}

groupshared bool s_is_last_workgroup;

[shader("compute")]
[numthreads(GEN_HIZ_X, GEN_HIZ_Y, 1)]
void entry_gen_hiz(
    uint2 gid : SV_GroupID,
    uint2 gtid : SV_GroupThreadID
)
{
    downsample_tile(-1, gid, gtid);

    if (all(equal(gtid, uint2(0,0))))
    {
        uint previously_finished_workgroups = 0;
        InterlockedAdd(*push.workgroup_finish_counter, 1, previously_finished_workgroups);
        const uint total_finished_workgroups = previously_finished_workgroups + 1;
        s_is_last_workgroup = total_finished_workgroups == push.total_workgroup_count;
    }

    GroupMemoryBarrierWithGroupSync();

    if (s_is_last_workgroup)
    {
        downsample_tile(6, gid, gtid);
    }
}