#include <daxa/daxa.inl>
#include "gen_hiz.inl"

#define TILE_WIDTH 64
#define TILE_WIDTH_LOG2 6
#define TILE_SIZE int2(TILE_WIDTH,TILE_WIDTH)
#define TILE_THREAD_WIDTH 16
#define TILE_THREAD_COUNT (TILE_THREAD_WIDTH * TILE_THREAD_WIDTH)

[[vk::push_constant]] GenHizPush2 push;

func make_gather_uv(float2 inv_size, uint2 top_left_index) -> float2
{
    return (float2(top_left_index) + 1.0f) * inv_size;
}

groupshared float s_mins[2 * TILE_THREAD_WIDTH * TILE_THREAD_WIDTH];

func downsample_tile_mip_n0(int start_mip, uint2 tile, uint2 tile_thread) -> float4
{
    float4 quad_values = {0.0f,0.0f,0.0f,0.0f};
    [[unroll]]
    for (uint quad_i = 0; quad_i < 4; ++quad_i)
    {
        const int2 sub_i = int2(quad_i >> 1, quad_i & 1);
        const int2 dst_index = int2((tile * TILE_THREAD_WIDTH + tile_thread) * 2 + sub_i);
        const int dst_mip = start_mip + 1;
        const int2 src_index = dst_index * 2;
        const int src_mip = start_mip;
        float4 fetch;
        const bool base_case = src_mip == -1;
        if (base_case)
        {
            const float2 src_inv_size = rcp(float2(push.data.dst_mip0_size) * 2.0f);
            fetch = push.data.attach.src.get().GatherRed(push.data.attach.globals.samplers.linear_clamp.get(), make_gather_uv(src_inv_size, src_index), 0);
        }
        else
        {
            const uint2 src_max_index = (push.data.dst_mip0_size >> start_mip) - 1;
            fetch.x = push.data.attach.hiz[src_mip].get_coherent().Load(min(src_index + int2(0,0), src_max_index));
            fetch.y = push.data.attach.hiz[src_mip].get_coherent().Load(min(src_index + int2(0,1), src_max_index));
            fetch.z = push.data.attach.hiz[src_mip].get_coherent().Load(min(src_index + int2(1,0), src_max_index));
            fetch.w = push.data.attach.hiz[src_mip].get_coherent().Load(min(src_index + int2(1,1), src_max_index));
        }
        const float min_v = min(min(fetch.x, fetch.y), min(fetch.z, fetch.w));

        const uint2 dst_mip_size = (push.data.dst_mip0_size >> dst_mip);
        if (all(lessThan(dst_index, dst_mip_size)))
        {
            push.data.attach.hiz[dst_mip].get()[dst_index] = min_v;
        }
        quad_values[quad_i] = min_v;
    }
    return quad_values;
}

func downsample_tile_mip_n1(int start_mip, uint2 tile, uint2 tile_thread, float4 quad_values)
{
    const int dst_mip = start_mip + 2;
    // Uniform return.
    if (dst_mip >= push.data.mip_count) return;
    const float min_v = min(min(quad_values.x, quad_values.y), min(quad_values.z, quad_values.w));
    const int2 dst_index = int2(tile * TILE_THREAD_WIDTH + tile_thread);
    
    const uint2 dst_mip_size = (push.data.dst_mip0_size >> dst_mip);
    if (all(lessThan(dst_index, dst_mip_size)))
    {
        push.data.attach.hiz[dst_mip].get()[dst_index] = min_v;
    }

    // At this point each thread has downsampled a 4x4 area.
    // Now each thread has to share its result with the rest of the workgroup.
    s_mins[tile_thread.y * TILE_THREAD_WIDTH + tile_thread.x] = min_v;
}

func downsample_tile_mip_n2_n5(int start_mip, uint2 tile, uint2 tile_thread, int mip_i)
{
    const int2 read_mip_tile_corner_index   = tile * TILE_SIZE;
    const uint dst_mip = start_mip + 3 + mip_i;
    const uint ping_pong_src_index = (mip_i & 1u);
    const uint ping_pong_dst_index = ((mip_i+1) & 1u);

    GroupMemoryBarrierWithGroupSync();

    // We immediately want half the thread indices on each axis, so 1/4th of threads.
    const uint2 remaining_tile_size = uint2(1,1) * (TILE_THREAD_WIDTH >> (mip_i + 1));

    // As the loop progesses, the tile shrinks as mips get smaller.
    // We need to deactivate threads with an if here.
    // IMPORTANT: We can not early out threads in the main kernel!
    //            Because we use GroupMemoryBarrierWithGroupSync(), 
    //            ALL workgroup threads MUST enter this function every iteration.
    const bool thread_active = all(lessThan(tile_thread, remaining_tile_size));
    if (thread_active)
    {
        const int2 dst_index = tile_thread + tile * remaining_tile_size;
        const int2 in_tile_src_index = tile_thread * 2;
        const float4 fetch = float4(
            s_mins[ping_pong_src_index * TILE_THREAD_COUNT + (in_tile_src_index.y + 0) * TILE_THREAD_WIDTH + (in_tile_src_index.x + 1)],
            s_mins[ping_pong_src_index * TILE_THREAD_COUNT + (in_tile_src_index.y + 1) * TILE_THREAD_WIDTH + (in_tile_src_index.x + 1)],
            s_mins[ping_pong_src_index * TILE_THREAD_COUNT + (in_tile_src_index.y + 0) * TILE_THREAD_WIDTH + (in_tile_src_index.x + 0)],
            s_mins[ping_pong_src_index * TILE_THREAD_COUNT + (in_tile_src_index.y + 1) * TILE_THREAD_WIDTH + (in_tile_src_index.x + 0)]
        );
        const float min_v = min(min(fetch.x,fetch.y), min(fetch.z,fetch.w));
        s_mins[((ping_pong_src_index + 1) & 1) * TILE_THREAD_COUNT + tile_thread.y * TILE_THREAD_WIDTH + tile_thread.x] = min_v;

        const uint2 dst_mip_size = (push.data.dst_mip0_size >> dst_mip);
        if (all(lessThan(dst_index, dst_mip_size)))
        {
            // The 6th mip is a special case as this mip level will be read by following workgroups IN THE SAME DISPATCH.
            // This means we need coherent reads, as otherwise the writes would not be visible between invocations.
            if (dst_mip == 5 || dst_mip == 11)
            {
                push.data.attach.hiz[dst_mip].get_coherent()[dst_index] = min_v;
            }
            else
            {
                push.data.attach.hiz[dst_mip].get()[dst_index] = min_v;
            }
        }
    }
}

func downsample_tile(int start_mip, int2 tile, int2 in_tile_thread_index)
{
    int dst_mip;
    
    dst_mip = start_mip + 1;
    if (dst_mip >= push.data.mip_count) return;
    float4 mip0_values = 
        downsample_tile_mip_n0(start_mip, tile, in_tile_thread_index);

    dst_mip = start_mip + 2;
    if (dst_mip >= push.data.mip_count) return;
    downsample_tile_mip_n1(start_mip, tile, in_tile_thread_index, mip0_values);

    for (int mip_i = 0; mip_i < 4; ++mip_i)
    {
        dst_mip = start_mip + 3 + mip_i;
        if (dst_mip >= push.data.mip_count) return;

        downsample_tile_mip_n2_n5(start_mip, tile, in_tile_thread_index, mip_i);
    }
}

groupshared bool s_is_last_workgroup;

[shader("compute")]
[numthreads(GEN_HIZ_X, GEN_HIZ_Y, 1)]
void entry_gen_hiz(
    uint2 dtid : SV_DispatchThreadID,
    uint2 gid : SV_GroupID,
    uint2 gtid : SV_GroupThreadID
)
{
    downsample_tile(-1, gid, gtid);

    if (all(equal(gtid, uint2(0,0))))
    {
        uint previously_finished_workgroups = 0;
        InterlockedAdd(*push.data.workgroup_finish_counter, 1, previously_finished_workgroups);
        const uint total_finished_workgroups = previously_finished_workgroups + 1;
        s_is_last_workgroup = total_finished_workgroups == push.data.total_workgroup_count;
    }

    GroupMemoryBarrierWithGroupSync();

    if (s_is_last_workgroup)
    {
        // Last workgroup potentially processes multiple tiles now.
        // It has to do this when the 5th mip size is larger then the window the downsample_tile function can process (64x64).
        uint2 mip5_size = push.data.dst_mip0_size / 64;
        uint2 mip5_tile_count = max((mip5_size >> 5), uint2(1,1));
        for (uint tile_y = 0; tile_y < mip5_tile_count.y; ++tile_y)
        {
            for (uint tile_x = 0; tile_x < mip5_tile_count.x; ++tile_x)
            {
                uint2 tile = uint2(tile_x, tile_y);
                downsample_tile(5, tile, gtid);
            }
        }
        
        GroupMemoryBarrierWithGroupSync();

        if (any(greaterThan(mip5_tile_count, uint2(1,1))))
        {
            downsample_tile(11, uint2(0,0), gtid);
        }
    }
}