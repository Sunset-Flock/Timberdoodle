#include <daxa.hlsl>
#include "gen_hiz.inl"

static let TILE_SIZE = uint2(128,128);
static let MINI_TILE_SIZE = uint2(8,8);

// Current Idea for 16 downsampling:
// Have two versions for downsampling
// - one for the bottom level, must be very optimized and fast, must be 64 -> 1.
// - one for upper levels, let one workgroup do all upper level downsamples in a loop.
//   - upper level downsampling should utalize as many threads as possible, downsample 64-> 4 to get more threads active.

[[vk::push_constant]] GenHizPush2 push;
// Tile = 128x128 pixel section.
// Group of 16x16 threads work on a tile.
// Each thread initially performs 8x8 fetches (4x4 Gather ops).
void downsample_tile(bool first_pass, uint2 tile_index, uint2 thread_index)
{
    // First each thread downsamples a mini tile, a 8x8 section.
    // THE MINI TILE SECTIONS DO NOT DIRECTLY NESSECARILY MAP TO PIXELS.
    // WHEN THE IMAGE IS NOT PO2 SIZED, THIS FUNCTION OVERSAMPLES PIXELS TO RESCALE THE IMAGE TO A POWER OF TWO.
    float downsampled_mip0[4][4];
    if (first_pass)
    {
        const uint2 po2_image_sample_index_start = tile_index * TILE_SIZE + thread_index * MINI_TILE_SIZE;
        const float2 po2_uv_stride = push.info.inv_src_image_po2_resolution;
        const float2 po2_gather_uv_stride = po2_uv_stride * 2;
        const float2 po2_gather_uv_start = float2(po2_image_sample_index_start + uint2(1,1) /*gather shift to be in between 4 texel*/) * push.info.inv_src_image_po2_resolution;

        // Downsample 16x16 minitile.
        for (uint y = 0; y < 4; ++y)
        {
            for (uint x = 0; x < 4; ++x)
            {
                const float2 uv = po2_gather_uv_start + po2_gather_uv_stride * float2(x,y);
                const float4 samples = push.at.src.get().GatherRed(SamplerState::get(push.at.globals.samplers.linear_clamp), uv);
                const float min_v = min(min(samples.x, samples.y), min(samples.z, samples.w));
                downsampled_mip0[x][y] = min_v;

                const uint2 out_po2_texel_index = tile_index * (TILE_SIZE/2) + thread_index * (MINI_TILE_SIZE/2) + uint2(x,y);
                push.at.mips[0].get()[uint3(out_po2_texel_index, 0)] = min_v;
            }
        }
    }        
    float downsampled_mip1[2][2];
    [unroll]
    for (uint y = 0; y < 2; ++y)
    {
        [unroll]
        for (uint x = 0; x < 2; ++x)
        {
            const float v0 = downsampled_mip0[x*2][y*2];
            const float v1 = downsampled_mip0[x*2+1][y*2];
            const float v2 = downsampled_mip0[x*2][y*2+1];
            const float v3 = downsampled_mip0[x*2+1][y*2+1];
            const float min_v = min(min(v0, v1), min(v2, v3));
            downsampled_mip1[x][y] = min_v;

            const uint2 out_po2_texel_index = tile_index * (TILE_SIZE/4) + thread_index * (MINI_TILE_SIZE/4) + uint2(x,y);
            push.at.mips[1].get()[uint3(out_po2_texel_index, 0)] = min_v;
        }
    }

    const float mip2_min_v = min(min(downsampled_mip1[0][0],downsampled_mip1[0][1]), min(downsampled_mip1[1][0], downsampled_mip1[1][1]));

    const uint2 out_po2_texel_index = tile_index * (TILE_SIZE/8) + thread_index * (MINI_TILE_SIZE/8);
    push.at.mips[2].get()[uint3(out_po2_texel_index, 0)] = mip2_min_v;
}

[shader("compute")]
[numthreads(GEN_HIZ_X, GEN_HIZ_Y, 1)]
void entry_gen_hiz(
    uint2 gid : SV_GroupID,
    uint2 gtid : SV_GroupThreadID
)
{
    downsample_tile(true, gid, gtid);
}