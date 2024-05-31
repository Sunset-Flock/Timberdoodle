#include "daxa/daxa.inl"
#include "shader_lib/aurora_util.glsl"

#include "aurora.inl"

#define PI 3.1415926535897932384626433832795

[[vk::push_constant]] AuroraBlurPush push;

groupshared float color_lds [BLUR_AURORA_IMAGE_WG + MAX_BLUR_RADIUS];

static float3x3 XYZ_TO_RGB = float3x3(
    2.3706743, -0.9000405, -0.4706338,
    -0.5138850,  1.4253036,  0.0885814,
    0.0052982, -0.0146949,  1.0093968
);

[numthreads(BLUR_AURORA_IMAGE_WG, 1, 1)]
[shader("compute")]
void main(
    uint3 svgid : SV_GroupID,
    uint3 svgtid : SV_GroupThreadID
)
{
#if defined(X_PASS)
    let kernel_width = push.uses.aurora_globals.rgb_blur_kernels[push.color_channel].width;
    let variation = push.uses.aurora_globals.rgb_blur_kernels[push.color_channel].variation;
    const int offset_kernel_width = kernel_width - 1;
    int group_x_offset = svgid.x * BLUR_AURORA_IMAGE_WG;
    let image_coords = int2(group_x_offset + svgtid.x, svgid.y);

    for(int wg_offset = 0; wg_offset < 2 * BLUR_AURORA_IMAGE_WG; wg_offset += BLUR_AURORA_IMAGE_WG)
    {
        let thread_x_image_coords = group_x_offset + wg_offset + svgtid.x - (offset_kernel_width / 2);
        let clamped_thread_x_image_coords = clamp(thread_x_image_coords, 0, push.uses.aurora_globals.aurora_image_resolution.x);

        if(svgtid.x + wg_offset < BLUR_AURORA_IMAGE_WG + offset_kernel_width)
        {
            float4 xyz_color_vec = RWTexture2D<float>::get(push.uses.color_image)[int2(clamped_thread_x_image_coords, svgid.y)];
            float3 rgb_color_vec = mul(XYZ_TO_RGB, xyz_color_vec.xyz);
            color_lds[svgtid.x + wg_offset] = rgb_color_vec[push.color_channel];
        }
    }
#else 
    let kernel_width = int(min(push.uses.aurora_globals.rgb_blur_kernels[push.color_channel].width * 3.0, 29));
    let variation = push.uses.aurora_globals.rgb_blur_kernels[push.color_channel].variation * 2.0;
    const int offset_kernel_width = kernel_width - 1;
    int group_y_offset = svgid.y * BLUR_AURORA_IMAGE_WG;
    let image_coords = int2(svgid.x, group_y_offset + svgtid.x);
    for(int wg_offset = 0; wg_offset < 2 * BLUR_AURORA_IMAGE_WG; wg_offset += BLUR_AURORA_IMAGE_WG)
    {
        let thread_y_image_coords = group_y_offset + wg_offset + svgtid.x - (offset_kernel_width / 2);
        let clamped_thread_y_image_coords = clamp(thread_y_image_coords, 0, push.uses.aurora_globals.aurora_image_resolution.y);

        if(svgtid.x + wg_offset < BLUR_AURORA_IMAGE_WG + offset_kernel_width)
        {
            float4 color_vec = RWTexture2D<float>::get(push.uses.blured_image)[int2(svgid.x, clamped_thread_y_image_coords)];
            color_lds[svgtid.x + wg_offset] = color_vec[push.color_channel];
        }
    }
#endif
    GroupMemoryBarrierWithGroupSync();
    float final_color = 0.0;
    float weight_sum = 0.0;
    for(int i = 0; i < kernel_width ; i++)
    {
        int kernel_i = i - offset_kernel_width / 2;
        float weight = (1.0 / (2.0 * PI * variation * variation)) * exp(-(kernel_i * kernel_i) / (2 * variation * variation));
        weight_sum += weight;
        final_color += weight * color_lds[svgtid.x + i];
    }

    if(all(lessThan(image_coords, push.uses.aurora_globals.aurora_image_resolution)))
    {
#if defined(X_PASS)
        float4 color_vec = RWTexture2D<float>::get(push.uses.color_image)[image_coords];
        color_vec[push.color_channel] = final_color / weight_sum;
        RWTexture2D<float>::get(push.uses.blured_image)[image_coords] = float4(color_vec.xyz, 1.0);
#else
        float4 color_vec = RWTexture2D<float>::get(push.uses.color_image)[image_coords];
        color_vec[push.color_channel] = final_color / weight_sum;
        RWTexture2D<float>::get( push.uses.color_image)[image_coords] = float4(color_vec.xyz, 1.0);
#endif
    }
}