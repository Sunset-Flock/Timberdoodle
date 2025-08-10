#include <daxa/daxa.inl>

#include "daxa_tg_debugger.inl"

[[vk::push_constant]] DebugTaskDrawDebugDisplayPush draw_debug_clone_push;

float3 hsv2rgb(float3 c) {
    float4 k = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * lerp(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}

float3 rainbow_maker(uint i)
{
    return (0.2987123 * float(i), 1.0f, 1.0f);
}
float3 rainbow_maker(int i)
{
    return (0.2987123 * float(i), 1.0f, 1.0f);
}

[shader("compute")]
[numthreads(DEBUG_DRAW_CLONE_X,DEBUG_DRAW_CLONE_Y,1)]
func entry_draw_debug_display(uint2 thread_index : SV_DispatchThreadID)
{
    let p = draw_debug_clone_push;

    if (any(thread_index >= p.src_size))
        return;

    float4 sample_color = float4(0,0,0,0);

    let readback_pixel = all(thread_index == p.mouse_over_index);

    switch (p.format)
    {
        case 0: 
        {
            var sample = RWTexture2D<float4>::get(p.src)[thread_index];
            if (readback_pixel)
            {
                ((float4*)p.readback_ptr)[p.readback_index * 2] = sample;
            }
            sample_color = float4((sample.rgb - p.float_min) * rcp(p.float_max - p.float_min), sample.a);
        }
        break;
        case 1: 
        {
            var sample = RWTexture2D<int4>::get(p.src)[thread_index];
            if (readback_pixel)
            {
                ((int4*)p.readback_ptr)[p.readback_index * 2] = sample;
            }
            if (p.rainbow_ints)
                sample_color = float4(rainbow_maker(sample.x), 1);
            else
                sample_color = float4((sample.rgb - p.int_min) * rcp(p.int_max - p.int_min), sample.a);
        }
        break;
        case 2: 
        {
            var sample = RWTexture2D<uint4>::get(p.src)[thread_index];
            if (readback_pixel)
            {
                ((uint4*)p.readback_ptr)[p.readback_index * 2] = sample;
            }
            if (p.rainbow_ints)
                sample_color = float4(rainbow_maker(sample.x), 1);
            else
                sample_color = float4((sample.rgb - p.uint_min) * rcp(p.uint_max - p.uint_min), sample.a);    
        }
        break;
    }
    
    let one_channel_active = (p.enabled_channels[0] + p.enabled_channels[1] + p.enabled_channels[2] + p.enabled_channels[3]) == 1;
    let only_alpha_active = one_channel_active && p.enabled_channels[3];

    if (only_alpha_active)
    {
        sample_color[3] = (sample_color[3] - p.float_min) * rcp(p.float_max - p.float_min);
    }

    sample_color[0] = p.enabled_channels[0] != 0 ? sample_color[0] : 0.0f;
    sample_color[1] = p.enabled_channels[1] != 0 ? sample_color[1] : 0.0f;
    sample_color[2] = p.enabled_channels[2] != 0 ? sample_color[2] : 0.0f;
    sample_color[3] = p.enabled_channels[3] != 0 ? sample_color[3] : 1.0f;

    if (one_channel_active)
    {
        let single_channel_color = 
            (p.enabled_channels[0] * sample_color[0]) + 
            (p.enabled_channels[1] * sample_color[1]) + 
            (p.enabled_channels[2] * sample_color[2]) + 
            (p.enabled_channels[3] * sample_color[3]);
        sample_color.xyz = single_channel_color;
        sample_color[3] = 1.0f;
    }

    if (readback_pixel)
    {
        p.readback_ptr[p.readback_index * 2 + 1] = sample_color;
        let color_max = max(max(sample_color.x, sample_color.y), max(sample_color.z, sample_color.w));
        let color_max_int = uint(color_max); 
        let color_min = max(max(sample_color.x, sample_color.y), max(sample_color.z, sample_color.w));
    }

    let previous_value = p.dst.get()[thread_index];
    p.dst.get()[thread_index] = float4(lerp(previous_value.rgb, sample_color.rgb, sample_color.a), 1.0f);
}