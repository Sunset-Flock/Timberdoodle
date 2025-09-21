#include <daxa/daxa.inl>

#include "clouds.inl"

[[vk::push_constant]] ComposeCloudsPush compose_clouds_push;

float4 cubic(float v){
    float4 n = float4(1.0, 2.0, 3.0, 4.0) - v;
    float4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return float4(x, y, z, w) * (1.0/6.0);
}

[shader("compute")]
[numthreads(COMPOSE_CLOUDS_DISPATCH_X, COMPOSE_CLOUDS_DISPATCH_Y)]
func entry_compose(uint2 svdtid : SV_DispatchThreadID)
{
    let push = compose_clouds_push;

    if(all(lessThan(svdtid, push.main_screen_resolution))) 
    {
        const float2 pixel_index = svdtid.xy;

        const float2 inv_tex_size = push.attach.globals.view_camera.inv_screen_size * 2.0f;
        const float2 tex_size = push.attach.globals.view_camera.screen_size / 2.0f;
        const float2 uv = pixel_index / push.main_screen_resolution;
   
        const float2 tex_coords = uv * tex_size - 0.5;
   
        float2 fxy = fract(tex_coords);
        const float2 sub_tex_coords = tex_coords - fxy;

        float4 xcubic = cubic(fxy.x);
        float4 ycubic = cubic(fxy.y);

        float4 c = tex_coords.xxyy + float2(-0.5, +1.5).xyxy;
    
        float4 s = float4(xcubic.yz + xcubic.yw, ycubic.xz + ycubic.yw);
        float4 offset = c + float4 (xcubic.yw, ycubic.yw) / s;
    
        offset *= inv_tex_size.xxyy;
        float4 samples[4];
        float2 offsets[4] = {offset.xz, offset.yz, offset.xw, offset.yw}; 

        for (int sample = 0; sample < 4; ++sample)
        {
            samples[sample] = push.attach.clouds_raymarched_result.get().SampleLevel(
                    push.attach.globals.samplers.linear_clamp.get(),
                    offsets[sample],
                    0
            );
        }

        float sx = s.x / (s.x + s.y);
        float sy = s.z / (s.z + s.w);

        // const float4 result = mix( mix(samples[3], samples[2], sx), mix(samples[1], samples[0], sx) , sy);
        const float4 result = push.attach.clouds_raymarched_result.get().SampleLevel( push.attach.globals.samplers.linear_clamp.get(), uv, 0);

             
        const float3 color = push.attach.color_image.get()[uint2(pixel_index)];

        const float3 composed_color = color * (result.a) + (result.rgb);
        push.attach.color_image.get()[uint2(pixel_index)] = composed_color;
    }
}