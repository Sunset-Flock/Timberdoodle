#include <daxa/daxa.inl>

#include "write_swapchain.inl"

#include "shader_lib/visbuffer.hlsl"

float3 index_to_color(uint index)
{
    return float3(cos(index), cos(index * 2 + 1), cos(index * 3 + 2)) * 0.5 + 0.5;
}

const float SRGB_ALPHA = 0.055;

static float4 DEBUG = (0.0f).xxxx;

#define SRGB_2_XYZ_MAT \
transpose(float3x3( \
	0.4124564, 0.3575761, 0.1804375, \
    0.2126729, 0.7151522, 0.0721750, \
    0.0193339, 0.1191920, 0.9503041 \
))

#define AXG_TRANSFORM \
 transpose(float3x3(\
    0.842479062253094 , 0.0423282422610123, 0.0423756549057051, \
    0.0784335999999992, 0.878468636469772 , 0.0784336, \
    0.0792237451477643, 0.0791661274605434, 0.879142973793104 \
))

#define INV_AGX_TRANSFORM \
transpose(float3x3( \
     1.19687900512017  , -0.0528968517574562, -0.0529716355144438, \
    -0.0980208811401368,  1.15190312990417  , -0.0980434501171241, \
    -0.0990297440797205, -0.0989611768448433,  1.15107367264116 \
))

#define XYZ_2_SRGB_MAT \
transpose(float3x3( \
	 3.2409699419,-1.5373831776,-0.4986107603, \
    -0.9692436363, 1.8759675015, 0.0415550574, \
	 0.0556300797,-0.2039769589, 1.0569715142 \
))

float luminance_from_col(float3 color) 
{
    float3 luminance_coefficients = transpose(SRGB_2_XYZ_MAT)[1];
    return dot(color, luminance_coefficients);
}

float3 agx_default_contrast_approximation(float3 x) 
{
    float3 x2 = x * x;
    float3 x4 = x2 * x2;
  
    return + 15.5     * x4 * x2
           - 40.14    * x4 * x
           + 31.96    * x4
           - 6.868    * x2 * x
           + 0.4298   * x2
           + 0.1191   * x
           - 0.00232;
}

void agx_eotf(inout float3 color) 
{
    color = mul(INV_AGX_TRANSFORM, color); 
}

void agx(inout float3 color) 
{
    const float min_EV = -12.47393;
    const float max_EV = 4.026069;

    color = mul(AXG_TRANSFORM, color);
    color = clamp(log2(color), min_EV, max_EV);
    color = (color - min_EV) / (max_EV - min_EV);
    color = agx_default_contrast_approximation(color);
}

void agx_look(inout float3 color) 
{
    // Punchy
    const float3 slope      = (1.1f).xxx;
    const float3 power      = (1.2f).xxx;
    const float saturation  = 1.3;

    float luma = luminance_from_col(color);
    
    color = pow(color * slope, power);
    color = max((0.0f).xxx, luma + saturation * (color - luma));
}

float3 agx_tonemapping(float3 exposed_color)
{
    agx(exposed_color);
    agx_look(exposed_color);
    agx_eotf(exposed_color);
    return exposed_color;
}

[[vk::push_constant]] WriteSwapchainPush push;
[numthreads(WRITE_SWAPCHAIN_WG_X,WRITE_SWAPCHAIN_WG_Y,1)]
void entry_write_swapchain(uint2 index : SV_DispatchThreadID)
{
    if (any(greaterThanEqual(index, push.size)))
    {
        return;
    }
    float3 color = float3(0,0,0);
    if (deref(push.attachments.globals).settings.anti_aliasing_mode == AA_MODE_SUPER_SAMPLE)
    {
        for (uint y = 0; y < 2; ++y)
        {
            for (uint x = 0; x < 2; ++x)
            {
                const float4 exposed_color = push.attachments.color_image.get()[index * 2 + int2(x,y)];
                const float3 tonemapped_color = agx_tonemapping(exposed_color.rgb);
                color += tonemapped_color * 0.25f;
            }
        }
    }
    else
    {
        const float4 exposed_color = push.attachments.color_image.get()[index];
        const float3 tonemapped_color = agx_tonemapping(exposed_color.rgb);
    
        color = tonemapped_color;
    }

    float4 debug_color = DEBUG;//push.attachments.debug_image.get()[index];
    color.rgb = mix(color.rgb, debug_color.rgb, debug_color.a);

    int crosshair_extent = 16;
    int crosshair_thickness = 2;

    if ((all(index > (push.size/2-(crosshair_extent/2)) && all(index < (push.size/2+(crosshair_extent/2))))) && (any(index > (push.size/2-(crosshair_thickness/2)) && index < (push.size/2+(crosshair_thickness/2)))))
    {
        color.rgb = fract(color.rgb + 0.5);
    }

    float3 gamma_correct = pow(color, float3(1.0/2.2));

    push.attachments.swapchain.get()[index] = float4(gamma_correct.rgb, 1);
}