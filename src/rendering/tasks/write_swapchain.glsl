#include <daxa/daxa.inl>

#include "write_swapchain.inl"

#include "shader_lib/visbuffer.glsl"

vec3 index_to_color(uint index)
{
    return vec3(cos(index), cos(index * 2 + 1), cos(index * 3 + 2)) * 0.5 + 0.5;
}

const float SRGB_ALPHA = 0.055;

const mat3 SRGB_2_XYZ_MAT = mat3(
	0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041
);

const mat3 XYZ_2_SRGB_MAT = mat3(
	 3.2409699419,-1.5373831776,-0.4986107603,
    -0.9692436363, 1.8759675015, 0.0415550574,
	 0.0556300797,-0.2039769589, 1.0569715142
);

const mat3 AXG_TRANSFORM = mat3(
    0.842479062253094 , 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772 , 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104
);

const mat3 INV_AGX_TRANSFORM = mat3(
     1.19687900512017  , -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368,  1.15190312990417  , -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433,  1.15107367264116
);

float luminance_from_col(vec3 color) 
{
    vec3 luminance_coefficients = SRGB_2_XYZ_MAT[1];
    return dot(color, luminance_coefficients);
}

vec3 agx_default_contrast_approximation(vec3 x) 
{
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
  
    return + 15.5     * x4 * x2
           - 40.14    * x4 * x
           + 31.96    * x4
           - 6.868    * x2 * x
           + 0.4298   * x2
           + 0.1191   * x
           - 0.00232;
}

void agx(inout vec3 color) 
{
    const float min_EV = -12.47393;
    const float max_EV = 4.026069;

    color = AXG_TRANSFORM * color;
    color = clamp(log2(color), min_EV, max_EV);
    color = (color - min_EV) / (max_EV - min_EV);
    color = agx_default_contrast_approximation(color);
}

void agx_eotf(inout vec3 color) 
{
    color = INV_AGX_TRANSFORM * color; 
}

void agx_look(inout vec3 color) 
{
    // Punchy
    const vec3 slope      = vec3(1.1);
    const vec3 power      = vec3(1.2);
    const float saturation = 1.3;

    float luma = luminance_from_col(color);
  
    color = pow(color * slope, power);
    color = max(vec3(0.0), luma + saturation * (color - luma));
}

vec3 agx_tonemapping(vec3 exposed_color)
{
    agx(exposed_color);
    agx_look(exposed_color);
    agx_eotf(exposed_color);
    return exposed_color;
}

DAXA_DECL_PUSH_CONSTANT(WriteSwapchainPush, push)
layout(local_size_x = WRITE_SWAPCHAIN_WG_X, local_size_y = WRITE_SWAPCHAIN_WG_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    if (index.x >= push.size.x || index.y >= push.size.y)
    {
        return;
    }
    vec3 color = vec3(0,0,0);
    if (deref(push.attachments.globals).settings.anti_aliasing_mode == AA_MODE_SUPER_SAMPLE)
    {
        for (uint y = 0; y < 2; ++y)
        {
            for (uint x = 0; x < 2; ++x)
            {
                const vec4 exposed_color = imageLoad(daxa_image2D(push.attachments.color_image), index * 2 + ivec2(x,y));
                const vec3 tonemapped_color = agx_tonemapping(exposed_color.rgb);
                color += tonemapped_color * 0.25f;
            }
        }
    }
    else
    {
        const vec4 exposed_color = imageLoad(daxa_image2D(push.attachments.color_image), index);
        const vec3 tonemapped_color = agx_tonemapping(exposed_color.rgb);
        color = tonemapped_color;
    }

    daxa_f32vec4 debug_color = texelFetch(daxa_texture2D(push.attachments.debug_image), index, 0);
    color.rgb = mix(color.rgb, debug_color.rgb, debug_color.a);

    vec3 gamma_correct = pow(color, vec3(1.0/2.2));

    imageStore(daxa_image2D(push.attachments.swapchain), index, vec4(gamma_correct.rgb, 1));
}