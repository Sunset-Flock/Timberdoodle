#include <daxa/daxa.inl>

#include "write_swapchain.inl"

#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/depth_util.glsl"

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

#define USE_OLD_AGX_APPROX 0

#if USE_OLD_AGX_APPROX
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

#else

float3 agxDefaultContrastApproximation(float3 x)
{
	float3 x2 = x * x;
	float3 x4 = x2 * x2;
	return +15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
void agxLook(inout float3 color)
{
    const float3 slope      = (1.0f).xxx;
    const float3 power      = (1.1f).xxx;
    const float saturation  = 1.1;
	float luma = luminance_from_col(color);
	color = pow(color * slope, power);
	color = max(luma + saturation * (color - luma), float3(0.0));
}
float3 agx_tonemapping(float3 color)
{
	// AgX constants
	const float3x3 AgXInsetMatrix = transpose(float3x3(
		float3(0.856627153315983, 0.137318972929847, 0.11189821299995),
		float3(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
		float3(0.0482516061458583, 0.101439036467562, 0.811302368396859)));
	// explicit AgXOutsetMatrix generated from Filaments AgXOutsetMatrixInv
	const float3x3 AgXOutsetMatrix = transpose(float3x3(
		float3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
		float3(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
		float3(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405)));
	const float3x3 LINEAR_REC2020_TO_LINEAR_SRGB = transpose(float3x3(
		float3(1.6605, -0.1246, -0.0182),
		float3(-0.5876, 1.1329, -0.1006),
		float3(-0.0728, -0.0083, 1.1187)));
	const float3x3 LINEAR_SRGB_TO_LINEAR_REC2020 = transpose(float3x3(
		float3(0.6274, 0.0691, 0.0164),
		float3(0.3293, 0.9195, 0.0880),
		float3(0.0433, 0.0113, 0.8956)));
	// LOG2_MIN      = -10.0
	// LOG2_MAX      =  +6.5
	// MIDDLE_GRAY   =  0.18
	const float AgxMinEv = -12.47393; // log2( pow( 2, LOG2_MIN ) * MIDDLE_GRAY )
	const float AgxMaxEv = 4.026069;  // log2( pow( 2, LOG2_MAX ) * MIDDLE_GRAY )
	color = mul(LINEAR_SRGB_TO_LINEAR_REC2020, color);
	color = mul(AgXInsetMatrix, color);
	// Log2 encoding
	color = max(color, 1e-10); // avoid 0 or negative numbers for log2
	color = log2(color);
	color = (color - AgxMinEv) / (AgxMaxEv - AgxMinEv);
	color = clamp(color, 0.0, 1.0);
	// Apply sigmoid
	color = agxDefaultContrastApproximation(color);
	// Apply AgX look
	agxLook(color);
	color = mul(AgXOutsetMatrix, color);
	// Linearize
	color = pow(max(float3(0.0), color), float3(2.2));
	color = mul(LINEAR_REC2020_TO_LINEAR_SRGB, color);
	// Gamut mapping. Simple clamp for now.
	color = clamp(color, 0.0, 1.0);
	return color;
}
#endif

float sRGB_OETF(float a) {
    return select(.0031308f >= a, 12.92f * a, 1.055f * pow(a, .4166666666666667f) - .055f);
}

float3 sRGB_OETF(float3 a) {
    return float3(sRGB_OETF(a.r), sRGB_OETF(a.g), sRGB_OETF(a.b));
}

float3 rotate_ldr_color(float3 v, float extra_rotation = 0.0f)
{
    return cos((v + extra_rotation) * 3.14);
}

func check_mark_selected(Texture2D<float> selecetd_mark, int2 size, int2 index, out bool self_marked, out bool mark_border)
{
    self_marked = selecetd_mark[index] > 0.0f;

    let BORDER = 3;
    bool surrounding_area_at_least_one_marked = false;
    for (int x = -2; x <= 2; ++x)
    {
        for (int y = -2; y <= 2; ++y)
        {
            if (length(float2(x,y)) > 2.5) continue;

            int2 sample_index = clamp(index + int2(x,y), int2(0,0), size-1);
            let marked = selecetd_mark[sample_index] > 0.0f;
            surrounding_area_at_least_one_marked = surrounding_area_at_least_one_marked || marked;
        }
    }

    mark_border = !self_marked && surrounding_area_at_least_one_marked;
}

[[vk::push_constant]] WriteSwapchainPush normal_push;
[numthreads(WRITE_SWAPCHAIN_WG_X,WRITE_SWAPCHAIN_WG_Y,1)]
void entry_write_swapchain(uint2 index : SV_DispatchThreadID)
{
    let push = normal_push;
    if (any(greaterThanEqual(index, push.size)))
    {
        return;
    }
    float3 color = float3(0,0,0);
    bool mark_selected = false;
    bool mark_selected_border = false;
    if (deref(push.attachments.globals).settings.anti_aliasing_mode == AA_MODE_SUPER_SAMPLE)
    {
        for (uint y = 0; y < 2; ++y)
        {
            for (uint x = 0; x < 2; ++x)
            {
                const float4 exposed_color = push.attachments.color_image.get()[index * 2 + int2(x,y)];
                const float3 tonemapped_color = agx_tonemapping(exposed_color.rgb);
                color += tonemapped_color * 0.25f;
                bool mark, mark_border;
                check_mark_selected(push.attachments.selected_mark_image.get(), push.size, index, mark, mark_border);
                mark_selected = mark_selected || mark;
                mark_selected_border = mark_selected_border || mark_border;
            }
        }
    }
    else
    {
        const float4 exposed_color = push.attachments.color_image.get()[index];
        const float3 tonemapped_color = agx_tonemapping(exposed_color.rgb);
        color = tonemapped_color;

        check_mark_selected(push.attachments.selected_mark_image.get(), push.size, index, mark_selected, mark_selected_border);
    }

    float4 debug_color = DEBUG;//push.attachments.debug_image.get()[index];
    color.rgb = mix(color.rgb, debug_color.rgb, debug_color.a);

    // Selected Mark Border:
    if (mark_selected_border)
    {
        color.rgb = lerp(color.rgb, rotate_ldr_color(color.rgb), 0.4f);
    }
    if (mark_selected)
    {
        let CHECKER_SIZE = 16;
        let checker2 = (index + push.attachments.globals.total_elapsed_us/40000) % (2*CHECKER_SIZE) > CHECKER_SIZE;
        let checker = checker2.x ^ checker2.y;
        let checker_boost = checker ? 0.3f : 0.0f;
        color.rgb = lerp(color.rgb, rotate_ldr_color(color.rgb, checker_boost), 0.1f);
    }

    // Crosshair:
    int crosshair_extent = 16;
    int crosshair_thickness = 2;
    if ((all(index > (push.size/2-(crosshair_extent/2)) && all(index < (push.size/2+(crosshair_extent/2))))) && (any(index > (push.size/2-(crosshair_thickness/2)) && index < (push.size/2+(crosshair_thickness/2)))))
    {
        color.rgb = rotate_ldr_color(color.rgb);
    }


    float3 gamma_correct = sRGB_OETF(color);

    push.attachments.swapchain.get()[index] = float4(gamma_correct.rgb, 1);
}


[[vk::push_constant]] WriteSwapchainDebugPush debug_push;
[numthreads(WRITE_SWAPCHAIN_WG_X,WRITE_SWAPCHAIN_WG_Y,1)]
void entry_write_swapchain_debug(uint2 index : SV_DispatchThreadID)
{
    let push = debug_push;
    if (any(greaterThanEqual(index, push.size)))
    {
        return;
    }

    float depth = push.attachments.depth_image.get()[index];
    float3 depth_color = unband_z_color(index.x, index.y, linearise_depth(push.attachments.globals.main_camera.near_plane, depth));

    float3 gamma_correct = sRGB_OETF(depth_color);
    push.attachments.swapchain.get()[index] = float4(gamma_correct, 1);
}