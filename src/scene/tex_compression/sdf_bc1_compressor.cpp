#include "sdf_bc1_compressor.hpp"
/*
	16 bit scalar to BC1 encoder.  The main entrypoint is:
		void sBuildBC1Block(uint64 *outBCBlock, std::span<float> inInputData, f32vec3 inUnpackDot)

	To convert this to your codebase, the following will need to be replaced with your version:
		f32vec3 - 3D vector, like float3
		i32vec3 - 3D integer, like int3 or uint3
		DBG_ASSERT_TRUE_M
		std::span<type> - Like std::span
		std::array<type, integer size> - Fixed size array.  A C-style array on the stack, or std::array
*/


/**
@brief BC1 block definition; 64 bits
**/
struct BC1Block
{
	u64 mEndpoint0 : 16;		// Bytes 0-1
	u64 mEndpoint1 : 16;		// Bytes 2-3
	u64 mIndices : 32;		// Bytes 4-7
};
static_assert(sizeof(BC1Block) == sizeof(u64));



/**
@brief BC1 interpolation mode
**/
enum class BC1InterpolationMode : int
{
	FourIntermediate,
	ThreeIntermediateWithZero
};



/**
@brief Extend a 16-bit/RGB565 color to 24-bit/RGB888.  Replicates GPU behaviour.
**/
static i32vec3 sConvert565To888(u16 in565Color)
{
	u32 channel_r = (in565Color >> 11) & 31;
	u32 channel_g = (in565Color >> 5) & 63;
	u32 channel_b = (in565Color) & 31;
	DBG_ASSERT_TRUE_M((channel_r < 32) && (channel_g < 64) && (channel_b < 32), "Values outside defined range");
	return i32vec3(
		(channel_r << 3) | (channel_r >> 2),
		(channel_g << 2) | (channel_g >> 4),
		(channel_b << 3) | (channel_b >> 2));
}



/**
@brief Unpack an RGB888 to a scalar.
	Factored out into shared code, because doing the dotproduct and /255 in different orders causes slight differences...
**/
static float sUnpackRGB888ToScalar(i32vec3 inRGB888, f32vec3 inUnpackDot)
{
	f32vec3 color_unit = f32vec3(inRGB888) / 255.0f;
	float unpacked_value = glm::dot(color_unit, inUnpackDot);
	return unpacked_value;
}



/**
@brief Unpack a 16-bit/RGB565 color to a float.
**/
static float sEvaluateEndpointColor(u16 inColor, f32vec3 inUnpackDot)
{
	return sUnpackRGB888ToScalar(sConvert565To888(inColor), inUnpackDot);
}



/**
@brief Find the four interpolated color for a BC1 block, replicating the values sampled by AMD GPUs.
	Input endpoint colors are expected to be 16-bit RGB565.
	Output colors are RGB888.
	
	Based on https://fgiesen.wordpress.com/2021/10/04/gpu-bcn-decoding/
**/
static void sBuildInterpolatedColors_AMD(std::span<i32vec3> outColors, u16 inEndpointColor0, u16 inEndpointColor1)
{
	DBG_ASSERT_TRUE_M(outColors.size() == 4, "There always must be 4 interpolated colors");

	i32vec3 color_0 = sConvert565To888(inEndpointColor0);
	i32vec3 color_1 = sConvert565To888(inEndpointColor1);

	BC1InterpolationMode interpolation_mode = (inEndpointColor0 > inEndpointColor1) ? BC1InterpolationMode::FourIntermediate : BC1InterpolationMode::ThreeIntermediateWithZero;

	if (interpolation_mode == BC1InterpolationMode::FourIntermediate)
	{
		outColors[0] = color_0;
		outColors[1] = color_1;
		outColors[2] = ((color_0 * 43) + (color_1 * 21) + i32vec3(32, 32, 32)) / 64;
		outColors[3] = ((color_0 * 21) + (color_1 * 43) + i32vec3(32, 32, 32)) / 64;
	}
	else if (interpolation_mode == BC1InterpolationMode::ThreeIntermediateWithZero)
	{
		outColors[0] = color_0;
		outColors[1] = color_1;
		outColors[2] = (color_0 + color_1 + i32vec3(1, 1, 1)) / 2;
		outColors[3] = i32vec3(0, 0, 0);	// We're ignoring alpha
	}
}



/**
@brief Find the four interpolated color for a BC1 block, replicating the values sampled by nVidia GPUs.
	Input endpoint colors are expected to be 16-bit RGB565.
	Output colors are RGB888.

	Based on https://fgiesen.wordpress.com/2021/10/04/gpu-bcn-decoding/
**/
static void sBuildInterpolatedColors_nVidia(std::span<i32vec3> outColors, u16 inEndpointColor0, u16 inEndpointColor1)
{
	DBG_ASSERT_TRUE_M(outColors.size() == 4, "There always must be 4 interpolated colors");

	i32vec3 color_0 = sConvert565To888(inEndpointColor0);
	i32vec3 color_1 = sConvert565To888(inEndpointColor1);

	// Extract 5-bit red and blue colors from input RGB565
	u32 channel_0_r = (inEndpointColor0 >> 11) & 31;
	u32 channel_0_b = (inEndpointColor0) & 31;

	u32 channel_1_r = (inEndpointColor1 >> 11) & 31;
	u32 channel_1_b = (inEndpointColor1) & 31;

	BC1InterpolationMode interpolation_mode = (inEndpointColor0 > inEndpointColor1) ? BC1InterpolationMode::FourIntermediate : BC1InterpolationMode::ThreeIntermediateWithZero;

	if (interpolation_mode == BC1InterpolationMode::FourIntermediate)
	{
		outColors[0] = color_0;
		outColors[1] = color_1;

		// Red and  blue channels: input is 5-bit, output is 8 bit
		outColors[2].x = ((2 * channel_0_r + channel_1_r) * 22) >> 3;
		outColors[2].z = ((2 * channel_0_b + channel_1_b) * 22) >> 3;

		outColors[3].x = ((channel_0_r + 2 * channel_1_r) * 22) >> 3;
		outColors[3].z = ((channel_0_b + 2 * channel_1_b) * 22) >> 3;


		// Green channel: input is 8 bit
		int diff = color_1.y - color_0.y;	
		int scaled_diff = 80 * diff + (diff >> 2);
		outColors[2].y = color_0.y + ((128 + scaled_diff) >> 8);
		outColors[3].y = color_1.y + ((128 - scaled_diff) >> 8);
	}
	else if (interpolation_mode == BC1InterpolationMode::ThreeIntermediateWithZero)
	{
		outColors[0] = color_0;
		outColors[1] = color_1;

		// Red and  blue channels: input is 5-bit, output is 8 bit
		outColors[2].x = ((channel_0_r + channel_1_r) * 33) >> 3;
		outColors[2].z = ((channel_0_b + channel_1_b) * 33) >> 3;

		// Green channel: input is 8 bit
		int diff = color_1.y - color_0.y;
		int scaled_diff = 128 * diff + (diff >> 2);
		outColors[2].y = color_0.y + ((128 * scaled_diff) >> 8);

		outColors[3] = i32vec3(0, 0, 0);	// We're ignoring alpha
	}
}



/**
@brief Find the four interpolated color for a BC1 block, replicating the values sampled by Intel GPUs.
	Input endpoint colors are expected to be 16-bit RGB565.
	Output colors are RGB888.

	Based on https://fgiesen.wordpress.com/2021/10/04/gpu-bcn-decoding/
**/
static void sBuildInterpolatedColors_Intel(std::span<i32vec3> outColors, u16 inEndpointColor0, u16 inEndpointColor1)
{
	DBG_ASSERT_TRUE_M(outColors.size() == 4, "There always must be 4 interpolated colors");

	i32vec3 color_0 = sConvert565To888(inEndpointColor0);
	i32vec3 color_1 = sConvert565To888(inEndpointColor1);

	BC1InterpolationMode interpolation_mode = (inEndpointColor0 > inEndpointColor1) ? BC1InterpolationMode::FourIntermediate : BC1InterpolationMode::ThreeIntermediateWithZero;

	if (interpolation_mode == BC1InterpolationMode::FourIntermediate)
	{
		outColors[0] = color_0;
		outColors[1] = color_1;
		outColors[2] = ((color_0 * 171) + (color_1 * 85) + i32vec3(128, 128, 128)) / 256;
		outColors[3] = ((color_0 * 85) + (color_1 * 171) + i32vec3(128, 128, 128)) / 256;
	}
	else if (interpolation_mode == BC1InterpolationMode::ThreeIntermediateWithZero)
	{
		outColors[0] = color_0;
		outColors[1] = color_1;
		outColors[2] = ((color_0 * 128) + (color_1 * 128) + i32vec3(128, 128, 128)) / 256;
		outColors[3] = i32vec3(0, 0, 0);	// We're ignoring alpha
	}
}



/**
@brief Given a 4x4 list of input values, and a pair of endpoint colors, build a final 64-bit BC1 block
	Input values are expected to be in [0, 1].
	Outputs the error and the BC block
**/
static void sBuildBC1BlockAndEvaluateError(float * outNetError, u64 * outBCBlock,
	std::span<float> inInputData, u16 inEndpointColorLo, u16 inEndpointColorHi, f32vec3 inUnpackDot)
{
	DBG_ASSERT_TRUE_M(inInputData.size() == 16, "BC1 Block always compresses 4x4 region (16 values)");

	// Declare and initialize the block
	BC1Block block = {};

	// Set endpoints.  
	block.mEndpoint0 = std::max(inEndpointColorLo, inEndpointColorHi);
	block.mEndpoint1 = std::min(inEndpointColorLo, inEndpointColorHi);
	DBG_ASSERT_TRUE_M(block.mEndpoint0 > block.mEndpoint1, "Endpoint 0 is expected to be greater than 1 to select 4-color interpolation");

	// Find the four interpolated colors
	std::array<i32vec3, 4> interpolated_colors_amd;
	sBuildInterpolatedColors_AMD(interpolated_colors_amd, block.mEndpoint0, block.mEndpoint1);

	std::array<i32vec3, 4> interpolated_colors_nvidia;
	sBuildInterpolatedColors_nVidia(interpolated_colors_nvidia, block.mEndpoint0, block.mEndpoint1);

	std::array<i32vec3, 4> interpolated_colors_intel;
	sBuildInterpolatedColors_Intel(interpolated_colors_intel, block.mEndpoint0, block.mEndpoint1);

	// Find the four interpolated values.  Work with the maximum possible value, to ensure the GPU sampled value is <= the source value on all platforms
	std::array<float, 4> interpolated_values;
	for (int i = 0; i < 4; i++)
	{
		interpolated_values[i] = std::max(
			std::max(
				sUnpackRGB888ToScalar(interpolated_colors_amd[i], inUnpackDot),
				sUnpackRGB888ToScalar(interpolated_colors_nvidia[i], inUnpackDot)
			),
			sUnpackRGB888ToScalar(interpolated_colors_intel[i], inUnpackDot));
	}

	// Choose the 2-bit index for each of the 4x4 values
	float net_error = 0.0f;
	for (int source_index = 0; source_index < 16; source_index++)
	{
		float source_value = inInputData[source_index];

		float least_error = 1e10f;
		int best_interpolated_index = -1;
		for (int i = 0; i < 4; i++)
		{
			float delta = source_value - interpolated_values[i];
			if ((delta >= 0.0f) && (delta < least_error))
			{
				// Interpolated value is lower than source, and improves error
				least_error = delta;
				best_interpolated_index = i;
			}
		}

		DBG_ASSERT_TRUE_M(best_interpolated_index >= 0, "We have not found interpolated value that meets our requirements");

		net_error += least_error * least_error;		// Accumulate squared error

		block.mIndices |= u64(best_interpolated_index) << (source_index * 2);
	}
	*outNetError = net_error;
	*outBCBlock = *(u64*)&block;
}



/**
@brief Find the highest 16-bit endpoint color that's <= the input value.
	This will quickly find a close approximation, but it's possible that it's not the best.
	A lookup table may be better ...
**/
static u16 sFindEndpointColor(float inValue, f32vec3 inUnpackDot)
{
	DBG_ASSERT_TRUE_M((inUnpackDot.x > inUnpackDot.y) && (inUnpackDot.y > inUnpackDot.z), "Expecting red is most significant and blue is least.");	// The code below sets up the components in order from most to least significant.
	u16 endpoint_color = 0;
	
	for (int r_step = 16; r_step > 0; r_step /= 2)		// R is in [0, 31]
	{
		u16 hypothetical_endpoint_color = static_cast<u16>(endpoint_color + (r_step << 11));
		if (sEvaluateEndpointColor(hypothetical_endpoint_color, inUnpackDot) <= inValue)
			endpoint_color = hypothetical_endpoint_color;
	}
	for (int g_step = 32; g_step > 0; g_step /= 2)		// G is in [0, 63]
	{
		u16 hypothetical_endpoint_color = static_cast<u16>(endpoint_color + (g_step << 5));
		if (sEvaluateEndpointColor(hypothetical_endpoint_color, inUnpackDot) <= inValue)
			endpoint_color = hypothetical_endpoint_color;
	}
	for (int b_step = 16; b_step > 0; b_step /= 2)		// B is in [0, 31]
	{
		u16 hypothetical_endpoint_color = static_cast<u16>(endpoint_color + b_step);
		if (sEvaluateEndpointColor(hypothetical_endpoint_color, inUnpackDot) <= inValue)
			endpoint_color = hypothetical_endpoint_color;
	}
	return endpoint_color;
}



/**
@brief BC1 block compression entrypoint.  Input is a 4x4 set of values in [0, 1]; output is a 64-bit BC1 block.
		A good value for inUnpackDot is (0.96414679f, 0.03518212f, 0.00067109f).

	Searches different endpoint choices to find the encoding with least error for inInputData.
	This implementation is best suited to input values that cover a relatively small range of the domain - such as when compressing smooth heightfields and distance fields.
	It may not give good results with other data.
**/
void CompressBlockBC1SDF(u64 *outBCBlock, std::span<float> inInputData, f32vec3 inUnpackDot)
{
	DBG_ASSERT_TRUE_M(inInputData.size() == 16, "BC1 Block always compresses 4x4 region (16 values)");

	// Find minimum and maximum input value
	f32 min_input_value(+1e10f);
	f32 max_input_value(-1e10f);
	for (int i = 0; i < inInputData.size(); i++)
	{
		min_input_value = std::min(min_input_value, inInputData[i]);
		max_input_value = std::max(max_input_value, inInputData[i]);
	}

	DBG_ASSERT_TRUE_M((min_input_value >= 0.0f) && (max_input_value <= 1.0f),
					  "Input values are expected to be in [0, 1]!");

	// Find 16-bit endpoints.  The two should be in order, and not equal.
	u16 endpoint_for_range_min = std::min(sFindEndpointColor(min_input_value, inUnpackDot), static_cast<u16>(65535 - 1));	// endpoint_for_range_max>min, so min can't be 65535
	u16 endpoint_for_range_max = std::max(sFindEndpointColor(max_input_value, inUnpackDot), static_cast<u16>(endpoint_for_range_min + 1));
	DBG_ASSERT_TRUE_M(endpoint_for_range_min < endpoint_for_range_max, "Invalid range calculated");

	float least_error = 1e10f;
	u64 best_bc_block = 0;

	for (u32 hypothetical_endpoint_hi = endpoint_for_range_min + 1; hypothetical_endpoint_hi <= endpoint_for_range_max; hypothetical_endpoint_hi++)
	{
		float curr_error = 1e10f;
		u64 curr_bc_block;
		sBuildBC1BlockAndEvaluateError(&curr_error, &curr_bc_block, inInputData, endpoint_for_range_min, static_cast<u16>(hypothetical_endpoint_hi), inUnpackDot);
		if (curr_error < least_error)
		{
			least_error = curr_error;
			best_bc_block = curr_bc_block;
		}
	}
	*outBCBlock = best_bc_block;
}
