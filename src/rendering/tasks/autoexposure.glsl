#include <daxa/daxa.inl>
#include "autoexposure.inl"

shared uint shared_histogram[LUM_HISTOGRAM_BIN_COUNT];

#if defined(GEN_HISTOGRAM)
const vec3 PERCEIVED_LUMINANCE_WEIGHTS = vec3(0.2127, 0.7152, 0.0722);
const float MIN_LUMINANCE_THRESHOLD = 2e-10;

DAXA_DECL_PUSH_CONSTANT(GenLuminanceHistogram, push)
layout(local_size_x = COMPUTE_HISTOGRAM_WG_X, local_size_y = COMPUTE_HISTOGRAM_WG_Y) in;
void main()
{
    shared_histogram[gl_LocalInvocationIndex % LUM_HISTOGRAM_BIN_COUNT] = 0;
    memoryBarrierShared();
    barrier();

    if(all(lessThan(gl_GlobalInvocationID.xy, deref(push.globals).settings.render_target_size)))
    {
        const vec3 color_value = imageLoad(daxa_image2D(push.color_image), daxa_i32vec2(gl_GlobalInvocationID.xy)).rgb;

        float luminance = dot(color_value, PERCEIVED_LUMINANCE_WEIGHTS);

        // Avoid log2 on values close to 0
        const float luminance_log2 = luminance < MIN_LUMINANCE_THRESHOLD ? 0.0 : log2(luminance);

        // map the luminance to be relative to [min, max] luminance values
        const float remapped_luminance = 
            (luminance_log2 - deref(push.globals).postprocess_settings.min_luminance_log2)
            * deref(push.globals).postprocess_settings.inv_luminance_log2_range;

        const float clamped_luminance = clamp(remapped_luminance, 0.0, 1.0);
        const uint bin_index = daxa_u32(clamped_luminance * 255);

        atomicAdd(shared_histogram[bin_index], 1);
    }
    memoryBarrierShared();
    barrier();

    if(gl_LocalInvocationIndex < LUM_HISTOGRAM_BIN_COUNT)
    {
        atomicAdd(deref(push.histogram[gl_LocalInvocationIndex]), shared_histogram[gl_LocalInvocationIndex]);
    }
}
#endif //GEN_HISTOGRAM
#if defined(GEN_AVERAGE)

DAXA_DECL_PUSH_CONSTANT(GenLuminanceAverage, push)
layout(local_size_x = LUM_HISTOGRAM_BIN_COUNT) in;
void main()
{
    const uint local_index = gl_LocalInvocationIndex;
    const uint bin_entry_count = deref(push.histogram[local_index]);

    shared_histogram[local_index] = bin_entry_count * local_index;
    memoryBarrierShared();
    barrier();

    // Sum up the bin counts by doing a merge sum, at the end shared_histogram[0] will hold the sum
    uint threshold = LUM_HISTOGRAM_BIN_COUNT / 2;
    for(int i = int(log2(LUM_HISTOGRAM_BIN_COUNT)); i > 0; i--)
    {
        if(local_index < threshold) { shared_histogram[local_index] += shared_histogram[local_index + threshold]; }
        threshold /= 2;
        memoryBarrierShared();
        barrier();
    }

    if(local_index == 0)
    {
        const uvec2 resolution = deref(push.globals).settings.render_target_size;
        const int total_pixel_count = int(resolution.x * resolution.y);
        // bin_count holds the value of the 0th bin in the histogram (because in the if we are selecting the 0th thread)
        // In the histogram generation we map all the completely black pixels (the ones that fall below the min luminance threshold)
        // to the 0th bin. We don't want to count totally black pixels in the average luminance computation, so we reject them here
        // Also note that because we weigh the bin count by the thread index, the black pixels do not contribute to the overall 
        // computed value
        const int valid_pixel_count = max(total_pixel_count - int(bin_entry_count), 1);
        const float weighed_average_log2 = shared_histogram[0] / valid_pixel_count;
        const float remapped_log2_average = 
            ((weighed_average_log2 / 254.0) * deref(push.globals).postprocess_settings.luminance_log2_range)
            + deref(push.globals).postprocess_settings.min_luminance_log2;
        const float luminance_average = exp2(remapped_log2_average);
        const float prev_lum_average = deref(push.luminance_average);

        const float tau = deref(push.globals).postprocess_settings.luminance_adaption_tau;
        const float luminance_adapt_factor = 1.0 - exp(-deref(push.globals).delta_time * tau);
        const float adapted_luminance = prev_lum_average + (luminance_average - prev_lum_average) * luminance_adapt_factor;
        deref(push.luminance_average) = adapted_luminance;
    }
}
#endif //GEN_AVERAGE