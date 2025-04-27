#include <daxa/daxa.inl>
#include "autoexposure.inl"

shared uint shared_histogram[LUM_HISTOGRAM_BIN_COUNT];

#if defined(GEN_HISTOGRAM)
const vec3 PERCEIVED_LUMINANCE_WEIGHTS = vec3(0.2127, 0.7152, 0.0722);
const float MIN_LUMINANCE_THRESHOLD = 2e-10;

DAXA_DECL_PUSH_CONSTANT(GenLuminanceHistogramH, push)

layout(local_size_x = COMPUTE_HISTOGRAM_WG_X, local_size_y = COMPUTE_HISTOGRAM_WG_Y) in;
void main()
{
    shared_histogram[gl_LocalInvocationIndex % LUM_HISTOGRAM_BIN_COUNT] = 0;
    memoryBarrierShared();
    barrier();

    if (all(lessThan(gl_GlobalInvocationID.xy, deref(push.globals).settings.render_target_size)))
    {
        const float exposure = deref(push.exposure_state).exposure;
        const vec3 color_value = imageLoad(daxa_image2D(push.color_image), daxa_i32vec2(gl_GlobalInvocationID.xy)).rgb / exposure;

        float luminance = dot(color_value, PERCEIVED_LUMINANCE_WEIGHTS);

        // Avoid log2 on values close to 0
        const float luminance_log2 = luminance < MIN_LUMINANCE_THRESHOLD ? deref(push.globals).postprocess_settings.min_luminance_log2 : log2(luminance);

        // map the luminance to be relative to [min, max] luminance values
        const float remapped_luminance =
            (luminance_log2 - deref(push.globals).postprocess_settings.min_luminance_log2) * deref(push.globals).postprocess_settings.inv_luminance_log2_range;

        const float clamped_luminance = clamp(remapped_luminance, 0.0, 1.0);
        const uint bin_index = daxa_u32(clamped_luminance * 255);

        atomicAdd(shared_histogram[bin_index], 1);
    }
    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationIndex < LUM_HISTOGRAM_BIN_COUNT)
    {
        atomicAdd(deref(push.histogram[gl_LocalInvocationIndex]), shared_histogram[gl_LocalInvocationIndex]);
    }
}
#endif // GEN_HISTOGRAM
#if defined(GEN_AVERAGE)

uint saturating_sub(uint a, uint b)
{
    return a - min(a, b);
}

#extension GL_EXT_shader_atomic_int64 : require

#define ae_hist_t uint

shared ae_hist_t sum;

DAXA_DECL_PUSH_CONSTANT(GenLuminanceAverageH, push)

float compute_exposure(float ev)
{
    const float exposure_bias = deref(push.globals).postprocess_settings.exposure_bias;
    const float calibration = deref(push.globals).postprocess_settings.calibration;
    const float sensor_sensitivity = deref(push.globals).postprocess_settings.sensor_sensitivity;
    const float ev100 = ev + log2(sensor_sensitivity * exposure_bias / calibration);
    const float exposure = 1.0 / (1.2 * exp2(ev100));
    return exposure;
}

layout(local_size_x = LUM_HISTOGRAM_BIN_COUNT) in;
void main()
{
    if (gl_LocalInvocationIndex == 0)
    {
        sum = 0;
    }
    ae_hist_t countForThisBin = deref(push.histogram[gl_LocalInvocationIndex]);
    shared_histogram[gl_LocalInvocationIndex] = countForThisBin;
    barrier();

    // Do an inclusive prefix sum on all the histogram values, store in shared memory (writing over the histogram)
    uint idx = gl_LocalInvocationIndex;
    [[unroll]]
    for (uint step = 0; step < int(log2(LUM_HISTOGRAM_BIN_COUNT)); step++)
    {
        if (idx < LUM_HISTOGRAM_BIN_COUNT / 2)
        {
            uint mask = (1u << step) - 1;
            uint rd_idx = ((idx >> step) << (step + 1)) + mask;
            uint wr_idx = rd_idx + 1 + (idx & mask);
            shared_histogram[wr_idx] += shared_histogram[rd_idx];
        }
        barrier();
    }

    barrier();

    // Use the prefix sum to eliminate the top and bottom outliers based on percentage.
    // Usually, you want to discard at least the bottom half of all histogram values, as
    // they represent the shaded/dark half of the image. This keeps good contrast
    float outlierFracLo = deref(push.globals).postprocess_settings.auto_exposure_histogram_clip_lo;
    float outlierFracHi = deref(push.globals).postprocess_settings.auto_exposure_histogram_clip_hi;
    ae_hist_t totalEntryCount = shared_histogram[LUM_HISTOGRAM_BIN_COUNT - 1];

    ae_hist_t rejectLoEntryCount = ae_hist_t(float(totalEntryCount) * outlierFracLo);
    ae_hist_t entryCountToUse = ae_hist_t(float(totalEntryCount) * (outlierFracHi - outlierFracLo));
    ae_hist_t rejectHiEntryCount = ae_hist_t(float(totalEntryCount) * outlierFracHi);

    uint binIndex = gl_LocalInvocationIndex;
    ae_hist_t count = countForThisBin;
    ae_hist_t exclusivePrefixSum = shared_histogram[binIndex] - count;
    ae_hist_t leftToReject = saturating_sub(rejectLoEntryCount, exclusivePrefixSum);
    ae_hist_t leftToUse = min(entryCountToUse, saturating_sub(rejectHiEntryCount, exclusivePrefixSum));
    ae_hist_t countToUse = min(saturating_sub(count, leftToReject), leftToUse);

    float t = (float(binIndex) + 0.5) / LUM_HISTOGRAM_BIN_COUNT;

    atomicAdd(sum, ae_hist_t(t * float(countToUse)));

    barrier();

    if (gl_LocalInvocationIndex == 0)
    {
        const float valid_pixel_count = max(float(entryCountToUse), 1);
        const float weighed_average_log2 = float(sum) / valid_pixel_count;
        const float desired_ev =
            (weighed_average_log2 * deref(push.globals).postprocess_settings.luminance_log2_range) + deref(push.globals).postprocess_settings.min_luminance_log2;

        const float prev_ev = deref(push.exposure_state).ev;

        const float tau = deref(push.globals).postprocess_settings.luminance_adaption_tau;
        const float luminance_adapt_factor = 1.0 - exp(-deref(push.globals).delta_time * tau);
        const float adapted_ev = prev_ev + (desired_ev - prev_ev) * luminance_adapt_factor;

        deref(push.exposure_state).ev = adapted_ev;
        deref(push.exposure_state).exposure = compute_exposure(adapted_ev);
    }
}
#endif // GEN_AVERAGE