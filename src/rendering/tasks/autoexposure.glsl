#include <daxa/daxa.inl>
#include "autoexposure.inl"

shared ae_hist_t shared_histogram[LUM_HISTOGRAM_BIN_COUNT];

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
        const float exposure = deref(push.globals).exposure;
        const vec3 color_value = imageLoad(daxa_image2D(push.color_image), daxa_i32vec2(gl_GlobalInvocationID.xy)).rgb / exposure;

        float luminance = dot(color_value, PERCEIVED_LUMINANCE_WEIGHTS);

        // Avoid log2 on values close to 0
        const float luminance_log2 = luminance < MIN_LUMINANCE_THRESHOLD ? deref(push.globals).postprocess_settings.min_luminance_log2 : log2(luminance);

        // map the luminance to be relative to [min, max] luminance values
        const float remapped_luminance =
            (luminance_log2 - deref(push.globals).postprocess_settings.min_luminance_log2) * deref(push.globals).postprocess_settings.inv_luminance_log2_range;

        const float clamped_luminance = clamp(remapped_luminance, 0.0, 1.0);
        const uint bin_index = daxa_u32(clamped_luminance * 255);

        vec2 uv = (vec2(gl_GlobalInvocationID.xy) + 0.5) / vec2(deref(push.globals).settings.render_target_size);
        // weight center pixels greater than edge pixels.
        float influence = exp(-8 * pow(length(uv - 0.5), 2));
        ae_hist_t quantizedInfluence = ae_hist_t(influence * 10.0);

        atomicAdd(shared_histogram[bin_index], quantizedInfluence);
    }
    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationIndex < LUM_HISTOGRAM_BIN_COUNT)
    {
        atomicAdd(deref(push.histogram).bins[gl_LocalInvocationIndex], shared_histogram[gl_LocalInvocationIndex]);
    }
}
#endif // GEN_HISTOGRAM
#if defined(GEN_AVERAGE)

ae_hist_t saturating_sub(ae_hist_t a, ae_hist_t b)
{
    return a - min(a, b);
}

#extension GL_EXT_shader_atomic_int64 : require

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

#include "exposure.glsl"

void auto_exposure_update(daxa_RWBufferPtr(AutoExposureHistogram) self, float ev)
{
	ev = clamp(ev, -16.0, 16.0);

	deref(self).desired_ev = ev;
	ev = eval_exposure_compensation_curve_ev(ev);

	float dt = deref(push.globals).delta_time * deref(push.globals).postprocess_settings.luminance_adaption_tau;

	float tFast = 1.0 - exp(-4.0 * dt);
	deref(self).ev_fast = (ev - deref(self).ev_fast) * tFast + deref(self).ev_fast;

	float tSlow = 1.0 - exp(-1.0 * dt);
	deref(self).ev_slow = (ev - deref(self).ev_slow) * tSlow + deref(self).ev_slow;

	if (isnan(deref(self).ev_fast))
		deref(self).ev_fast = 1;
	if (isnan(deref(self).ev_slow))
		deref(self).ev_slow = 1;

	deref(self).ev_slow = clamp(deref(self).ev_slow, -16.0, 16.0);
	deref(self).ev_fast = clamp(deref(self).ev_fast, -16.0, 16.0);
}


layout(local_size_x = LUM_HISTOGRAM_BIN_COUNT) in;
void main()
{
    if (gl_LocalInvocationIndex == 0)
    {
        sum = 0;
    }
    ae_hist_t countForThisBin = deref(push.histogram).bins[gl_LocalInvocationIndex];
    shared_histogram[gl_LocalInvocationIndex] = countForThisBin;
    memoryBarrierShared();
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
        memoryBarrierShared();
        barrier();
    }

    memoryBarrierShared();
    barrier();

    // Use the prefix sum to eliminate the top and bottom outliers based on percentage.
    // Usually, you want to discard at least the bottom half of all histogram values, as
    // they represent the shaded/dark half of the image. This keeps good contrast
    float outlierFracLo = deref(push.globals).postprocess_settings.auto_exposure_histogram_clip_lo;
    float outlierFracHi = deref(push.globals).postprocess_settings.auto_exposure_histogram_clip_hi;
    ae_hist_t totalEntryCount = shared_histogram[LUM_HISTOGRAM_BIN_COUNT - 1];

    atomicMax(deref(push.histogram).max_bin_value, countForThisBin);
    atomicAdd(deref(push.histogram).bins_total_count, countForThisBin);

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

    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationIndex == 0)
    { 
        float mean = float(sum) / max(float(entryCountToUse), 1);
        float image_log2_lum = deref(push.globals).postprocess_settings.min_luminance_log2 + mean * deref(push.globals).postprocess_settings.luminance_log2_range;
        float desiredEv = -image_log2_lum;
        if (entryCountToUse < 10)
            desiredEv = 0;

        auto_exposure_update(push.histogram, desiredEv);

        const float adapted_ev = auto_exposure_get_ev_smoothed(push.histogram);
        const float exposure_value = compute_exposure(-adapted_ev);
        deref(push.exposure_state).ev = adapted_ev;
        deref(push.exposure_state).exposure = exposure_value;
        deref(push.exposure_state).inv_exposure = 1.0 / exposure_value;
    }
}
#endif // GEN_AVERAGE
