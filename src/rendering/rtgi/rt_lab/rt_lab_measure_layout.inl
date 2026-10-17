#pragma once

// Layout of the RT Lab convergence measurement histograms, shared by the shaders (rt_lab_measure.inl) and the CPU
// analytics (rt_lab_state.hpp).

// Convergence measurement (RtLabState, rt_lab_state.hpp): every frame of a measurement run, two per-tile metrics of the
// FINAL (post-spatial, full res) output are binned into log histograms and appended to a host visible buffer:
//   metric 0, convergence error:    post-spatial split relative error |A - B| / min(A, B)
//   metric 1, temporal instability: relative temporal std of the output luma, sqrt(var) / mean, with a running
//                                   mean / variance per pixel (average until measure_window frames, then an EMA),
//                                   reprojected with the camera (follows the surface point, restarts on disocclusion)
// Per tile (RT_LAB_MEASURE_TILE^2 pixels) the RMS of the metric is binned, much less noisy than single pixels.
#define RT_LAB_MEASURE_TILE 4
#define RT_LAB_MEASURE_X 8
#define RT_LAB_MEASURE_Y 8
#define RT_LAB_MEASURE_BINS 512
// Range of NEW measurements (each measurement keeps the range it was recorded with: RtLabMeasurement::log_min/max).
#define RT_LAB_MEASURE_LOG_MIN -5.5f // ~0.0003 %
#define RT_LAB_MEASURE_LOG_MAX 2.0f  // 10000 %
// One metric's row: [bins][below range][above range][tiles with geometry][pad]. A tile with geometry but no valid metric
// (split history halves without samples) counts as above range.
#define RT_LAB_MEASURE_UNDER (RT_LAB_MEASURE_BINS + 0)
#define RT_LAB_MEASURE_OVER (RT_LAB_MEASURE_BINS + 1)
#define RT_LAB_MEASURE_TILES (RT_LAB_MEASURE_BINS + 2)
#define RT_LAB_MEASURE_METRIC_STRIDE (RT_LAB_MEASURE_BINS + 4)
#define RT_LAB_MEASURE_METRICS 2
#define RT_LAB_MEASURE_METRIC_ERROR 0
#define RT_LAB_MEASURE_METRIC_INSTABILITY 1
#define RT_LAB_MEASURE_ROW_SIZE (RT_LAB_MEASURE_METRIC_STRIDE * RT_LAB_MEASURE_METRICS)

// Temporal lag (step response): every measured frame the mean UNEXPOSED output luminance of every
// RT_LAB_MEASURE_LAG_TILE^2 tile (-1 = no geometry) is appended to a host visible buffer; the CPU finds per tile the frame
// from which on it stays within a tolerance of its final value (rt_lab_measure_lag, rt_lab.hpp).
#define RT_LAB_MEASURE_LAG_TILE 16
#define RT_LAB_MEASURE_LAG_FINAL_FRAMES 8 // the final value = mean of the run's last this many frames
