#include "rt_lab_telemetry.hpp"

#include "rt_lab.hpp"
#include "rt_lab_telemetry.inl"
#include "rt_lab_trace.inl" // RT_LAB_RAY_T_MAX

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numbers>

#include <fmt/format.h>
#include <imgui.h>
#include <implot.h>

static constexpr f64 TELEMETRY_ZERO = 1e-9;       // values at or below count as zero (excluded from a log axis)
static constexpr f64 TELEMETRY_FIT_PLOT_POINTS = 160;

static auto rt_lab_telemetry_luma(f32 const (&rgb)[3]) -> f64
{
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

static auto rt_lab_telemetry_dot(f32 const (&a)[3], f32 const (&b)[3]) -> f64
{
    return static_cast<f64>(a[0]) * b[0] + static_cast<f64>(a[1]) * b[1] + static_cast<f64>(a[2]) * b[2];
}

// == Distributions ====================================================================================================

static auto rt_lab_normal_pdf(f64 z) -> f64
{
    return std::exp(-0.5 * z * z) / std::sqrt(2.0 * std::numbers::pi);
}

static auto rt_lab_normal_cdf(f64 z) -> f64
{
    return 0.5 * std::erfc(-z / std::numbers::sqrt2);
}

static constexpr f64 TELEMETRY_TAIL_OFFSET = 2.0;           // RT_LAB_DIST_TAIL_OFFSET: tail thresholds = the perceptual mean +- this
static constexpr f64 TELEMETRY_BODY_VARIANCE_FLOOR = 1e-4;  // plotted / KS distributions only: a body of identical values would be a spike
static constexpr f64 TELEMETRY_TAIL_VARIANCE_FLOOR = 0.02;

static auto rt_lab_mixture_body_pdf(RtLabTelemetryMixture const & m, f64 u) -> f64
{
    f64 const sigma = std::sqrt(std::max(m.body_variance, TELEMETRY_BODY_VARIANCE_FLOOR));
    return rt_lab_normal_pdf((u - m.body_mean) / sigma) / sigma;
}

static auto rt_lab_mixture_body_cdf(RtLabTelemetryMixture const & m, f64 u) -> f64
{
    return rt_lab_normal_cdf((u - m.body_mean) / std::sqrt(std::max(m.body_variance, TELEMETRY_BODY_VARIANCE_FLOOR)));
}

// Density of x (>= 0) under (1 - p) body + p tail.
static auto rt_lab_mixture_pdf(RtLabTelemetryMixture const & m, f64 x) -> f64
{
    f64 const shifted = std::max(x, 0.0) + m.eps;
    f64 const u = std::log(shifted);
    f64 const tail_sigma = std::sqrt(std::max(m.tail_variance, TELEMETRY_TAIL_VARIANCE_FLOOR));
    f64 const tail = m.tail_count > 0 ? rt_lab_normal_pdf((u - m.tail_mean) / tail_sigma) / tail_sigma : 0.0;
    return ((1.0 - m.tail_fraction) * rt_lab_mixture_body_pdf(m, u) + m.tail_fraction * tail) / shifted;
}

static auto rt_lab_mixture_cdf(RtLabTelemetryMixture const & m, f64 x) -> f64
{
    f64 const u = std::log(std::max(x, 0.0) + m.eps);
    f64 const tail_sigma = std::sqrt(std::max(m.tail_variance, TELEMETRY_TAIL_VARIANCE_FLOOR));
    f64 const tail = m.tail_count > 0 ? rt_lab_normal_cdf((u - m.tail_mean) / tail_sigma) : 0.0;
    return (1.0 - m.tail_fraction) * rt_lab_mixture_body_cdf(m, u) + m.tail_fraction * tail;
}

// Mean and variance (population) of y[first, last).
static void rt_lab_telemetry_range_moments(std::vector<f64> const & y, size_t first, size_t last, f64 & mean, f64 & variance)
{
    f64 const count = static_cast<f64>(last - first);
    mean = 0.0;
    for (size_t i = first; i < last; ++i)
    {
        mean += y[i];
    }
    mean /= count;
    variance = 0.0;
    for (size_t i = first; i < last; ++i)
    {
        variance += (y[i] - mean) * (y[i] - mean);
    }
    variance /= count;
}

// Batch version of the temporal pass's per pixel body / tail tracking (rt_lab_temporal.hlsl rt_lab_temporal_distribution) over all
// collected values: y = ln(x + eps), eps = 1e-3 mean; the perceptual mean is the mean of ALL y, tau = that + 2; body = y <= tau,
// tail = y > tau. `sorted_x` ascending.
static void rt_lab_telemetry_fit_mixture(RtLabTelemetryMixture & m, std::vector<f64> const & sorted_x, f64 mean_x, std::vector<f64> const & ks_sorted)
{
    m = {};
    size_t const n = sorted_x.size();
    if (n < 2 || !(mean_x > 0.0)) // no sample count threshold: the fit is shown from the first rays (variances are floored)
    {
        return;
    }
    m.eps = 1e-3 * mean_x;
    std::vector<f64> y(n);
    for (size_t i = 0; i < n; ++i)
    {
        y[i] = std::log(std::max(sorted_x[i], 0.0) + m.eps); // monotone: stays sorted
    }
    f64 ignored = 0.0;
    rt_lab_telemetry_range_moments(y, 0, n, m.perceptual_mean, ignored);
    m.tau = m.perceptual_mean + TELEMETRY_TAIL_OFFSET;
    size_t const first_tail = static_cast<size_t>(std::upper_bound(y.begin(), y.end(), m.tau) - y.begin());
    m.body_count = first_tail;
    m.tail_count = n - first_tail;
    if (m.body_count < 1)
    {
        return;
    }
    rt_lab_telemetry_range_moments(y, 0, first_tail, m.body_mean, m.body_variance);
    if (m.tail_count > 0)
    {
        rt_lab_telemetry_range_moments(y, first_tail, n, m.tail_mean, m.tail_variance);
    }
    m.tail_fraction = static_cast<f64>(m.tail_count) / static_cast<f64>(n);
    m.valid = true;
    // KS distance of the whole series to the mixture (an even subset of the sorted values).
    size_t const ks_n = ks_sorted.size();
    size_t const stride = std::max<size_t>(ks_n / 1500, 1);
    for (size_t i = 0; i < ks_n; i += stride)
    {
        f64 const upper = static_cast<f64>(i + 1) / static_cast<f64>(ks_n);
        f64 const lower = static_cast<f64>(i) / static_cast<f64>(ks_n);
        f64 const lognormal = rt_lab_mixture_cdf(m, ks_sorted[i]);
        m.ks_lognormal = std::max({m.ks_lognormal, std::abs(upper - lognormal), std::abs(lognormal - lower)});
    }
}

static auto rt_lab_mixture3_pdf(RtLabTelemetryMixture3 const & m, f64 x) -> f64
{
    f64 const shifted = std::max(x, 0.0) + m.eps;
    f64 const u = std::log(shifted);
    f64 const body_fraction = std::max(1.0 - m.low_fraction - m.high_fraction, 0.0);
    f64 const body_sigma = std::sqrt(std::max(m.body_variance, TELEMETRY_BODY_VARIANCE_FLOOR));
    f64 const body = rt_lab_normal_pdf((u - m.body_mean) / body_sigma) / body_sigma;
    f64 const high_sigma = std::sqrt(std::max(m.high_variance, TELEMETRY_TAIL_VARIANCE_FLOOR));
    f64 const low_sigma = std::sqrt(std::max(m.low_variance, TELEMETRY_TAIL_VARIANCE_FLOOR));
    f64 const high = m.high_count > 0 ? rt_lab_normal_pdf((u - m.high_mean) / high_sigma) / high_sigma : 0.0;
    f64 const low = m.low_count > 0 ? rt_lab_normal_pdf((u - m.low_mean) / low_sigma) / low_sigma : 0.0;
    return (body_fraction * body + m.high_fraction * high + m.low_fraction * low) / shifted;
}

static auto rt_lab_mixture3_cdf(RtLabTelemetryMixture3 const & m, f64 x) -> f64
{
    f64 const u = std::log(std::max(x, 0.0) + m.eps);
    f64 const body_fraction = std::max(1.0 - m.low_fraction - m.high_fraction, 0.0);
    f64 const body = rt_lab_normal_cdf((u - m.body_mean) / std::sqrt(std::max(m.body_variance, TELEMETRY_BODY_VARIANCE_FLOOR)));
    f64 const high = m.high_count > 0 ? rt_lab_normal_cdf((u - m.high_mean) / std::sqrt(std::max(m.high_variance, TELEMETRY_TAIL_VARIANCE_FLOOR))) : 0.0;
    f64 const low = m.low_count > 0 ? rt_lab_normal_cdf((u - m.low_mean) / std::sqrt(std::max(m.low_variance, TELEMETRY_TAIL_VARIANCE_FLOOR))) : 0.0;
    return body_fraction * body + m.high_fraction * high + m.low_fraction * low;
}

auto rt_lab_telemetry_mixture3_pdf(RtLabTelemetryMixture3 const & m, f64 x) -> f64
{
    return rt_lab_mixture3_pdf(m, x);
}

// Batch version of the temporal pass's three component tracking (distribution guide mode 3) over all collected values: y = ln(x + eps),
// eps = 1e-3 mean; thresholds = the mean of ALL y +- 2. Body = between, bright tail = above, dark tail = below. `sorted_x` ascending.
static void rt_lab_telemetry_fit_mixture3(RtLabTelemetryMixture3 & m, std::vector<f64> const & sorted_x, f64 mean_x, std::vector<f64> const & ks_sorted)
{
    m = {};
    size_t const n = sorted_x.size();
    if (n < 2 || !(mean_x > 0.0))
    {
        return;
    }
    m.eps = 1e-3 * mean_x;
    std::vector<f64> y(n);
    for (size_t i = 0; i < n; ++i)
    {
        y[i] = std::log(std::max(sorted_x[i], 0.0) + m.eps);
    }
    f64 ignored = 0.0;
    rt_lab_telemetry_range_moments(y, 0, n, m.perceptual_mean, ignored);
    m.tau_high = m.perceptual_mean + TELEMETRY_TAIL_OFFSET;
    m.tau_low = m.perceptual_mean - TELEMETRY_TAIL_OFFSET;
    size_t const first_tail = static_cast<size_t>(std::upper_bound(y.begin(), y.end(), m.tau_high) - y.begin());
    size_t const last_low = static_cast<size_t>(std::lower_bound(y.begin(), y.end(), m.tau_low) - y.begin());
    m.low_count = last_low;
    m.high_count = n - first_tail;
    if (first_tail <= last_low)
    {
        return;
    }
    m.body_count = first_tail - last_low;
    rt_lab_telemetry_range_moments(y, last_low, first_tail, m.body_mean, m.body_variance);
    if (m.low_count > 0)
    {
        rt_lab_telemetry_range_moments(y, 0, last_low, m.low_mean, m.low_variance);
    }
    if (m.high_count > 0)
    {
        rt_lab_telemetry_range_moments(y, first_tail, n, m.high_mean, m.high_variance);
    }
    m.low_fraction = static_cast<f64>(m.low_count) / static_cast<f64>(n);
    m.high_fraction = static_cast<f64>(m.high_count) / static_cast<f64>(n);
    m.valid = true;
    size_t const ks_n = ks_sorted.size();
    size_t const stride = std::max<size_t>(ks_n / 1500, 1);
    for (size_t i = 0; i < ks_n; i += stride)
    {
        f64 const upper = static_cast<f64>(i + 1) / static_cast<f64>(ks_n);
        f64 const lower = static_cast<f64>(i) / static_cast<f64>(ks_n);
        f64 const lognormal = rt_lab_mixture3_cdf(m, ks_sorted[i]);
        m.ks_lognormal = std::max({m.ks_lognormal, std::abs(upper - lognormal), std::abs(lognormal - lower)});
    }
}

// == Statistics =======================================================================================================

// `values`: the series (linear). Moments, quantiles and the "how many rays carry the energy" measures are computed on the
// values as they are; the gaussian fit + histogram live in the fit domain (linear, or log10 of the non zero values).
// `values`: the data (bars, statistics, KS, MSE). `fit_values` (null = the data itself): what the distributions are fitted on.
static void rt_lab_telemetry_compute_series(RtLabTelemetrySeries & s, std::vector<f64> const & values, std::vector<f64> const * fit_values, bool log_requested, i32 bins, bool auto_bins, bool fit_mixture, u64 excluded)
{
    s = {};
    s.excluded = excluded;
    u64 const total = values.size();
    s.total = total;
    if (total < 2)
    {
        return;
    }
    f64 const n = static_cast<f64>(total);
    f64 sum = 0.0, sum_sq = 0.0, top = 0.0;
    u64 zeros = 0;
    for (f64 v : values)
    {
        sum += v;
        sum_sq += v * v;
        top = std::max(top, v);
        zeros += v <= TELEMETRY_ZERO ? 1 : 0;
    }
    s.mean = sum / n;
    f64 m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (f64 v : values)
    {
        f64 const d = v - s.mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
    }
    s.variance = m2 / (n - 1.0);
    s.std_dev = std::sqrt(s.variance);
    if (m2 > 0.0)
    {
        s.skewness = (m3 / n) / std::pow(m2 / n, 1.5);
        s.excess_kurtosis = (m4 / n) / ((m2 / n) * (m2 / n)) - 3.0;
    }
    s.std_error = s.std_dev / std::sqrt(n);
    if (s.mean > 0.0)
    {
        s.coefficient_of_variation = s.std_dev / s.mean;
        s.relative_std_error = s.std_error / s.mean;
        s.rays_for_5_percent = std::pow(1.96 * s.coefficient_of_variation / 0.05, 2.0);
    }
    s.effective_sample_ratio = sum_sq > 0.0 ? (sum * sum) / (n * sum_sq) : 0.0;
    s.top_share = sum > 0.0 ? top / sum : 0.0;
    s.zero_fraction = static_cast<f64>(zeros) / n;

    std::vector<f64> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    auto quantile = [&](std::vector<f64> const & v, f64 q) -> f64
    {
        return v[std::min(static_cast<size_t>(q * static_cast<f64>(v.size())), v.size() - 1)];
    };
    s.min = sorted.front();
    s.max = sorted.back();
    s.median = quantile(sorted, 0.5);
    s.p95 = quantile(sorted, 0.95);
    s.p99 = quantile(sorted, 0.99);
    std::vector<f64> sorted_fit;
    if (fit_values != nullptr)
    {
        sorted_fit = *fit_values;
        std::sort(sorted_fit.begin(), sorted_fit.end());
    }
    std::vector<f64> const & fit_sorted = fit_values != nullptr ? sorted_fit : sorted;
    f64 fit_mean_value = 0.0;
    for (f64 v : fit_sorted)
    {
        fit_mean_value += v;
    }
    fit_mean_value = fit_sorted.empty() ? 0.0 : fit_mean_value / static_cast<f64>(fit_sorted.size());
    if (fit_mixture && fit_sorted.size() >= 2)
    {
        rt_lab_telemetry_fit_mixture(s.mixture, fit_sorted, fit_mean_value, sorted);
        rt_lab_telemetry_fit_mixture3(s.mixture3, fit_sorted, fit_mean_value, sorted);
    }

    // Auto Histogram picks a log axis when the non zero values are too spread for a linear one: the 0.5 % .. 99.5 % range is more than
    // 30 interquartile ranges wide, so the bulk would sit in a single bar.
    bool log_axis = log_requested;
    if (auto_bins && !log_axis)
    {
        std::vector<f64> nonzero;
        nonzero.reserve(sorted.size());
        for (f64 v : sorted)
        {
            if (v > TELEMETRY_ZERO)
            {
                nonzero.push_back(v);
            }
        }
        if (nonzero.size() >= 8)
        {
            f64 const iqr = quantile(nonzero, 0.75) - quantile(nonzero, 0.25);
            log_axis = iqr > 0.0 && (quantile(nonzero, 0.995) - quantile(nonzero, 0.005)) > 30.0 * iqr;
        }
    }
    s.log_domain = log_axis;

    std::vector<f64> domain;
    if (log_axis)
    {
        domain.reserve(sorted.size());
        for (f64 v : sorted)
        {
            if (v > TELEMETRY_ZERO)
            {
                domain.push_back(std::log10(v));
            }
        }
        s.excluded += total - domain.size();
    }
    else
    {
        domain = std::move(sorted);
    }
    s.count = domain.size();
    if (s.count < 2)
    {
        return;
    }
    f64 const dn = static_cast<f64>(s.count);
    // Gaussian fit: the moments of the fit values in the display domain (log10 of the positive values, or linear).
    std::vector<f64> fit_domain_storage;
    if (fit_values != nullptr)
    {
        for (f64 v : fit_sorted)
        {
            if (!log_axis)
            {
                fit_domain_storage.push_back(v);
            }
            else if (v > TELEMETRY_ZERO)
            {
                fit_domain_storage.push_back(std::log10(v));
            }
        }
    }
    std::vector<f64> const & fit_domain = fit_values != nullptr ? fit_domain_storage : domain;
    f64 const fn = static_cast<f64>(fit_domain.size());
    if (fn >= 2.0)
    {
        for (f64 v : fit_domain)
        {
            s.fit_mean += v;
        }
        s.fit_mean /= fn;
        for (f64 v : fit_domain)
        {
            s.fit_variance += (v - s.fit_mean) * (v - s.fit_mean);
        }
        s.fit_variance /= fn - 1.0;
        s.fit_std_dev = std::sqrt(s.fit_variance);
    }
    if (s.fit_std_dev > 0.0)
    {
        for (size_t i = 0; i < domain.size(); ++i)
        {
            f64 const cdf = 0.5 * std::erfc(-(domain[i] - s.fit_mean) / (s.fit_std_dev * std::numbers::sqrt2));
            f64 const upper = static_cast<f64>(i + 1) / dn;
            f64 const lower = static_cast<f64>(i) / dn;
            s.ks_distance = std::max({s.ks_distance, std::abs(upper - cdf), std::abs(cdf - lower)});
        }
    }

    // Histogram range: the SHORTEST interval of the display domain that holds up to 99.99 % of the non zero values (padded by 5 % each side),
    // so the bars sit where the rays are. Exact zeros (black rays) are a clump of their own: not in the bars (counted below range, the
    // "black rays" row has their share). Every ray is then bucketed into the bins below; what lies outside is counted.
    // Auto bins: Freedman-Diaconis on the values inside the interval, 24 .. 200.
    {
        std::vector<f64> nonzero;
        nonzero.reserve(domain.size());
        for (f64 v : domain)
        {
            if (v > TELEMETRY_ZERO || log_axis)
            {
                nonzero.push_back(v);
            }
        }
        std::vector<f64> const & ranged = nonzero.size() >= 8 ? nonzero : domain;
        // The shortest interval with the highest coverage (99.99 %, 99.9 %, 99.5 %, 99 %, 98 %) whose width stays within 40 interquartile ranges of
        // the bulk: shares as much of the tail as possible, extreme outliers cannot squash the bulk.
        f64 const bulk_iqr = quantile(ranged, 0.75) - quantile(ranged, 0.25);
        f64 const max_width = bulk_iqr > 0.0 ? 40.0 * bulk_iqr : std::numeric_limits<f64>::max();
        constexpr f64 COVERAGES[5] = {0.9999, 0.999, 0.995, 0.99, 0.98};
        for (i32 level = 0; level < 5; ++level)
        {
            size_t const window = std::max<size_t>(static_cast<size_t>(std::ceil(COVERAGES[level] * static_cast<f64>(ranged.size()))), 2);
            size_t best = 0;
            f64 best_width = std::numeric_limits<f64>::max();
            for (size_t i = 0; i + window <= ranged.size(); ++i)
            {
                f64 const width = ranged[i + window - 1] - ranged[i];
                if (width < best_width)
                {
                    best_width = width;
                    best = i;
                }
            }
            s.hist_lo = ranged[best];
            s.hist_hi = ranged[std::min(best + window - 1, ranged.size() - 1)];
            if (best_width <= max_width)
            {
                break;
            }
        }
        f64 const pad = 0.05 * (s.hist_hi - s.hist_lo);
        s.hist_lo -= pad;
        s.hist_hi += pad;
        if (!log_axis && ranged.front() >= 0.0)
        {
            s.hist_lo = std::max(s.hist_lo, 0.0);
        }
        if (s.hist_hi <= s.hist_lo)
        {
            s.hist_lo = ranged.front();
            s.hist_hi = ranged.back();
        }
        if (s.hist_hi <= s.hist_lo)
        {
            s.hist_hi = s.hist_lo + 1.0;
        }
        if (auto_bins)
        {
            std::vector<f64> inside;
            inside.reserve(ranged.size());
            for (f64 v : ranged)
            {
                if (v >= s.hist_lo && v <= s.hist_hi)
                {
                    inside.push_back(v);
                }
            }
            f64 iqr = inside.size() >= 4 ? quantile(inside, 0.75) - quantile(inside, 0.25) : 0.0;
            if (!(iqr > 0.0))
            {
                iqr = s.fit_std_dev;
            }
            f64 const width = 2.0 * iqr * std::pow(std::max(static_cast<f64>(inside.size()), 1.0), -1.0 / 3.0);
            bins = width > 0.0 ? static_cast<i32>(std::clamp(std::round((s.hist_hi - s.hist_lo) / width), 24.0, 200.0)) : 48;
        }
    }
    bins = std::max(bins, 4);
    s.bin_width = (s.hist_hi - s.hist_lo) / static_cast<f64>(bins);
    s.bin_count.assign(static_cast<size_t>(bins), 0.0f);
    s.bin_center.resize(static_cast<size_t>(bins));
    for (i32 b = 0; b < bins; ++b)
    {
        s.bin_center[static_cast<size_t>(b)] = static_cast<f32>(s.hist_lo + (static_cast<f64>(b) + 0.5) * s.bin_width);
    }
    for (f64 v : domain)
    {
        if (v < s.hist_lo)
        {
            ++s.below_range;
        }
        else if (v > s.hist_hi)
        {
            ++s.above_range;
        }
        else
        {
            size_t const bin = std::min(static_cast<size_t>((v - s.hist_lo) / s.bin_width), static_cast<size_t>(bins - 1));
            s.bin_count[bin] += 1.0f;
        }
    }
    // MSE of the distributions against the bars: expected count of a bin = values x probability mass of the bin.
    {
        auto gaussian_cdf = [&](f64 v) { return rt_lab_normal_cdf((v - s.fit_mean) / s.fit_std_dev); };
        auto mixture3_cdf = [&](f64 v) { return rt_lab_mixture3_cdf(s.mixture3, s.log_domain ? std::pow(10.0, v) : v); };
        auto mixture_cdf = [&](f64 v) { return rt_lab_mixture_cdf(s.mixture, s.log_domain ? std::pow(10.0, v) : v); };
        f64 squared[3] = {};
        f64 observed_squared = 0.0;
        for (size_t b = 0; b < s.bin_count.size(); ++b)
        {
            f64 const low = s.hist_lo + static_cast<f64>(b) * s.bin_width;
            f64 const high = low + s.bin_width;
            f64 const observed = s.bin_count[b];
            observed_squared += observed * observed;
            f64 const expected[3] = {
                s.fit_std_dev > 0.0 ? static_cast<f64>(s.count) * (gaussian_cdf(high) - gaussian_cdf(low)) : 0.0,
                s.mixture.valid ? static_cast<f64>(s.total) * (mixture_cdf(high) - mixture_cdf(low)) : 0.0,
                s.mixture3.valid ? static_cast<f64>(s.total) * (mixture3_cdf(high) - mixture3_cdf(low)) : 0.0,
            };
            for (i32 k = 0; k < 3; ++k)
            {
                squared[k] += (observed - expected[k]) * (observed - expected[k]);
            }
        }
        s.mse_valid[0] = s.fit_std_dev > 0.0;
        s.mse_valid[1] = s.mixture.valid;
        s.mse_valid[2] = s.mixture3.valid;
        for (i32 k = 0; k < 3; ++k)
        {
            s.mse_abs[k] = squared[k] / std::max<f64>(static_cast<f64>(s.bin_count.size()), 1.0);
            s.mse_rel[k] = observed_squared > 0.0 ? squared[k] / observed_squared : 0.0;
        }
    }
    s.valid = true;
}

static void rt_lab_telemetry_compute_stats(RtLabTelemetryState & t)
{
    RtLabTelemetryStats & st = t.stats;
    st = {};
    st.rays = t.rays.size();
    st.frames = t.frames;
    st.mean_rays_per_frame = t.frames > 0 ? static_cast<f64>(t.rays_seen) / static_cast<f64>(t.frames) : 0.0;
    if (t.rays.empty())
    {
        return;
    }
    f64 const scale = t.exposed_units ? static_cast<f64>(t.surface.exposure) * std::numbers::pi : 1.0;
    std::vector<f64> modulated, incoming, lengths;
    modulated.reserve(t.rays.size());
    incoming.reserve(t.rays.size());
    u64 misses = 0, near_hits = 0;
    f64 weight_sum = 0.0, mean_dir[3] = {};
    for (RtLabTelemetryRay const & r : t.rays)
    {
        f64 const mod_luma = rt_lab_telemetry_luma(r.modulated);
        f64 const in_luma = rt_lab_telemetry_luma(r.radiance);
        modulated.push_back(mod_luma * scale);
        incoming.push_back(in_luma * scale);
        if (r.hit_t >= RT_LAB_RAY_T_MAX * 0.5f)
        {
            ++misses;
        }
        else
        {
            lengths.push_back(r.hit_t);
            near_hits += r.hit_t < 1.0f ? 1 : 0;
        }
        for (i32 c = 0; c < 3; ++c)
        {
            st.mean_modulated[c] += static_cast<f64>(r.modulated[c]) * scale;
            st.mean_incoming[c] += static_cast<f64>(r.radiance[c]) * scale;
            mean_dir[c] += in_luma * r.direction[c];
        }
        weight_sum += in_luma;
        st.mean_cos_detail += r.cos_detail;
        st.mean_cos_face += r.cos_face;
        st.mean_importance_weight += r.cos_detail / std::max(r.cos_face, 1e-3f);
    }
    f64 const n = static_cast<f64>(t.rays.size());
    for (i32 c = 0; c < 3; ++c)
    {
        st.mean_modulated[c] /= n;
        st.mean_incoming[c] /= n;
    }
    st.mean_cos_detail /= n;
    st.mean_cos_face /= n;
    st.mean_importance_weight /= n;
    st.miss_fraction = static_cast<f64>(misses) / n;
    st.near_hit_fraction = static_cast<f64>(near_hits) / n;
    f64 const dir_length = std::sqrt(mean_dir[0] * mean_dir[0] + mean_dir[1] * mean_dir[1] + mean_dir[2] * mean_dir[2]);
    if (weight_sum > 0.0 && dir_length > 0.0)
    {
        st.direction_concentration = dir_length / weight_sum;
        for (i32 c = 0; c < 3; ++c)
        {
            st.mean_direction[c] = mean_dir[c] / dir_length;
        }
        f64 const cosine = st.mean_direction[0] * t.surface.detail_normal[0] + st.mean_direction[1] * t.surface.detail_normal[1] + st.mean_direction[2] * t.surface.detail_normal[2];
        st.mean_direction_angle = std::acos(std::clamp(cosine, -1.0, 1.0)) * 180.0 / std::numbers::pi;
    }
    st.detail_face_angle = std::acos(std::clamp(rt_lab_telemetry_dot(t.surface.detail_normal, t.surface.face_normal), -1.0, 1.0)) * 180.0 / std::numbers::pi;
    st.view_cos = -rt_lab_telemetry_dot(t.surface.view_direction, t.surface.detail_normal);

    // The reference rays (traced in one frame when the button was released) are the data of the plots, statistics, KS and MSE; the distributions are
    // fitted on the collected rays (what a pixel could estimate). "Use Collected Rays": the collected rays are both.
    bool const show_reference = t.reference_valid && !t.use_collected && !t.reference_rays.empty();
    std::vector<f64> reference_modulated, reference_lengths;
    u64 reference_misses = 0;
    if (show_reference)
    {
        reference_modulated.reserve(t.reference_rays.size());
        for (RtLabTelemetryRay const & r : t.reference_rays)
        {
            reference_modulated.push_back(rt_lab_telemetry_luma(r.modulated) * scale);
            if (r.hit_t >= RT_LAB_RAY_T_MAX * 0.5f)
            {
                ++reference_misses;
            }
            else
            {
                reference_lengths.push_back(r.hit_t);
            }
        }
    }
    rt_lab_telemetry_compute_series(st.modulated, show_reference ? reference_modulated : modulated, show_reference ? &modulated : nullptr, t.log_axis, t.bins, t.auto_histogram, true, 0);
    rt_lab_telemetry_compute_series(st.ray_length, show_reference ? reference_lengths : lengths, show_reference ? &lengths : nullptr, t.log_axis, t.bins, t.auto_histogram, true, show_reference ? reference_misses : misses);
    st.modulated.reference = show_reference;
    st.ray_length.reference = show_reference;
}

// == Collection =======================================================================================================

static void rt_lab_telemetry_clear(RtLabTelemetryState & t)
{
    t.rays.clear();
    t.reference_rays.clear();
    t.reference_valid = false;
    t.reference_in_flight = false;
    t.reference_dispatch_pending = false;
    t.frames = 0;
    t.rays_seen = 0;
    t.full = false;
    t.surface = {};
    t.stats = {};
    t.stats_dirty = true;
}

// Reads the slots whose frames are done, oldest first.
static void rt_lab_telemetry_ingest(RenderContext & render_context, RtLabTelemetryState & t)
{
    if (t.buffer.id().is_empty())
    {
        return;
    }
    u32 const * words = render_context.gpu_context->device.buffer_host_address_as<u32>(t.buffer.id()).value();
    auto const as_float = [](u32 word) -> f32
    {
        f32 value = 0.0f;
        std::memcpy(&value, &word, sizeof(f32));
        return value;
    };
    std::vector<std::pair<u32, u32>> ready = {}; // (stamp, slot)
    for (u32 slot = 0; slot < RT_LAB_TELEMETRY_SLOTS; ++slot)
    {
        u32 const stamp = words[slot * RT_LAB_TELEMETRY_SLOT_WORDS];
        bool const finished = t.frame_counter >= t.issued_frame[slot] + MAX_GPU_FRAMES_IN_FLIGHT + 2;
        if (stamp != 0 && stamp == t.issued_stamp[slot] && finished && stamp > t.last_ingested && stamp >= t.first_stamp)
        {
            ready.emplace_back(stamp, slot);
        }
    }
    std::sort(ready.begin(), ready.end());
    for (auto const & [stamp, slot] : ready)
    {
        t.last_ingested = stamp;
        u32 const * slot_words = words + slot * RT_LAB_TELEMETRY_SLOT_WORDS;
        if (static_cast<i32>(slot_words[4]) != t.target_pixel[0] || static_cast<i32>(slot_words[5]) != t.target_pixel[1])
        {
            continue; // another pixel (the cursor moved)
        }
        auto f = [&](u32 index) -> f32 { return as_float(slot_words[index]); };
        ++t.frames;
        t.stats_dirty = true;
        RtLabTelemetrySurface & surface = t.surface;
        surface.valid = (slot_words[3] & 1u) != 0;
        surface.screen_pixel[0] = slot_words[4];
        surface.screen_pixel[1] = slot_words[5];
        surface.lab_pixel[0] = slot_words[6];
        surface.lab_pixel[1] = slot_words[7];
        for (u32 c = 0; c < 3; ++c)
        {
            surface.position[c] = f(8 + c);
            surface.face_normal[c] = f(12 + c);
            surface.detail_normal[c] = f(16 + c);
            surface.albedo[c] = f(20 + c);
            surface.view_direction[c] = f(24 + c);
        }
        surface.linear_depth = f(11);
        surface.metalness = f(15);
        surface.pixel_width = f(19);
        surface.exposure = f(23);
        t.rays_seen += slot_words[1];
        u32 const stored = std::min(slot_words[2], static_cast<u32>(RT_LAB_TELEMETRY_MAX_RAYS));
        for (u32 k = 0; k < stored; ++k)
        {
            if (t.rays.size() >= static_cast<size_t>(std::max(t.max_samples, 1)))
            {
                t.full = true;
                break;
            }
            RtLabTelemetryRay ray = {};
            u32 const base = RT_LAB_TELEMETRY_HEADER_WORDS + k * RT_LAB_TELEMETRY_RAY_WORDS;
            static_assert(sizeof(RtLabTelemetryRay) == sizeof(u32) * RT_LAB_TELEMETRY_RAY_WORDS);
            std::memcpy(&ray, slot_words + base, sizeof(ray));
            t.rays.push_back(ray);
        }
    }
}

void rt_lab_telemetry_input(RenderContext & render_context, bool mouse_down, i32 cursor_x, i32 cursor_y, i32 window_width, i32 window_height)
{
    auto & t = render_context.rt_lab.telemetry;
    t.collecting = false;
    if (!t.enabled || !mouse_down || window_width <= 0 || window_height <= 0)
    {
        return;
    }
    // Holding keeps the pixel the press started on (moving the mouse neither changes the pixel nor restarts the collection).
    if (t.was_collecting)
    {
        t.pixel[0] = t.target_pixel[0];
        t.pixel[1] = t.target_pixel[1];
        t.collecting = true;
        return;
    }
    auto const size = render_context.render_data.settings.render_target_size;
    t.pixel[0] = std::clamp(static_cast<i32>(static_cast<i64>(cursor_x) * size.x / window_width), 0, static_cast<i32>(size.x) - 1);
    t.pixel[1] = std::clamp(static_cast<i32>(static_cast<i64>(cursor_y) * size.y / window_height), 0, static_cast<i32>(size.y) - 1);
    t.collecting = true;
}

auto rt_lab_telemetry_frame_update(RenderContext & render_context) -> bool
{
    auto & t = render_context.rt_lab.telemetry;
    auto & device = render_context.gpu_context->device;
    ++t.frame_counter;
    t.dispatch_pending = false;
    t.reference_dispatch_pending = false;
    // Test hook (no mouse needed): TIDO_TELEMETRY_PROBE="x y start_frame frames" holds the button on screen pixel (x, y) for `frames` frames starting at
    // frame_update call `start_frame`, then prints a summary when the reference rays arrived.
    static i32 probe[4] = {-1, -1, 0, 0};
    static bool probe_parsed = false;
    if (!probe_parsed)
    {
        probe_parsed = true;
        if (char const * text = std::getenv("TIDO_TELEMETRY_PROBE"))
        {
            std::sscanf(text, "%d %d %d %d", &probe[0], &probe[1], &probe[2], &probe[3]);
        }
    }
    bool const probing = probe[0] >= 0;
    if (probing)
    {
        t.enabled = true;
        bool const hold = static_cast<i32>(t.frame_counter) >= probe[2] && static_cast<i32>(t.frame_counter) < probe[2] + probe[3];
        t.collecting = hold;
        t.pixel[0] = probe[0];
        t.pixel[1] = probe[1];
    }
    bool rebuild = t.enabled != t.built;
    if (!t.enabled)
    {
        t.collecting = false;
        t.was_collecting = false;
        return rebuild;
    }
    if (t.buffer.id().is_empty())
    {
        u64 const size = sizeof(u32) * RT_LAB_TELEMETRY_SLOT_WORDS * RT_LAB_TELEMETRY_SLOTS;
        t.buffer.set_buffer(device.create_buffer({
            .size = size,
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "rt_lab_telemetry",
        }));
        std::memset(device.buffer_host_address(t.buffer.id()).value(), 0, size);
        t.issued_stamp.fill(0);
        t.issued_frame.fill(0);
        rebuild = true;
    }
    if (t.reference_buffer.id().is_empty())
    {
        u64 const size = sizeof(u32) * (RT_LAB_TELEMETRY_REFERENCE_HEADER_WORDS + RT_LAB_TELEMETRY_REFERENCE_RAYS * RT_LAB_TELEMETRY_REFERENCE_RAY_WORDS);
        t.reference_buffer.set_buffer(device.create_buffer({
            .size = size,
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "rt_lab_telemetry_reference",
        }));
        std::memset(device.buffer_host_address(t.reference_buffer.id()).value(), 0, size);
        rebuild = true;
    }
    // A new press or another pixel starts over.
    if (t.collecting && (!t.was_collecting || t.pixel[0] != t.target_pixel[0] || t.pixel[1] != t.target_pixel[1]))
    {
        rt_lab_telemetry_clear(t);
        t.target_pixel[0] = t.pixel[0];
        t.target_pixel[1] = t.pixel[1];
        t.first_stamp = t.next_stamp;
    }
    rt_lab_telemetry_ingest(render_context, t);
    // Button released with rays collected: shoot the reference rays of the pixel in this frame.
    if (t.was_collecting && !t.collecting && !t.rays.empty())
    {
        t.reference_stamp = t.next_stamp++;
        t.reference_issued_frame = t.frame_counter;
        t.reference_screen_pixel[0] = static_cast<u32>(t.target_pixel[0]);
        t.reference_screen_pixel[1] = static_cast<u32>(t.target_pixel[1]);
        t.reference_dispatch_pending = true;
        t.reference_in_flight = true;
        t.reference_valid = false;
    }
    // The reference rays arrive once the frames in flight are done.
    if (t.reference_in_flight && t.frame_counter >= t.reference_issued_frame + MAX_GPU_FRAMES_IN_FLIGHT + 2)
    {
        u32 const * words = device.buffer_host_address_as<u32>(t.reference_buffer.id()).value();
        t.reference_in_flight = false;
        if (words[0] == t.reference_stamp && words[1] != 0u)
        {
            auto as_float = [&](u32 index) -> f32 { f32 value = 0.0f; std::memcpy(&value, &words[index], sizeof(f32)); return value; };
            f32 const albedo_scale = 1.0f - t.surface.metalness;
            t.reference_rays.clear();
            t.reference_rays.reserve(RT_LAB_TELEMETRY_REFERENCE_RAYS);
            for (u32 i = 0; i < RT_LAB_TELEMETRY_REFERENCE_RAYS; ++i)
            {
                u32 const base = RT_LAB_TELEMETRY_REFERENCE_HEADER_WORDS + i * RT_LAB_TELEMETRY_REFERENCE_RAY_WORDS;
                RtLabTelemetryRay ray = {};
                for (u32 k = 0; k < 3; ++k)
                {
                    ray.radiance[k] = as_float(base + k);
                    ray.direction[k] = as_float(base + 4 + k);
                }
                ray.hit_t = as_float(base + 3);
                // Same modulation as the collected rays (rt_lab_telemetry.hlsl): albedo x radiance x cos(detail) / cos(face).
                ray.cos_face = std::max(static_cast<f32>(rt_lab_telemetry_dot(t.surface.face_normal, ray.direction)), 1e-3f);
                ray.cos_detail = std::max(static_cast<f32>(rt_lab_telemetry_dot(t.surface.detail_normal, ray.direction)), 0.0f);
                for (u32 k = 0; k < 3; ++k)
                {
                    ray.modulated[k] = t.surface.albedo[k] * albedo_scale * ray.radiance[k] * (ray.cos_detail / ray.cos_face);
                }
                t.reference_rays.push_back(ray);
            }
            t.reference_valid = true;
            t.stats_dirty = true;
            if (probing)
            {
                RtLabTelemetryStats const probe_stats = (rt_lab_telemetry_compute_stats(t), t.stats);
                f64 collected_mean = 0.0;
                for (RtLabTelemetryRay const & r : t.rays)
                {
                    collected_mean += rt_lab_telemetry_luma(r.modulated);
                }
                collected_mean /= static_cast<f64>(std::max<size_t>(t.rays.size(), 1));
                RtLabTelemetrySeries const & m = probe_stats.modulated;
                fmt::print("[TelemetryProbe] collected {} rays mean {:.6g} | reference {} rays mean {:.6g} std {:.6g} (log axis {}) tail {:.2f} %\n", t.rays.size(), collected_mean,
                    t.reference_rays.size(), m.mean, m.std_dev, m.log_domain, m.mixture.valid ? m.mixture.tail_fraction * 100.0 : -1.0);
                f64 reference_incoming = 0.0, collected_incoming = 0.0;
                for (RtLabTelemetryRay const & r : t.reference_rays) { reference_incoming += rt_lab_telemetry_luma(r.radiance); }
                for (RtLabTelemetryRay const & r : t.rays) { collected_incoming += rt_lab_telemetry_luma(r.radiance); }
                fmt::print("[TelemetryProbe] incoming (unmodulated) mean: collected {:.6g} reference {:.6g} | albedo ({:.3f} {:.3f} {:.3f}) metal {:.2f} valid {}\n",
                    collected_incoming / static_cast<f64>(std::max<size_t>(t.rays.size(), 1)), reference_incoming / static_cast<f64>(std::max<size_t>(t.reference_rays.size(), 1)),
                    t.surface.albedo[0], t.surface.albedo[1], t.surface.albedo[2], t.surface.metalness, t.surface.valid);
                fmt::print("[TelemetryProbe] MSE rel vs reference: gauss {:.4f} ln {:.4f} ln3 {:.4f} | KS gauss {:.3f} ln {:.3f} ln3 {:.3f}\n",
                    m.mse_rel[0], m.mse_rel[1], m.mse_rel[2], m.ks_distance, m.mixture.ks_lognormal, m.mixture3.ks_lognormal);
                t.use_collected = true;
                RtLabTelemetryStats const collected_stats = (rt_lab_telemetry_compute_stats(t), t.stats);
                t.use_collected = false;
                RtLabTelemetrySeries const & c = collected_stats.modulated;
                fmt::print("[TelemetryProbe] MSE rel vs collected : gauss {:.4f} ln {:.4f} ln3 {:.4f}\n", c.mse_rel[0], c.mse_rel[1], c.mse_rel[2]);
                rt_lab_telemetry_compute_stats(t);
            }
        }
    }
    if (t.collecting && !t.full)
    {
        u32 const stamp = t.next_stamp++;
        u32 const slot = stamp % RT_LAB_TELEMETRY_SLOTS;
        t.issued_stamp[slot] = stamp;
        t.issued_frame[slot] = t.frame_counter;
        t.dispatch_stamp = stamp;
        t.dispatch_pixel[0] = static_cast<u32>(t.target_pixel[0]);
        t.dispatch_pixel[1] = static_cast<u32>(t.target_pixel[1]);
        t.dispatch_pending = true;
    }
    t.was_collecting = t.collecting;
    t.collecting = false;
    return rebuild;
}

void rt_lab_telemetry_destroy(RenderContext & render_context)
{
    auto & t = render_context.rt_lab.telemetry;
    if (!t.reference_buffer.id().is_empty())
    {
        render_context.gpu_context->device.destroy_buffer(t.reference_buffer.id());
        t.reference_buffer.set_buffer(daxa::BufferId{});
    }
    if (!t.buffer.id().is_empty())
    {
        render_context.gpu_context->device.destroy_buffer(t.buffer.id());
        t.buffer.set_buffer(daxa::BufferId{});
    }
}

void rt_lab_telemetry_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto & t = render_context->rt_lab.telemetry;
    if (!t.dispatch_pending)
    {
        return;
    }
    auto const & AT = RtLabTelemetryH::Info::AT;
    auto const lab_size = ti.info(AT.lab_depth).value().size;
    auto const full_size = render_context->render_data.settings.render_target_size;
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab_telemetry_compile_info().name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    RtLabTelemetryPush push = {};
    push.attach = alloc.device_address;
    push.screen_pixel = {t.dispatch_pixel[0], t.dispatch_pixel[1]};
    push.lab_pixel = {
        std::min(static_cast<u32>(static_cast<u64>(t.dispatch_pixel[0]) * lab_size.x / full_size.x), lab_size.x - 1),
        std::min(static_cast<u32>(static_cast<u64>(t.dispatch_pixel[1]) * lab_size.y / full_size.y), lab_size.y - 1),
    };
    push.slot = t.dispatch_stamp % RT_LAB_TELEMETRY_SLOTS;
    push.stamp = t.dispatch_stamp;
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({1, 1, 1});
}

void rt_lab_telemetry_reference_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto & t = render_context->rt_lab.telemetry;
    if (!t.reference_dispatch_pending)
    {
        return;
    }
    auto const & AT = RtLabTelemetryReferenceH::Info::AT;
    auto const lab_size = ti.info(AT.lab_depth).value().size;
    auto const full_size = render_context->render_data.settings.render_target_size;
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab_telemetry_reference_compile_info().name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    RtLabTelemetryReferencePush push = {};
    push.attach = alloc.device_address;
    push.lab_pixel = {
        std::min(static_cast<u32>(static_cast<u64>(t.reference_screen_pixel[0]) * lab_size.x / full_size.x), lab_size.x - 1),
        std::min(static_cast<u32>(static_cast<u64>(t.reference_screen_pixel[1]) * lab_size.y / full_size.y), lab_size.y - 1),
    };
    push.size = {lab_size.x, lab_size.y};
    push.stamp = t.reference_stamp;
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({RT_LAB_TELEMETRY_REFERENCE_RAYS / RT_LAB_TELEMETRY_REFERENCE_THREADS, 1, 1});
}

// == UI ===============================================================================================================

// Series colors of the histogram plot (the legend below the plot is drawn with the same colors, in two lines).
static constexpr ImVec4 TELEMETRY_COLOR_BARS = ImVec4(0.35f, 0.55f, 0.90f, 1.0f);
static constexpr ImVec4 TELEMETRY_COLOR_GAUSSIAN = ImVec4(0.70f, 0.50f, 0.95f, 1.0f);
static constexpr ImVec4 TELEMETRY_COLOR_LOGNORMAL = ImVec4(1.00f, 0.60f, 0.20f, 1.0f);
static constexpr ImVec4 TELEMETRY_COLOR_LOGNORMAL_3 = ImVec4(0.25f, 0.85f, 0.95f, 1.0f);
static constexpr ImVec4 TELEMETRY_COLOR_THRESHOLD = ImVec4(0.95f, 0.30f, 0.30f, 1.0f);
static constexpr ImVec4 TELEMETRY_COLOR_MEAN = ImVec4(0.75f, 0.75f, 0.75f, 1.0f);

static void rt_lab_telemetry_legend_item(ImVec4 color, char const * label, bool same_line)
{
    if (same_line)
    {
        ImGui::SameLine(0.0f, 14.0f);
    }
    ImGui::PushID(label);
    ImGui::ColorButton("##color", color, ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoAlpha, ImVec2(12.0f, 12.0f));
    ImGui::PopID();
    ImGui::SameLine(0.0f, 4.0f);
    ImGui::TextUnformatted(label);
}

static void rt_lab_telemetry_plot(char const * title, char const * unit_label, RtLabTelemetrySeries const & s)
{
    bool const log_axis = s.log_domain;
    std::string const x_label = std::string(log_axis ? "log10 " : "") + unit_label;
    ImGui::PushID(title);
    ImGui::SeparatorText(s.reference ? (std::string(title) + "  [4096 reference rays]").c_str() : title);
    if (!s.valid)
    {
        ImGui::TextDisabled("Not enough data (%llu values in the fitted domain).", static_cast<unsigned long long>(s.count));
        ImGui::PopID();
        return;
    }
    if (ImPlot::BeginPlot("##histogram", ImVec2(-1.0f, 210.0f), ImPlotFlags_NoLegend))
    {
        ImPlot::SetupAxes(x_label.c_str(), "rays", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        f32 max_count = 1.0f;
        for (f32 count : s.bin_count)
        {
            max_count = std::max(max_count, count);
        }
        ImPlot::SetNextFillStyle(TELEMETRY_COLOR_BARS);
        ImPlot::PlotBars("rays", s.bin_center.data(), s.bin_count.data(), static_cast<i32>(s.bin_center.size()), s.bin_width);
        f32 const cap = 1.5f * max_count; // a narrow fit (many identical values) is a spike that would squash the bars
        if (s.fit_std_dev > 0.0)
        {
            std::array<f32, static_cast<size_t>(TELEMETRY_FIT_PLOT_POINTS)> xs, ys;
            for (size_t i = 0; i < xs.size(); ++i)
            {
                f64 const x = s.hist_lo + (s.hist_hi - s.hist_lo) * static_cast<f64>(i) / (TELEMETRY_FIT_PLOT_POINTS - 1.0);
                f64 const z = (x - s.fit_mean) / s.fit_std_dev;
                xs[i] = static_cast<f32>(x);
                ys[i] = std::min(static_cast<f32>(static_cast<f64>(s.count) * s.bin_width * std::exp(-0.5 * z * z) / (s.fit_std_dev * std::sqrt(2.0 * std::numbers::pi))), cap);
            }
            ImPlot::SetNextLineStyle(TELEMETRY_COLOR_GAUSSIAN, 1.5f);
            ImPlot::PlotLine("gaussian", xs.data(), ys.data(), static_cast<i32>(xs.size()));
        }
        if (s.mixture.valid)
        {
            // Body / tail mixture of the distribution guide, scaled to rays per bin. The density of the log axis is per log10 unit.
            constexpr size_t MIXTURE_POINTS = 480;
            std::array<f32, MIXTURE_POINTS> xs, lognormal_ys;
            for (size_t i = 0; i < MIXTURE_POINTS; ++i)
            {
                f64 const v = s.hist_lo + (s.hist_hi - s.hist_lo) * static_cast<f64>(i) / static_cast<f64>(MIXTURE_POINTS - 1);
                f64 const x = log_axis ? std::pow(10.0, v) : v;
                f64 const jacobian = log_axis ? x * std::numbers::ln10 : 1.0;
                f64 const scale = static_cast<f64>(s.total) * s.bin_width * jacobian;
                xs[i] = static_cast<f32>(v);
                lognormal_ys[i] = std::min(static_cast<f32>(scale * rt_lab_mixture_pdf(s.mixture, x)), cap);
            }
            ImPlot::SetNextLineStyle(TELEMETRY_COLOR_LOGNORMAL, 1.5f);
            ImPlot::PlotLine("lognormal", xs.data(), lognormal_ys.data(), static_cast<i32>(MIXTURE_POINTS));
            f64 const tau_x = std::exp(s.mixture.tau) - s.mixture.eps;
            if (tau_x > 0.0)
            {
                f64 const tau_line = log_axis ? std::log10(tau_x) : tau_x;
                ImPlot::SetNextLineStyle(TELEMETRY_COLOR_THRESHOLD, 1.0f);
                ImPlot::PlotInfLines("threshold", &tau_line, 1);
            }
        }
        if (s.mixture3.valid)
        {
            constexpr size_t POINTS = 480;
            std::array<f32, POINTS> xs, lognormal_ys;
            for (size_t i = 0; i < POINTS; ++i)
            {
                f64 const v = s.hist_lo + (s.hist_hi - s.hist_lo) * static_cast<f64>(i) / static_cast<f64>(POINTS - 1);
                f64 const x = log_axis ? std::pow(10.0, v) : v;
                f64 const jacobian = log_axis ? x * std::numbers::ln10 : 1.0;
                f64 const scale = static_cast<f64>(s.total) * s.bin_width * jacobian;
                xs[i] = static_cast<f32>(v);
                lognormal_ys[i] = std::min(static_cast<f32>(scale * rt_lab_mixture3_pdf(s.mixture3, x)), cap);
            }
            ImPlot::SetNextLineStyle(TELEMETRY_COLOR_LOGNORMAL_3, 1.5f);
            ImPlot::PlotLine("lognormal 3", xs.data(), lognormal_ys.data(), static_cast<i32>(POINTS));
        }
        f64 const mean_line = s.fit_mean;
        ImPlot::SetNextLineStyle(TELEMETRY_COLOR_MEAN, 1.0f);
        ImPlot::PlotInfLines("mean", &mean_line, 1);
        ImPlot::EndPlot();
    }
    // Legend, two lines (the plot's own legend is one row and cuts off).
    rt_lab_telemetry_legend_item(TELEMETRY_COLOR_BARS, "rays", false);
    rt_lab_telemetry_legend_item(TELEMETRY_COLOR_GAUSSIAN, "gaussian", true);
    rt_lab_telemetry_legend_item(TELEMETRY_COLOR_LOGNORMAL, "lognormal body + lognormal tail", true);
    rt_lab_telemetry_legend_item(TELEMETRY_COLOR_LOGNORMAL_3, "dark tail + lognormal body + bright tail", false);
    rt_lab_telemetry_legend_item(TELEMETRY_COLOR_THRESHOLD, "body / tail threshold", true);
    rt_lab_telemetry_legend_item(TELEMETRY_COLOR_MEAN, "mean", true);
    if (s.mixture3.valid)
    {
        RtLabTelemetryMixture3 const & m3 = s.mixture3;
        ImGui::Text("three components: dark %.2f %% (%llu)  body %.2f %% (%llu)  bright %.2f %% (%llu)   dark ln mean %.3f std %.3f   bright ln mean %.3f std %.3f",
            m3.low_fraction * 100.0, static_cast<unsigned long long>(m3.low_count), (1.0 - m3.low_fraction - m3.high_fraction) * 100.0, static_cast<unsigned long long>(m3.body_count),
            m3.high_fraction * 100.0, static_cast<unsigned long long>(m3.high_count), m3.low_mean, std::sqrt(m3.low_variance), m3.high_mean, std::sqrt(m3.high_variance));
    }
    RtLabTelemetryMixture const & m = s.mixture;
    if (m.valid)
    {
        // Parameters in the log domain of the guide (y = ln(x + eps)).
        ImGui::Text("tail %.2f %% (%llu of %llu rays), threshold %.4g (%s)", m.tail_fraction * 100.0, static_cast<unsigned long long>(m.tail_count),
            static_cast<unsigned long long>(m.body_count + m.tail_count), std::exp(m.tau) - m.eps, "perceptual mean + 2");
        ImGui::Text("perceptual mean ln %.3f   body ln mean %.3f std %.3f   tail ln mean %.3f std %.3f", m.perceptual_mean, m.body_mean, std::sqrt(m.body_variance), m.tail_mean, std::sqrt(m.tail_variance));
    }
    ImGui::TextDisabled("outside the bars: %llu below, %llu above (%llu left out)", static_cast<unsigned long long>(s.below_range),
        static_cast<unsigned long long>(s.above_range), static_cast<unsigned long long>(s.excluded));
    ImGui::PopID();
}

void rt_lab_telemetry_ui(RenderContext & render_context)
{
    auto & t = render_context.rt_lab.telemetry;
    ImGui::Checkbox("Enable Pixel Telemetry", &t.enabled);
    ImGui::SetItemTooltip("Hold the left mouse button on a pixel of the viewport: the rays + surface detail of that RT Lab pixel are collected\n"
                          "every frame (the pixel stays the one the press started on) and the statistics below are computed from them. Keep the\n"
                          "camera and the lights still while collecting. Statistics use the rays after the ray filter (clamp).\n"
                          "Enabling rebuilds the task graph (the lab keeps its hit distances for the telemetry).");
    if (!t.enabled)
    {
        return;
    }
    if (render_context.render_data.rt_lab_settings.enabled == 0)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "RT Lab is off: no rays are traced.");
    }
    if (ImGui::Button("Clear"))
    {
        rt_lab_telemetry_clear(t);
        t.first_stamp = t.next_stamp;
    }
    ImGui::SameLine();
    if (t.collecting || t.was_collecting)
    {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "collecting pixel (%d, %d)", t.target_pixel[0], t.target_pixel[1]);
    }
    else
    {
        ImGui::TextDisabled("hold left click on a pixel");
    }
    bool settings_changed = false;
    settings_changed |= ImGui::Checkbox("Log Axis", &t.log_axis);
    ImGui::SetItemTooltip("Histograms and the gaussian fit over log10 of the values (zeros are left out).");
    ImGui::SameLine();
    settings_changed |= ImGui::Checkbox("Exposed Units", &t.exposed_units);
    ImGui::SetItemTooltip("Luminance x exposure x pi: the INDIRECT_DIFFUSE debug view's units. Off: scene radiance.");
    ImGui::SameLine();
    settings_changed |= ImGui::Checkbox("Use Collected Rays", &t.use_collected);
    ImGui::SetItemTooltip("Histogram, statistics, KS and MSE of the collected rays instead of the 4096 reference rays shot in one frame when the button is\n"
                          "released. The distribution fits are always made on the collected rays.");
    settings_changed |= ImGui::Checkbox("Auto Histogram", &t.auto_histogram);
    ImGui::SetItemTooltip("Range (the shortest interval with up to 99.99 %% of the non zero rays, at most 40 interquartile ranges wide) and bin count follow the collected rays, and a series with a very\n"
                          "wide spread switches to a log10 axis by itself. Off: the bins slider + the Log Axis option.");
    if (!t.auto_histogram)
    {
        settings_changed |= ImGui::SliderInt("Bins", &t.bins, 8, 128);
    }
    ImGui::SliderInt("Max Rays Kept", &t.max_samples, 500, 100000, "%d", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Collection stops when this many rays are kept (hold again to start over).");
    t.stats_dirty |= settings_changed;

    // Statistics are recomputed while collecting every other frame (the graph fills up live).
    ++t.stats_age;
    if (t.stats_dirty && (t.stats_age >= 2 || !t.was_collecting || settings_changed))
    {
        rt_lab_telemetry_compute_stats(t);
        t.stats_dirty = false;
        t.stats_age = 0;
    }
    RtLabTelemetryStats const & st = t.stats;
    RtLabTelemetrySurface const & surface = t.surface;

    // The graph first: it fills up while the button is held.
    ImGui::Text("%llu rays in %llu frames (%.2f rays / frame)%s", static_cast<unsigned long long>(st.rays), static_cast<unsigned long long>(st.frames),
        st.mean_rays_per_frame, t.full ? "   FULL" : "");
    if (t.reference_in_flight)
    {
        ImGui::TextDisabled("reference rays are being shot...");
    }
    else if (t.reference_valid)
    {
        ImGui::TextDisabled("%s: %zu reference rays (one frame, frame counter advanced per ray); distributions fitted on the %llu collected rays", t.use_collected ? "showing the collected rays" : "showing the reference rays",
            t.reference_rays.size(), static_cast<unsigned long long>(st.rays));
    }
    if (st.mean_rays_per_frame > RT_LAB_TELEMETRY_MAX_RAYS)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "More than %d rays in a frame: the rest of the pixel's rays is not collected.", RT_LAB_TELEMETRY_MAX_RAYS);
    }
    if (st.rays == 0 && !surface.valid)
    {
        ImGui::TextDisabled(t.frames > 0 ? "No geometry at this pixel (sky)." : "Nothing collected yet.");
        return;
    }
    // MSE of the three distributions against the histogram bars (absolute in rays^2 per bin, relative to the bars' energy), best one green.
    if (st.modulated.valid || st.ray_length.valid)
    {
        if (ImGui::BeginTable("telemetry_mse", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("MSE vs histogram");
            ImGui::TableSetupColumn("modulated abs");
            ImGui::TableSetupColumn("modulated rel");
            ImGui::TableSetupColumn("ray length abs");
            ImGui::TableSetupColumn("ray length rel");
            ImGui::TableHeadersRow();
            char const * const names[3] = {"gaussian", "lognormal body + tail", "dark + lognormal body + bright"};
            RtLabTelemetrySeries const * const series[2] = {&st.modulated, &st.ray_length};
            for (i32 k = 0; k < 3; ++k)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(names[k]);
                for (i32 c = 0; c < 2; ++c)
                {
                    RtLabTelemetrySeries const & s = *series[c];
                    bool best = false;
                    if (s.valid && s.mse_valid[k])
                    {
                        best = true;
                        for (i32 other = 0; other < 3; ++other)
                        {
                            best &= !s.mse_valid[other] || s.mse_rel[k] <= s.mse_rel[other];
                        }
                    }
                    for (i32 column = 0; column < 2; ++column)
                    {
                        ImGui::TableSetColumnIndex(1 + c * 2 + column);
                        if (!s.valid || !s.mse_valid[k])
                        {
                            ImGui::TextUnformatted("-");
                        }
                        else
                        {
                            ImVec4 const color = best ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f) : ImGui::GetStyleColorVec4(ImGuiCol_Text);
                            if (column == 0)
                            {
                                ImGui::TextColored(color, "%.5g", s.mse_abs[k]);
                            }
                            else
                            {
                                ImGui::TextColored(color, "%.3f %%", s.mse_rel[k] * 100.0);
                            }
                        }
                    }
                }
            }
            ImGui::EndTable();
            ImGui::SetItemTooltip("Per bar of the histogram: observed rays vs the rays the distribution expects there (values x its probability mass of the bin).\n"
                                  "abs = mean of (observed - expected)^2 over the bars (rays^2). rel = sum (observed - expected)^2 / sum observed^2.");
        }
    }
    char const * value_label = t.exposed_units ? "exposed luminance" : "luminance";
    rt_lab_telemetry_plot("Modulated incoming color (cos + albedo + normal)", value_label, st.modulated);
    rt_lab_telemetry_plot("Ray length (hits)", "distance (m)", st.ray_length);
    if (st.rays < 2)
    {
        return;
    }

    ImGui::SeparatorText("Statistics");
    if (ImGui::BeginTable("telemetry_statistics", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("");
        ImGui::TableSetupColumn("modulated");
        ImGui::TableSetupColumn("ray length");
        ImGui::TableHeadersRow();
        RtLabTelemetrySeries const * const series[2] = {&st.modulated, &st.ray_length};
        auto row = [&](char const * label, char const * tooltip, auto getter)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(label);
            if (tooltip != nullptr)
            {
                ImGui::SetItemTooltip("%s", tooltip);
            }
            for (i32 c = 0; c < 2; ++c)
            {
                ImGui::TableSetColumnIndex(c + 1);
                std::string text = getter(*series[c], c);
                ImGui::TextUnformatted(text.c_str());
            }
        };
        auto number = [](f64 v) -> std::string { return fmt::format("{:.5g}", v); };
        auto linear = [&](auto member) { return [=](RtLabTelemetrySeries const & s, i32) { return s.total >= 2 ? number(s.*member) : std::string("-"); }; };
        row("values", "Values the series has (hits only for the ray length).", [&](RtLabTelemetrySeries const & s, i32 c) {
            return c == 1 ? fmt::format("{} hits ({} misses)", s.total, s.excluded) : fmt::format("{}", s.total);
        });
        row("mean", nullptr, linear(&RtLabTelemetrySeries::mean));
        row("std dev", nullptr, linear(&RtLabTelemetrySeries::std_dev));
        row("relative std error", "std dev / sqrt(rays) / mean: the noise of the pixel's estimate with all collected rays.", linear(&RtLabTelemetrySeries::relative_std_error));
        row("median", nullptr, linear(&RtLabTelemetrySeries::median));
        row("99 %", nullptr, linear(&RtLabTelemetrySeries::p99));
        row("max", nullptr, linear(&RtLabTelemetrySeries::max));
        row("skewness", "> 0: a heavy bright tail (fireflies).", linear(&RtLabTelemetrySeries::skewness));
        row("black rays", "Share of rays with no light (zero luminance).", [&](RtLabTelemetrySeries const & s, i32 c) { return c == 1 || s.total < 2 ? std::string("-") : fmt::format("{:.1f} %", s.zero_fraction * 100.0); });
        row("tail fraction", "Share of rays above the body / tail threshold.",
            [&](RtLabTelemetrySeries const & s, i32) { return s.mixture.valid ? fmt::format("{:.2f} %", s.mixture.tail_fraction * 100.0) : std::string("-"); });
        row("KS gaussian", "Kolmogorov-Smirnov distance of the data to the fitted gaussian (in the plotted domain): 0 = perfect fit.",
            [&](RtLabTelemetrySeries const & s, i32) { return s.valid ? number(s.ks_distance) : std::string("-"); });
        row("KS lognormal body", "KS distance of the whole series to the lognormal body + lognormal tail mixture (distribution guide, scheme 1).",
            [&](RtLabTelemetrySeries const & s, i32) { return s.mixture.valid ? number(s.mixture.ks_lognormal) : std::string("-"); });
        row("KS dark + lognormal + bright", "KS distance of the whole series to the dark tail + lognormal body + bright tail mixture (distribution guide mode 3).",
            [&](RtLabTelemetrySeries const & s, i32) { return s.mixture3.valid ? number(s.mixture3.ks_lognormal) : std::string("-"); });
        ImGui::EndTable();
    }

    // Surface detail, last.
    ImGui::SeparatorText("Surface");
    ImGui::Text("pixel (%u, %u)   depth %.3f m   position (%.2f, %.2f, %.2f)", surface.screen_pixel[0], surface.screen_pixel[1], surface.linear_depth,
        surface.position[0], surface.position[1], surface.position[2]);
    ImGui::Text("face normal (%.2f, %.2f, %.2f)   detail normal (%.2f, %.2f, %.2f)", surface.face_normal[0], surface.face_normal[1], surface.face_normal[2],
        surface.detail_normal[0], surface.detail_normal[1], surface.detail_normal[2]);
    ImGui::ColorButton("##albedo", ImVec4(surface.albedo[0], surface.albedo[1], surface.albedo[2], 1.0f), ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoAlpha, ImVec2(14.0f, 14.0f));
    ImGui::SameLine();
    ImGui::Text("albedo (%.2f, %.2f, %.2f)   metalness %.2f   sky visibility %.0f %%", surface.albedo[0], surface.albedo[1], surface.albedo[2], surface.metalness,
        st.miss_fraction * 100.0);
}
