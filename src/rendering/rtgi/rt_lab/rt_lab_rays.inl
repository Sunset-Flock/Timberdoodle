#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

// Ray redistribution (rt_lab_rays.hlsl): a simplified port of the RTGI ray budget (rtgi_distribute_rays.hlsl), diffuse
// only. Per frame:
//   demand     (per lab pixel) writes the lab geometry, reprojects the history length exactly like the temporal pass
//              (rt_lab_reproject.hlsl: bilinear footprint, occluder aware soft normalize) and turns it into a desired
//              ray count: 1 base ray + calc_desired_extra_rays extras, at most max_rays_per_pixel. Totals the base /
//              extra demand into RtLabRayCounters.
//   distribute (one workgroup per 8x8 tile) drains the totals to the hard budget like RTGI: min_ray_budget base coverage
//              first, the extras get the rest at one global ratio, leftover budget raises the base coverage. Each tile
//              gets exactly floor(fraction x its demand): base rays in rotated Bayer order, extras by a cumulative
//              demand scan. Reserves the tile's rays in the ray list (one atomic), writes ray_alloc (.x offset, .y count).
//   trace      (per ray list entry, rt_lab_trace.hlsl) writes one RtLabRay per entry.
//   gather     (per lab pixel) averages the pixel's rays into lab_raw_diffuse (.rgb mean, .a = ray count = samples).
// Redistribution off: every geometry pixel wants exactly 1 ray and the budget is 1 ray per pixel (= the old trace).
//
// Temporal gradient (RtLabSettings::temporal_gradient_*, A-SVGF style), per frame:
//   gradient trace (rt_lab_trace.inl) re-traces last frame's gradient records -> gradient_samples (per cell)
//   gradient blur  screen space gaussian over the cells, edge aware (bilateral: each tap's cell geometry vs the center
//                  cell's, relative world distance + loose normal weight), separable (horizontal, vertical) -> temporal_gradient (per cell:
//                  lambda + the cell's geometry)
//   demand / temporal read lambda per pixel with an edge aware 2x2 bilinear upscale (rt_lab_temporal_gradient)
//   demand / temporal scale the reprojected history lengths by (1 - lambda) BEFORE the ray redistribution
//   gather picks one traced ray per RT_LAB_GRADIENT_CELL^2 cell (first ray of a pixel with rays, pixel order rotated)
//                  and stores its launch data + result as the cell's record for the next frame's re-trace.
#define RT_LAB_RAYS_X 8
#define RT_LAB_RAYS_Y 8
#define RT_LAB_TRACE_THREADS 64
#define RT_LAB_GRADIENT_CELL 4                     // one gradient sample per 4x4 lab pixels
// temporal_gradient_all_base_rays: per cell fixed point sums of the re-traced samples (exposed delta / max x this,
// each sample clamped to +-RT_LAB_GRADIENT_SUM_CLAMP, so 16 samples stay inside int32).
#define RT_LAB_GRADIENT_SUM_SCALE 16384.0f
// RtLabSettings::temporal_gradient_pixel_relative: what lambda is relative to.
#define RT_LAB_GRADIENT_RELATIVE_SAMPLES 0 // A-SVGF: |sum w delta| / sum w max of the re-traced samples
#define RT_LAB_GRADIENT_RELATIVE_PIXEL 1   // |mean delta| (absolute through blur + dilation) / the pixel's own max(old, new)
#define RT_LAB_GRADIENT_RELATIVE_MIN 2     // min of both (bright pixel near a dark change: pixel; dark pixel near a bright change: samples)
#define RT_LAB_GRADIENT_SUM_CLAMP 4000.0f
#define RT_LAB_GRADIENT_HISTORY_PLANE_TOLERANCE 1.0f // temporal_gradient_history_alpha: plane distance in cell widths
// Kalman temporal gradient (RT_LAB_GRADIENT_CAP_KALMAN), per cell in the blur's vertical pass:
//   measurement z = log(sum w new / sum w old) of the re-traced samples in reach, variance R = sum w^2 (new - r old)^2 /
//   (sum w new)^2 (ratio estimator, delta method; the blur carries the w^2 moment sums, separable as w_h^2 w_v^2)
//   filter (x, P) per cell, reprojected (nearest previous cell, plane test): P += q; innovation^2 > gate^2 (P + R) =
//   reset to (z, R) (shock: explosions / flashes react at once), else the usual Kalman update
//   correction c = clamp(x^3 / (x^2 + P), +-RT_LAB_GRADIENT_KALMAN_MAX_LOG) (Wiener shrunk: noise -> ~0, a measured
//   change in full), information loss L = P + (x - c)^2 (uncertainty + what the clamp could not correct)
//   temporal_gradient .x = L (dilated like lambda), .w = c (undilated). Consumers: history x exp(c) (rescale),
//   length N' = N / (1 + N k L / v).
#define RT_LAB_GRADIENT_KALMAN_MAX_LOG 2.9957323f    // ln 20, the rescale clamp
#define RT_LAB_GRADIENT_KALMAN_MIN_VARIANCE 1e-6f    // measurement variance floor
#define RT_LAB_GRADIENT_KALMAN_NO_SAMPLE_DECAY 0.5f  // no samples in reach: the tracked change decays (x *= this)
// gradient_record_data.w of a valid record = RT_LAB_GRADIENT_RECORD_MAGIC (top 8 bits) | hit shading seed (low 24 bits)
#define RT_LAB_GRADIENT_RECORD_MAGIC 0xA5000000u
#define RT_LAB_GRADIENT_RECORD_MAGIC_MASK 0xFF000000u
#define RT_LAB_HIT_SEED_MASK 0x00FFFFFFu
#define RT_LAB_GRADIENT_NO_RAY 0xFFFFFFFFu         // gradient selection: the cell has no ray this frame
#define RT_LAB_GRADIENT_RESCALE_MAX_LAMBDA 0.95f   // history rescale: |signed lambda| clamp (ratio within [0.05, 20])
// Edge aware gradient filtering, COARSE tests (rt_lab_gradient_surface_weight). Each cell is represented by the geometry
// of its lab pixel cell x CELL + CELL / 2.
#define RT_LAB_GRADIENT_NORMAL_POWER 1.0f              // normal weight ((1 + dot) / 2)^power
// Gradient blur + upscale edge stopping (rt_lab_gradient_distance_weight): no plane test, world distance relative to the
// screen offset (falloff tolerance like coarse_distance_tolerance) x a somewhat loose normal weight ((1 + dot) / 2)^power.
#define RT_LAB_GRADIENT_DISTANCE_TOLERANCE 2.0f
#define RT_LAB_GRADIENT_DISTANCE_NORMAL_POWER 2.0f
#define RT_LAB_GRADIENT_UPSCALE_PLANE_TOLERANCE 4.0f   // anti-lag upscale (rt_lab_coarse_anti_lag_stats): plane distance falloff in pixel widths

struct RtLabRayCounters
{
    daxa_u32 base_demand;  // geometry pixels (1 base ray each)
    daxa_u32 extra_demand; // extra rays wanted on top of the base rays
    daxa_u32 ray_count;    // ray list entries reserved by the distribute pass (never above the budget)
    daxa_u32 pad;
};
DAXA_DECL_BUFFER_PTR(RtLabRayCounters)

struct RtLabRay
{
    daxa_f32vec3 radiance; // rgb * RTGI_RADIANCE_SCALE
    daxa_u32 direction;    // octahedral 32
};
DAXA_DECL_BUFFER_PTR(RtLabRay)

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabDemandH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)                 // full res NDC depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)        // half res linear depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, lab_depth)                     // linear depth, 0 = sky
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, lab_face_normal)               // octahedral 32
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, history)                    // last frame's history, null without temporal
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, ray_demand)                    // desired rays, 0 = sky
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtLabRayCounters), ray_counters)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec4>, temporal_gradient)          // per cell (blur output), null when off
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, variance_history)           // last frame's per sample variance (.rgb), null without temporal
DAXA_DECL_TASK_HEAD_END

struct RtLabDemandPush
{
    daxa_BufferPtr(RtLabDemandH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_u32 history_bound;  // history + variance_history are bound (temporal on)
    daxa_u32 gradient_bound; // temporal_gradient is bound
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabDistributeH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, ray_demand)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtLabRayCounters), ray_counters)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(daxa_u32), ray_list)                             // packed pixel x | y << 16
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec2>, ray_alloc)                 // .x list offset, .y ray count
DAXA_DECL_TASK_HEAD_END

struct RtLabDistributePush
{
    daxa_BufferPtr(RtLabDistributeH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_u32 ray_capacity; // ray list entries
};

// Ray filter (RtLabSettings::ray_filter_*, rt_lab_rays.hlsl entry_rt_lab_ray_filter): one thread per ray list entry.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabRayFilterH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRayCounters), ray_counters)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ray_list)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRay), rays)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(RtLabRay), filtered_rays)
DAXA_DECL_TASK_HEAD_END

struct RtLabRayFilterPush
{
    daxa_BufferPtr(RtLabRayFilterH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;     // lab resolution
    daxa_u32 ray_capacity; // ray list entries
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabGatherH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRay), rays)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, lab_raw_diffuse)           // .rgb mean * RTGI_RADIANCE_SCALE, .a = rays
// Temporal gradient sample selection (null when off): the corner thread of each cell picks one of the cell's traced rays
// and writes the cell's record (launch data from the lab geometry, direction + result from the ray).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, gradient_record_origin)    // .xyz origin, .w t min
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, gradient_record_surface)   // .xyz surface, .w luminance
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec4>, gradient_record_data)      // dir, face normal, unused, magic
// Ray filter output (null when off): the pixel means come from these; the gradient selection keeps the raw `rays`.
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRay), filtered_rays)
// Firefly energy weighting (bound with the ray filter + temporal, else null): mean RAW (unclamped) luminance of the pixel's
// rays (RTGI_RADIANCE_SCALE scaled; f32, raw tails overflow f16), 0 without rays.
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, lab_raw_luminance)
DAXA_DECL_TASK_HEAD_END

struct RtLabGatherPush
{
    daxa_BufferPtr(RtLabGatherH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_u32 gradient_enabled; // the gradient images are bound
    daxa_u32 filtered_bound;   // filtered_rays is bound
    daxa_u32 gradient_per_pixel; // temporal_gradient_all_base_rays: every pixel records its base ray (lab res records)
    daxa_u32 raw_luminance_bound; // lab_raw_luminance is bound
};

// Screen space gaussian over the gradient cells, edge aware and separable: tap weight = gaussian x the tap cell's
// geometry vs the center cell's (rt_lab_gradient_distance_weight: relative world distance + loose normal, no plane test),
// sky cells skipped. lambda = |sum w delta| / sum w max (A-SVGF), 0 without samples
// in reach. Cell geometry comes straight from the g-buffer (the lab geometry of this frame is written later, by the demand).
//   horizontal: gradient_samples (packed) -> blur_sums_dst (.x sum w delta, .y sum w max, .z sum w samples, .w sum w old)
//   vertical:   blur_sums -> temporal_gradient (.x lambda bits, .y the cell's linear depth bits (0 = sky), .z its packed
//               face normal, .w signed undilated lambda), read by rt_lab_temporal_gradient (rt_lab_shared.hlsl).
//               temporal_gradient_pixel_relative != 0: .x / .w = f16 pairs (lambda | |mean delta| << 16) / (signed lambda |
//               signed mean delta << 16), mean delta = exposed sum w delta / sum w samples, made relative per pixel by
//               the consumers (rt_lab_gradient_pixel_relative).
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabGradientBlurH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)                 // full res NDC depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)        // half res linear depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, gradient_samples)        // horizontal pass input
// temporal_gradient_all_base_rays: per cell fixed point sums (delta as int bits, max) instead of gradient_samples.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, gradient_delta_sum)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, gradient_max_sum)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, gradient_count_sum)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, blur_sums_dst)     // horizontal pass output
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, blur_sums)          // vertical pass input
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec4>, temporal_gradient) // vertical pass output
// Vertical pass: smoothed sums per cell (sum w delta, sum w max, sum w samples, cell linear depth), last frame's + this
// frame's (temporal_gradient_history_alpha).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, gradient_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, gradient_history_dst)
// Kalman: per cell sum of the old luminance (temporal_gradient_all_base_rays), the w^2 moment sums (sum w^2 delta^2,
// sum w^2 delta old, sum w^2 old^2) horizontal output / vertical input.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, gradient_old_sum)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, blur_moments_dst)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, blur_moments)
DAXA_DECL_TASK_HEAD_END

// Weighted max dilation of lambda after the blur (temporal_gradient_dilation_radius > 0): the blur estimates lambda (noise),
// the dilation gives it reach without diluting it at its origin: lambda_out = max over taps of lambda_tap x gauss(d)
// (sigma = radius / 2) x rt_lab_gradient_distance_weight. Separable (max of products of a separable kernel), the edge
// weight per pass like the blur. Cell depth / normal come from the input (temporal_gradient layout), passed through.
//   horizontal: src (blurred temporal_gradient) -> dst (intermediate), vertical: intermediate -> temporal_gradient
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabGradientDilateH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec4>, src)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec4>, dst)
DAXA_DECL_TASK_HEAD_END

struct RtLabGradientDilatePush
{
    daxa_BufferPtr(RtLabGradientDilateH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;     // gradient cells
    daxa_u32vec2 lab_size; // lab pixels
    daxa_u32 vertical;     // 0 = horizontal pass, 1 = vertical pass
};

struct RtLabGradientBlurPush
{
    daxa_BufferPtr(RtLabGradientBlurH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;     // gradient cells
    daxa_u32vec2 lab_size; // lab pixels
    daxa_u32 vertical;     // 0 = horizontal pass, 1 = vertical pass
    daxa_u32 per_pixel;    // horizontal pass input = gradient_delta_sum / gradient_max_sum
};
