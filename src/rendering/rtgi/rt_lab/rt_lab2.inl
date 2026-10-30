#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"
#include "rt_lab2_settings.inl"
#include "rt_lab_rays.inl" // RtLabRayCounters

// RT Lab 2 (rt_lab2.hpp): a second denoiser for the RT Lab trace. Same rays (rt_lab_rays / rt_lab_trace), nothing of the
// old temporal / spatial passes. A world space hash grid (a copy of the RTGI one) is the temporal integrator:
//   query      (per pixel)     find / insert the pixel's cell, count the pixel into it (cell exposure)
//   prepare    (per cell slot) exposure (visible pixel count, smoothed) + its importance scaling of the history, age / free cells
//   [ray demand / distribute / trace / gather of the old lab: a selection of full res pixels shoots the rays]
//   accumulate (per pixel)     the pixel's ray mean is summed into its cell (fixed point 64 bit atomics)
//   integrate  (per cell slot) merge the frame's sums into the cell history (sample count blend, capped)
//   lookup     (per pixel)     stochastic trilinear / cubic B-spline interpolation of the cells around the pixel
//   integrator (per pixel)     window variance of the lookup, variance driven history keep, neighborhood clamp -> output
//   resolve                    the old lab's resolve (full res copy, RTGI_RADIANCE_SCALE divided out)

#define RT_LAB2_MAX_PROBES 16
#define RT_LAB2_MAX_LEVEL 15
#define RT_LAB2_INVALID_CELL 0xFFFFFFFFu
#define RT_LAB2_PASS_X 8
#define RT_LAB2_PASS_Y 8
#define RT_LAB2_CELL_PASS_X 64
// Radiance (RTGI_RADIANCE_SCALE units) sums are 64 bit fixed point: value x this. A cell holds at most ~1e4 rays of ~1e5 -> 1e9 x 256 fits.
#define RT_LAB2_RADIANCE_FIXED_SCALE 256.0f

// One hash grid cell: (level, integer coordinate, face normal bucket) -> radiance history.
struct RtLab2Cell
{
    daxa_u32 key;              // checksum of the cell key, 0 = empty slot
    daxa_u32 frames_untouched; // frames since a pixel queried the cell
    daxa_u32 pixel_count;      // pixels that queried the cell this frame (query atomic, cleared by the integrate pass)
    daxa_u32 ray_count;        // rays summed into the cell this frame
    daxa_u64 sum_r;            // this frame's fixed point radiance sums (x RT_LAB2_RADIANCE_FIXED_SCALE)
    daxa_u64 sum_g;
    daxa_u64 sum_b;
    daxa_f32 radiance_r;       // history mean (RTGI_RADIANCE_SCALE units)
    daxa_f32 radiance_g;
    daxa_f32 radiance_b;
    daxa_f32 sample_count;     // effective samples in the history (0 = no data yet)
    daxa_f32 exposure;         // smoothed visible pixel count of the cell (0 = never seen)
    // The last integrated frame (written by the cell integrate pass, shown by the live cell probes).
    daxa_u32 last_pixel_count; // pixels that queried the cell
    daxa_u32 last_ray_count;   // rays summed into the cell
    daxa_f32 last_mean_r;      // mean of those rays (RTGI_RADIANCE_SCALE units), kept from the last frame that had rays
    daxa_f32 last_mean_g;
    daxa_f32 last_mean_b;
    daxa_u32 level;            // cascade level of the key (written by the inserting pixel, shown by the statistics)
    daxa_u32 pad1;
    // Temporal gradient (see RtLab2Settings): ray records of the last frame, the measurement, the Kalman state, the correction applied this frame.
    daxa_u32 rec_next;         // records written this frame (atomic, may exceed the slots)
    daxa_u32 rec_valid;        // records the last frame left (min(rec_next, RT_LAB2_GRADIENT_RECORDS)), re-traced this frame
    daxa_f32 meas_z;           // measurement of the re-trace: ln(new / old) (valid when meas_r > 0)
    daxa_f32 meas_r;           // its variance
    daxa_f32 gk_x;             // Kalman state: estimated log change per frame
    daxa_f32 gk_p;             // its variance (0 = not initialized)
    daxa_f32 gk_pending;       // correction not applied yet (glide)
    daxa_f32 gk_applied;       // log correction applied to the history this frame
    daxa_f32 gk_loss;          // uncertainty left after the correction (cuts the sample count)
    daxa_u32 grad_count;       // half res pixels that integrated their blurred gradient into this cell this frame
    daxa_u64 grad_sum_prec;    // sums of those pixels: precision x RT_LAB2_GRADIENT_PREC_SCALE
    daxa_u64 grad_sum_zprec;   // precision x (z + RT_LAB2_GRADIENT_Z_BIAS) x RT_LAB2_GRADIENT_PREC_SCALE
};
DAXA_DECL_BUFFER_PTR(RtLab2Cell)

// One recorded ray of a cell (the re-trace launches it bit identically: quantized direction, stored hit shading seed).
#define RT_LAB2_GRADIENT_RECORDS 4
#define RT_LAB2_GRADIENT_MAX_LOG 2.9957323f   // ln 20: limit of a correction
#define RT_LAB2_GRADIENT_PREC_SCALE 1024.0f
#define RT_LAB2_GRADIENT_Z_BIAS 4.0f
#define RT_LAB2_GRADIENT_PREC_MAX 10000.0f
struct RtLab2GradientRecord
{
    daxa_f32vec3 origin;
    daxa_f32 t_min;
    daxa_u32 direction;      // octahedral 32
    daxa_u32 data;           // RT_LAB_GRADIENT_RECORD_MAGIC | hit seed
    daxa_f32 old_luminance;  // of the ray when it was traced (RTGI_RADIANCE_SCALE units, raw)
    daxa_u32 pad;
};
DAXA_DECL_BUFFER_PTR(RtLab2GradientRecord)

// == Live cell probes (rt_lab2_probe.hpp) ==
// Every frame the cell under up to RT_LAB2_PROBE_COUNT screen pixels is copied into a host visible ring of slots. Words (u32 / bit cast f32) of a slot:
//   header (RT_LAB2_PROBE_HEADER_WORDS): 0 stamp (never 0)   1 active mask
//   probe i at RT_LAB2_PROBE_HEADER_WORDS + i * RT_LAB2_PROBE_WORDS:
//     0 flags (bit 0 geometry, bit 1 cell found, bit 2 lookup valid)   1,2 screen pixel   3 cell index   4 level   5 normal bucket   6 frames untouched   7 key
//     8..10 cell coordinate (int)   11 cell size (m)   12..14 position   15 linear depth   16..18 normal
//     19 history samples   20..22 history radiance rgb (scene units)   23 camera exposure
//     24 exposure (smoothed visible pixels)   25 last frame pixels   26 last frame rays   27 unused   28..30 last frame mean rgb
//     31 lookup valid   32..34 lookup rgb   35 unused   36..38 integrator output rgb   39 integrator frames
//     40 + t * 4 (t < 8): trilinear corner tap t: found (0 / 1), weight, samples, luminance
//     72 gk_x   73 gk_p   74 gk_applied   75 gk_loss   76 meas_z   77 meas_r   78 rec_valid   79 gk_pending
#define RT_LAB2_PROBE_COUNT 2
#define RT_LAB2_PROBE_SLOTS 8
#define RT_LAB2_PROBE_HEADER_WORDS 4
#define RT_LAB2_PROBE_WORDS 80
#define RT_LAB2_PROBE_SLOT_WORDS (RT_LAB2_PROBE_HEADER_WORDS + RT_LAB2_PROBE_COUNT * RT_LAB2_PROBE_WORDS)
#define RT_LAB2_PROBE_FLAG_GEOMETRY 1u
#define RT_LAB2_PROBE_FLAG_CELL 2u
#define RT_LAB2_PROBE_FLAG_LOOKUP 4u

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2ProbeH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLab2Cell), cells)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lookup)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, integrated)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), probe) // host visible
DAXA_DECL_TASK_HEAD_END

struct RtLab2ProbePush
{
    daxa_BufferPtr(RtLab2ProbeH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
    daxa_u32vec2 screen_pixel[RT_LAB2_PROBE_COUNT];
    daxa_u32vec2 size;
    daxa_u32 slot;
    daxa_u32 stamp;
    daxa_u32 active_mask; // bit i: probe i has a pixel
};

// == Ray redistribution on quad candidates ==
// The ray redistribution runs on a half res grid: ONE candidate pixel per 2x2 quad of the lab pixels, a different one every frame (rotation + a
// per quad offset). The select pass picks the candidate (a geometry pixel, the next sub pixel when the chosen one is sky) and its ray demand: 1 base
// ray + the extras its CELL asks for. The distribute pass (parallel per 8x8 tile of quads, scans instead of the old serial loop) spends the budget
// with the old rules and writes the ray list (full res pixel coordinates, so the trace / filter / gather stay as they are) and ray_alloc (full res).
// candidate: .x bits 0..7 ray demand (0 = none), bits 8..9 the sub pixel of the quad. counters: [0] candidates with geometry, [1] extra rays asked.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2SelectH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, pixel_cell)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLab2Cell), cells)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, candidate)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), counters)
DAXA_DECL_TASK_HEAD_END

struct RtLab2SelectPush
{
    daxa_BufferPtr(RtLab2SelectH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
    daxa_u32vec2 size;     // the candidate grid
    daxa_u32vec2 lab_size; // the lab pixels
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2DistributeH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, candidate)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), counters)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtLabRayCounters), ray_counters)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(daxa_u32), ray_list)                       // packed pixel x | y << 16
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec2>, ray_alloc)           // full res, .x list offset, .y ray count
DAXA_DECL_TASK_HEAD_END

struct RtLab2DistributePush
{
    daxa_BufferPtr(RtLab2DistributeH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
    daxa_u32vec2 size;     // the candidate grid
    daxa_u32vec2 lab_size; // the lab pixels
    daxa_u32 ray_capacity;
};

// == Demand (cell driven ray redistribution, the first version: every pixel; unused now, kept so older builds still find the entry) ==
// After the old lab's demand pass (1 base ray per geometry pixel, totals in ray_counters) and the cell prepare: raises ray_demand of every pixel
// by the extras its cell asks for and adds them to the counters' extra demand. The old distribute pass drains the budget.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2DemandH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, pixel_cell)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLab2Cell), cells)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DIndex<daxa_u32>, ray_demand)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtLabRayCounters), ray_counters)
DAXA_DECL_TASK_HEAD_END

struct RtLab2DemandPush
{
    daxa_BufferPtr(RtLab2DemandH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
    daxa_u32vec2 size;
};

// == Rays shot debug view ==
// The old distribute pass draws DEBUG_DRAW_MODE_RTGI_DIFFUSE_RAYS_SHOT / _TOTAL_RAYS_SHOT normalized by max_rays_per_pixel, which RT Lab 2 keeps at 1
// for that pass (the cell driven extras are added by the demand pass): it would be white. This pass redraws the view after it, normalized by
// RtLab2Settings::max_rays_per_pixel. Dispatched every frame, exits at once unless one of those debug modes is active.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2RaysDebugH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc) // .y ray count
DAXA_DECL_TASK_HEAD_END

struct RtLab2RaysDebugPush
{
    daxa_BufferPtr(RtLab2RaysDebugH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
    daxa_u32vec2 size;
};

// == Statistics (rt_lab2_stats.hpp) ==
// One thread per cell slot after the cell integrate pass: counters + histograms of the live cells into a host visible slot (zeroed by the CPU before
// the frame). Words (u32) of a slot:
//   0 live cells   1 cells queried last frame   2 cells with data   3 cells not queried last frame   4 pixels (sum of the coverage)   5 rays   6 cells that got rays
//   8 + level (16): live cells per level     24 + level: queried cells per level
//   40 + b (16): queried cells with data by history fill (samples / window) in 16 steps
//   56 + b (16): queried cells by log2(1 + coverage)     72 + b (16): queried cells by log2(1 + rays last frame)     88 + b (16): live cells by log2(1 + frames untouched)
#define RT_LAB2_STATS_BUCKETS 16
#define RT_LAB2_STATS_SLOTS 8
#define RT_LAB2_STATS_SLOT_WORDS 128
#define RT_LAB2_STATS_LEVEL_LIVE 8
#define RT_LAB2_STATS_LEVEL_TOUCHED 24
#define RT_LAB2_STATS_FILL 40
#define RT_LAB2_STATS_COVERAGE 56
#define RT_LAB2_STATS_RAYS 72
#define RT_LAB2_STATS_UNTOUCHED 88

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2StatsH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLab2Cell), cells)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), stats) // host visible
DAXA_DECL_TASK_HEAD_END

struct RtLab2StatsPush
{
    daxa_BufferPtr(RtLab2StatsH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
    daxa_u32 slot;
};

// == Temporal gradient passes (the re-trace is in rt_lab2_gradient.inl: it needs the ray tracing shading attachments) ==
// Record: one thread per ray list entry (after the trace): the rays that win a record slot of their cell (probability by the cell's last ray count, at most the
// coverage limited number of slots) write their launch data + luminance.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2GradientRecordH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRayCounters), ray_counters)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ray_list)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRay), rays)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, pixel_cell)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtLab2Cell), cells)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(RtLab2GradientRecord), records)
DAXA_DECL_TASK_HEAD_END

struct RtLab2GradientRecordPush
{
    daxa_BufferPtr(RtLab2GradientRecordH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
    daxa_u32vec2 size;     // lab pixels
    daxa_u32 ray_capacity;
};

// Spatial: one thread per half res pixel (a candidate pixel of its 2x2 quad): stochastic geometry aware taps of the measurements of the cells around, precision
// weighted mean, integrated into the pixel's cell (wave grouped 64 bit atomics).
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2GradientSpatialH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, pixel_cell)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, pixel_cell2)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtLab2Cell), cells)
DAXA_DECL_TASK_HEAD_END

struct RtLab2GradientSpatialPush
{
    daxa_BufferPtr(RtLab2GradientSpatialH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
    daxa_u32vec2 size;     // the half res grid
    daxa_u32vec2 lab_size; // lab pixels
};

// == Query ==
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2QueryH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtLab2Cell), cells)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, pixel_cell) // cell index, RT_LAB2_INVALID_CELL = none
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, pixel_cell2) // the neighbor cascade's cell inside the level blend region, else none
DAXA_DECL_TASK_HEAD_END

struct RtLab2QueryPush
{
    daxa_BufferPtr(RtLab2QueryH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
    daxa_u32vec2 size;
};

// == Cell prepare / integrate (one thread per cell slot) ==
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2CellPassH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtLab2Cell), cells)
DAXA_DECL_TASK_HEAD_END

struct RtLab2CellPassPush
{
    daxa_BufferPtr(RtLab2CellPassH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
};

// == Accumulate (per pixel, the old gather's lab_raw_diffuse: .rgb ray mean, .a ray count) ==
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2AccumulateH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_raw_diffuse)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, pixel_cell)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, pixel_cell2)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtLab2Cell), cells)
DAXA_DECL_TASK_HEAD_END

struct RtLab2AccumulatePush
{
    daxa_BufferPtr(RtLab2AccumulateH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
    daxa_u32vec2 size;
};

// == Lookup ==
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2LookupH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLab2Cell), cells)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, lookup) // .rgb interpolated radiance, .a = 1 valid / 0 none
DAXA_DECL_TASK_HEAD_END

struct RtLab2LookupPush
{
    daxa_BufferPtr(RtLab2LookupH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
    daxa_u32vec2 size;
};

// == Integrator ==
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2IntegratorH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lookup)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, history)     // last frame's output: .rgb radiance, .a frames
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, integrated) // this frame's output (the resolve's lab_diffuse)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, pixel_cell)       // the temporal gradient's applied correction scales the reprojected history
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLab2Cell), cells)
DAXA_DECL_TASK_HEAD_END

struct RtLab2IntegratorPush
{
    daxa_BufferPtr(RtLab2IntegratorH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
    daxa_u32vec2 size;
};

// == Cell debug draw (DEBUG_DRAW_MODE_RT_LAB2_CELLS / _CELL_SAMPLES / _CELL_EXPOSURE), full res, own cell of every pixel ==
// Folded into the lookup pass (it has the pixel's cell geometry anyway).
