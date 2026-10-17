#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

#define RT_LAB_COARSE_X 8
#define RT_LAB_COARSE_Y 8

// One pass of the coarse spatial fallback (rt_lab_coarse.hlsl): 5x5 a-trous over the diffuse as RTGI style SH-Y + CoCg
// (rtgi_shared.hlsl), so the loose normal test can gather from surfaces facing elsewhere and the resolve re-evaluates the
// gathered directional irradiance for the receiver's normal. Edge stopping: RtLabSettings::coarse_* (no plane test,
// smooth relative distance + smooth normal test). cocg_count: .xy CoCg, .z effective sample count (the plain temporal
// history length in the first pass's input, / RT_LAB_SPATIAL_COUNT_SCALE after). The split difference takes the same
// weights. Optional variance guide (RtLabSettings::coarse_variance_guide_*): same likelihood test as the fine filter on
// the mean rgb (Y of the SH-Y + CoCg), with its own (wider) tolerance.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabCoarseH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, src_sh_y)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, src_cocg_count)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, dst_sh_y)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, dst_cocg_count)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, split_src)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, split_dst)
// Per sample rgb variance from the temporal pass (the coarse fallback needs temporal, so always bound).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_variance)
DAXA_DECL_TASK_HEAD_END

struct RtLabCoarsePush
{
    daxa_BufferPtr(RtLabCoarseH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_i32 step;              // tap spacing in pixels (1, 2, 4, 8, ...)
    daxa_u32 split_enabled;     // split_src / split_dst are bound
    daxa_u32 clamp_input_count; // first pass: the input count is the plain temporal history length, clamp it
};
