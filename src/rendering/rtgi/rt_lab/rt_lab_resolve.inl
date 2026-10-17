#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

#define RT_LAB_RESOLVE_X 8
#define RT_LAB_RESOLVE_Y 8

// Full res output (rt_lab_resolve.hlsl): divides out RTGI_RADIANCE_SCALE; at half res bilateral upsamples first.
// The split difference (split history on, else null) is upsampled with the same weights (stays RTGI_RADIANCE_SCALE scaled)
// and drives the live post-spatial RT_LAB_SPLIT_ERROR / RELATIVE_ERROR debug views.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabResolveH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_diffuse)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, resolved)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, lab_split_difference)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, split_difference_resolved)
// Coarse spatial fallback output (null when off), SH-Y + CoCg (.xy) / effective sample count (.z, / RT_LAB_SPATIAL_COUNT_SCALE):
// resolved for the receiver's normal and blended in to top up the effective sample count (rt_lab_coarse_top_up).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_coarse_sh_y)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_coarse_cocg_count)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, lab_coarse_split_difference)
// Post stabilization (push.stabilize, RtLabState::post_stabilization_*): TAA style variance clipped history of the output,
// .rgb stabilized radiance, .a linear depth (0 = none). Null when off.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, stabilize_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, stabilize_history_out)
// Post stabilization short history of the NEIGHBORHOOD (clamp box) variance (push.stabilize & RT_LAB_STABILIZE_VARIANCE_BOUND),
// packed f16 std dev YCoCg: .x (Y, Co), .y (Cg, -), .z unused, .w frames (asuint f32, <= RT_LAB_STABILIZE_VARIANCE_MAX_FRAMES).
// Appended at the end: an older C++ build never sets the bound bit, so the shader never reads these.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec4>, stabilize_variance)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec4>, stabilize_variance_out)
DAXA_DECL_TASK_HEAD_END

struct RtLabResolvePush
{
    daxa_BufferPtr(RtLabResolveH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_u32 split_enabled; // lab_split_difference / split_difference_resolved are bound
    daxa_u32 clamp_input_count; // no spatial pass ran: clamp the sample count (sample_count_input_clamp) here
    daxa_u32 coarse_enabled;    // lab_coarse_sh_y / lab_coarse_cocg_count / lab_coarse_split_difference are bound
    daxa_u32 stabilize;         // post stabilization bits RT_LAB_STABILIZE_* (an older C++ build sends 1 = enabled only)
    daxa_f32 stabilize_alpha;   // blend factor of the current output
    daxa_f32 stabilize_gamma;   // neighborhood box half width in standard deviations
    // Appended (0 = the previous behavior, so an older C++ build that does not set them keeps working):
    daxa_u32 stabilize_radius;        // variance estimator neighborhood (2r + 1)^2, 0 = 1, max RT_LAB_STABILIZE_MAX_RADIUS
    daxa_u32 stabilize_disable_clamp; // 1 = no variance clamp (plain exponential history)
};
#define RT_LAB_STABILIZE_MAX_RADIUS 3
#define RT_LAB_STABILIZE_ENABLED (1u << 0u)        // stabilize_history / stabilize_history_out are bound
#define RT_LAB_STABILIZE_VARIANCE_BOUND (1u << 1u) // stabilize_variance / stabilize_variance_out are bound (history kept up to date)
#define RT_LAB_STABILIZE_VARIANCE_CLAMP (1u << 2u) // the neighborhood variance history loosens the clamp box
#define RT_LAB_STABILIZE_VARIANCE_MAX_FRAMES 4.0f // window cap of the short temporal variance history
