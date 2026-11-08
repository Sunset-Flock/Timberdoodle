#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

// Temporal gradient 2: the re-traced samples as a directional pyramid, resolved to a per pixel lighting change.
//
// The temporal gradient re-trace writes one sample per gradient cell (4x4 lab pixels): the exposed luminance the ray had
// last frame, the luminance it has now, the ray direction and the world position of the surface. Per level:
//   positions: one world position per area, the SMOOTH MAJORITY of its children: start at the weighted mean, then a few mean
//              shift iterations with a shrinking gaussian (the estimate slowly locks into the densest cluster, outliers
//              lose their weight). .w = the accepted weight (0 = empty area). Level 0 children are cells, level n children
//              are the areas of level n - 1 (weighted by their own accepted weight).
//   sums:      one RT_LAB_GRADIENT_OCT x RT_LAB_GRADIENT_OCT octahedral map per area. Each texel holds the weighted sums of old, new,
//              the weight w and w^2 of the children whose direction falls into it, plus the second moments (old^2, old new, new^2)
//              in a second image. A child's weight is a gaussian of its distance to the area position (plain world space
//              distance, no plane test), so far off samples are not accepted already while downsampling.
//   denoise (optional): per texel the weighted sum over the 5x5 neighbouring areas of the same level. The weight is a gaussian
//              of the distance between the two area positions, i.e. the denoising only reaches across probes that are close in
//              world space.
//   convolve:  per texel the cosine weighted mean of old and new over the covered texels, the lighting arriving at a surface
//              facing that direction (E_old, E_new), the covered share of the lobe (support) and the variance of the log change
//              (delta method over the ratio estimator, from the second moments).
//   resolve:   per lab pixel and level each of the 4 nearest areas that passes the world distance gate between the pixel and the area
//              position is read at the pixel's normal (4 wrapped taps): c = ln(E_new / E_old), its variance, shrunk (James-Stein) to
//              c max(0, 1 - var / c^2). The level takes the largest shrunk change of its areas (max, no averaging). The highest valid
//              level is the baseline, every finer level can only enlarge it (a smaller change never calms it down).
//   level 0: one map per RT_LAB_GRADIENT_OCT_AREA^2 cells (16x16 lab pixels), level 1: per 4x4 level 0 areas (64x64),
//   level 2: per 4x4 level 1 areas (256x256).
#define RT_LAB_GRADIENT_OCT 4      // octahedral map resolution (texels per axis)
#define RT_LAB_GRADIENT_OCT_AREA 4 // children per axis: cells per level 0 area, level 0 areas per level 1 area, ...
#define RT_LAB_GRADIENT_OCT_DENOISE_RADIUS 2 // 5x5 areas
#define RT_LAB_GRADIENT_OCT_MAX_LOG 2.9957323f // ln 20: the largest change a pixel gets
#define RT_LAB_GRADIENT_OCT_GATE_MIN 0.1f     // resolve: an area counts when the gaussian of its world distance to the pixel is at least this
#define RT_LAB_GRADIENT_OCT_SUPPORT_MIN 0.05f // resolve: and its map at the pixel's normal covers at least this share of the lobe

#define RT_LAB_GRADIENT_OCT_MODE_BIN_CELLS 0       // level 0: cell samples -> maps
#define RT_LAB_GRADIENT_OCT_MODE_BIN_MAPS 1        // level 1+: child maps -> maps (weighted sums)
#define RT_LAB_GRADIENT_OCT_MODE_CONVOLVE 2        // cosine convolution of a map image
#define RT_LAB_GRADIENT_OCT_MODE_POSITIONS_CELLS 3 // level 0 area positions from the cell positions
#define RT_LAB_GRADIENT_OCT_MODE_POSITIONS_AREAS 4 // level 1+ area positions from the child area positions
#define RT_LAB_GRADIENT_OCT_MODE_DENOISE 5         // 5x5 accumulation of the map sums over close areas

// Generic pass (one thread per output texel, or per output area for the position modes; a 4x4 workgroup = one area of maps).
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabGradientOctH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, src_cells)      // (old, new, direction u + 1, v) per gradient cell, z = 0: none
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, src_maps)       // maps: (sum w old, sum w new, sum w, sum w^2) / convolved: (E old, E new, support, variance of ln E new / E old)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, src_moments)    // (sum w old^2, sum w old new, sum w new^2, 0)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, src_positions)  // child positions: cells (xyz, 1 = valid) or areas of the level below (xyz, weight)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, area_positions) // the positions of this level's areas
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, dst_maps)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, dst_moments)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, dst_positions)
DAXA_DECL_TASK_HEAD_END

struct RtLabGradientOctPush
{
    daxa_BufferPtr(RtLabGradientOctH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;     // the dispatch image (dst_maps, or dst_positions in the position modes)
    daxa_u32vec2 src_size; // the source image (cells or maps or positions)
    daxa_u32 mode;
    daxa_u32 area_pixels;  // lab pixels per area of this level
    daxa_u32 lab_height;   // lab resolution height (world space pixel width)
    daxa_u32 denoise;      // the DENOISE mode copies when 0
    daxa_f32 distance_scale; // world space tolerance = area footprint x this
};

// Per lab pixel: the resolve (writes gradient2) and the debug views (the balls, the gradient2 views).
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabGradientOctResolveH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
// The g-buffer geometry (the resolve runs before the ray demand, which writes the lab geometry).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)                 // full res NDC depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)        // half res linear depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, convolved0)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, convolved1)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, convolved2)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, positions0)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, positions1)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, positions2)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, sums0) // unconvolved maps (weighted sums), denoised if on (raw views)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, sums1)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, sums2)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, gradient2) // per pixel result, RGBA16F: shrunk log change, bias variance, level, valid
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_DECL_TASK_HEAD_END

struct RtLabGradientOctResolvePush
{
    daxa_BufferPtr(RtLabGradientOctResolveH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;       // lab resolution
    daxa_u32 lab_height;
    daxa_u32 resolve;        // write gradient2 (the history scale or a gradient2 view needs it)
    daxa_f32 distance_scale; // see RtLabGradientOctPush
};
