#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/asteroids.inl"
#include "../../shader_shared/globals.inl"

#if CPU_SIMULATION
DAXA_DECL_TASK_HEAD_BEGIN(DebugDrawAsteroidsH)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUAsteroid), asteroids);
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

struct DebugDrawAsteroidsPush
{
    DebugDrawAsteroidsH::AttachmentShaderBlob attach;
    daxa_f32vec3* asteroid_mesh_positions;
};

#else

struct AsteroidParameters
{
    daxa_f32vec3 *position;
    daxa_f32vec3 *velocity;
    daxa_f32vec3 *velocity_derivative;
    daxa_f32 *velocity_divergence;
    daxa_f32 *smoothing_radius;
    daxa_f32 *mass;
    daxa_f32 *density;
    daxa_f32 *energy;
    daxa_f32 *pressure;
    daxa_f32 *scale;
};

#define MATERIAL_UPDATE_WORKGROUP_X 256
DAXA_DECL_TASK_HEAD_BEGIN(MaterialUpdateH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_BufferPtr(daxa_f32), asteroid_params);
DAXA_DECL_TASK_HEAD_END

struct MaterialUpdatePush
{
    MaterialUpdateH::AttachmentShaderBlob attach;
    daxa_u32 asteroid_count;

    daxa_f32 *asteroid_density;
    daxa_f32 *asteroid_energy;
    daxa_f32 *asteroid_pressure;

    daxa_f32 start_density;
    daxa_f32 A;
    daxa_f32 c;
};

#define SPATIAL_HASH_INITIALIZE_WORKGROUP_X 256
DAXA_DECL_TASK_HEAD_BEGIN(InitializeHashingH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32), asteroid_params);
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(daxa_u32vec2), spatial_hash);
DAXA_DECL_TASK_HEAD_END

struct InitalizeHashingPush
{
    daxa_u32vec2 *spatial_hash;
    daxa_f32vec3 *asteroid_position;
    daxa_f32 cell_size;
    daxa_u32 asteroid_count;
};

#define RADIX_DOWNSWEEP_PASS_WORKGROUP_X 256
DAXA_DECL_TASK_HEAD_BEGIN(RadixDownsweepPassH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32vec2), spatial_hash_src);
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(daxa_u32), wg_bin_counts);
DAXA_DECL_TASK_HEAD_END

struct RadixDownsweepPassPush
{
    RadixDownsweepPassH::AttachmentShaderBlob attach;
    daxa_u32 asteroid_count;
    daxa_u32 pass_index;
};

#define RADIX_SCAN_PASS_WORKGROUP_X 1024
DAXA_DECL_TASK_HEAD_BEGIN(RadixScanPassH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_BufferPtr(daxa_u32), wg_bin_counts);
DAXA_DECL_TASK_HEAD_END

struct RadixScanPassPush
{
    RadixScanPassH::AttachmentShaderBlob attach;
    daxa_u32 downsweep_pass_wg_count;
};

#define RADIX_SCAN_FINALIZE_PASS_WORKGROUP_X 256
DAXA_DECL_TASK_HEAD_BEGIN(RadixScanFinalizePassH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_BufferPtr(daxa_u32), wg_bin_counts);
DAXA_DECL_TASK_HEAD_END

struct RadixScanFinalizePassPush
{
    RadixScanFinalizePassH::AttachmentShaderBlob attach;
    daxa_u32 downsweep_pass_wg_count;
};

#define RADIX_UPSWEEP_PASS_WORKGROUP_X 256

#ifdef __cplusplus
static_assert(RADIX_UPSWEEP_PASS_WORKGROUP_X == RADIX_DOWNSWEEP_PASS_WORKGROUP_X, "upsweep and downsweep require the same workgroup sizes");
#endif
DAXA_DECL_TASK_HEAD_BEGIN(RadixUpsweepPassH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32vec2), src_spatial_hash);
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(daxa_u32vec2), dst_spatial_hash);
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), wg_bin_counts);
DAXA_DECL_TASK_HEAD_END

struct RadixUpsweepPassPush
{
    RadixUpsweepPassH::AttachmentShaderBlob attach;
    daxa_u32 asteroid_count;
    daxa_u32 pass_index;
    daxa_u32 downsweep_pass_wg_count;
};

#define SPATIAL_HASH_FINALIZE_WORKGROUP_X 256
DAXA_DECL_TASK_HEAD_BEGIN(FinalizeHashingH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32vec2), spatial_hash);
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(daxa_u32), cell_start_indices);
DAXA_DECL_TASK_HEAD_END

struct FinalizeHashingPush
{
    FinalizeHashingH::AttachmentShaderBlob attach;
    daxa_u32 asteroid_count;
};

#define DERIVATIVES_CALCULATION_WORKGROUP_X 256
DAXA_DECL_TASK_HEAD_BEGIN(DerivativesCalculationH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_BufferPtr(daxa_f32), asteroid_params);
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32vec2), spatial_hash);
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), cell_start_indices);
DAXA_DECL_TASK_HEAD_END

struct DerivativesCalculationPush
{
    daxa_u32 asteroid_count;
    daxa_f32 max_smoothing_radius;
    daxa_f32 cell_size;

    daxa_f32vec3 *position;
    daxa_f32vec3 *velocity;
    daxa_f32vec3 *velocity_derivative;
    daxa_f32 *velocity_divergence;
    daxa_f32 *smoothing_radius;
    daxa_f32 *mass;
    daxa_f32 *density;
    daxa_f32 *pressure;

    daxa_u32vec2 *spatial_lookup;
    daxa_u32 *cell_start_indices;
};

#define EQUATION_UPDATE_WORKGROUP_X 256
DAXA_DECL_TASK_HEAD_BEGIN(EquationUpdateH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_BufferPtr(daxa_f32), asteroid_params);
DAXA_DECL_TASK_HEAD_END

struct EquationUpdatePush
{
    daxa_u32 asteroid_count;
    daxa_f32 dt;
    AsteroidParameters params;
};

DAXA_DECL_TASK_HEAD_BEGIN(DebugDrawAsteroidsH)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_f32), asteroid_params);
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

struct DebugDrawAsteroidsPush
{
    DebugDrawAsteroidsH::AttachmentShaderBlob attach;
    daxa_f32vec3* asteroid_mesh_positions;

    daxa_f32vec3 *positions;
    AsteroidParameters parameters;
};

#endif