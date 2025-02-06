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

DAXA_DECL_TASK_HEAD_BEGIN(MaterialUpdateH)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE, daxa_BufferPtr(daxa_f32), asteroid_params);
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(DebugDrawAsteroidsH)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_f32), asteroid_params);
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

struct AsteroidParameters
{
    daxa_f32vec3 *position;
    daxa_f32vec3 *velocity;
    daxa_f32vec3 *velocity_derivative;
    daxa_f32 *velocity_divergence;
    daxa_f32 *smoothing_radius;
    daxa_f32 *mass;
    daxa_f32 *density;
    daxa_f32 *density_derivative;
    daxa_f32 *energy;
    daxa_f32 *energy_derivative;
    daxa_f32 *pressure;
    daxa_f32 *scale;
};

struct DebugDrawAsteroidsPush
{
    DebugDrawAsteroidsH::AttachmentShaderBlob attach;
    daxa_f32vec3* asteroid_mesh_positions;

    daxa_f32vec3 *positions;
    AsteroidParameters parameters;
};
#endif