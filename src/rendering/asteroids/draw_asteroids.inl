#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/asteroids.inl"
#include "../../shader_shared/globals.inl"

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