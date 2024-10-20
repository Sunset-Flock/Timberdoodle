#pragma once

// Shared:

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/volumetric.inl"


DAXA_DECL_TASK_HEAD_BEGIN(VolUpdateFroxels)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(VolRaymarchFroxels)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_DECL_TASK_HEAD_END

#if !defined(__cplusplus)



#endif