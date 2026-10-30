#pragma once

// MUST be the first include (it defines DAXA_RAY_TRACING before the first daxa include).
#include "rt_lab_trace.inl" // RT_LAB_SHADE_ATTACHMENTS

#include "rt_lab2.inl"

// Temporal gradient re-trace (rt_lab2_gradient_trace.hlsl): one thread per cell slot. Re-traces the cell's recorded rays of the last frame (rt_lab2.inl,
// RtLab2GradientRecord) with the same shading as the trace (rt_lab_trace_radiance, bit identical: quantized direction, stored hit seed), and writes the cell's
// measurement meas_z = ln(sum new / sum old luminance), meas_r = its variance.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab2GradientTraceH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtLab2Cell), cells)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLab2GradientRecord), records)
RT_LAB_SHADE_ATTACHMENTS
DAXA_DECL_TASK_HEAD_END

struct RtLab2GradientTracePush
{
    daxa_BufferPtr(RtLab2GradientTraceH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab2Settings) settings;
};
