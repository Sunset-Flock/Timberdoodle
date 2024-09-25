#pragma once

#include "timberdoodle.hpp"

namespace tido
{
    auto make_task_buffer(daxa::Device & device, u32 size, std::string_view name, daxa::MemoryFlags flags = {}) -> daxa::TaskBuffer;

    auto upgrade_compute_pipeline_compile_info(daxa::ComputePipelineCompileInfo const & old) -> daxa::ComputePipelineCompileInfo2;

    auto channel_count_of_format(daxa::Format format) -> u32;

    enum struct ScalarKind
    {
        FLOAT,
        INT,
        UINT
    };
    auto scalar_kind_of_format(daxa::Format format) -> ScalarKind;

    auto is_format_depth_stencil(daxa::Format format) -> bool;

    auto compute_shader_info(char const * ident) -> daxa::ComputePipelineCompileInfo2;
}