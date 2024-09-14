#pragma once

#include "timberdoodle.hpp"

namespace tido
{
    auto make_task_buffer(daxa::Device & device, u32 size, std::string_view name, daxa::MemoryFlags flags = {}) -> daxa::TaskBuffer;

    auto upgrade_compute_pipeline_compile_info(daxa::ComputePipelineCompileInfo const & old) -> daxa::ComputePipelineCompileInfo2;

    auto channel_count_of_format(daxa::Format format) -> u32;
}