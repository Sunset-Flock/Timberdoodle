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

#define MAKE_COMPUTE_COMPILE_INFO(NAME, PATH, ENTRY)                                                   \
    auto NAME() -> daxa::ComputePipelineCompileInfo2 const &                                           \
    {                                                                                                  \
        static const daxa::ComputePipelineCompileInfo2 info = []() {                                   \
            return daxa::ComputePipelineCompileInfo2{                                                  \
                .source = daxa::ShaderSource{daxa::ShaderFile{PATH}},                                  \
                .entry_point = std::string(ENTRY),                                                     \
                .name = (std::filesystem::path(PATH).filename().string() + "::") + std::string(ENTRY), \
            };                                                                                         \
        }();                                                                                           \
        return info;                                                                                   \
    }
} // namespace tido