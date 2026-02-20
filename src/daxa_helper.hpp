#pragma once

#include "timberdoodle.hpp"

namespace tido
{
    auto make_task_buffer(daxa::Device & device, u32 size, std::string_view name, daxa::MemoryFlags flags = {}) -> daxa::ExternalTaskBuffer;

    auto compute_shader_info(char const * ident) -> daxa::ComputePipelineCompileInfo2;

#define MAKE_COMPUTE_COMPILE_INFO(NAME, PATH, ENTRY)                                                   \
    inline auto NAME() -> daxa::ComputePipelineCompileInfo2 const &                                    \
    {                                                                                                  \
        static const daxa::ComputePipelineCompileInfo2 info = []() {                                   \
            return daxa::ComputePipelineCompileInfo2{                                                  \
                .source = daxa::ShaderSource{daxa::ShaderFile{PATH}},                                  \
                .entry_point = std::string(ENTRY),                                                     \
                .required_subgroup_size = WARP_SIZE,                                                   \
                .name = (std::filesystem::path(PATH).filename().string() + "::") + std::string(ENTRY), \
            };                                                                                         \
        }();                                                                                           \
        return info;                                                                                   \
    }
} // namespace tido