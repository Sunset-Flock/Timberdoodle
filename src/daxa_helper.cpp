#include "daxa_helper.hpp"

namespace tido
{
    auto tido::make_task_buffer(daxa::Device & device, u32 size, std::string_view name, daxa::MemoryFlags flags) -> daxa::TaskBuffer
    {
        return daxa::TaskBuffer{
            device,
            daxa::BufferInfo{
                .size = size,
                .allocate_info = flags,
                .name = name,
            },
        };
    }

    auto upgrade_compute_pipeline_compile_info(daxa::ComputePipelineCompileInfo const & old) -> daxa::ComputePipelineCompileInfo2
    {
        daxa::ComputePipelineCompileInfo2 info = {};
        info.source = std::move(old.shader_info.source);
        info.entry_point = std::move(old.shader_info.compile_options.entry_point);
        info.language = std::move(old.shader_info.compile_options.language);
        info.defines = std::move(old.shader_info.compile_options.defines);
        info.enable_debug_info = std::move(old.shader_info.compile_options.enable_debug_info);
        info.create_flags = std::move(old.shader_info.compile_options.create_flags);
        info.required_subgroup_size = std::move(old.shader_info.compile_options.required_subgroup_size);
        info.push_constant_size = std::move(old.push_constant_size);
        info.name = std::move(old.name);
        return info;
    }

    auto compute_shader_info(char const * ident) -> daxa::ComputePipelineCompileInfo2
    {
        std::string str(ident);
        auto offset = str.find(':');
        std::string path = str.substr(0, offset);
        std::string entry = str.substr(offset+1, str.size());
        daxa::ComputePipelineCompileInfo2 ret = {
            .source = daxa::ShaderFile{path},
            .entry_point = entry,
            .language = daxa::ShaderLanguage::SLANG,
            .defines = {},
            .enable_debug_info = {},
            .create_flags = {},
            .required_subgroup_size = {},
            .push_constant_size = {},
            .name = str,
        };
        return ret;
    }
} // namespace tido