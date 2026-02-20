#include "daxa_helper.hpp"

namespace tido
{
    auto tido::make_task_buffer(daxa::Device & device, u32 size, std::string_view name, daxa::MemoryFlags flags) -> daxa::ExternalTaskBuffer
    {
        return daxa::ExternalTaskBuffer{{
            .buffer = device.create_buffer(daxa::BufferInfo{
                .size = size,
                .memory_flags = flags,
                .name = name,
            }),
            .name = name,
        }};
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