#include "daxa_helper.hpp"

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