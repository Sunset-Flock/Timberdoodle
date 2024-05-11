#include "daxa_helper.hpp"

auto tido::make_task_buffer(daxa::Device & device, u32 size, std::string_view name) -> daxa::TaskBuffer
{
    return daxa::TaskBuffer{
        device,
        daxa::BufferInfo{
            .size = size,
            .name = name,
        },
    };
}