#include "daxa_helper.hpp"

auto tido::make_task_buffer(daxa::Device & device, u32 size, std::string_view name) -> daxa::TaskBuffer
{
    return daxa::TaskBuffer{daxa::TaskBufferInfo{
        .initial_buffers = daxa::TrackedBuffers{ 
            .buffers = std::array{
                device.create_buffer({
                    .size = size,
                    .name = name,
                }),
            },
        },
        .name = std::string(name),
    }};
}