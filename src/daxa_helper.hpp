#pragma once

#include "timberdoodle.hpp"

namespace tido
{
    auto make_task_buffer(daxa::Device & device, u32 size, std::string_view name) -> daxa::TaskBuffer;
}