#pragma once

#pragma once
#include <filesystem>

#include "../../timberdoodle.hpp"
#include "../../multithreading/thread_pool.hpp"
using namespace tido::types;

enum struct Compression
{
    BC1,
    BC1_SDF,
    BC4,
    BC6,
    BC7,
    UNDEFINED
};

struct CreateCompressedImageInfo
{
    std::span<const std::byte> in_data;
    std::span<std::byte> out_data;
    u32vec3 image_dimensions;
    Compression compression;
};

auto compress_image(CreateCompressedImageInfo const & info) -> std::shared_ptr<Task>;