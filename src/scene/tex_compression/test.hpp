#pragma once

#include "../../timberdoodle.hpp"
using namespace tido::types;

#include "../../multithreading/thread_pool.hpp"

#include "test.inl"
#include "../../window.hpp"

struct TestContext
{
    daxa::Instance instance = {};
    daxa::Device device = {};
    daxa::PipelineManager pipeline_manager = {};
    daxa::Swapchain swapchain = {};
    std::unique_ptr<Window> window = {};

    int shader_iterations = 64;
    int dispatch_iterations = 1;

    std::unique_ptr<ThreadPool> thread_pool;

    std::shared_ptr<daxa::ComputePipeline> test_pipeline = {};
    // First texture is 2D, second is 2D array thrid is 3D
    std::array<daxa::ImageId, 3> raw_test_images = {};
    std::array<daxa::ImageId, 3> BC1_test_images = {};
    std::array<daxa::ImageId, 3> BC4_test_images = {};
    std::array<daxa::ImageId, 3> BC6_test_images = {};
    std::array<daxa::ImageId, 3> BC7_test_images = {};
};

enum struct TextureContent : u8
{
    CHECKERBOARD,
    UV_GRADIENT,
    BLACK,
    UNDEFINED
};

void prepare_test_textures(u32vec3 const test_textures_dimensions, TextureContent const content, TestContext & context);
void test_image(TestContext & ctx, std::array<daxa::ImageId, 3> images, u32vec3 texture_dimensions);

void test_main();