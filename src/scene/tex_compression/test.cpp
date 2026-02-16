#include "test.hpp"
#include "tex_compression.hpp"

#include <iostream>

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_NATIVE_INCLUDE_NONE
using HWND = void *;
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#include <GLFW/glfw3native.h>

void test_main()
{
    static bool initialized = false;
    static TestContext test_context;
    const u32vec3 test_dimensions = {64, 64, 32};
    if(! initialized)
    {
        initialized = true;
        test_context.thread_pool = std::make_unique<ThreadPool>(7);

        test_context.instance = daxa::create_instance({});
        test_context.device = [&](){
                auto required_implicit =
                    daxa::ImplicitFeatureFlagBits::BASIC_RAY_TRACING |
                    daxa::ImplicitFeatureFlagBits::MESH_SHADER |
                    daxa::ImplicitFeatureFlagBits::SHADER_CLOCK |
                    // daxa::ImplicitFeatureFlagBits::HOST_IMAGE_COPY |
                    daxa::ImplicitFeatureFlagBits::SWAPCHAIN;

                auto device_info = daxa::DeviceInfo2{};
                device_info.name = "TextureTest";
                device_info.physical_device_index = 0;

                //device_info = this->instance.choose_device(required_implicit, device_info);

                fmt::println("Choosen GPU: {}", reinterpret_cast<char const * const>(&test_context.instance.list_devices_properties()[device_info.physical_device_index].device_name));

                return test_context.instance.create_device_2(device_info);
        }();

        test_context.window = std::make_unique<Window>(1024, 1024, "Compression test");

        test_context.swapchain = test_context.device.create_swapchain({
            .native_window_info = daxa::NativeWindowInfoWin32{ .hwnd = glfwGetWin32Window(test_context.window->glfw_handle) },
            .surface_format_selector = [](daxa::Format format, daxa::ColorSpace) -> i32
            {
                switch (format)
                {
                    case daxa::Format::R8G8B8A8_UNORM: return 80;
                    case daxa::Format::B8G8R8A8_UNORM: return 60;
                    default:                           return 0;
                }
            },
            .present_mode = daxa::PresentMode::IMMEDIATE,
            .image_usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .name = "GPU Compression test swapchain",
        });

        test_context.pipeline_manager = daxa::PipelineManager{daxa::PipelineManagerInfo2{
            .device = test_context.device,
            .root_paths =
                {
                    "./src",
                    DAXA_SHADER_INCLUDE_DIR,
                },
            .write_out_preprocessed_code = "./preproc",
            .write_out_spirv = "./spv_raw",
            .spirv_cache_folder = "spv",
            .register_null_pipelines_when_first_compile_fails = true,
            .default_language = daxa::ShaderLanguage::GLSL,
            .default_enable_debug_info = true,
            .name = "Compression Test PipelineCompiler",
        }};

        auto const compilation_result = test_context.pipeline_manager.add_compute_pipeline2({
            .source = daxa::ShaderSource(daxa::ShaderFile("./src/tex_compression/test.hlsl")),
            .entry_point = "entry_compressed_sampling_test",
            .name = "Compressed sampling test pipeline"
        });

        if(compilation_result.value()->is_valid())
        {
            std::cout << fmt::format("[Compression test] SUCCESFULLY compiled test pipeline") << std::endl;
        }
        else
        {
            std::cout << fmt::format("[Renderer::compile_pipelines()] FAILED to compile test pipeline with message \n {}", compilation_result.message()) << std::endl;
            DAXA_DBG_ASSERT_TRUE_M(false, "Failed to create pipeline!");
        }
        test_context.test_pipeline = compilation_result.value();
        prepare_test_textures(test_dimensions, TextureContent::UV_GRADIENT, test_context);
    }
    while(true)
    {
        std::cout << "============================================================" << std::endl;
        std::cout << "=========================== RAW ============================" << std::endl;
        std::cout << "============================================================" << std::endl;
        test_image(test_context, test_context.raw_test_images, test_dimensions);
        std::cout << "============================================================" << std::endl;
        std::cout << "=========================== BC1 ============================" << std::endl;
        std::cout << "============================================================" << std::endl;
        test_image(test_context, test_context.BC1_test_images, test_dimensions);
        std::cout << "============================================================" << std::endl;
        std::cout << "=========================== BC4 ============================" << std::endl;
        std::cout << "============================================================" << std::endl;
        test_image(test_context, test_context.BC4_test_images, test_dimensions);
        std::cout << "============================================================" << std::endl;
        std::cout << "=========================== BC6 ============================" << std::endl;
        std::cout << "============================================================" << std::endl;
        test_image(test_context, test_context.BC6_test_images, test_dimensions);
        std::cout << "============================================================" << std::endl;
        std::cout << "=========================== BC7 ============================" << std::endl;
        std::cout << "============================================================" << std::endl;
        test_image(test_context, test_context.BC7_test_images, test_dimensions);

        auto swapchain_image = test_context.swapchain.acquire_next_image();
        auto recorder = test_context.device.create_command_recorder({
            .name = ("recorder (clearcolor)"),
        });

        recorder.pipeline_image_barrier({
            .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
            .image = swapchain_image,
            .layout_operation = daxa::ImageLayoutOperation::TO_GENERAL,
        });

        recorder.clear_image({
            .clear_value = {std::array<f32, 4>{0, 0, 1, 1}},
            .image = swapchain_image,
        });

        recorder.pipeline_image_barrier({
            .src_access = daxa::AccessConsts::TRANSFER_WRITE,
            .image = swapchain_image,
            .layout_operation = daxa::ImageLayoutOperation::TO_PRESENT_SRC,
        });

        auto executalbe_commands = recorder.complete_current_commands();
        recorder.~CommandRecorder();

        test_context.device.submit_commands({
            .command_lists = std::array{executalbe_commands},
            .wait_binary_semaphores = std::array{test_context.swapchain.current_acquire_semaphore()},
            .signal_binary_semaphores = std::array{test_context.swapchain.current_present_semaphore()},
            .signal_timeline_semaphores = std::array{test_context.swapchain.current_timeline_pair()},
        });

        test_context.device.present_frame({
            .wait_binary_semaphores = std::array{test_context.swapchain.current_present_semaphore()},
            .swapchain = test_context.swapchain,
        });
    }
}

void prepare_test_textures(u32vec3 const test_textures_dimensions, TextureContent const content, TestContext & context)
{
    DBG_ASSERT_TRUE_M((test_textures_dimensions.x % 4) == 0 && (test_textures_dimensions.y % 4) == 0,
                       "Invalid texture dimensions, compressed textures need to be 4 aligned!");


    u32 const texel_count = test_textures_dimensions.x * test_textures_dimensions.y * test_textures_dimensions.z;

    std::vector<std::byte> half_fp_raw_data;
    u32 const half_fp_byte_count = 3 * sizeof(short) * texel_count;
    half_fp_raw_data.resize(half_fp_byte_count);

    std::vector<std::byte> rgba_unorm_raw_data;
    u32 const rgba_raw_data_byte_count = 4 * sizeof(std::byte) * texel_count;
    rgba_unorm_raw_data.resize(rgba_raw_data_byte_count);

    std::vector<std::byte> r_unorm_raw_data;
    u32 const r_unorm_raw_data_byte_count = sizeof(std::byte) * texel_count;
    r_unorm_raw_data.resize(r_unorm_raw_data_byte_count);

    for(u32 x = 0; x < test_textures_dimensions.x; ++x)
    {
        for(u32 y = 0; y < test_textures_dimensions.y; ++y)
        {
            for(u32 z = 0; z < test_textures_dimensions.z; ++z)
            {
                u32 const linear_pixel_index = x + y * test_textures_dimensions.x + z * test_textures_dimensions.y * test_textures_dimensions.z;
                u32 const in_fp_byte_array_pixel_index = linear_pixel_index * 3 * sizeof(short);
                u32 const in_rgb_byte_array_pixel_index = linear_pixel_index * 3;
                u32 const in_r_byte_array_pixel_index = linear_pixel_index;
                switch(content)
                {
                    case TextureContent::UV_GRADIENT:
                    {
                        std::array<short, 3> color = {
                            glm::detail::toFloat16(s_cast<f32>(x)),
                            glm::detail::toFloat16(s_cast<f32>(y)),
                            glm::detail::toFloat16(s_cast<f32>(z)),
                        };

                        memcpy(&half_fp_raw_data[in_fp_byte_array_pixel_index], &color, sizeof(color));

                        rgba_unorm_raw_data[in_rgb_byte_array_pixel_index + 0] = s_cast<std::byte>(x & 0xFFu);
                        rgba_unorm_raw_data[in_rgb_byte_array_pixel_index + 1] = s_cast<std::byte>(y & 0xFFu);
                        rgba_unorm_raw_data[in_rgb_byte_array_pixel_index + 2] = s_cast<std::byte>(z & 0xFFu);
                        rgba_unorm_raw_data[in_rgb_byte_array_pixel_index + 3] = s_cast<std::byte>(z & 0xFFu);

                        r_unorm_raw_data[in_r_byte_array_pixel_index] = s_cast<std::byte>((x + y + z) & 0xFFu);
                    }
                }
            }
        }
    }

    struct PerFormatData
    {
        daxa::Format format;
        std::array<daxa::ImageId, 3> * textures;
        u32 compressed_bytesize;
        std::vector<std::byte> * source;
        Compression compression;
    };

    std::array per_format_data = {
       PerFormatData{daxa::Format::R16G16B16A16_SFLOAT, &context.raw_test_images, (texel_count * 8)     ,  &half_fp_raw_data,     Compression::UNDEFINED},
       PerFormatData{daxa::Format::BC1_RGB_UNORM_BLOCK, &context.BC1_test_images, (texel_count / 16) * 8,  &rgba_unorm_raw_data,  Compression::BC1},
       PerFormatData{daxa::Format::BC4_UNORM_BLOCK    , &context.BC4_test_images, (texel_count / 16) * 8,  &r_unorm_raw_data,     Compression::BC4},
       PerFormatData{daxa::Format::BC6H_UFLOAT_BLOCK  , &context.BC6_test_images, (texel_count / 16) * 16, &half_fp_raw_data,     Compression::BC6},
       PerFormatData{daxa::Format::BC7_UNORM_BLOCK    , &context.BC7_test_images, (texel_count / 16) * 16, &rgba_unorm_raw_data,  Compression::BC7},
    };

    for(u32 format_index = 0; format_index < per_format_data.size(); ++format_index)
    {
        auto cr = context.device.create_command_recorder({.name = "upload images"});

        std::vector<std::byte> compressed_data;
        auto const & format_data = per_format_data.at(format_index);
        if (format_data.compression != Compression::UNDEFINED)
        {
            compressed_data.resize(format_data.compressed_bytesize);

            auto compress_image_task = compress_image({
                .in_data = *format_data.source,
                .out_data = compressed_data,
                .image_dimensions = test_textures_dimensions,
                .compression = format_data.compression
            });

            context.thread_pool->blocking_dispatch(compress_image_task);
        }

        // =================================================================
        // ========================== CREATE IMAGES ========================
        // =================================================================
        {
            // 2D image
            format_data.textures->at(0) = context.device.create_image({
                .dimensions = 2,
                .format = format_data.format,
                .size = {test_textures_dimensions.x, test_textures_dimensions.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED
            });

            // 2D arary image
            format_data.textures->at(1) = context.device.create_image({
                .flags = daxa::ImageCreateFlagBits::COMPATIBLE_2D_ARRAY,
                .dimensions = 2,
                .format = format_data.format,
                .size = {test_textures_dimensions.x, test_textures_dimensions.y, 1},
                .array_layer_count = test_textures_dimensions.z,
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED
            });

            // 3D image
            format_data.textures->at(2) = context.device.create_image({
                .dimensions = 3,
                .format = format_data.format,
                .size = {test_textures_dimensions.x, test_textures_dimensions.y, test_textures_dimensions.z},
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED
            });
        }

        // =================================================================
        // ============== TRANSITION IMAGES TO GENERAL =====================
        // =================================================================
        {
            daxa::ImageBarrierInfo barrier_info = {
                .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                .layout_operation = daxa::ImageLayoutOperation::TO_GENERAL,
            };

            // Transition images to general layout.
            for(auto const & tex : *format_data.textures)
            {
                barrier_info.image = tex;
                cr.pipeline_image_barrier(barrier_info);
            }
        }

        // =================================================================
        // ===================== UPLOAD TEXTURE DATA =======================
        // =================================================================
        {
            auto const & compressed_data_ = (format_data.compression != Compression::UNDEFINED) ? compressed_data : *format_data.source;
            daxa::BufferId staging_buffer = context.device.create_buffer({
                .size = compressed_data_.size(),
                .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
                .name = "staging buffer",
            });
            cr.destroy_buffer_deferred(staging_buffer);
            std::memcpy(context.device.buffer_host_address(staging_buffer).value(), compressed_data_.data(), compressed_data_.size());

            daxa::BufferImageCopyInfo buffer_copy_info = {
                .src_buffer = staging_buffer,
                .buffer_offset = 0,
                .dst_image = format_data.textures->at(0),
                .image_slice = {
                    .mip_level = 0,
                    .base_array_layer = 0,
                },
                .image_offset = {0, 0, 0},
            };

            // 2D image
            buffer_copy_info.image_slice.layer_count = 1;
            buffer_copy_info.image_extent = {test_textures_dimensions.x, test_textures_dimensions.y, 1};
            cr.copy_buffer_to_image(buffer_copy_info);

            // 2D array image
            buffer_copy_info.image_slice.layer_count = test_textures_dimensions.z;
            buffer_copy_info.image_extent = {test_textures_dimensions.x, test_textures_dimensions.y, 1};
            cr.copy_buffer_to_image({buffer_copy_info});

            // 3D image
            buffer_copy_info.image_slice.layer_count = 1;
            buffer_copy_info.image_extent = {test_textures_dimensions.x, test_textures_dimensions.y, test_textures_dimensions.z};
            cr.copy_buffer_to_image({buffer_copy_info});


        }

        // =================================================================
        // ========================== FLUSH CACHES =========================
        // =================================================================
        {
            daxa::ImageBarrierInfo barrier_info = {
                .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                .dst_access = daxa::AccessConsts::READ,
            };

            // Transition images to general layout.
            for(auto const & tex : *format_data.textures)
            {
                barrier_info.image = tex;
                cr.pipeline_image_barrier(barrier_info);
            }
        }


        context.device.wait_on_submit({
            daxa::QUEUE_MAIN,
            context.device.submit_commands({
                .command_lists = std::array{cr.complete_current_commands()},
            }),
        });
        context.device.collect_garbage();
    }
}

auto to_string(TestSamplerType type) -> std::string_view
{
    switch (type)
    {
        case TEST_SAMPLER_TYPE_LINEAR_CLAMP: return "LINEAR_CLAMP";
        case TEST_SAMPLER_TYPE_LINEAR_REPEAT: return "LINEAR_REPEAT";
        case TEST_SAMPLER_TYPE_NEAREST_CLAMP: return "NEAREST_CLAMP";
        case TEST_SAMPLER_TYPE_NEAREST_REPEAT: return "NEAREST_REPEAT";
    }
    return "";
}

auto to_string(TestTextureType type) -> std::string_view
{
    switch (type)
    {
        case TEST_TEXTURE_TYPE_2D: return "2D";
        case TEST_TEXTURE_TYPE_2D_ARRAY: return "2D_ARRAY";
        case TEST_TEXTURE_TYPE_3D: return "3D";
        case TEST_TEXTURE_TYPE_3D_EMULATED: return "3D_EMULATED";
    }
    return "";
}

auto to_string(TestIndexingType type) -> std::string_view
{
    switch (type)
    {
        case TEST_INDEXING_TYPE_ZERO: return "ZERO";
        case TEST_INDEXING_TYPE_DTID: return "DTID";
        case TEST_INDEXING_TYPE_RANDOM: return "RANDOM";
        case TEST_INDEXING_TYPE_RANDOM_WALK: return "RANDOM_WALK";
        case TEST_INDEXING_TYPE_RANDOM_WALK_LINE: return "RANDOM_WALK_LINE";
    }
    return "";
}

void test_image(TestContext & ctx, std::array<daxa::ImageId, 3> images, u32vec3 texture_dimensions)
{
    // Setup

    daxa::u32 z_dim = 1u << 14u;
    
    daxa::SamplerId samplers[TEST_SAMPLER_TYPE_COUNT] = {
        ctx.device.create_sampler({}), // linear clamp
        ctx.device.create_sampler({
            .address_mode_u = daxa::SamplerAddressMode::REPEAT,
            .address_mode_v = daxa::SamplerAddressMode::REPEAT,
            .address_mode_w = daxa::SamplerAddressMode::REPEAT,
        }), // linear repeat
        ctx.device.create_sampler({
            .magnification_filter = daxa::Filter::NEAREST,
            .minification_filter = daxa::Filter::NEAREST,
        }), // nearest clamp
        ctx.device.create_sampler({
            .magnification_filter = daxa::Filter::NEAREST,
            .minification_filter = daxa::Filter::NEAREST,
            .address_mode_u = daxa::SamplerAddressMode::REPEAT,
            .address_mode_v = daxa::SamplerAddressMode::REPEAT,
            .address_mode_w = daxa::SamplerAddressMode::REPEAT,
        }), // nearest repeat
    };

    daxa::CommandRecorder cmd = ctx.device.create_command_recorder({});

    cmd.set_pipeline(*ctx.test_pipeline);

    u32 const query_pool_size = TEST_INDEXING_TYPE_COUNT * TEST_TEXTURE_TYPE_COUNT * TEST_SAMPLER_TYPE_COUNT * 2/*2 values for the before and after timestamp*/;
    daxa::TimelineQueryPool query_pool = ctx.device.create_timeline_query_pool({
        .query_count = query_pool_size,
    });

    // Record

    for (daxa::u32 test_indexing_type = 0; test_indexing_type < TEST_INDEXING_TYPE_COUNT; ++test_indexing_type)
    {
        for (daxa::u32 test_tex_type = 0; test_tex_type < TEST_TEXTURE_TYPE_COUNT; ++test_tex_type)
        {
            for (daxa::u32 test_sampler_type = 0; test_sampler_type < TEST_SAMPLER_TYPE_COUNT; ++test_sampler_type)
            {
                daxa::u32 const before_query_index = 2 *  (
                    test_sampler_type + 
                    test_tex_type * TEST_SAMPLER_TYPE_COUNT + 
                    test_indexing_type * TEST_SAMPLER_TYPE_COUNT * TEST_TEXTURE_TYPE_COUNT
                );
                DAXA_DBG_ASSERT_TRUE_M(before_query_index < query_pool_size, "OUT OF BOUNDS");
                daxa::u32 const after_query_index = before_query_index + 1;

                cmd.write_timestamp({
                    .query_pool = query_pool,
                    .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER,
                    .query_index = before_query_index,
                });

                daxa::ImageId image_id = {};
                switch(test_tex_type)
                {
                    case TEST_TEXTURE_TYPE_2D: image_id = images[0]; break;
                    case TEST_TEXTURE_TYPE_2D_ARRAY: image_id = images[1]; break;
                    case TEST_TEXTURE_TYPE_3D: image_id = images[2]; break;
                    case TEST_TEXTURE_TYPE_3D_EMULATED: image_id = images[1]; break; // MUST BE 1 AS IT NEEDS 2d ARRAY LAYER!
                }

                CompressedSamplingTestPush push = {};
                push.test_indexing_type = static_cast<TestIndexingType>(test_indexing_type);
                push.test_tex_type = static_cast<TestTextureType>(test_tex_type);
                push.image = std::bit_cast<daxa_ImageViewId>(image_id.default_view());
                push.tex_size = std::bit_cast<daxa_u32vec3>(texture_dimensions);
                push.dummy = {}; // never written, can safely be 0.
                push.iterations = ctx.shader_iterations;
                push.test_sampler_id = std::bit_cast<daxa_SamplerId>(samplers[test_sampler_type]);
                push.sampler_linear = test_sampler_type < TEST_SAMPLER_TYPE_NEAREST_CLAMP;
                cmd.push_constant(push);

                for (u32 i = 0; i < ctx.dispatch_iterations; ++i)
                {
                    cmd.dispatch({
                        (push.tex_size.x + COMPRESSED_SAMPLING_TEST_DISPATCH_X - 1) / COMPRESSED_SAMPLING_TEST_DISPATCH_X, 
                        (push.tex_size.y + COMPRESSED_SAMPLING_TEST_DISPATCH_Y - 1) / COMPRESSED_SAMPLING_TEST_DISPATCH_Y, 
                        z_dim,
                    });
                }

                cmd.write_timestamp({
                    .query_pool = query_pool,
                    .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER,
                    .query_index = after_query_index,
                });

                // Prevent tests from overlapping with each other
                cmd.pipeline_barrier({
                    .src_access = daxa::AccessConsts::COMPUTE_SHADER_READ_WRITE,
                    .dst_access = daxa::AccessConsts::COMPUTE_SHADER_READ_WRITE,
                });
            }
        }
    }

    // Submit and read results

    ctx.device.submit_commands({
        .command_lists = std::array{cmd.complete_current_commands()},
    });
    ctx.device.wait_idle();
    ctx.device.collect_garbage();
    
    auto const results = query_pool.get_query_results(0, query_pool_size);

    for (daxa::u32 test_indexing_type = 0; test_indexing_type < TEST_INDEXING_TYPE_COUNT; ++test_indexing_type)
    {
        for (daxa::u32 test_tex_type = 0; test_tex_type < TEST_TEXTURE_TYPE_COUNT; ++test_tex_type)
        {
            for (daxa::u32 test_sampler_type = 0; test_sampler_type < TEST_SAMPLER_TYPE_COUNT; ++test_sampler_type)
            {
                daxa::u32 const before_query_index = 4 * (
                    test_sampler_type + 
                    test_tex_type * TEST_SAMPLER_TYPE_COUNT + 
                    test_indexing_type * TEST_SAMPLER_TYPE_COUNT * TEST_TEXTURE_TYPE_COUNT
                );
                daxa::u32 const after_query_index = before_query_index + 2;
                DAXA_DBG_ASSERT_TRUE_M(after_query_index < query_pool_size * 2, "OUT OF BOUNDS");

                u64 const time_taken = results[after_query_index] - results[before_query_index];

                std::cout << std::format(
                    "Indexing {:<16.16} Tex Dimension {:12.12} Sampler {:<16.16} took {:14.4f}us",
                    to_string(static_cast<TestIndexingType>(test_indexing_type)),
                    to_string(static_cast<TestTextureType>(test_tex_type)), 
                    to_string(static_cast<TestSamplerType>(test_sampler_type)), 
                    float(time_taken) * 0.001f
                ) << std::endl;
            }
        }
    }
}