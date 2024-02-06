#include "gpu_context.hpp"

#include "shader_shared/asset.inl"
#include "shader_shared/scene.inl"

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_NATIVE_INCLUDE_NONE
using HWND = void *;
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#include <GLFW/glfw3native.h>

// Not needed, this is set by cmake.
// Intellisense doesnt get it, so this prevents it from complaining.
#if !defined(DAXA_SHADER_INCLUDE_DIR)
#define DAXA_SHADER_INCLUDE_DIR "."
#endif

GPUContext::GPUContext(Window const & window)
    : context{daxa::create_instance({})}, device{this->context.create_device({
          .max_allowed_images = 100000, .max_allowed_buffers = 100000,
#if COMPILE_IN_MESH_SHADER
          .enable_mesh_shader = true,
#endif
          .name = "Sandbox Device"
      })},
      swapchain{this->device.create_swapchain({
          .native_window = glfwGetWin32Window(window.glfw_handle),
          .native_window_platform = daxa::NativeWindowPlatform::WIN32_API,
          .surface_format_selector = [](daxa::Format format) -> i32
          {
              switch (format)
              {
                  case daxa::Format::R8G8B8A8_UNORM: return 80;
                  case daxa::Format::B8G8R8A8_UNORM: return 60;
                  default:                           return 0;
              }
          },
          .present_mode = daxa::PresentMode::IMMEDIATE,
          .image_usage = daxa::ImageUsageFlagBits::SHADER_STORAGE,
          .name = "Sandbox Swapchain",
      })},
      pipeline_manager{daxa::PipelineManager{{
          .device = this->device,
          .shader_compile_options =
              []()
          {
            // msvc time!
              return daxa::ShaderCompileOptions{
                  .root_paths =
                      {
                          "./src",
                          DAXA_SHADER_INCLUDE_DIR,
                      },
                  .write_out_preprocessed_code = "./preproc",
                  .spirv_cache_folder = "spv",
                  .language = daxa::ShaderLanguage::GLSL,
                  .enable_debug_info = true,
              };
          }(),
          .register_null_pipelines_when_first_compile_fails = true,
          .name = "Sandbox PipelineCompiler",
      }}},
      transient_mem{{
          .device = this->device,
          .capacity = 4096,
          .name = "transient memory pool",
      }},
      shader_globals_buffer{this->device.create_buffer({
          .size = sizeof(ShaderGlobals),
          .name = "globals",
      })},
      shader_globals_task_buffer{daxa::TaskBuffer{{
        .initial_buffers = std::array{shader_globals_buffer},
        .name = "globals",
      }}}
{
    shader_globals.samplers = {.linear_clamp = this->device.create_sampler({
                                   .name = "linear clamp sampler",
                               }),
        .linear_repeat = this->device.create_sampler({
            .address_mode_u = daxa::SamplerAddressMode::REPEAT,
            .address_mode_v = daxa::SamplerAddressMode::REPEAT,
            .address_mode_w = daxa::SamplerAddressMode::REPEAT,
            .name = "linear repeat sampler",
        }),
        .nearest_clamp = this->device.create_sampler({
            .magnification_filter = daxa::Filter::NEAREST,
            .minification_filter = daxa::Filter::NEAREST,
            .mipmap_filter = daxa::Filter::NEAREST,
            .name = "nearest clamp sampler",
        })};
    debug_draw_info.init(device);
    shader_globals.debug_draw_info = device.get_device_address(debug_draw_info.buffer).value();
}

auto GPUContext::dummy_string() -> std::string
{
    return std::string(" - ") + std::to_string(counter++);
}

GPUContext::~GPUContext()
{
    device.destroy_buffer(shader_globals_buffer);
    device.destroy_sampler(std::bit_cast<daxa::SamplerId>(shader_globals.samplers.linear_clamp));
    device.destroy_sampler(std::bit_cast<daxa::SamplerId>(shader_globals.samplers.linear_repeat));
    device.destroy_sampler(std::bit_cast<daxa::SamplerId>(shader_globals.samplers.nearest_clamp));
    device.destroy_buffer(debug_draw_info.buffer);
}