#include "gpu_context.hpp"

#include "shader_shared/geometry.inl"
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

template<typename T>
struct _MAP_Test { constexpr static int I = 0; };

#define TEST(X) _MAP_ ## X::I

GPUContext::GPUContext(Window const & window)
    : gpu_context{daxa::create_instance({})}, device{this->gpu_context.create_device({
          .max_allowed_images = 100000, 
          .max_allowed_buffers = 100000,
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
                  .write_out_shader_binary = "./spv_raw",
                  .spirv_cache_folder = "spv",
                  .language = daxa::ShaderLanguage::GLSL,
                  .enable_debug_info = true,
              };
          }(),
          .register_null_pipelines_when_first_compile_fails = true,
          .name = "Sandbox PipelineCompiler",
      }}},
      dummy_tlas_id{
        device.create_tlas({
            .size = 1u,
            .name = "dummy tlas",
        })},
      lin_clamp_sampler{this->device.create_sampler({.name = "default linear clamp sampler"})},
      nearest_clamp_sampler{this->device.create_sampler({  
        .magnification_filter = daxa::Filter::NEAREST,
        .minification_filter = daxa::Filter::NEAREST,
        .mipmap_filter = daxa::Filter::NEAREST,
        .name = "default nearest clamp sampler",
    })}
{
    shader_debug_context.init(device);
}

auto GPUContext::dummy_string() -> std::string
{
    int i = TEST(Test<int>);
    return std::string(" - ") + std::to_string(counter++);
}

GPUContext::~GPUContext()
{
    device.destroy_buffer(shader_debug_context.buffer);
    device.destroy_buffer(shader_debug_context.readback_queue);
    device.destroy_image(shader_debug_context.debug_lens_image);
    device.destroy_image(shader_debug_context.vsm_debug_meta_memory_table.get_state().images[0]);
    device.destroy_image(shader_debug_context.vsm_debug_page_table.get_state().images[0]);
    device.destroy_sampler(lin_clamp_sampler);
    device.destroy_sampler(nearest_clamp_sampler);
}