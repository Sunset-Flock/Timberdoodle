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

template <typename T>
struct _MAP_Test
{
    constexpr static int I = 0;
};

#define TEST(X) _MAP_##X::I

GPUContext::GPUContext(Window const & window)
    : instance{daxa::create_instance({})},
      device{[&]()
          {
              auto required_implicit =
                  daxa::ImplicitFeatureFlagBits::BASIC_RAY_TRACING |
                  daxa::ImplicitFeatureFlagBits::MESH_SHADER |
                  daxa::ImplicitFeatureFlagBits::SHADER_CLOCK |
                  // daxa::ImplicitFeatureFlagBits::HOST_IMAGE_COPY |
                  daxa::ImplicitFeatureFlagBits::SWAPCHAIN;

              auto device_info = daxa::DeviceInfo2{};
              device_info.max_allowed_images = 1u << 16u;
              device_info.max_allowed_buffers = 1u << 17u;
              device_info.max_allowed_acceleration_structures = 1u << 17u;
              device_info.name = "Timberdoodle";
              device_info.physical_device_index = 1;

              //device_info = this->instance.choose_device(required_implicit, device_info);

              fmt::println("Choosen GPU: {}", reinterpret_cast<char const * const>(&this->instance.list_devices_properties()[device_info.physical_device_index].device_name));

              return this->instance.create_device_2(device_info);
          }()},
      swapchain{this->device.create_swapchain({
          .native_window = glfwGetWin32Window(window.glfw_handle),
          .native_window_platform = daxa::NativeWindowPlatform::WIN32_API,
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
          .name = "Timberdoodle Swapchain",
      })},
      pipeline_manager{daxa::PipelineManager{daxa::PipelineManagerInfo2{
          .device = this->device,
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
          .name = "Timberdoodle PipelineCompiler",
      }}},
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
    device.destroy_image(shader_debug_context.vsm_debug_meta_memory_table.get_state().images[0]);
    device.destroy_image(shader_debug_context.vsm_debug_page_table.get_state().images[0]);
    device.destroy_image(shader_debug_context.vsm_recreated_shadowmap_memory_table.get_state().images[0]);
    device.destroy_sampler(lin_clamp_sampler);
    device.destroy_sampler(nearest_clamp_sampler);
}