#pragma once

#include "timberdoodle.hpp"
#include "window.hpp"
#include "shader_shared/shared.inl"
#include "shader_shared/globals.inl"

struct ShaderDebugDrawContext
{
    u32 max_circle_draws = 256'000;
    u32 max_rectangle_draws = 256'000;
    u32 max_aabb_draws = 256'000;
    u32 circle_vertices = 64; // Uses line strip
    u32 rectangle_vertices = 5; // Uses line strip
    u32 aabb_vertices = 24; // Uses line list
    daxa::BufferId buffer = {};
    ShaderDebugInput shader_debug_input = {};
    ShaderDebugOutput shader_debug_output = {};
    i32 debug_magnifier_pixel_span = 15;
    i32 old_debug_magnifier_pixel_span = 0;
    bool draw_magnified_area_rect = true;
    
    daxa::ImageId shader_debug_magnified_image = {};
    daxa::TaskImage tshader_debug_magnified_image = {};

    std::vector<ShaderDebugCircleDraw> cpu_debug_circle_draws = {};
    std::vector<ShaderDebugRectangleDraw> cpu_debug_rectangle_draws = {};
    std::vector<ShaderDebugAABBDraw> cpu_debug_aabb_draws = {};

    void init(daxa::Device & device)
    {
        usize size = sizeof(ShaderDebugBufferHead);
        size += sizeof(ShaderDebugCircleDraw) * max_circle_draws;
        size += sizeof(ShaderDebugRectangleDraw) * max_rectangle_draws;
        size += sizeof(ShaderDebugRectangleDraw) * max_aabb_draws;
        buffer = device.create_buffer({
            .size = size,
            .name = "shader debug buffer",
        });
        shader_debug_magnified_image = device.create_image({
            .format = daxa::Format::B10G11R11_UFLOAT_PACK32, 
            .size = { 7, 7, 1 },
            .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .name = "debug magnified image",
        });
        tshader_debug_magnified_image = daxa::TaskImage({
            .initial_images = {.images = std::array{shader_debug_magnified_image}}, 
            .name = "tshader_debug_magnified_image",
        });
    }

    void update(daxa::Device & device, u32 render_image_size_x, u32 render_image_size_y)
    {
        if (old_debug_magnifier_pixel_span != debug_magnifier_pixel_span)
        {
            if (device.is_id_valid(shader_debug_magnified_image))
            {
                device.destroy_image(shader_debug_magnified_image);
            }
            shader_debug_magnified_image = device.create_image({
                .format = daxa::Format::B10G11R11_UFLOAT_PACK32, 
                .size = { static_cast<u32>(debug_magnifier_pixel_span), static_cast<u32>(debug_magnifier_pixel_span), 1 },
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "debug magnified image",
            });
            tshader_debug_magnified_image.set_images({.images=std::array{shader_debug_magnified_image}});
            old_debug_magnifier_pixel_span = debug_magnifier_pixel_span;
        }
        if (draw_magnified_area_rect)
        {
            auto u = (static_cast<f32>(shader_debug_input.texel_detector_pos.x) + 0.5f) / static_cast<f32>(render_image_size_x);
            auto v = (static_cast<f32>(shader_debug_input.texel_detector_pos.y) + 0.5f) / static_cast<f32>(render_image_size_y);
            auto span_u = (static_cast<f32>(debug_magnifier_pixel_span + 2)) / static_cast<f32>(render_image_size_x);
            auto span_v = (static_cast<f32>(debug_magnifier_pixel_span + 2)) / static_cast<f32>(render_image_size_y);
            
            cpu_debug_aabb_draws.push_back(ShaderDebugAABBDraw{
                .position = {u * 2.0f - 1.0f, v * 2.0f - 1.0f, 0.5},
                .size = {span_u * 2.0f, span_v * 2.0f, 0.99 },
                .color = daxa_f32vec3(1,0,0),
                .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC,
            });
        }
    }

    void update_debug_buffer(daxa::Device & device, daxa::CommandRecorder & recorder, daxa::TransferMemoryPool & allocator)
    {
        u32 const circle_buffer_offset = sizeof(ShaderDebugBufferHead);
        u32 const rectangle_buffer_offset = circle_buffer_offset + sizeof(ShaderDebugCircleDraw) * max_circle_draws;
        u32 const aabb_buffer_offset = rectangle_buffer_offset + sizeof(ShaderDebugRectangleDraw) * max_rectangle_draws;
        
        auto head = ShaderDebugBufferHead{
            .circle_draw_indirect_info = {
                .vertex_count = circle_vertices,
                .instance_count = std::min(static_cast<u32>(cpu_debug_circle_draws.size()), max_circle_draws),
                .first_vertex = 0,
                .first_instance = 0,
            },
            .rectangle_draw_indirect_info = {
                .vertex_count = rectangle_vertices,
                .instance_count = std::min(static_cast<u32>(cpu_debug_rectangle_draws.size()), max_rectangle_draws),
                .first_vertex = 0,
                .first_instance = 0,
            },
            .aabb_draw_indirect_info = {
                .vertex_count = aabb_vertices,
                .instance_count = std::min(static_cast<u32>(cpu_debug_aabb_draws.size()), max_aabb_draws),
                .first_vertex = 0,
                .first_instance = 0,
            },
            .circle_draw_capacity = max_circle_draws,
            .rectangle_draw_capacity = max_rectangle_draws,
            .aabb_draw_capacity = max_aabb_draws,
            .cpu_input = shader_debug_input,
            .gpu_output = {},
            .circle_draws = device.get_device_address(buffer).value() + circle_buffer_offset,
            .rectangle_draws = device.get_device_address(buffer).value() + rectangle_buffer_offset,
            .aabb_draws = device.get_device_address(buffer).value() + aabb_buffer_offset,
        };
        auto alloc = allocator.allocate_fill(head).value();
        recorder.copy_buffer_to_buffer({
            .src_buffer = allocator.buffer(),
            .dst_buffer = buffer,
            .src_offset = alloc.buffer_offset,
            .dst_offset = 0,
            .size = sizeof(ShaderDebugBufferHead),
        });
        
        auto stage_circle_draws_size = sizeof(ShaderDebugCircleDraw) * head.circle_draw_indirect_info.instance_count;
        if (stage_circle_draws_size > 0)
        {
            auto stage_circle_draws = allocator.allocate(stage_circle_draws_size,4).value();
            std::memcpy(stage_circle_draws.host_address, cpu_debug_circle_draws.data(), stage_circle_draws_size);
            recorder.copy_buffer_to_buffer({
                .src_buffer = allocator.buffer(),
                .dst_buffer = buffer,
                .src_offset = stage_circle_draws.buffer_offset,
                .dst_offset = circle_buffer_offset,
                .size = stage_circle_draws_size,
            });
            cpu_debug_circle_draws.clear();
        }
        
        auto stage_rectangle_draws_size = sizeof(ShaderDebugRectangleDraw) * head.rectangle_draw_indirect_info.instance_count;
        if (stage_rectangle_draws_size > 0)
        {
            auto stage_rectangle_draws = allocator.allocate(stage_rectangle_draws_size,4).value();
            std::memcpy(stage_rectangle_draws.host_address, cpu_debug_rectangle_draws.data(), stage_rectangle_draws_size);
            recorder.copy_buffer_to_buffer({
                .src_buffer = allocator.buffer(),
                .dst_buffer = buffer,
                .src_offset = stage_rectangle_draws.buffer_offset,
                .dst_offset = rectangle_buffer_offset,
                .size = stage_rectangle_draws_size,
            });
            cpu_debug_rectangle_draws.clear();
        }
        
        auto stage_aabb_draws_size = sizeof(ShaderDebugAABBDraw) * head.aabb_draw_indirect_info.instance_count;
        if (stage_aabb_draws_size > 0)
        {
            auto stage_aabb_draws = allocator.allocate(stage_aabb_draws_size,4).value();
            std::memcpy(stage_aabb_draws.host_address, cpu_debug_aabb_draws.data(), stage_aabb_draws_size);
            recorder.copy_buffer_to_buffer({
                .src_buffer = allocator.buffer(),
                .dst_buffer = buffer,
                .src_offset = stage_aabb_draws.buffer_offset,
                .dst_offset = aabb_buffer_offset,
                .size = stage_aabb_draws_size,
            });
            cpu_debug_aabb_draws.clear();
        }
    }
};

struct GPUContext
{
    GPUContext(Window const & window);
    GPUContext(GPUContext &&) = default;
    ~GPUContext();

    // common unique:
    daxa::Instance context = {};
    daxa::Device device = {};
    daxa::Swapchain swapchain = {};
    daxa::PipelineManager pipeline_manager = {};
    daxa::TransferMemoryPool transient_mem;

    ShaderGlobals shader_globals = {};
    daxa::BufferId shader_globals_buffer = {};
    daxa::TaskBuffer shader_globals_task_buffer = {};
    daxa::types::DeviceAddress shader_globals_address = {};

    ShaderDebugDrawContext debug_draw_info = {};

    // Pipelines:
    std::unordered_map<std::string, std::shared_ptr<daxa::RasterPipeline>> raster_pipelines = {};
    std::unordered_map<std::string_view, std::shared_ptr<daxa::ComputePipeline>> compute_pipelines = {};

    // Data
    Settings prev_settings = {};
    Settings settings = {};
    SkySettings prev_sky_settings = {};
    SkySettings sky_settings = {};

    u32 counter = {};
    auto dummy_string() -> std::string;
};