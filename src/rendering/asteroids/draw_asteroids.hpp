#pragma once
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../scene_renderer_context.hpp"
#include "../../gpu_context.hpp"
#include "../../asteroids/asteroids.hpp"

#include "draw_asteroids.inl"

struct AsteroidsState
{
    daxa::BufferId debug_asteroid_mesh_buffer = {};

#if CPU_SIMULATION
    daxa::TaskBufferView asteroids = {};
#else
    daxa::TaskBuffer gpu_asteroids = {};
    daxa::TaskGraph gen_asteroids_graph = {};
#endif

    daxa::u32 debug_probe_mesh_triangles = {};
    daxa::u32 debug_probe_mesh_vertices = {};
    daxa_f32vec3* debug_asteroid_mesh_vertex_positions_addr = {};
    std::array<GPUAsteroid, MAX_ASTEROID_COUNT> cpu_asteroids = {};

    bool last_simulation_started = {};
    bool simulation_just_started = {};

    u32 asteroids_count = {};

    void initialize_gpu_simulation(daxa::Device & device, AsteroidsWrapper const & asteroids);
    void initialize_persistent_state(daxa::Device& device);
    void initalize_transient_state(daxa::TaskGraph & tg);
    void update_cpu_data(daxa::Device & device, AsteroidSimulation & simulation, ShaderDebugDrawContext & debug_context);
    void cleanup(daxa::Device & device);
};

struct TaskDrawAsteroidsInfo
{
    RenderContext * render_context;
    AsteroidsState * asteroids_state;
    daxa::TaskGraph * tg;
    daxa::TaskImageView depth;  
    daxa::TaskImageView color;
};

void task_draw_asteroids(TaskDrawAsteroidsInfo const & info);

static constexpr inline char const ASTEROID_SHADER_PATH[] = "./src/rendering/asteroids/asteroids_debug_draw.hlsl";
inline daxa::RasterPipelineCompileInfo debug_draw_asteroids_compile_info()
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.color_attachments = std::vector{
        daxa::RenderAttachment{
            .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
        },
    };
    ret.depth_test = {
        .depth_attachment_format = daxa::Format::D32_SFLOAT,
        .enable_depth_write = true,
        .depth_test_compare_op = daxa::CompareOp::GREATER,
        .min_depth_bounds = 0.0f,
        .max_depth_bounds = 1.0f,
    };
    ret.raster = {
        .primitive_topology = daxa::PrimitiveTopology::TRIANGLE_LIST,
        .primitive_restart_enable = {},
        .polygon_mode = daxa::PolygonMode::FILL,
        .face_culling = daxa::FaceCullFlagBits::NONE,
        .front_face_winding = daxa::FrontFaceWinding::COUNTER_CLOCKWISE,
    };
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{ASTEROID_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_fragment_debug_draw_asteroids",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{ASTEROID_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_vertex_debug_draw_asteroids",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.push_constant_size = sizeof(DebugDrawAsteroidsPush);
    ret.name = "DebugDrawAsteroids";
    return ret;
}

struct DebugDrawAsteroidsTask : DebugDrawAsteroidsH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    AsteroidsState* asteroid_state = {};

    void callback(daxa::TaskInterface ti)
    {
#if !CPU_SIMULATION
        if(ti.get(AT.asteroid_params).ids.size() != (ASTEROID_SCALE + 1)) { return; }
#endif

        auto const colorImageSize = ti.device.image_info(ti.get(AT.color_image).ids[0]).value().size;
        daxa::RenderPassBeginInfo render_pass_begin_info{
            .depth_attachment =
                daxa::RenderAttachmentInfo{
                    .image_view = ti.get(AT.depth_image).view_ids[0],
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = daxa::AttachmentLoadOp::LOAD,
                    .store_op = daxa::AttachmentStoreOp::STORE,
                },
            .render_area = daxa::Rect2D{.width = colorImageSize.x, .height = colorImageSize.y},
        };
        render_pass_begin_info.color_attachments = {
            daxa::RenderAttachmentInfo{
                .image_view = ti.get(AT.color_image).view_ids[0],
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = daxa::AttachmentLoadOp::LOAD,
                .store_op = daxa::AttachmentStoreOp::STORE,
            },
        };

        auto render_cmd = std::move(ti.recorder).begin_renderpass(render_pass_begin_info);
        render_cmd.set_pipeline(*render_context->gpu_context->raster_pipelines.at(debug_draw_asteroids_compile_info().name));

        DebugDrawAsteroidsPush push = {
            .attach = ti.attachment_shader_blob,
            .asteroid_mesh_positions = asteroid_state->debug_asteroid_mesh_vertex_positions_addr,
#if !CPU_SIMULATION
            .parameters = {
                .position = r_cast<daxa_f32vec3*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_POSITION]).value()),
                .velocity = r_cast<daxa_f32vec3*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_VELOCITY]).value()),
                .velocity_derivative = r_cast<daxa_f32vec3*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_VELOCITY_DERIVATIVE]).value()),
                .velocity_divergence = r_cast<daxa_f32*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_VELOCITY_DIVERGENCE]).value()),
                .smoothing_radius = r_cast<daxa_f32*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_SMOOTHING_RADIUS]).value()),
                .mass = r_cast<daxa_f32*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_MASS]).value()),
                .density = r_cast<daxa_f32*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_DENSITY]).value()),
                .density_derivative = r_cast<daxa_f32*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_DENSITY_DERIVATIVE]).value()),
                .energy = r_cast<daxa_f32*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_ENERGY]).value()),
                .energy_derivative = r_cast<daxa_f32*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_ENERGY_DERIVATIVE]).value()),
                .pressure = r_cast<daxa_f32*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_PRESSURE]).value()),
                .scale = r_cast<daxa_f32*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_SCALE]).value()),
            }
#endif
        };
        render_cmd.push_constant(push);

        render_cmd.set_index_buffer({ .id = asteroid_state->debug_asteroid_mesh_buffer}); 

        render_cmd.draw_indexed({
            .index_count = asteroid_state->debug_probe_mesh_triangles * 3,
            .instance_count = asteroid_state->asteroids_count,
        });

        ti.recorder = std::move(render_cmd).end_renderpass();
    }
};