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

static constexpr inline char const ASTEROID_SIMULATION_SHADER_PATH[] = "./src/rendering/asteroids/asteroids_simulation.hlsl";
inline daxa::ComputePipelineCompileInfo material_update_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{ASTEROID_SIMULATION_SHADER_PATH},
            .compile_options = {
                .entry_point = "update_material",
                .language = daxa::ShaderLanguage::SLANG
            }
        },
        .push_constant_size = static_cast<u32>(sizeof(MaterialUpdatePush)),
        .name = std::string{MaterialUpdateH::NAME},
    };
}

inline daxa::ComputePipelineCompileInfo derivative_update_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{ASTEROID_SIMULATION_SHADER_PATH},
            .compile_options = {
                .entry_point = "calculate_derivatives",
                .language = daxa::ShaderLanguage::SLANG
            }
        },
        .push_constant_size = static_cast<u32>(sizeof(DerivativesCalculationPush)),
        .name = std::string{DerivativesCalculationH::NAME},
    };
}

inline daxa::ComputePipelineCompileInfo equation_update_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{ASTEROID_SIMULATION_SHADER_PATH},
            .compile_options = {
                .entry_point = "equation_update",
                .language = daxa::ShaderLanguage::SLANG
            }
        },
        .push_constant_size = static_cast<u32>(sizeof(EquationUpdatePush)),
        .name = std::string{EquationUpdateH::NAME},
    };
}