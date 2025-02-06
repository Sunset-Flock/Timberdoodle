#include "draw_asteroids.hpp"
#include "../tasks/misc.hpp"

static const daxa_f32vec3 PROBE_MESH_POSITIONS[] = {
    daxa_f32vec3{0.270598, 0.923880, -0.270598},
    daxa_f32vec3{0.500000, 0.707107, -0.500000},
    daxa_f32vec3{0.653281, 0.382683, -0.653281},
    daxa_f32vec3{0.707107, 0.000000, -0.707107},
    daxa_f32vec3{0.653281, -0.382683, -0.653281},
    daxa_f32vec3{0.500000, -0.707107, -0.500000},
    daxa_f32vec3{0.270598, -0.923880, -0.270598},
    daxa_f32vec3{0.382683, 0.923880, 0.000000},
    daxa_f32vec3{0.707107, 0.707107, 0.000000},
    daxa_f32vec3{0.923879, 0.382683, 0.000000},
    daxa_f32vec3{1.000000, 0.000000, 0.000000},
    daxa_f32vec3{0.923879, -0.382683, 0.000000},
    daxa_f32vec3{0.707107, -0.707107, 0.000000},
    daxa_f32vec3{0.382683, -0.923880, 0.000000},
    daxa_f32vec3{0.270598, 0.923880, 0.270598},
    daxa_f32vec3{0.500000, 0.707107, 0.500000},
    daxa_f32vec3{0.653281, 0.382683, 0.653281},
    daxa_f32vec3{0.707107, 0.000000, 0.707107},
    daxa_f32vec3{0.653281, -0.382683, 0.653281},
    daxa_f32vec3{0.500000, -0.707107, 0.500000},
    daxa_f32vec3{0.270598, -0.923880, 0.270598},
    daxa_f32vec3{0.000000, 0.923880, 0.382683},
    daxa_f32vec3{0.000000, 0.707107, 0.707107},
    daxa_f32vec3{0.000000, 0.382683, 0.923879},
    daxa_f32vec3{0.000000, 0.000000, 1.000000},
    daxa_f32vec3{0.000000, -0.382683, 0.923879},
    daxa_f32vec3{0.000000, -0.707107, 0.707107},
    daxa_f32vec3{0.000000, -0.923880, 0.382683},
    daxa_f32vec3{-0.270598, 0.923880, 0.270598},
    daxa_f32vec3{-0.500000, 0.707107, 0.500000},
    daxa_f32vec3{-0.653281, 0.382683, 0.653281},
    daxa_f32vec3{-0.707107, 0.000000, 0.707107},
    daxa_f32vec3{-0.653281, -0.382683, 0.653281},
    daxa_f32vec3{-0.500000, -0.707107, 0.500000},
    daxa_f32vec3{-0.270598, -0.923880, 0.270598},
    daxa_f32vec3{-0.382683, 0.923880, 0.000000},
    daxa_f32vec3{-0.707107, 0.707107, 0.000000},
    daxa_f32vec3{-0.923879, 0.382683, 0.000000},
    daxa_f32vec3{-1.000000, 0.000000, 0.000000},
    daxa_f32vec3{-0.923879, -0.382683, 0.000000},
    daxa_f32vec3{-0.707107, -0.707107, 0.000000},
    daxa_f32vec3{-0.382683, -0.923880, 0.000000},
    daxa_f32vec3{-0.270598, 0.923880, -0.270598},
    daxa_f32vec3{-0.500000, 0.707107, -0.500000},
    daxa_f32vec3{-0.653281, 0.382683, -0.653281},
    daxa_f32vec3{-0.707107, 0.000000, -0.707107},
    daxa_f32vec3{-0.653281, -0.382683, -0.653281},
    daxa_f32vec3{-0.500000, -0.707107, -0.500000},
    daxa_f32vec3{-0.270598, -0.923880, -0.270598},
    daxa_f32vec3{0.000000, -1.000000, 0.000000},
    daxa_f32vec3{0.000000, 1.000000, 0.000000},
    daxa_f32vec3{0.000000, 0.923880, -0.382683},
    daxa_f32vec3{0.000000, 0.707107, -0.707107},
    daxa_f32vec3{0.000000, 0.382683, -0.923879},
    daxa_f32vec3{0.000000, 0.000000, -1.000000},
    daxa_f32vec3{0.000000, -0.382683, -0.923879},
    daxa_f32vec3{0.000000, -0.707107, -0.707107},
    daxa_f32vec3{0.000000, -0.923880, -0.382683}
};
    

static const daxa_i32 PROBE_MESH_INDICES[] = {
    57, 5, 6,
    55, 3, 4,
    53, 1, 2,
    51, 50, 0,
    49, 57, 6,
    56, 4, 5,
    54, 2, 3,
    52, 0, 1,
    2, 8, 9,
    0, 50, 7,
    49, 6, 13,
    5, 11, 12,
    3, 9, 10,
    1, 7, 8,
    6, 12, 13,
    4, 10, 11,
    49, 13, 20,
    12, 18, 19,
    10, 16, 17,
    8, 14, 15,
    13, 19, 20,
    11, 17, 18,
    9, 15, 16,
    7, 50, 14,
    15, 21, 22,
    20, 26, 27,
    18, 24, 25,
    16, 22, 23,
    14, 50, 21,
    49, 20, 27,
    19, 25, 26,
    17, 23, 24,
    27, 33, 34,
    25, 31, 32,
    23, 29, 30,
    21, 50, 28,
    49, 27, 34,
    26, 32, 33,
    24, 30, 31,
    22, 28, 29,
    28, 50, 35,
    49, 34, 41,
    33, 39, 40,
    31, 37, 38,
    29, 35, 36,
    34, 40, 41,
    32, 38, 39,
    30, 36, 37,
    40, 46, 47,
    38, 44, 45,
    36, 42, 43,
    41, 47, 48,
    39, 45, 46,
    37, 43, 44,
    35, 50, 42,
    49, 41, 48,
    48, 56, 57,
    46, 54, 55,
    44, 52, 53,
    42, 50, 51,
    49, 48, 57,
    47, 55, 56,
    45, 53, 54,
    43, 51, 52,
    57, 56, 5,
    55, 54, 3,
    53, 52, 1,
    56, 55, 4,
    54, 53, 2,
    52, 51, 0,
    2, 1, 8,
    5, 4, 11,
    3, 2, 9,
    1, 0, 7,
    6, 5, 12,
    4, 3, 10,
    12, 11, 18,
    10, 9, 16,
    8, 7, 14,
    13, 12, 19,
    11, 10, 17,
    9, 8, 15,
    15, 14, 21,
    20, 19, 26,
    18, 17, 24,
    16, 15, 22,
    19, 18, 25,
    17, 16, 23,
    27, 26, 33,
    25, 24, 31,
    23, 22, 29,
    26, 25, 32,
    24, 23, 30,
    22, 21, 28,
    33, 32, 39,
    31, 30, 37,
    29, 28, 35,
    34, 33, 40,
    32, 31, 38,
    30, 29, 36,
    40, 39, 46,
    38, 37, 44,
    36, 35, 42,
    41, 40, 47,
    39, 38, 45,
    37, 36, 43,
    48, 47, 56,
    46, 45, 54,
    44, 43, 52,
    47, 46, 55,
    45, 44, 53,
    43, 42, 51
};

struct MaterialUpdateTask : MaterialUpdateH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    AsteroidsState* asteroid_state = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(material_update_compile_info().name));
        MaterialUpdatePush push = {
            .attach = ti.attachment_shader_blob,
            .asteroid_count = asteroid_state->asteroids_count,
            .asteroid_density = r_cast<daxa_f32*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_DENSITY]).value()),
            .asteroid_energy = r_cast<daxa_f32*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_ENERGY]).value()),
            .asteroid_pressure = r_cast<daxa_f32*>(ti.device.buffer_device_address(ti.get(AT.asteroid_params).ids[ASTEROID_PRESSURE]).value()),
            .start_density = INITAL_DENSITY,
            .A = 26700000000.0f,
            .c = 2.0f,
        };
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({round_up_div(asteroid_state->asteroids_count, MATERIAL_UPDATE_WORKGROUP_X), 1, 1});
    }
};

struct DerivativesCalculationTask : DerivativesCalculationH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    AsteroidsState* asteroid_state = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(derivative_update_compile_info().name));
        DerivativesCalculationPush push = {
            .asteroid_count = asteroid_state->asteroids_count,
            .params = {
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
        };
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({asteroid_state->asteroids_count, 1, 1});
    }
};

struct EquationUpdateTask : EquationUpdateH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    AsteroidsState* asteroid_state = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(equation_update_compile_info().name));
        EquationUpdatePush push = {
            .asteroid_count = asteroid_state->asteroids_count,
            .dt = 0.01f,
            .params = {
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
        };
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({round_up_div(asteroid_state->asteroids_count, EQUATION_UPDATE_WORKGROUP_X), 1, 1});
    }
};

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

void AsteroidsState::initialize_persistent_state(daxa::Device& device)
{
    debug_probe_mesh_triangles = (sizeof(PROBE_MESH_INDICES) / sizeof(daxa_i32)) / 3;
    auto const probe_triangles_mem_size = debug_probe_mesh_triangles * 3 * sizeof(daxa_i32);
    debug_probe_mesh_vertices = sizeof(PROBE_MESH_POSITIONS) / sizeof(daxa_f32vec3);
    auto const probe_vertex_mem_size = debug_probe_mesh_vertices * sizeof(daxa_f32vec3);

    debug_asteroid_mesh_buffer = device.create_buffer({
        .size = probe_triangles_mem_size + probe_vertex_mem_size,
        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "debug asteroids mesh buffer",
    });

    std::byte* host_addr = device.buffer_host_address(debug_asteroid_mesh_buffer).value();
    std::memcpy(host_addr, PROBE_MESH_INDICES, probe_triangles_mem_size);
    std::memcpy(host_addr + probe_triangles_mem_size, PROBE_MESH_POSITIONS, probe_vertex_mem_size);

    daxa::DeviceAddress device_addr = device.buffer_device_address(debug_asteroid_mesh_buffer).value();

    debug_asteroid_mesh_vertex_positions_addr = reinterpret_cast<daxa_f32vec3*>(device_addr + probe_triangles_mem_size);

#if !CPU_SIMULATION 
    gpu_asteroids = daxa::TaskBuffer(daxa::TaskBufferInfo{
        .initial_buffers = {std::array{ device.create_buffer({.size = 1, .name = "dummy"})}},
        .name = "GPU Asteroids Task buffer"
        }
    );
#endif
}

void AsteroidsState::initalize_transient_state(daxa::TaskGraph & tg)
{
#if CPU_SIMULATION
    asteroids = tg.create_transient_buffer({
        .size = sizeof(GPUAsteroid) * MAX_ASTEROID_COUNT,
        .name = "asteroids buffer"
    });
#endif
}

void AsteroidsState::initialize_gpu_simulation(daxa::Device & device, AsteroidsWrapper const & asteroids)
{
#if !CPU_SIMULATION
    asteroids_count = asteroids.positions.size();

    if(!gpu_asteroids.get_state().buffers.empty())
    {
        for(auto const & buffer : gpu_asteroids.get_state().buffers)
        {
            device.destroy_buffer(buffer);
        }
    }

    auto const ai = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE;
    gpu_asteroids.set_buffers({
        .buffers = std::array{
            device.create_buffer({ .size = sizeof(f32vec3) * asteroids_count, .allocate_info = ai, .name = "gpu asteroids positions"}),
            device.create_buffer({ .size = sizeof(f32vec3) * asteroids_count, .allocate_info = ai, .name = "gpu asteroids velocities"}),
            device.create_buffer({ .size = sizeof(f32vec3) * asteroids_count, .allocate_info = ai, .name = "gpu asteroids velocity derivatives"}),
            device.create_buffer({ .size = sizeof(f32) * asteroids_count, .allocate_info = ai, .name = "gpu asteroids velocity divergences"}),
            device.create_buffer({ .size = sizeof(f32) * asteroids_count, .allocate_info = ai, .name = "gpu asteroids smoothing radii" }),
            device.create_buffer({ .size = sizeof(f32) * asteroids_count, .allocate_info = ai, .name = "gpu asteroids smoothing masses" }),
            device.create_buffer({ .size = sizeof(f32) * asteroids_count, .allocate_info = ai, .name = "gpu asteroids densities" }),
            device.create_buffer({ .size = sizeof(f32) * asteroids_count, .allocate_info = ai, .name = "gpu asteroids density derivatives" }),
            device.create_buffer({ .size = sizeof(f32) * asteroids_count, .allocate_info = ai, .name = "gpu asteroids energies" }),
            device.create_buffer({ .size = sizeof(f32) * asteroids_count, .allocate_info = ai, .name = "gpu asteroids energy derivatives" }),
            device.create_buffer({ .size = sizeof(f32) * asteroids_count, .allocate_info = ai, .name = "gpu asteroids pressures" }),
            device.create_buffer({ .size = sizeof(f32) * asteroids_count, .allocate_info = ai, .name = "gpu asteroids scales" }),
        },
    });

    std::array<const void*, 3> vec3_data_pointers = {asteroids.positions.data(), asteroids.velocities.data(), asteroids.velocity_derivatives.data()};
    for(i32 vec3_buffer = ASTEROID_POSITION; vec3_buffer < ASTEROID_VELOCITY_DIVERGENCE; vec3_buffer++)
    {
        std::byte * host_addr = device.buffer_host_address_as<std::byte>(gpu_asteroids.get_state().buffers[vec3_buffer]).value();
        std::memcpy(host_addr, vec3_data_pointers.at(vec3_buffer), sizeof(f32vec3) * asteroids_count);
    }

    std::array<const void*, 9> float_data_pointers = {
        asteroids.velocity_divergences.data(),
        asteroids.smoothing_radii.data(),
        asteroids.masses.data(),
        asteroids.densities.data(),
        asteroids.density_derivatives.data(),
        asteroids.energies.data(),
        asteroids.energy_derivatives.data(),
        asteroids.pressures.data(),
        asteroids.particle_scales.data(),
    };

    for(i32 float_buffer = ASTEROID_VELOCITY_DIVERGENCE; float_buffer <= ASTEROID_SCALE; float_buffer++)
    {
        std::byte * host_addr = device.buffer_host_address_as<std::byte>(gpu_asteroids.get_state().buffers[float_buffer]).value();
        std::memcpy(host_addr, float_data_pointers.at(float_buffer - ASTEROID_VELOCITY_DIVERGENCE), sizeof(f32) * asteroids_count);
    }
#endif
}

void AsteroidsState::update_cpu_data(daxa::Device & device, AsteroidSimulation & simulation, ShaderDebugDrawContext & debug_context)
{
#if CPU_SIMULATION
    AsteroidsWrapper asteroids = simulation.get_asteroids();
    if(asteroids.simulation_started)
    {
        DBG_ASSERT_TRUE_M(asteroids.positions.size() <= MAX_ASTEROID_COUNT, "Asteroids must fit into GPU buffer");
        for(i32 asteroid_index = 0; asteroid_index < asteroids.positions.size(); ++asteroid_index)
        {
            auto const & position = asteroids.positions.at(asteroid_index);
            cpu_asteroids.at(asteroid_index).position = daxa_f32vec3(
                position.x, position.y, position.z
            );
            auto const & velocity = asteroids.velocities.at(asteroid_index);
            cpu_asteroids.at(asteroid_index).velocity = daxa_f32vec3(
                velocity.x, velocity.y, velocity.z
            );
            auto const & velocity_derivative = asteroids.velocity_derivatives.at(asteroid_index);
            cpu_asteroids.at(asteroid_index).acceleration = daxa_f32vec3(
                velocity_derivative.x, velocity_derivative.y, velocity_derivative.z
            );
            cpu_asteroids.at(asteroid_index).velocity_divergence = asteroids.velocity_divergences.at(asteroid_index);
            cpu_asteroids.at(asteroid_index).pressure = asteroids.pressures.at(asteroid_index);
            cpu_asteroids.at(asteroid_index).density = asteroids.densities.at(asteroid_index);
            cpu_asteroids.at(asteroid_index).particle_scale = asteroids.particle_scales.at(asteroid_index);
        }
        asteroids_count = asteroids.positions.size();
    }
#else
    if(simulation.simulation_started.load())
    {
        simulation_just_started = !last_simulation_started && simulation.simulation_started.load();
        if(simulation_just_started)
        {
            initialize_gpu_simulation(device, simulation.get_asteroids());
        }
    }
#endif
    else
    {
        // auto const & asteroids = simulation.get_asteroids();
        // for(i32 simulation_body_index = 0; simulation_body_index < asteroids.simulation_bodies.size(); ++simulation_body_index)
        // {
        //     auto const & simulation_body = asteroids.simulation_bodies.at(simulation_body_index);
        //     cpu_asteroids.at(simulation_body_index).position = std::bit_cast<daxa_f32vec3>(simulation_body.position);
        //     cpu_asteroids.at(simulation_body_index).velocity = std::bit_cast<daxa_f32vec3>(simulation_body.velocity_vector * simulation_body.velocity_magnitude);
        //     cpu_asteroids.at(simulation_body_index).particle_scale = simulation_body.radius * POSITION_SCALING_FACTOR;

        //     f32 const & velocity_vector_scaling_factor = (simulation_body.radius + glm::length(simulation_body.velocity_magnitude));
        //     debug_context.line_draws.draw({
        //         .start = std::bit_cast<daxa_f32vec3>(simulation_body.position * s_cast<f32>(POSITION_SCALING_FACTOR)),
        //         .end = std::bit_cast<daxa_f32vec3>((simulation_body.position + simulation_body.velocity_vector * velocity_vector_scaling_factor) * s_cast<f32>(POSITION_SCALING_FACTOR)),
        //         .color = daxa_f32vec3{1.0f, 0.0f, 0.0f},
        //         .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE,
        //     });
        // }
        // asteroids_count = asteroids.simulation_bodies.size();
    }

#if !CPU_SIMULATION
    last_simulation_started = simulation.simulation_started.load();
#endif
}

void task_draw_asteroids(TaskDrawAsteroidsInfo const & info)
{

#if CPU_SIMULATION
    info.tg->add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, info.asteroids_state->asteroids)
        },
        .task = [info](daxa::TaskInterface ti)
        {
            allocate_fill_copy(ti, info.asteroids_state->cpu_asteroids, ti.get(info.asteroids_state->asteroids));
        }
    });

    info.tg->add_task(DebugDrawAsteroidsTask{
        .views = std::array{
            DebugDrawAsteroidsTask::AT.globals | info.render_context->tgpu_render_data,
            DebugDrawAsteroidsTask::AT.asteroids | info.asteroids_state->asteroids,
            DebugDrawAsteroidsTask::AT.color_image | info.color,
            DebugDrawAsteroidsTask::AT.depth_image | info.depth
        },
        .render_context = info.render_context,
        .asteroid_state = info.asteroids_state,
    });
#else
    info.tg->conditional({
        .condition_index = 0,
        .when_true = [&]{
            info.tg->add_task(MaterialUpdateTask{
                .views = std::array{
                    DebugDrawAsteroidsTask::AT.globals | info.render_context->tgpu_render_data,
                    DebugDrawAsteroidsTask::AT.asteroid_params | info.asteroids_state->gpu_asteroids,
                },
                .render_context = info.render_context,
                .asteroid_state = info.asteroids_state,
            });

            info.tg->add_task(DerivativesCalculationTask{
                .views = std::array{
                    DebugDrawAsteroidsTask::AT.globals | info.render_context->tgpu_render_data,
                    DebugDrawAsteroidsTask::AT.asteroid_params | info.asteroids_state->gpu_asteroids,
                },
                .render_context = info.render_context,
                .asteroid_state = info.asteroids_state,
            });

            info.tg->add_task(EquationUpdateTask{
                .views = std::array{
                    DebugDrawAsteroidsTask::AT.globals | info.render_context->tgpu_render_data,
                    DebugDrawAsteroidsTask::AT.asteroid_params | info.asteroids_state->gpu_asteroids,
                },
                .render_context = info.render_context,
                .asteroid_state = info.asteroids_state,
            });
        }
    });

    info.tg->add_task(DebugDrawAsteroidsTask{
        .views = std::array{
            DebugDrawAsteroidsTask::AT.globals | info.render_context->tgpu_render_data,
            DebugDrawAsteroidsTask::AT.asteroid_params | info.asteroids_state->gpu_asteroids,
            DebugDrawAsteroidsTask::AT.color_image | info.color,
            DebugDrawAsteroidsTask::AT.depth_image | info.depth
        },
        .render_context = info.render_context,
        .asteroid_state = info.asteroids_state,
    });
#endif
}