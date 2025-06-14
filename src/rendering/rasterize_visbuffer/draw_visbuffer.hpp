#pragma once

#include "draw_visbuffer.inl"

#include "../scene_renderer_context.hpp"
#include "../tasks/misc.hpp"

inline MAKE_COMPUTE_COMPILE_INFO(cull_meshlets_compute_pipeline_compile_info, "./src/rendering/rasterize_visbuffer/draw_visbuffer.hlsl", "entry_compute_meshlet_cull")

static constexpr inline char const SLANG_DRAW_VISBUFFER_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/draw_visbuffer.hlsl";

using DrawVisbuffer_WriteCommandTask2 = SimpleComputeTask<
    DrawVisbuffer_WriteCommandH::Task,
    DrawVisbufferPush_WriteCommand,
    SLANG_DRAW_VISBUFFER_SHADER_PATH,
    "entry_write_commands">;

struct DrawVisbufferPipelineConfig
{
    bool task_shader_cull = {};
    bool alpha_masked_geo = {};
    auto to_index() const -> u32
    {
        return (task_shader_cull ? 1 : 0) + (alpha_masked_geo ? 2 : 0);
    }
    static auto from_index(u32 index) -> DrawVisbufferPipelineConfig
    {
        DrawVisbufferPipelineConfig ret = {};
        ret.task_shader_cull = (index & 1) != 0;
        ret.alpha_masked_geo = (index & 2) != 0;
        return ret;
    }
    static constexpr auto index_count() -> u32
    {
        return 8;
    }
};
static inline auto create_draw_visbuffer_compile_info(DrawVisbufferPipelineConfig config) -> daxa::RasterPipelineCompileInfo
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.depth_test = {
        .depth_attachment_format = daxa::Format::D32_SFLOAT,
        .enable_depth_write = true,
        .depth_test_compare_op = daxa::CompareOp::GREATER,
        .min_depth_bounds = 0.0f,
        .max_depth_bounds = 1.0f,
    };
    ret.color_attachments = {
        daxa::RenderAttachment{
            .format = daxa::Format::R32_UINT,
        },
    };
    char const * task_cull_suffix = config.task_shader_cull ? "_meshlet_cull" : "";
    char const * geo_suffix = config.alpha_masked_geo ? "_masked" : "_opaque";
    ret.name = std::string("DrawVisbuffer") + task_cull_suffix + geo_suffix;
    ret.raster.static_state_sample_count = daxa::None; // Set to use dynamic state for msaa.
    ret.push_constant_size = config.task_shader_cull ? s_cast<u32>(sizeof(CullMeshletsDrawVisbufferPush)) : s_cast<u32>(sizeof(DrawVisbufferPush));
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = std::string("entry_fragment") + task_cull_suffix + geo_suffix,
            .language = daxa::ShaderLanguage::SLANG,
        },
    };    
    ret.mesh_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = std::string("entry_mesh") + task_cull_suffix + geo_suffix,
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    if (config.task_shader_cull)
    {
        ret.task_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
            .compile_options = {
            .entry_point = std::string("entry_task") + task_cull_suffix,
                .language = daxa::ShaderLanguage::SLANG,
            },
        };
    }
    return ret;
}

inline static std::array<daxa::RasterPipelineCompileInfo, DrawVisbufferPipelineConfig::index_count()> draw_visbuffer_mesh_shader_pipelines = {
    create_draw_visbuffer_compile_info(DrawVisbufferPipelineConfig::from_index(0)),
    create_draw_visbuffer_compile_info(DrawVisbufferPipelineConfig::from_index(1)),
    create_draw_visbuffer_compile_info(DrawVisbufferPipelineConfig::from_index(2)),
    create_draw_visbuffer_compile_info(DrawVisbufferPipelineConfig::from_index(3)),
};

struct DrawVisbufferTask : DrawVisbufferH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    u32 pass = {};
    bool observer = {};
    bool clear_render_targets = {};
    void callback(daxa::TaskInterface ti)
    {
        u32 render_time_index = RenderTimes::INVALID_RENDER_TIME_INDEX;
        if (pass == VISBUF_FIRST_PASS)
        {
            render_time_index = RenderTimes::index<"VISBUFFER","FIRST_PASS_DRAW">();
        }
        if (pass == VISBUF_SECOND_PASS)
        {
            render_time_index = RenderTimes::index<"VISBUFFER","SECOND_PASS_DRAW">();
        }

        render_context->render_times.start_gpu_timer(ti.recorder, render_time_index);

        auto [x, y, z] = ti.device.image_info(ti.id(AT.depth_image)).value().size;
        auto load_op = clear_render_targets ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD;
        daxa::RenderPassBeginInfo render_pass_begin_info = {
            .render_area = daxa::Rect2D{.width = x, .height = y},
        };
        render_pass_begin_info.depth_attachment =
            daxa::RenderAttachmentInfo{
                .image_view = ti.id(AT.depth_image).default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::DepthValue{0.0f, 0},
            };
        render_pass_begin_info.color_attachments.push_back(
            daxa::RenderAttachmentInfo{
                .image_view = ti.id(AT.vis_image).default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = std::array<u32, 4>{INVALID_TRIANGLE_ID, 0, 0, 0},
            }
        );

        auto render_cmd = std::move(ti.recorder).begin_renderpass(render_pass_begin_info);
        render_cmd.set_rasterization_samples(daxa::RasterizationSamples::E1);

        for (u32 opaque_draw_list_type = 0; opaque_draw_list_type < 2; ++opaque_draw_list_type)
        {
            auto const & pipeline_info = draw_visbuffer_mesh_shader_pipelines[DrawVisbufferPipelineConfig{
                .task_shader_cull = false, 
                .alpha_masked_geo = opaque_draw_list_type != 0,
            }.to_index()];
            render_cmd.set_pipeline(*render_context->gpu_context->raster_pipelines.at(pipeline_info.name));
            DrawVisbufferPush push{ 
                .attach = ti.attachment_shader_blob,
                .draw_data = {
                    .pass_index = pass,
                    .draw_list_section_index = opaque_draw_list_type,
                    .observer = observer,
                },
                .meshes = render_context->render_data.scene.meshes,
                .materials = render_context->render_data.scene.materials,
                .entity_combined_transforms = render_context->render_data.scene.entity_combined_transforms,
            };
            render_cmd.push_constant(push);
            render_cmd.draw_mesh_tasks_indirect({
                .indirect_buffer = ti.id(AT.draw_commands),
                .offset = sizeof(DispatchIndirectStruct) * opaque_draw_list_type,
                .draw_count = 1,
                .stride = sizeof(DispatchIndirectStruct),
            });
        }
        ti.recorder = std::move(render_cmd).end_renderpass();

        render_context->render_times.end_gpu_timer(ti.recorder, render_time_index);
    }
};

struct CullMeshletsDrawVisbufferTask : CullMeshletsDrawVisbufferH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    bool first_pass = {};
    bool clear_render_targets = {};
    void callback(daxa::TaskInterface ti)
    {
        auto load_op = clear_render_targets ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD;
        auto [x, y, z] = ti.info(AT.vis_image).value().size;
        daxa::RenderPassBeginInfo render_pass_begin_info = {
            .render_area = daxa::Rect2D{.width = x, .height = y},
        };
        render_pass_begin_info.depth_attachment =
            daxa::RenderAttachmentInfo{
                .image_view = ti.view(AT.depth_image),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::DepthValue{0.0f, 0},
            };
        render_pass_begin_info.color_attachments.push_back(
            daxa::RenderAttachmentInfo{
                .image_view = ti.view(AT.vis_image),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = std::array<u32, 4>{INVALID_TRIANGLE_ID, 0, 0, 0},
            }
        );

        u32 render_time_index = first_pass ? RenderTimes::index<"VISBUFFER","FIRST_PASS_CULL_AND_DRAW">() : RenderTimes::index<"VISBUFFER","SECOND_PASS_CULL_AND_DRAW">(); 

        render_context->render_times.start_gpu_timer(ti.recorder, render_time_index);
        auto render_cmd = std::move(ti.recorder).begin_renderpass(render_pass_begin_info);
        
        for (u32 opaque_draw_list_type = 0; opaque_draw_list_type < PREPASS_DRAW_LIST_TYPE_COUNT; ++opaque_draw_list_type)
        {
            auto buffer = ti.id(opaque_draw_list_type == PREPASS_DRAW_LIST_OPAQUE ? AT.po2expansion : AT.masked_po2expansion);
            auto const & pipeline_info = draw_visbuffer_mesh_shader_pipelines[DrawVisbufferPipelineConfig{
                .task_shader_cull = true, 
                .alpha_masked_geo = opaque_draw_list_type != 0,
            }.to_index()];
            render_cmd.set_pipeline(*render_context->gpu_context->raster_pipelines.at(pipeline_info.name));

            const bool prefix_sum_expansion = render_context->render_data.settings.enable_prefix_sum_work_expansion;
            CullMeshletsDrawVisbufferPush push = {
                .attach = ti.attachment_shader_blob,
                .draw_data = {
                    .pass_index = first_pass ? 0u : 1u,
                    .draw_list_section_index = opaque_draw_list_type,
                    .observer = false,
                },
                .meshes = render_context->render_data.scene.meshes,
                .materials = render_context->render_data.scene.materials,
                .entity_combined_transforms = render_context->render_data.scene.entity_combined_transforms,
            };
            render_cmd.push_constant(push);
            render_cmd.draw_mesh_tasks_indirect({
                .indirect_buffer = buffer,
                .offset = 0,
                .draw_count = 1,
                .stride = sizeof(DispatchIndirectStruct),
            });
        }
        ti.recorder = std::move(render_cmd).end_renderpass();
        render_context->render_times.end_gpu_timer(ti.recorder, render_time_index);
    }
};

struct CullMeshletsComputeTask : CullMeshletsDrawVisbufferH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    bool first_pass = {};
    static inline constexpr std::string_view NAME = "CullMeshletsCompute";
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(cull_meshlets_compute_pipeline_compile_info().name));

        u32 render_time_index = first_pass ? RenderTimes::index<"VISBUFFER","FIRST_PASS_CULL_MESHLETS_COMPUTE">() : RenderTimes::index<"VISBUFFER","SECOND_PASS_CULL_MESHLETS_COMPUTE">(); 

        render_context->render_times.start_gpu_timer(ti.recorder, render_time_index);
        
        for (u32 opaque_draw_list_type = 0; opaque_draw_list_type < PREPASS_DRAW_LIST_TYPE_COUNT; ++opaque_draw_list_type)
        {
            auto buffer = ti.id(opaque_draw_list_type == PREPASS_DRAW_LIST_OPAQUE ? AT.po2expansion : AT.masked_po2expansion);
            for (u32 i = 0; i < 1; ++i)
            {
                CullMeshletsDrawVisbufferPush push = {
                    .attach = ti.attachment_shader_blob,
                    .draw_data = {
                        .pass_index = first_pass ? 0u : 1u,
                        .draw_list_section_index = opaque_draw_list_type,
                        .observer = false,
                    },
                    .meshes = render_context->render_data.scene.meshes,
                    .materials = render_context->render_data.scene.materials,
                    .entity_combined_transforms = render_context->render_data.scene.entity_combined_transforms,
                };
                ti.recorder.push_constant(push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = buffer,
                    .offset = sizeof(DispatchIndirectStruct) * i,
                });
            }
        }
        render_context->render_times.end_gpu_timer(ti.recorder, render_time_index);
    }
};

struct TaskCullAndDrawVisbufferInfo
{
    RenderContext * render_context = {};
    daxa::TaskGraph & tg;
    bool first_pass = {};
    bool clear_render_targets = {};
    daxa::TaskBufferView first_pass_meshlet_bitfield = {};
    std::array<daxa::TaskBufferView, PREPASS_DRAW_LIST_TYPE_COUNT> meshlet_cull_po2expansion = {};
    daxa::TaskImageView hiz = {};
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskBufferView mesh_instances = {};
    daxa::TaskImageView vis_image = {};
    daxa::TaskImageView debug_image = {};    
    daxa::TaskImageView depth_image = {};
    daxa::TaskImageView overdraw_image = {};
};
inline void task_cull_and_draw_visbuffer(TaskCullAndDrawVisbufferInfo const & info)
{
    u32 const pass = info.first_pass ? VISBUF_FIRST_PASS : VISBUF_SECOND_PASS;

    if (info.render_context->render_data.settings.enable_separate_compute_meshlet_culling)
    {
        auto const stage = daxa::TaskStage::COMPUTE_SHADER;
        info.tg.add_task(CullMeshletsComputeTask{
            .views = CullMeshletsComputeTask::Views{
                .globals = info.render_context->tgpu_render_data.view().override_stage(stage),
                .hiz = info.hiz.override_stage(stage),
                .po2expansion = info.meshlet_cull_po2expansion[0].override_stage(stage),
                .masked_po2expansion = info.meshlet_cull_po2expansion[1].override_stage(stage),
                .first_pass_meshlet_bitfield = info.first_pass_meshlet_bitfield.override_stage(stage),
                .meshlet_instances = info.meshlet_instances.override_stage(stage),
                .mesh_instances = info.mesh_instances.override_stage(stage),
                .overdraw_image = info.overdraw_image.override_stage(stage),
                .vis_image = info.vis_image,
                .depth_image = info.depth_image,
            },
            .render_context = info.render_context,
            .first_pass = info.first_pass,
        });

        auto draw_commands_array = info.tg.create_transient_buffer({
            .size = 2 * static_cast<u32>(std::max(sizeof(DrawIndirectStruct), sizeof(DispatchIndirectStruct))),
            .name = std::string("draw visbuffer command buffer array") + info.render_context->gpu_context->dummy_string(),
        });

        DrawVisbuffer_WriteCommandTask2 write_task = {
            .views = DrawVisbuffer_WriteCommandTask2::Views{
                .globals = info.render_context->tgpu_render_data,
                .meshlet_instances = info.meshlet_instances,
                .draw_commands = draw_commands_array,
            },
            .gpu_context = info.render_context->gpu_context,
            .push = DrawVisbufferPush_WriteCommand{.pass = pass},
            .dispatch_callback = [](){ return daxa::DispatchInfo{1,1,1}; },
        };
        info.tg.add_task(write_task);

        DrawVisbufferTask draw_task = {
            .views = DrawVisbufferTask::Views{
                .globals = info.render_context->tgpu_render_data,
                .draw_commands = draw_commands_array,
                .hiz = info.hiz,
                .meshlet_instances = info.meshlet_instances,
                .overdraw_image = info.overdraw_image,
                .vis_image = info.vis_image,
                .depth_image = info.depth_image,
            },
            .render_context = info.render_context,
            .pass = pass,
            .clear_render_targets = info.clear_render_targets,
        };
        info.tg.add_task(draw_task);
    }
    else
    {
        auto const stage = daxa::TaskStage::RASTER_SHADER;
        info.tg.add_task(CullMeshletsDrawVisbufferTask{
            .views = CullMeshletsDrawVisbufferTask::Views{
                .globals = info.render_context->tgpu_render_data.view().override_stage(stage),
                .hiz = info.hiz.override_stage(stage),
                .po2expansion = info.meshlet_cull_po2expansion[0].override_stage(stage),
                .masked_po2expansion = info.meshlet_cull_po2expansion[1].override_stage(stage),
                .first_pass_meshlet_bitfield = info.first_pass_meshlet_bitfield.override_stage(stage),
                .meshlet_instances = info.meshlet_instances.override_stage(stage),
                .mesh_instances = info.mesh_instances.override_stage(stage),
                .overdraw_image = info.overdraw_image.override_stage(stage),
                .vis_image = info.vis_image,
                .depth_image = info.depth_image,
            },
            .render_context = info.render_context,
            .first_pass = info.first_pass,
            .clear_render_targets = info.clear_render_targets,
        });
    }
}

struct TaskDrawVisbufferInfo
{
    RenderContext * render_context = {};
    daxa::TaskGraph & tg;
    u32 pass = {};
    bool observer = {};
    bool clear_render_targets = {};
    daxa::TaskImageView hiz = daxa::NullTaskImage;
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView material_manifest = {};
    daxa::TaskBufferView combined_transforms = {};
    daxa::TaskImageView vis_image = {};
    daxa::TaskImageView debug_image = {};    
    daxa::TaskImageView depth_image = {};
    daxa::TaskImageView overdraw_image = {};
};

inline void task_draw_visbuffer(TaskDrawVisbufferInfo const & info)
{
    auto draw_commands_array = info.tg.create_transient_buffer({
        .size = 2 * static_cast<u32>(std::max(sizeof(DrawIndirectStruct), sizeof(DispatchIndirectStruct))),
        .name = std::string("draw visbuffer command buffer array") + info.render_context->gpu_context->dummy_string(),
    });

    DrawVisbuffer_WriteCommandTask2 write_task = {
        .views = DrawVisbuffer_WriteCommandTask2::Views{
            .globals = info.render_context->tgpu_render_data,
            .meshlet_instances = info.meshlet_instances,
            .draw_commands = draw_commands_array,
        },
        .gpu_context = info.render_context->gpu_context,
        .push = DrawVisbufferPush_WriteCommand{.pass = info.pass},
        .dispatch_callback = [](){ return daxa::DispatchInfo{1,1,1}; },
    };
    info.tg.add_task(write_task);

    if (info.overdraw_image != daxa::NullTaskImage && info.clear_render_targets)
    {
        info.tg.clear_image({info.overdraw_image, std::array{0u, 0u, 0u, 0u}});
    }

    DrawVisbufferTask draw_task = {
        .views = DrawVisbufferTask::Views{
            .globals = info.render_context->tgpu_render_data,
            .draw_commands = draw_commands_array,
            .hiz = info.hiz,
            .meshlet_instances = info.meshlet_instances,
            .overdraw_image = info.overdraw_image,
            .vis_image = info.vis_image,
            .depth_image = info.depth_image,
        },
        .render_context = info.render_context,
        .pass = info.pass,
        .observer = info.observer,
        .clear_render_targets = info.clear_render_targets,
    };
    info.tg.add_task(draw_task);
}