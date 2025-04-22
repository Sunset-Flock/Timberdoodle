#pragma once

#include "daxa/daxa.inl"
#include "daxa/utils/task_graph.inl"

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry_pipeline.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/gpu_work_expansion.inl"

#define SPLIT_ATOMIC_VISBUFFER_X 16
#define SPLIT_ATOMIC_VISBUFFER_Y 16

#define COMPUTE_RASTERIZE_WORKGROUP_X 64

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(SplitAtomicVisbufferH)
DAXA_TH_IMAGE_ID(READ, REGULAR_2D, atomic_visbuffer)
DAXA_TH_IMAGE_ID(WRITE, REGULAR_2D, visbuffer)
DAXA_TH_IMAGE_ID(WRITE, REGULAR_2D, depth)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(DrawVisbuffer_WriteCommandH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(daxa_u32), draw_commands)
DAXA_DECL_TASK_HEAD_END

// When drawing triangles, this draw command has triangle ids appended to the end of the command.
DAXA_DECL_RASTER_TASK_HEAD_BEGIN(DrawVisbufferH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER(INDIRECT_COMMAND_READ, draw_commands)
// Used by observer to cull:
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, hiz)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_IMAGE_ID(READ_WRITE, REGULAR_2D, atomic_visbuffer)
DAXA_TH_IMAGE_ID(READ_WRITE, REGULAR_2D, overdraw_image)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vis_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

// Used as compute or raster task depending on if its used for atomic or normal attachment
DAXA_DECL_TASK_HEAD_BEGIN(CullMeshletsDrawVisbufferH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
// Cull Attachments:
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, hiz)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, po2expansion)
DAXA_TH_BUFFER_PTR(READ, daxa_u64, masked_po2expansion)
DAXA_TH_BUFFER_PTR(READ_WRITE, SFPMBitfieldRef, first_pass_meshlets_bitfield_arena)
// Draw Attachments:
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_IMAGE_ID(READ_WRITE, REGULAR_2D, atomic_visbuffer) // Optional
DAXA_TH_IMAGE_ID(READ_WRITE, REGULAR_2D, overdraw_image)   // Optional
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vis_image)     // Optional
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)   // Optional
DAXA_DECL_TASK_HEAD_END

struct DrawVisbufferDrawData
{
    daxa_u32 pass_index;
    daxa_u32 draw_list_section_index;
    daxa_b32 observer;
};

struct SplitAtomicVisbufferPush
{
    SplitAtomicVisbufferH::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
};

struct DrawVisbufferPush_WriteCommand
{
    DrawVisbuffer_WriteCommandH::AttachmentShaderBlob attach;
    daxa_u32 pass;
};

struct DrawVisbufferPush
{
    DrawVisbufferH::AttachmentShaderBlob attach;
    DrawVisbufferDrawData draw_data;
    daxa_BufferPtr(GPUMesh) meshes;
    daxa_BufferPtr(GPUMaterial) materials;
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms;
};

#if DAXA_LANGUAGE != DAXA_LANGUAGE_GLSL
struct CullMeshletsDrawVisbufferPush
{
    CullMeshletsDrawVisbufferH::AttachmentShaderBlob attach;
    DrawVisbufferDrawData draw_data;
    daxa_BufferPtr(GPUMesh) meshes;
    daxa_BufferPtr(GPUMaterial) materials;
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms;
};
#endif

#if defined(__cplusplus)
#include "../scene_renderer_context.hpp"
#include "../tasks/misc.hpp"

inline MAKE_COMPUTE_COMPILE_INFO(cull_meshlets_compute_pipeline_compile_info, "./src/rendering/rasterize_visbuffer/draw_visbuffer.hlsl", "entry_compute_meshlet_cull")
inline MAKE_COMPUTE_COMPILE_INFO(draw_meshlets_compute_pipeline_compile_info, "./src/rendering/rasterize_visbuffer/draw_visbuffer.hlsl", "entry_mesh_opaque_compute_raster")

static constexpr inline char const SLANG_DRAW_VISBUFFER_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/draw_visbuffer.hlsl";

using DrawVisbuffer_WriteCommandTask2 = SimpleComputeTask<
    DrawVisbuffer_WriteCommandH::Task,
    DrawVisbufferPush_WriteCommand,
    SLANG_DRAW_VISBUFFER_SHADER_PATH,
    "entry_write_commands">;

using SplitAtomicVisbufferTask = SimpleComputeTask<
    SplitAtomicVisbufferH::Task,
    SplitAtomicVisbufferPush,
    SLANG_DRAW_VISBUFFER_SHADER_PATH,
    "entry_split_atomic_visbuffer">;

struct DrawVisbufferPipelineConfig
{
    bool atomic_visbuffer = {};
    bool task_shader_cull = {};
    bool alpha_masked_geo = {};
    auto to_index() const -> u32
    {
        return (atomic_visbuffer ? 1 : 0) + (task_shader_cull ? 2 : 0) + (alpha_masked_geo ? 4 : 0);
    }
    static auto from_index(u32 index) -> DrawVisbufferPipelineConfig
    {
        DrawVisbufferPipelineConfig ret = {};
        ret.atomic_visbuffer = (index & 1) != 0;
        ret.task_shader_cull = (index & 2) != 0;
        ret.alpha_masked_geo = (index & 4) != 0;
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
    if (!config.atomic_visbuffer)
    {    
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
    }
    char const * task_cull_suffix = config.task_shader_cull ? "_meshlet_cull" : "";
    char const * geo_suffix = config.alpha_masked_geo ? "_masked" : "_opaque";
    char const * atomic_fragment_suffix = config.atomic_visbuffer ? "_atomicvis" : "";
    ret.name = std::string("DrawVisbuffer") + task_cull_suffix + geo_suffix + atomic_fragment_suffix;
    ret.raster.static_state_sample_count = daxa::None; // Set to use dynamic state for msaa.
    ret.push_constant_size = config.task_shader_cull ? s_cast<u32>(sizeof(CullMeshletsDrawVisbufferPush)) : s_cast<u32>(sizeof(DrawVisbufferPush));
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {
            .entry_point = std::string("entry_fragment") + task_cull_suffix + geo_suffix + atomic_fragment_suffix,
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
    create_draw_visbuffer_compile_info(DrawVisbufferPipelineConfig::from_index(4)),
    create_draw_visbuffer_compile_info(DrawVisbufferPipelineConfig::from_index(5)),
    create_draw_visbuffer_compile_info(DrawVisbufferPipelineConfig::from_index(6)),
    create_draw_visbuffer_compile_info(DrawVisbufferPipelineConfig::from_index(7)),
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

        bool const atomic_visbuffer = !ti.id(AT.atomic_visbuffer).is_empty();
        bool const compute_raster = atomic_visbuffer;
        if (compute_raster)
        {
            for (u32 opaque_draw_list_type = 0; opaque_draw_list_type < 2; ++opaque_draw_list_type)
            {
                ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(draw_meshlets_compute_pipeline_compile_info().name));
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
                ti.recorder.push_constant(push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.id(AT.draw_commands),
                    .offset = sizeof(DispatchIndirectStruct) * opaque_draw_list_type,
                });
            }
        }
        else
        {
            auto [x, y, z] = ti.device.image_info(ti.id(AT.depth_image)).value().size;
            auto load_op = clear_render_targets ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD;
            daxa::RenderPassBeginInfo render_pass_begin_info = {
                .render_area = daxa::Rect2D{.width = x, .height = y},
            };
            if (!atomic_visbuffer)
            {
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
            }

            auto render_cmd = std::move(ti.recorder).begin_renderpass(render_pass_begin_info);
            render_cmd.set_rasterization_samples(daxa::RasterizationSamples::E1);

            for (u32 opaque_draw_list_type = 0; opaque_draw_list_type < 2; ++opaque_draw_list_type)
            {
                auto const & pipeline_info = draw_visbuffer_mesh_shader_pipelines[DrawVisbufferPipelineConfig{
                    .atomic_visbuffer = atomic_visbuffer,
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
        }

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
        bool const atomic_visbuffer = !ti.id(AT.atomic_visbuffer).is_empty();
        auto [x, y, z] = ti.info(AT.vis_image).value().size;
        daxa::RenderPassBeginInfo render_pass_begin_info = {
            .render_area = daxa::Rect2D{.width = x, .height = y},
        };
        if (!atomic_visbuffer)
        {
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
        }

        u32 render_time_index = first_pass ? RenderTimes::index<"VISBUFFER","FIRST_PASS_CULL_AND_DRAW">() : RenderTimes::index<"VISBUFFER","SECOND_PASS_CULL_AND_DRAW">(); 

        render_context->render_times.start_gpu_timer(ti.recorder, render_time_index);
        auto render_cmd = std::move(ti.recorder).begin_renderpass(render_pass_begin_info);
        
        for (u32 opaque_draw_list_type = 0; opaque_draw_list_type < PREPASS_DRAW_LIST_TYPE_COUNT; ++opaque_draw_list_type)
        {
            auto buffer = ti.id(opaque_draw_list_type == PREPASS_DRAW_LIST_OPAQUE ? AT.po2expansion : AT.masked_po2expansion);
            auto const & pipeline_info = draw_visbuffer_mesh_shader_pipelines[DrawVisbufferPipelineConfig{
                .atomic_visbuffer = atomic_visbuffer,
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
    std::array<daxa::TaskBufferView, PREPASS_DRAW_LIST_TYPE_COUNT> meshlet_cull_po2expansion = {};
    daxa::TaskBufferView first_pass_meshlets_bitfield_arena = {};
    daxa::TaskImageView hiz = {};
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskBufferView mesh_instances = {};
    daxa::TaskImageView vis_image = {};
    daxa::TaskImageView atomic_visbuffer = {};
    daxa::TaskImageView debug_image = {};    
    daxa::TaskImageView depth_image = {};
    daxa::TaskImageView overdraw_image = {};
};
inline void task_cull_and_draw_visbuffer(TaskCullAndDrawVisbufferInfo const & info)
{
    bool const atomic_visbuffer = !info.atomic_visbuffer.is_null();
    bool const atomic_visbuf_clear = info.clear_render_targets && atomic_visbuffer;
    if (atomic_visbuf_clear)
    {
        info.tg.clear_image({info.atomic_visbuffer, std::array{INVALID_TRIANGLE_ID, 0u, 0u, 0u}});
    }

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
                .first_pass_meshlets_bitfield_arena = info.first_pass_meshlets_bitfield_arena.override_stage(stage),
                .meshlet_instances = info.meshlet_instances.override_stage(stage),
                .mesh_instances = info.mesh_instances.override_stage(stage),
                .atomic_visbuffer = info.atomic_visbuffer.override_stage(stage),
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
                .atomic_visbuffer = info.atomic_visbuffer,
                .overdraw_image = info.overdraw_image,
                .vis_image = info.vis_image,
                .depth_image = info.depth_image,
            },
            .render_context = info.render_context,
            .pass = pass,
            .clear_render_targets = !atomic_visbuf_clear && info.clear_render_targets,
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
                .first_pass_meshlets_bitfield_arena = info.first_pass_meshlets_bitfield_arena.override_stage(stage),
                .meshlet_instances = info.meshlet_instances.override_stage(stage),
                .mesh_instances = info.mesh_instances.override_stage(stage),
                .atomic_visbuffer = info.atomic_visbuffer.override_stage(stage),
                .overdraw_image = info.overdraw_image.override_stage(stage),
                .vis_image = info.vis_image,
                .depth_image = info.depth_image,
            },
            .render_context = info.render_context,
            .first_pass = info.first_pass,
            .clear_render_targets = !atomic_visbuf_clear && info.clear_render_targets,
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
    daxa::TaskImageView atomic_visbuffer = {};
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

    bool const atomic_visbuf = !info.atomic_visbuffer.is_null() && info.depth_image.is_null() && info.vis_image.is_null();
    bool const atomic_visbuf_clear = atomic_visbuf && info.clear_render_targets;
    if (atomic_visbuf_clear)
    {
        info.tg.clear_image({info.atomic_visbuffer, std::array{INVALID_TRIANGLE_ID, 0u, 0u, 0u}});
    }

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

    if (info.overdraw_image != daxa::NullTaskImage)
    {
        info.tg.clear_image({info.overdraw_image, std::array{0u, 0u, 0u, 0u}});
    }

    DrawVisbufferTask draw_task = {
        .views = DrawVisbufferTask::Views{
            .globals = info.render_context->tgpu_render_data,
            .draw_commands = draw_commands_array,
            .hiz = info.hiz,
            .meshlet_instances = info.meshlet_instances,
            .atomic_visbuffer = info.atomic_visbuffer,
            .overdraw_image = info.overdraw_image,
            .vis_image = info.vis_image,
            .depth_image = info.depth_image,
        },
        .render_context = info.render_context,
        .pass = info.pass,
        .observer = info.observer,
        .clear_render_targets = !atomic_visbuf_clear && info.clear_render_targets,
    };
    info.tg.add_task(draw_task);
}
#endif // #if defined(__cplusplus)