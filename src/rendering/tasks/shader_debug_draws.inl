#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/debug.inl"

DAXA_DECL_TASK_HEAD_BEGIN(DebugDrawH)
DAXA_TH_BUFFER_PTR(VERTEX_SHADER_READ_WRITE, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image)
DAXA_TH_IMAGE_ID(FRAGMENT_SHADER_SAMPLED, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

struct DebugDrawPush
{
    DebugDrawH::AttachmentShaderBlob attachments;
    daxa::u32 draw_as_observer;
};

#define DEBUG_DRAW_CLONE_X 16
#define DEBUG_DRAW_CLONE_Y 16
struct DebugTaskDrawDebugDisplayPush
{
    daxa::ImageViewId src;
    daxa::RWTexture2DIndex<daxa_f32vec4> dst;
    daxa_u32vec2 src_size;
    daxa::u32 image_view_type;
    daxa_i32 format;
    daxa::f32 float_min;
    daxa::f32 float_max;
    daxa::i32 int_min;
    daxa::i32 int_max;
    daxa::u32 uint_min;
    daxa::u32 uint_max;
    daxa::i32 rainbow_ints;
    daxa_i32vec4 enabled_channels;
    daxa_i32vec2 mouse_over_index;
    daxa_BufferPtr(daxa_f32vec4) readback_ptr;
    daxa_u32 readback_index;
};

#if defined(__cplusplus)

#include "../scene_renderer_context.hpp"
#include "../../daxa_helper.hpp"

static constexpr inline char const DRAW_SHADER_DEBUG_PATH[] = "./src/rendering/tasks/shader_debug_draws.hlsl";
inline daxa::RasterPipelineCompileInfo draw_shader_debug_common_pipeline_compile_info()
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    // ret.depth_test = {
    //     .depth_attachment_format = daxa::Format::D32_SFLOAT,
    //     .enable_depth_write = false,
    //     .depth_test_compare_op = daxa::CompareOp::GREATER,
    //     .min_depth_bounds = 0.0f,
    //     .max_depth_bounds = 1.0f,
    // };
    ret.color_attachments = std::vector{
        daxa::RenderAttachment{
            .format = daxa::Format::R8G8B8A8_UNORM,
        },
    };
    ret.raster = {
        .primitive_topology = daxa::PrimitiveTopology::LINE_STRIP,
        .primitive_restart_enable = {},
        .polygon_mode = daxa::PolygonMode::FILL,
        .face_culling = daxa::FaceCullFlagBits::NONE,
        .front_face_winding = daxa::FrontFaceWinding::CLOCKWISE,
        .depth_clamp_enable = {},
        .rasterizer_discard_enable = {},
        .depth_bias_enable = {},
        .depth_bias_constant_factor = 0.0f,
        .depth_bias_clamp = 0.0f,
        .depth_bias_slope_factor = 0.0f,
        .line_width = 1.7f,
    };
    return ret;
}

inline daxa::RasterPipelineCompileInfo draw_shader_debug_circles_pipeline_compile_info()
{
    auto ret = draw_shader_debug_common_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {
            .entry_point = "entry_fragment",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {
            .entry_point = "entry_vertex_circle",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "DrawShaderDebugCircles";
    ret.push_constant_size = sizeof(DebugDrawPush);
    return ret;
};

inline daxa::RasterPipelineCompileInfo draw_shader_debug_rectangles_pipeline_compile_info()
{
    auto ret = draw_shader_debug_common_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {
            .entry_point = "entry_fragment",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {
            .entry_point = "entry_vertex_rectangle",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "DrawShaderDebugRectangles";
    ret.push_constant_size = sizeof(DebugDrawPush);
    return ret;
};

inline daxa::RasterPipelineCompileInfo draw_shader_debug_aabb_pipeline_compile_info()
{
    auto ret = draw_shader_debug_common_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {
            .entry_point = "entry_fragment",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},        
        .compile_options = {
            .entry_point = "entry_vertex_aabb",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "DrawShaderDebugAABB";
    ret.push_constant_size = sizeof(DebugDrawPush);
    ret.raster.primitive_topology = daxa::PrimitiveTopology::LINE_LIST;
    return ret;
};

inline daxa::RasterPipelineCompileInfo draw_shader_debug_box_pipeline_compile_info()
{
    auto ret = draw_shader_debug_common_pipeline_compile_info();
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {
            .entry_point = "entry_fragment",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .compile_options = {
            .entry_point = "entry_vertex_box",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "DrawShaderDebugBox";
    ret.push_constant_size = sizeof(DebugDrawPush);
    ret.raster.primitive_topology = daxa::PrimitiveTopology::LINE_LIST;
    return ret;
};

struct DebugDrawTask : DebugDrawH::Task
{
    AttachmentViews views = {};
    RenderContext * rctx = {};
    void callback(daxa::TaskInterface ti)
    {
        auto const colorImageSize = ti.device.image_info(ti.get(AT.color_image).ids[0]).value().size;
        daxa::RenderPassBeginInfo render_pass_begin_info{
            // .depth_attachment =
            //     daxa::RenderAttachmentInfo{
            //         .image_view = ti.get(AT.depth_image).view_ids[0],
            //         .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
            //         .load_op = daxa::AttachmentLoadOp::LOAD,
            //         .store_op = daxa::AttachmentStoreOp::STORE,
            //         .clear_value = daxa::DepthValue{0.0f, 0},
            //     },
            .render_area = daxa::Rect2D{.width = colorImageSize.x, .height = colorImageSize.y},
        };
        render_pass_begin_info.color_attachments = {
            daxa::RenderAttachmentInfo{
                .image_view = ti.get(AT.color_image).view_ids[0],
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = daxa::AttachmentLoadOp::LOAD,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::ClearValue{std::array<u32, 4>{0, 0, 0, 0}},
            },
        };
        auto render_cmd = std::move(ti.recorder).begin_renderpass(render_pass_begin_info);

        render_cmd.set_pipeline(*rctx->gpuctx->raster_pipelines.at(draw_shader_debug_circles_pipeline_compile_info().name));

        DebugDrawPush push{
            .attachments = ti.attachment_shader_blob,
            .draw_as_observer = rctx->render_data.settings.draw_from_observer,
        };
        render_cmd.push_constant(push);

        render_cmd.draw_indirect({
            .draw_command_buffer = rctx->gpuctx->shader_debug_context.buffer,
            .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, circle_draw_indirect_info),
            .draw_count = 1,
            .draw_command_stride = sizeof(DrawIndirectStruct),
            .is_indexed = false,
        });
        render_cmd.set_pipeline(*rctx->gpuctx->raster_pipelines.at(draw_shader_debug_rectangles_pipeline_compile_info().name));
        render_cmd.draw_indirect({
            .draw_command_buffer = rctx->gpuctx->shader_debug_context.buffer,
            .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, rectangle_draw_indirect_info),
            .draw_count = 1,
            .draw_command_stride = sizeof(DrawIndirectStruct),
            .is_indexed = false,
        });
        render_cmd.set_pipeline(*rctx->gpuctx->raster_pipelines.at(draw_shader_debug_aabb_pipeline_compile_info().name));
        render_cmd.draw_indirect({
            .draw_command_buffer = rctx->gpuctx->shader_debug_context.buffer,
            .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, aabb_draw_indirect_info),
            .draw_count = 1,
            .draw_command_stride = sizeof(DrawIndirectStruct),
            .is_indexed = false,
        });
        render_cmd.set_pipeline(*rctx->gpuctx->raster_pipelines.at(draw_shader_debug_box_pipeline_compile_info().name));
        render_cmd.draw_indirect({
            .draw_command_buffer = rctx->gpuctx->shader_debug_context.buffer,
            .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, box_draw_indirect_info),
            .draw_count = 1,
            .draw_command_stride = sizeof(DrawIndirectStruct),
            .is_indexed = false,
        });

        ti.recorder = std::move(render_cmd).end_renderpass();
    }
};

inline auto debug_task_draw_display_image_pipeline_info()
{
    return daxa::ComputePipelineCompileInfo2
    {
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .entry_point = "entry_draw_debug_display",
        .language = daxa::ShaderLanguage::SLANG,
        .push_constant_size = sizeof(DebugTaskDrawDebugDisplayPush),
        .name = "debug_task_pipeline",
    };
}


void debug_task(daxa::TaskInterface ti, TgDebugContext & tg_debug, daxa::ComputePipeline& pipeline, bool pre_task)
{
    if (pre_task)
    {
        std::string task_name = std::string(ti.task_name);
        usize name_duplication = tg_debug.this_frame_duplicate_task_name_counter[task_name]++;
        if (name_duplication > 0)
        {
            task_name = task_name + " (" + std::to_string(name_duplication) + ")";
        }
        tg_debug.this_frame_task_attachments.push_back(TgDebugContext::TgDebugTask{.task_name = task_name});
    }
    usize this_frame_task_index = tg_debug.this_frame_task_attachments.size() - 1ull;
    for (u32 i = 0; i < ti.attachment_infos.size(); ++i)
    {
        if (ti.attachment_infos[i].type != daxa::TaskAttachmentType::IMAGE)
            continue;
        daxa::TaskImageAttachmentIndex src = {i};
        auto& attach_info = ti.get(src);

        std::string key = std::string(ti.task_name) + "::AT." + ti.attachment_infos[i].name();
        if (pre_task)
        {
            tg_debug.this_frame_task_attachments[this_frame_task_index].attachments.push_back(ti.attachment_infos[i]);
        }

        if (!tg_debug.inspector_states.contains(key))
            continue;
        
        auto& state = tg_debug.inspector_states.at(key);
        if (!state.active)
            continue;

        if (state.pre_task != pre_task)
            continue;
        
        // Destroy Stale images from last frame.
        if (!state.stale_image.is_empty())
        {
            ti.device.destroy_image(state.stale_image);
            state.stale_image = {};
        }   
        if (!state.stale_image1.is_empty())
        {
            ti.device.destroy_image(state.stale_image1);
            state.stale_image1 = {};
        }
        if (state.readback_buffer.is_empty())
        {
            state.readback_buffer = ti.device.create_buffer({
                .size = sizeof(daxa_f32vec4) * 2 /*raw,color*/ * 4 /*frames in flight*/,
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = std::string("readback buffer for ") + key,
            });
        }

        daxa::ImageId src_id = ti.get(src).ids[0];          // either src image id or frozen image id
        daxa::ImageInfo src_info = ti.info(src).value();    // either src image info ir frozen image info
        tido::ScalarKind scalar_kind = tido::scalar_kind_of_format(src_info.format);
        if (state.freeze_image)
        {
            bool const freeze_image_this_frame = state.frozen_image.is_empty() && state.freeze_image;
            if (freeze_image_this_frame)
            {
                daxa::ImageInfo image_frozen_info = src_info;
                image_frozen_info.usage |= daxa::ImageUsageFlagBits::TRANSFER_DST;
                image_frozen_info.name = std::string(src_info.name.data()) + " frozen copy";
                state.frozen_image = ti.device.create_image(image_frozen_info);

                daxa::ClearValue frozen_copy_clear = {};
                if (tido::is_format_depth_stencil(src_info.format))
                {
                    frozen_copy_clear = daxa::DepthValue{ .depth = 0.0f, .stencil = 0u };
                }
                else
                {
                    switch(scalar_kind)
                    {
                        case tido::ScalarKind::FLOAT: frozen_copy_clear = std::array{1.0f,0.0f,1.0f,1.0f}; break;
                        case tido::ScalarKind::INT: frozen_copy_clear = std::array{1,0,0,1}; break;
                        case tido::ScalarKind::UINT: frozen_copy_clear = std::array{1u, 0u, 0u, 1u}; break;
                    }
                }

                daxa::ImageMipArraySlice slice = {
                    .level_count = src_info.mip_level_count,
                    .layer_count = src_info.array_layer_count,
                };

                ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                    .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .dst_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .image_slice = slice,
                    .image_id = state.frozen_image,
                });

                ti.recorder.clear_image({
                    .clear_value = std::array{1.0f,0.0f,1.0f,1.0f}, 
                    .dst_image = state.frozen_image,
                    .dst_slice = slice,
                });

                ti.recorder.pipeline_barrier(daxa::MemoryBarrierInfo{
                    .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                });

                ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                    .src_access = ti.get(src).access,
                    .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .src_layout = ti.get(src).layout,
                    .dst_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    .image_slice = ti.get(src).view.slice,
                    .image_id = src_id,
                });

                // Copy src image data to frozen image.
                for (u32 mip = slice.base_mip_level; mip < (slice.base_mip_level + slice.level_count); ++mip)
                {
                    ti.recorder.copy_image_to_image({
                        .src_image = src_id,
                        .dst_image = state.frozen_image,
                        .src_slice = daxa::ImageArraySlice::slice(slice, mip),
                        .dst_slice = daxa::ImageArraySlice::slice(slice, mip),
                        .extent = {
                            std::max(1u, src_info.size.x >> mip),
                            std::max(1u, src_info.size.y >> mip),
                            std::max(1u, src_info.size.z >> mip)
                        },
                    });
                }

                ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                    .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .dst_access = ti.get(src).access,
                    .src_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    .dst_layout = ti.get(src).layout,
                    .image_slice = ti.get(src).view.slice,
                    .image_id = src_id,
                });
                ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                    .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .dst_access = daxa::AccessConsts::COMPUTE_SHADER_READ,
                    .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .dst_layout = daxa::ImageLayout::READ_ONLY_OPTIMAL,
                    .image_slice = slice,
                    .image_id = state.frozen_image,
                });
            }
            // Frozen image overwrites.
            src_id = state.frozen_image;
            src_info = ti.device.image_info(state.frozen_image).value();
        }
        else 
        {
            // If not frozen, copy over new data for ui.
            state.attachment_info = ti.get(src);
            state.runtime_image_info = src_info;
            // Mark frozen copy for deletion in next frame.
            state.stale_image1 = state.frozen_image;
            state.frozen_image = {};
        }

        if (src_id.is_empty())
        {
            return;
        }

        state.slice_valid = state.attachment_info.view.slice.contains(daxa::ImageMipArraySlice{
            .base_mip_level = state.mip,
            .base_array_layer = state.layer,
        });

        daxa::ImageInfo display_image_info = {};
        display_image_info.dimensions = 2u;
        display_image_info.size.x = std::max(1u, src_info.size.x >> state.mip);
        display_image_info.size.y = std::max(1u, src_info.size.y >> state.mip);
        display_image_info.size.z = 1u;
        display_image_info.mip_level_count = 1u;
        display_image_info.array_layer_count = 1u;
        display_image_info.sample_count = 1u;
        display_image_info.format = daxa::Format::R16G16B16A16_SFLOAT;
        display_image_info.name = "tg image debug clone";
        display_image_info.usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST;
        
        if (!state.display_image.is_empty())
        {
            auto current_clone_size = ti.device.image_info(state.display_image).value().size;
            if (display_image_info.size.x != current_clone_size.x || display_image_info.size.y != current_clone_size.y)
            {
                state.stale_image = state.display_image;
                state.display_image = ti.device.create_image(display_image_info);
            }
        }
        else
        {
            state.display_image = ti.device.create_image(display_image_info);
        }

        ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
            .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .image_id = state.display_image,
        });

        ti.recorder.clear_image({
            .clear_value = std::array{1.0f,0.0f,1.0f,1.0f}, 
            .dst_image = state.display_image,
        });

        if (state.slice_valid)
        {
            ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                .dst_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE,
                .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                .dst_layout = daxa::ImageLayout::GENERAL,
                .image_slice = ti.device.image_view_info(state.display_image.default_view()).value().slice,
                .image_id = state.display_image,
            });
            if (!state.freeze_image)
            {
                ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                    .src_access = ti.get(src).access,
                    .dst_access = daxa::AccessConsts::COMPUTE_SHADER_READ,
                    .src_layout = ti.get(src).layout,
                    .dst_layout = daxa::ImageLayout::READ_ONLY_OPTIMAL,
                    .image_slice = ti.get(src).view.slice,
                    .image_id = src_id,
                });
            }

            ti.recorder.set_pipeline(pipeline);

            daxa::ImageViewInfo src_image_view_info = ti.device.image_view_info(src_id.default_view()).value();
            src_image_view_info.slice.base_mip_level = state.mip;
            src_image_view_info.slice.base_array_layer = state.layer;
            daxa::ImageViewId src_view = ti.device.create_image_view(src_image_view_info);
            ti.recorder.destroy_image_view_deferred(src_view);
            ti.recorder.push_constant(DebugTaskDrawDebugDisplayPush{
                .src = src_view,
                .dst = state.display_image.default_view(),
                .src_size = { display_image_info.size.x, display_image_info.size.y },
                .image_view_type = static_cast<u32>(src_image_view_info.type),
                .format = static_cast<i32>(scalar_kind),
                .float_min = static_cast<f32>(state.min_v),
                .float_max = static_cast<f32>(state.max_v),
                .int_min = static_cast<i32>(state.min_v),
                .int_max = static_cast<i32>(state.max_v),
                .uint_min = static_cast<u32>(state.min_v),
                .uint_max = static_cast<u32>(state.max_v),
                .rainbow_ints = state.rainbow_ints,
                .enabled_channels = state.enabled_channels,
                .mouse_over_index = state.mouse_pos_relative_to_image,
                .readback_ptr = ti.device.device_address(state.readback_buffer).value(),
                .readback_index = tg_debug.readback_index,
            }); 
            ti.recorder.dispatch({
                round_up_div(display_image_info.size.x, DEBUG_DRAW_CLONE_X),
                round_up_div(display_image_info.size.y, DEBUG_DRAW_CLONE_Y),
                1,
            });
            if (!state.freeze_image)
            {
                ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                    .src_access = daxa::AccessConsts::COMPUTE_SHADER_READ,
                    .dst_access = ti.get(src).access,
                    .src_layout = daxa::ImageLayout::READ_ONLY_OPTIMAL,
                    .dst_layout = ti.get(src).layout,
                    .image_slice = ti.get(src).view.slice,
                    .image_id = src_id,
                });
            }
            ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                .src_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE,
                .dst_access = daxa::AccessConsts::FRAGMENT_SHADER_READ,
                .src_layout = daxa::ImageLayout::GENERAL,
                .dst_layout = daxa::ImageLayout::READ_ONLY_OPTIMAL,
                .image_id = state.display_image,
            });
        }
        else // ui image slice NOT valid.
        {
            ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                .dst_access = daxa::AccessConsts::FRAGMENT_SHADER_READ,
                .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                .dst_layout = daxa::ImageLayout::READ_ONLY_OPTIMAL,
                .image_id = state.display_image,
            });
        }
    }
}

#endif // #if __cplusplus