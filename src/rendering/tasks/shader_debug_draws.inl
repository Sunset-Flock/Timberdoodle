#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/debug.inl"

DAXA_DECL_RASTER_TASK_HEAD_BEGIN(DebugDrawH)
DAXA_TH_BUFFER_PTR(VERTEX_SHADER_READ_WRITE, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_DECL_TASK_HEAD_END

#define DEBUG_OBJECT_DRAW_MODE_LINE 0
#define DEBUG_OBJECT_DRAW_MODE_CIRCLE 1
#define DEBUG_OBJECT_DRAW_MODE_RECTANGLE 2
#define DEBUG_OBJECT_DRAW_MODE_AABB 3
#define DEBUG_OBJECT_DRAW_MODE_BOX 4
#define DEBUG_OBJECT_DRAW_MODE_CONE 5
#define DEBUG_OBJECT_DRAW_MODE_SPHERE 6

#define DEBUG_OBJECT_LINE_VERTICES 2
#define DEBUG_OBJECT_CIRCLE_VERTICES 128
#define DEBUG_OBJECT_RECTANGLE_VERTICES 8
#define DEBUG_OBJECT_AABB_VERTICES 24
#define DEBUG_OBJECT_BOX_VERTICES 24
#define DEBUG_OBJECT_CONE_VERTICES 64
#define DEBUG_OBJECT_SPHERE_VERTICES (5*64)

struct DebugDrawPush
{
    DebugDrawH::AttachmentShaderBlob attachments;
    daxa::u32 draw_as_observer;
    daxa::u32 mode;
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

inline daxa::RasterPipelineCompileInfo draw_shader_debug_lines_pipeline_compile_info()
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.depth_test = {
        .depth_attachment_format = daxa::Format::D32_SFLOAT,
        .enable_depth_write = false,
        .depth_test_compare_op = daxa::CompareOp::GREATER,
        .min_depth_bounds = 0.0f,
        .max_depth_bounds = 1.0f,
    };
    ret.color_attachments = std::vector{
        daxa::RenderAttachment{
            .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
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
            .entry_point = "entry_vertex_line",
            .language = daxa::ShaderLanguage::SLANG,
        },
    };
    ret.name = "DrawShaderDebugLines";
    ret.push_constant_size = sizeof(DebugDrawPush);
    ret.raster.primitive_topology = daxa::PrimitiveTopology::LINE_LIST;
    return ret;
};

struct DebugDrawTask : DebugDrawH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto const colorImageSize = ti.info(AT.color_image).value().size;
        daxa::RenderPassBeginInfo render_pass_begin_info{
            .depth_attachment =
                daxa::RenderAttachmentInfo{
                    .image_view = ti.get(AT.depth_image).view_ids[0],
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = daxa::AttachmentLoadOp::LOAD,
                    .store_op = daxa::AttachmentStoreOp::STORE,
                    .clear_value = daxa::DepthValue{0.0f, 0},
                },
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

        DebugDrawPush push{
            .attachments = ti.attachment_shader_blob,
            .draw_as_observer = render_context->render_data.settings.draw_from_observer,
        };

        render_cmd.set_pipeline(*render_context->gpu_context->raster_pipelines.at(draw_shader_debug_lines_pipeline_compile_info().name));
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_LINE;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, line_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_CIRCLE;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, circle_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_RECTANGLE;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, rectangle_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_AABB;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, aabb_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_BOX;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, box_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_CONE;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, cone_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }
        {
            push.mode = DEBUG_OBJECT_DRAW_MODE_SPHERE;
            render_cmd.push_constant(push);
            render_cmd.draw_indirect({
                .draw_command_buffer = render_context->gpu_context->shader_debug_context.buffer,
                .indirect_buffer_offset = offsetof(ShaderDebugBufferHead, sphere_draws),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
                .is_indexed = false,
            });
        }

        ti.recorder = std::move(render_cmd).end_renderpass();
    }
};

inline MAKE_COMPUTE_COMPILE_INFO(debug_task_draw_display_image_pipeline_info, "./src/rendering/tasks/shader_debug_draws.hlsl", "entry_draw_debug_display")

void debug_task(daxa::TaskInterface ti, TgDebugContext & tg_debug, daxa::ComputePipeline& pipeline, bool pre_task)
{
    if (pre_task)
    {
        std::string name = std::string(ti.task_name) + "(IDX:" + std::to_string(ti.task_index) + ')';
        tg_debug.this_frame_task_attachments.push_back(TgDebugContext::TgDebugTask{.task_index = ti.task_index, .task_name = name });
    }
    usize this_frame_task_index = tg_debug.this_frame_task_attachments.size() - 1ull;
    for (u32 i = 0; i < ti.attachment_infos.size(); ++i)
    {
        if (ti.attachment_infos[i].type != daxa::TaskAttachmentType::IMAGE)
            continue;
        daxa::TaskImageAttachmentIndex src = {i};
        auto& attach_info = ti.get(src);

        std::string key = std::string(ti.task_name) + "(IDX:" + std::to_string(ti.task_index) + ')' + "::AT." + ti.attachment_infos[i].name();
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

        if (ti.id(src).is_empty())
        {
            return;
        }

        // First frame this is always unfroozen, so we always initialize the image.
        if (!state.freeze_image)
        {
            daxa::ImageId src_id = ti.id(src);          // either src image id or frozen image id
            daxa::ImageInfo src_info = ti.info(src).value();    // either src image info ir frozen image info
            tido::ScalarKind scalar_kind = tido::scalar_kind_of_format(src_info.format);

            // If not frozen, copy over new data for ui.
            state.attachment_info = ti.get(src);
            state.runtime_image_info = src_info;
            
            daxa::ImageInfo raw_copy_image_info = src_info;
            if (raw_copy_image_info.format == daxa::Format::D32_SFLOAT)
            {
                raw_copy_image_info.format = daxa::Format::R32_SFLOAT;
            }
            if (raw_copy_image_info.format == daxa::Format::D16_UNORM)
            {
                raw_copy_image_info.format = daxa::Format::R16_UNORM;
            }
            raw_copy_image_info.usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_STORAGE; // STORAGE is better than SAMPLED as it supports 64bit images.
            raw_copy_image_info.name = std::string(src_info.name.data()) + " raw image copy";
            
            if (!state.raw_image_copy.is_empty())
            {
                auto const current_info = ti.device.image_info(state.raw_image_copy).value();
                auto const changed = 
                    raw_copy_image_info.size.x != current_info.size.x || 
                    raw_copy_image_info.size.y != current_info.size.y ||
                    raw_copy_image_info.mip_level_count != current_info.mip_level_count ||
                    raw_copy_image_info.array_layer_count != current_info.array_layer_count || 
                    raw_copy_image_info.format != current_info.format;
                if (changed)
                {
                    state.stale_image = state.raw_image_copy;
                    state.raw_image_copy =ti.device.create_image(raw_copy_image_info);
                }
            }
            else
            {
                state.raw_image_copy = ti.device.create_image(raw_copy_image_info);
            }
            
            daxa::ImageInfo display_image_info = {};
            display_image_info.dimensions = 2u;
            display_image_info.size.x = std::max(1u, raw_copy_image_info.size.x >> state.mip);
            display_image_info.size.y = std::max(1u, raw_copy_image_info.size.y >> state.mip);
            display_image_info.size.z = 1u;
            display_image_info.mip_level_count = 1u;
            display_image_info.array_layer_count = 1u;
            display_image_info.sample_count = 1u;
            display_image_info.format = daxa::Format::R16G16B16A16_SFLOAT;
            display_image_info.name = "tg image debug clone";
            display_image_info.usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST;

            if (!state.display_image.is_empty())
            {
                auto const current_size = ti.device.image_info(state.display_image).value().size;
                auto const size_changed = (display_image_info.size.x != current_size.x || display_image_info.size.y != current_size.y);
                if (size_changed)
                {
                    state.stale_image1 = state.display_image;
                    state.display_image = ti.device.create_image(display_image_info);
                }
            }
            else
            {
                state.display_image = ti.device.create_image(display_image_info);
            }

            // CLear before copy. Magenta marks mips/array layers that are not within the image slice of this attachment!
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
                .image_id = state.raw_image_copy,
            });

            ti.recorder.clear_image({
                .clear_value = std::array{1.0f,0.0f,1.0f,1.0f}, 
                .dst_image = state.raw_image_copy,
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
                    .dst_image = state.raw_image_copy,
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
                .dst_layout = daxa::ImageLayout::GENERAL,               // STORAGE always uses general in daxa
                .image_slice = slice,
                .image_id = state.raw_image_copy,
            });
        }
        auto const raw_image_copy_info = ti.device.info(state.raw_image_copy).value();
        auto const raw_image_copy = state.raw_image_copy;
        auto const display_image_info = ti.device.info(state.display_image).value();
        auto const scalar_kind = tido::scalar_kind_of_format(raw_image_copy_info.format);

        state.slice_valid = state.attachment_info.view.slice.contains(daxa::ImageMipArraySlice{
            .base_mip_level = state.mip,
            .base_array_layer = state.layer,
        });

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

            ti.recorder.set_pipeline(pipeline);

            daxa::ImageViewInfo src_image_view_info = ti.device.image_view_info(raw_image_copy.default_view()).value();
            src_image_view_info.slice.level_count = 1;
            src_image_view_info.slice.layer_count = 1;
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
                .mouse_over_index = {
                    state.mouse_pos_relative_to_image_mip0.x >> state.mip,
                    state.mouse_pos_relative_to_image_mip0.y >> state.mip,
                },
                .readback_ptr = ti.device.device_address(state.readback_buffer).value(),
                .readback_index = tg_debug.readback_index,
            }); 
            ti.recorder.dispatch({
                round_up_div(display_image_info.size.x, DEBUG_DRAW_CLONE_X),
                round_up_div(display_image_info.size.y, DEBUG_DRAW_CLONE_Y),
                1,
            });
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