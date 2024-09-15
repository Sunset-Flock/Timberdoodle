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

enum DrawDebugClone_Format
{
    DrawDebugClone_Format_FLOAT,
    DrawDebugClone_Format_INT,
    DrawDebugClone_Format_UINT,
};

#define DEBUG_DRAW_CLONE_X 16
#define DEBUG_DRAW_CLONE_Y 16
struct DrawDebugClonePush
{
    daxa_BufferPtr(RenderGlobalData) globals;
    daxa::ImageViewId src;
    daxa::RWTexture2DIndex<daxa_f32vec4> dst;
    daxa_u32vec2 src_size;
    daxa::u32 image_view_type;
    DrawDebugClone_Format format;
    daxa::f32 float_min;
    daxa::f32 float_max;
    daxa::i32 int_min;
    daxa::i32 int_max;
    daxa::u32 uint_min;
    daxa::u32 uint_max;
    daxa::i32 rainbow_ints;
    daxa_i32vec4 enabled_channels;
};

#if defined(__cplusplus)

#include "../scene_renderer_context.hpp"

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

inline auto draw_debug_clone_pipeline_info()
{
    return daxa::ComputePipelineCompileInfo2
    {
        .source = daxa::ShaderFile{DRAW_SHADER_DEBUG_PATH},
        .entry_point = "entry_draw_debug_clone",
        .language = daxa::ShaderLanguage::SLANG,
        .push_constant_size = sizeof(DrawDebugClonePush),
        .name = "draw_debug_clone",
    };
}


void draw_debug_clone(daxa::TaskInterface ti, daxa::TaskBufferAttachmentIndex globals, RenderContext * rctx)
{
    for (u32 i = 0; i < ti.attachment_infos.size(); ++i)
    {
        if (ti.attachment_infos[i].type != daxa::TaskAttachmentType::IMAGE)
            continue;
        daxa::TaskImageAttachmentIndex src = {i};
        auto& attach_info = ti.get(src);

        std::string key = std::string(ti.task_name) + " + " + ti.attachment_infos[i].name();
        rctx->tg_debug.this_frame_task_attachments[std::string(ti.task_name)].emplace(std::string(ti.attachment_infos[i].name()));

        if (!rctx->tg_debug.inspector_states.contains(key))
            continue;
        
        auto& state = rctx->tg_debug.inspector_states.at(key);
        if (!state.active)
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

        daxa::ImageId src_id = ti.get(src).ids[0];          // either src image id or frozen image id
        daxa::ImageInfo src_info = ti.info(src).value();    // either src image info ir frozen image info
        if (state.freeze_image)
        {
            bool const freeze_image_this_frame = state.frozen_image.is_empty() && state.freeze_image;
            if (freeze_image_this_frame)
            {
                daxa::ImageInfo image_frozen_info = src_info;
                image_frozen_info.usage |= daxa::ImageUsageFlagBits::TRANSFER_DST;
                image_frozen_info.name = std::string(src_info.name.data()) + " frozen copy";
                state.frozen_image = ti.device.create_image(image_frozen_info);

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

            ti.recorder.set_pipeline(*rctx->gpuctx->compute_pipelines.at(std::string("draw_debug_clone")));

            DrawDebugClone_Format format = {};
            switch (src_info.format)
            {
                case daxa::Format::UNDEFINED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R4G4_UNORM_PACK8: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R4G4B4A4_UNORM_PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B4G4R4A4_UNORM_PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R5G6B5_UNORM_PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B5G6R5_UNORM_PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R5G5B5A1_UNORM_PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B5G5R5A1_UNORM_PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A1R5G5B5_UNORM_PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8_SNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8_USCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8_SSCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R8_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R8_SRGB: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8_SNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8_USCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8_SSCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R8G8_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R8G8_SRGB: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8B8_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8B8_SNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8B8_USCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8B8_SSCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8B8_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R8G8B8_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R8G8B8_SRGB: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B8G8R8_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B8G8R8_SNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B8G8R8_USCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B8G8R8_SSCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B8G8R8_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::B8G8R8_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::B8G8R8_SRGB: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8B8A8_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8B8A8_SNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8B8A8_USCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8B8A8_SSCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R8G8B8A8_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R8G8B8A8_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R8G8B8A8_SRGB: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B8G8R8A8_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B8G8R8A8_SNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B8G8R8A8_USCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B8G8R8A8_SSCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B8G8R8A8_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::B8G8R8A8_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::B8G8R8A8_SRGB: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A8B8G8R8_UNORM_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A8B8G8R8_SNORM_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A8B8G8R8_USCALED_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A8B8G8R8_SSCALED_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A8B8G8R8_UINT_PACK32: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::A8B8G8R8_SINT_PACK32: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::A8B8G8R8_SRGB_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A2R10G10B10_UNORM_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A2R10G10B10_SNORM_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A2R10G10B10_USCALED_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A2R10G10B10_SSCALED_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A2R10G10B10_UINT_PACK32: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::A2R10G10B10_SINT_PACK32: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::A2B10G10R10_UNORM_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A2B10G10R10_SNORM_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A2B10G10R10_USCALED_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A2B10G10R10_SSCALED_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A2B10G10R10_UINT_PACK32: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::A2B10G10R10_SINT_PACK32: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R16_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16_SNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16_USCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16_SSCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R16_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R16_SFLOAT: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16_SNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16_USCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16_SSCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R16G16_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R16G16_SFLOAT: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16B16_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16B16_SNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16B16_USCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16B16_SSCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16B16_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R16G16B16_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R16G16B16_SFLOAT: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16B16A16_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16B16A16_SNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16B16A16_USCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16B16A16_SSCALED: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R16G16B16A16_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R16G16B16A16_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R16G16B16A16_SFLOAT: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R32_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R32_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R32_SFLOAT: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R32G32_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R32G32_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R32G32_SFLOAT: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R32G32B32_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R32G32B32_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R32G32B32_SFLOAT: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R32G32B32A32_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R32G32B32A32_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R32G32B32A32_SFLOAT: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R64_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R64_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R64_SFLOAT: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R64G64_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R64G64_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R64G64_SFLOAT: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R64G64B64_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R64G64B64_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R64G64B64_SFLOAT: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R64G64B64A64_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::R64G64B64A64_SINT: format = DrawDebugClone_Format_INT; break;
                case daxa::Format::R64G64B64A64_SFLOAT: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B10G11R11_UFLOAT_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::E5B9G9R9_UFLOAT_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::D16_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::X8_D24_UNORM_PACK32: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::D32_SFLOAT: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::S8_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::D16_UNORM_S8_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::D24_UNORM_S8_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::D32_SFLOAT_S8_UINT: format = DrawDebugClone_Format_UINT; break;
                case daxa::Format::BC1_RGB_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC1_RGB_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC1_RGBA_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC1_RGBA_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC2_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC2_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC3_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC3_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC4_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC4_SNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC5_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC5_SNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC6H_UFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC6H_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC7_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::BC7_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ETC2_R8G8B8_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ETC2_R8G8B8_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ETC2_R8G8B8A1_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ETC2_R8G8B8A1_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ETC2_R8G8B8A8_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ETC2_R8G8B8A8_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::EAC_R11_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::EAC_R11_SNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::EAC_R11G11_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::EAC_R11G11_SNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_4x4_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_4x4_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_5x4_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_5x4_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_5x5_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_5x5_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_6x5_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_6x5_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_6x6_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_6x6_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_8x5_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_8x5_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_8x6_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_8x6_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_8x8_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_8x8_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_10x5_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_10x5_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_10x6_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_10x6_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_10x8_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_10x8_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_10x10_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_10x10_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_12x10_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_12x10_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_12x12_UNORM_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_12x12_SRGB_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G8B8G8R8_422_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B8G8R8G8_422_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G8_B8_R8_3PLANE_420_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G8_B8R8_2PLANE_420_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G8_B8_R8_3PLANE_422_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G8_B8R8_2PLANE_422_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G8_B8_R8_3PLANE_444_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R10X6_UNORM_PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R10X6G10X6_UNORM_2PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R10X6G10X6B10X6A10X6_UNORM_4PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G10X6B10X6G10X6R10X6_422_UNORM_4PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B10X6G10X6R10X6G10X6_422_UNORM_4PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R12X4_UNORM_PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R12X4G12X4_UNORM_2PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::R12X4G12X4B12X4A12X4_UNORM_4PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G12X4B12X4G12X4R12X4_422_UNORM_4PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B12X4G12X4R12X4G12X4_422_UNORM_4PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G16B16G16R16_422_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::B16G16R16G16_422_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G16_B16_R16_3PLANE_420_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G16_B16R16_2PLANE_420_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G16_B16_R16_3PLANE_422_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G16_B16R16_2PLANE_422_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G16_B16_R16_3PLANE_444_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G8_B8R8_2PLANE_444_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::G16_B16R16_2PLANE_444_UNORM: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A4R4G4B4_UNORM_PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::A4B4G4R4_UNORM_PACK16: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_4x4_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_5x4_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_5x5_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_6x5_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_6x6_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_8x5_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_8x6_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_8x8_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_10x5_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_10x6_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_10x8_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_10x10_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_12x10_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::ASTC_12x12_SFLOAT_BLOCK: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::PVRTC1_2BPP_UNORM_BLOCK_IMG: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::PVRTC1_4BPP_UNORM_BLOCK_IMG: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::PVRTC2_2BPP_UNORM_BLOCK_IMG: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::PVRTC2_4BPP_UNORM_BLOCK_IMG: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::PVRTC1_2BPP_SRGB_BLOCK_IMG: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::PVRTC1_4BPP_SRGB_BLOCK_IMG: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::PVRTC2_2BPP_SRGB_BLOCK_IMG: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::PVRTC2_4BPP_SRGB_BLOCK_IMG: format = DrawDebugClone_Format_FLOAT; break;
                case daxa::Format::MAX_ENUM: format = DrawDebugClone_Format_FLOAT; break;
            }

            daxa::ImageViewInfo src_image_view_info = ti.device.image_view_info(src_id.default_view()).value();
            src_image_view_info.slice.base_mip_level = state.mip;
            src_image_view_info.slice.base_array_layer = state.layer;
            daxa::ImageViewId src_view = ti.device.create_image_view(src_image_view_info);
            ti.recorder.destroy_image_view_deferred(src_view);
            ti.recorder.push_constant(DrawDebugClonePush{
                .globals = ti.device_address(globals).value(),
                .src = src_view,
                .dst = state.display_image.default_view(),
                .src_size = { display_image_info.size.x, display_image_info.size.y },
                .image_view_type = static_cast<u32>(src_image_view_info.type),
                .format = format,
                .float_min = static_cast<f32>(state.min_v),
                .float_max = static_cast<f32>(state.max_v),
                .int_min = static_cast<i32>(state.min_v),
                .int_max = static_cast<i32>(state.max_v),
                .uint_min = static_cast<u32>(state.min_v),
                .uint_max = static_cast<u32>(state.max_v),
                .rainbow_ints = state.rainbow_ints,
                .enabled_channels = state.enabled_channels,
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