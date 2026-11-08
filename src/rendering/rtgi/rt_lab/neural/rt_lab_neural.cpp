#include "rt_lab_neural.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include <fmt/format.h>

// == Weights =========================================================================================================

// File: "RTLN", u32 version (1), u32 float count, u32 reserved, then the floats (neural/train/train.py save_params).
// Layout of the floats = RT_LAB_NEURAL_* offsets in rt_lab_neural_core.slang.
// Parameter count by input size: HID = 32, OUT = 16. Input size by feature mask: 192 raw 3x3, 64 window, 48 ray length 3x3, else 32.
static constexpr u32 rt_lab_neural_param_count(u32 in) { return 32 * in + 32 + 32 * 32 + 32 + 16 * 32 + 16; }
static constexpr auto rt_lab_neural_input_size(u32 feature_mask) -> u32
{
    if (feature_mask & RT_LAB_NEURAL_FEATURE_LITE)
    {
        return 40;
    }
    if (feature_mask & RT_LAB_NEURAL_FEATURE_RAW3)
    {
        return 192;
    }
    if (feature_mask & RT_LAB_NEURAL_FEATURE_WINDOW)
    {
        return 64;
    }
    return (feature_mask & RT_LAB_NEURAL_FEATURE_RAYLENGTH3) ? 48 : 32;
}

auto rt_lab_neural_coopvec_supported(daxa::Device const & device) -> bool
{
    return static_cast<bool>(device.properties().implicit_features & daxa::ImplicitFeatureFlagBits::COOPERATIVE_VECTOR);
}

// IEEE binary32 -> binary16, round to nearest even (weights are small, no inf / denormal care beyond clamping).
static auto rt_lab_neural_to_half(f32 value) -> u16
{
    u32 bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u32 const sign = (bits >> 16) & 0x8000u;
    i32 exponent = static_cast<i32>((bits >> 23) & 0xFFu) - 127 + 15;
    u32 mantissa = bits & 0x7FFFFFu;
    if (exponent >= 31)
    {
        return static_cast<u16>(sign | 0x7BFFu); // clamp to the largest finite half
    }
    if (exponent <= 0)
    {
        if (exponent < -10)
        {
            return static_cast<u16>(sign);
        }
        mantissa |= 0x800000u;
        u32 const shift = static_cast<u32>(14 - exponent);
        u32 half_mantissa = mantissa >> shift;
        u32 const remainder = mantissa & ((1u << shift) - 1u);
        u32 const halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u)))
        {
            half_mantissa += 1;
        }
        return static_cast<u16>(sign | half_mantissa);
    }
    u32 half_bits = (static_cast<u32>(exponent) << 10) | (mantissa >> 13);
    u32 const remainder = mantissa & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_bits & 1u)))
    {
        half_bits += 1; // may carry into the exponent, which is the correct rounding
    }
    return static_cast<u16>(sign | half_bits);
}

static void rt_lab_neural_destroy_weights(RenderContext & render_context)
{
    auto & state = render_context.rt_lab.neural;
    if (!state.weights.id().is_empty())
    {
        render_context.gpu_context->device.destroy_buffer(state.weights.id());
        state.weights.set_buffer(daxa::BufferId{});
    }
    if (!state.weights_f16.id().is_empty())
    {
        render_context.gpu_context->device.destroy_buffer(state.weights_f16.id());
        state.weights_f16.set_buffer(daxa::BufferId{});
    }
    state.weights_loaded = false;
    state.loaded_path = std::filesystem::path{};
}

auto rt_lab_neural_ensure_weights(RenderContext & render_context) -> bool
{
    auto & state = render_context.rt_lab.neural;
    if (state.weights_loaded && state.loaded_path == state.weights_path)
    {
        return true;
    }
    std::ifstream file(state.weights_path, std::ios::binary);
    char header[16] = {};
    if (!file || !file.read(header, sizeof(header)) || std::memcmp(header, "RTLN", 4) != 0)
    {
        fmt::print("[RtLabNeural] cannot read weights \"{}\" (train one with neural/train/train.py)\n", state.weights_path.string());
        state.loaded_path = state.weights_path; // do not retry (and print) every frame
        state.weights_loaded = false;
        return false;
    }
    u32 version = 0;
    u32 count = 0;
    u32 feature_mask = 0;
    std::memcpy(&version, header + 4, 4);
    std::memcpy(&count, header + 8, 4);
    std::memcpy(&feature_mask, header + 12, 4);
    if (feature_mask & (RT_LAB_NEURAL_FEATURE_V2 | RT_LAB_NEURAL_FEATURE_W2))
    {
        fmt::print("[RtLabNeural] weights \"{}\" use the SH feature set (v2: 3x3 footprint, shared pixel encoder) or the variance guide style weighting: the engine does not run those networks yet (the trainer does)\n", state.weights_path.string());
        state.loaded_path = state.weights_path;
        state.weights_loaded = false;
        return false;
    }
    if (feature_mask == 0 && count == rt_lab_neural_param_count(64))
    {
        feature_mask = RT_LAB_NEURAL_FEATURE_WINDOW; // old files (reserved word 0): the 5x5 window network is recognized by its size
    }
    if (version != 1 || count != rt_lab_neural_param_count(rt_lab_neural_input_size(feature_mask)))
    {
        fmt::print("[RtLabNeural] weights \"{}\": version {} with {} parameters and feature mask {}, expected version 1 and {} parameters for that mask\n", state.weights_path.string(), version,
            count, feature_mask, rt_lab_neural_param_count(rt_lab_neural_input_size(feature_mask)));
        state.loaded_path = state.weights_path;
        state.weights_loaded = false;
        return false;
    }
    if (feature_mask & RT_LAB_NEURAL_FEATURE_RAW3)
    {
        fmt::print("[RtLabNeural] weights \"{}\" use the raw 3x3 feature set: the engine does not provide its inputs (albedo, per pixel ray length accumulation of the 3x3 window); the lite feature set runs\n",
            state.weights_path.string());
        state.loaded_path = state.weights_path;
        state.weights_loaded = false;
        return false;
    }
    std::vector<f32> values(count);
    if (!file.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(f32))))
    {
        fmt::print("[RtLabNeural] weights \"{}\" are truncated\n", state.weights_path.string());
        state.loaded_path = state.weights_path;
        state.weights_loaded = false;
        return false;
    }
    auto & device = render_context.gpu_context->device;
    // Replacing the buffer invalidates the task graph that references it: wait, the caller rebuilds on the shape change.
    device.wait_idle();
    rt_lab_neural_destroy_weights(render_context);
    state.weights.set_buffer(device.create_buffer({
        .size = count * sizeof(f32),
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "rt_lab_neural_weights",
    }));
    std::memcpy(device.buffer_host_address(state.weights.id()).value(), values.data(), count * sizeof(f32));
    // The same weights as half for the cooperative vector MLP (row major matrices + biases at the same offsets).
    state.weights_f16.set_buffer(device.create_buffer({
        .size = count * sizeof(u16),
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "rt_lab_neural_weights_f16",
    }));
    u16 * const halves = reinterpret_cast<u16 *>(device.buffer_host_address(state.weights_f16.id()).value());
    for (u32 i = 0; i < count; ++i)
    {
        halves[i] = rt_lab_neural_to_half(values[i]);
    }
    state.weights_loaded = true;
    state.loaded_path = state.weights_path;
    state.param_count = count;
    state.feature_mask = feature_mask;
    state.recurrent = (feature_mask & RT_LAB_NEURAL_FEATURE_RECURRENT) != 0; // trained with its previous latent: it runs recurrent
    fmt::print("[RtLabNeural] loaded {} weights from \"{}\" (feature mask {}, {})\n", count, state.weights_path.string(), feature_mask, state.recurrent ? "recurrent" : "stateless");
    return true;
}

auto rt_lab_neural_active(RtLabSettings const & settings, RtLabNeuralState const & state) -> bool
{
    return settings.neural_enabled != 0 && state.weights_loaded;
}

// == Dump ============================================================================================================

namespace
{
    struct DumpFileHeader
    {
        char magic[8] = {'R', 'T', 'L', 'N', 'D', 'U', 'M', 'P'};
        u32 version = 3; // 2: geometry block has a third uint4 (detail normal, albedo), frame blocks carry per pixel ray statistics; 3: 4 uint4 per pixel and frame (+ SH-Y, CoCg, squared moments)
        u32 crop = 0;
        u32 frames = 0;
        u32 first_frame = 0;
        u32 lab_width = 0;
        u32 lab_height = 0;
        u32 crop_x = 0;
        u32 crop_y = 0;
    };
} // namespace

// uint4 count of a dump with `frames` frame blocks of a crop x crop crop.
static auto rt_lab_neural_dump_uint4_count(u32 crop, u32 frames) -> u64
{
    u64 const pixels = static_cast<u64>(crop) * crop;
    return pixels * 3 + static_cast<u64>(frames) * (1 + pixels * 4);
}

static void rt_lab_neural_destroy_dump(RenderContext & render_context)
{
    auto & dump = render_context.rt_lab.neural.dump;
    if (!dump.buffer.id().is_empty())
    {
        render_context.gpu_context->device.destroy_buffer(dump.buffer.id());
        dump.buffer.set_buffer(daxa::BufferId{});
    }
    dump.allocated_blocks = 0;
    dump.allocated_crop = 0;
}

static void rt_lab_neural_flush_dump(RenderContext & render_context)
{
    auto & dump = render_context.rt_lab.neural.dump;
    auto & device = render_context.gpu_context->device;
    device.wait_idle();
    DumpFileHeader header = {};
    header.crop = dump.actual_crop;
    header.frames = static_cast<u32>(dump.written_frames);
    header.first_frame = static_cast<u32>(dump.first_frame);
    header.lab_width = dump.lab_width;
    header.lab_height = dump.lab_height;
    header.crop_x = dump.actual_origin_x;
    header.crop_y = dump.actual_origin_y;
    u64 const bytes = rt_lab_neural_dump_uint4_count(dump.actual_crop, header.frames) * 16;
    std::filesystem::create_directories(dump.path.parent_path());
    std::ofstream file(dump.path, std::ios::binary);
    file.write(reinterpret_cast<char const *>(&header), sizeof(header));
    file.write(reinterpret_cast<char const *>(device.buffer_host_address(dump.buffer.id()).value()), static_cast<std::streamsize>(bytes));
    file.close();
    fmt::print("[RtLabNeural] dump: {} frames of {}x{} at ({}, {}) -> \"{}\" ({:.1f} MB)\n", header.frames, header.crop, header.crop, header.crop_x, header.crop_y,
        dump.path.string(), static_cast<f64>(bytes + sizeof(header)) / (1024.0 * 1024.0));
    dump.flushed = true;
    dump.enabled = false; // one shot: the graph drops the task on the next rebuild
}

// == Frame update ====================================================================================================

auto rt_lab_neural_frame_update(RenderContext & render_context) -> bool
{
    auto & state = render_context.rt_lab.neural;
    auto const & settings = render_context.render_data.rt_lab_settings; // the applied preset
    if (settings.neural_enabled != 0)
    {
        rt_lab_neural_ensure_weights(render_context);
    }
    auto & dump = state.dump;
    if (dump.enabled)
    {
        u32 const crop = static_cast<u32>(std::max(dump.crop_size, 16));
        u32 const blocks = 1u + static_cast<u32>(std::max(dump.frames, 1));
        if (dump.buffer.id().is_empty() || dump.allocated_blocks < blocks || dump.allocated_crop < crop)
        {
            render_context.gpu_context->device.wait_idle();
            rt_lab_neural_destroy_dump(render_context);
            dump.buffer.set_buffer(render_context.gpu_context->device.create_buffer({
                .size = rt_lab_neural_dump_uint4_count(crop, static_cast<u32>(std::max(dump.frames, 1))) * 16,
                .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "rt_lab_neural_dump",
            }));
            dump.allocated_blocks = blocks;
            dump.allocated_crop = crop;
            dump.written_frames = 0;
            dump.frames_since_done = 0;
            dump.flushed = false;
        }
        if (dump.written_frames >= dump.frames && !dump.flushed)
        {
            dump.frames_since_done += 1;
            if (dump.frames_since_done >= 4) // GPU latency: the last blocks are written by now (flush waits for idle anyway)
            {
                rt_lab_neural_flush_dump(render_context);
            }
        }
    }
    return state.built_active != rt_lab_neural_active(settings, state) || state.built_dump != state.dump.enabled;
}

void rt_lab_neural_dump_begin(RtLabNeuralState & state, std::filesystem::path const & path, i32 first_frame, i32 frames, i32 crop_x, i32 crop_y, i32 crop_size,
    std::vector<i32> frame_list)
{
    auto & dump = state.dump;
    dump.frame_list = std::move(frame_list);
    if (!dump.frame_list.empty())
    {
        frames = static_cast<i32>(dump.frame_list.size());
    }
    dump.enabled = frames > 0;
    dump.path = path;
    dump.first_frame = first_frame;
    dump.frames = frames;
    dump.crop_x = crop_x;
    dump.crop_y = crop_y;
    dump.crop_size = crop_size;
    dump.written_frames = 0;
    dump.frames_since_done = 0;
    dump.flushed = false;
}

void rt_lab_neural_record_built(RtLabSettings const & settings, RtLabNeuralState & state)
{
    state.built_active = rt_lab_neural_active(settings, state);
    state.built_dump = state.dump.enabled;
}

void rt_lab_neural_destroy(RenderContext & render_context)
{
    rt_lab_neural_destroy_weights(render_context);
    rt_lab_neural_destroy_dump(render_context);
}

// == Passes ==========================================================================================================

// push.size = lab size, one thread per pixel of dispatch_size.
template <typename Push>
static void rt_lab_neural_dispatch(daxa::TaskInterface ti, RenderContext * render_context, Push push, daxa::Extent3D const & size, daxa::Extent3D const & dispatch_size,
    u32 block_size, u32 render_time, std::string const & pipeline_name)
{
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, render_time);
    auto const pipeline = render_context->gpu_context->compute_pipelines.find(pipeline_name);
    if (pipeline == render_context->gpu_context->compute_pipelines.end())
    {
        throw std::runtime_error(fmt::format("neural pass pipeline \"{}\" is not registered (renderer.cpp feature mask list)", pipeline_name));
    }
    ti.recorder.set_pipeline(*pipeline->second);
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    push.attach = alloc.device_address;
    push.size = {size.x, size.y};
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(dispatch_size.x, block_size), round_up_div(dispatch_size.y, block_size), 1});
}

static void rt_lab_neural_ray_stats_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtLabNeuralRayStatsH::Info::AT;
    RtLabNeuralRayStatsPush push = {};
    push.reset = render_context->render_data.rt_lab_settings.reset_history != 0 ? 1u : 0u;
    auto const size = ti.info(AT.ray_capped).value().size;
    rt_lab_neural_dispatch(ti, render_context, push, size, size, RT_LAB_NEURAL_X, RenderTimes::index<"RT_LAB", "NEURAL_RAY_STATS">(), rt_lab_neural_ray_stats_compile_info().name);
}

void rt_lab_neural_add_ray_stats_task(daxa::TaskGraph & tg, RenderContext & render_context, RtLabNeuralRayStatsViews const & views)
{
    tg.add_task(daxa::HeadTask<RtLabNeuralRayStatsH::Info>()
            .head_views(RtLabNeuralRayStatsH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .lab_depth = views.lab_depth,
                .lab_face_normal = views.lab_face_normal,
                .lab_depth_history = views.lab_depth_history,
                .lab_face_normal_history = views.lab_face_normal_history,
                .accumulated = views.accumulated,
                .lab_raw_diffuse = views.lab_raw_diffuse,
                .ray_alloc = views.ray_alloc,
                .ray_lengths = views.ray_lengths,
                .ray_capped_history = views.ray_capped_history,
                .ray_capped = views.ray_capped,
            })
            .executes(rt_lab_neural_ray_stats_callback, &render_context));
}

static void rt_lab_neural_latent_callback(daxa::TaskInterface ti, RenderContext * render_context, bool ray_length_bound, bool detail_normal_bound, bool ray_capped_bound)
{
    bool const coopvec = render_context->render_data.rt_lab_settings.neural_use_coopvec != 0 && rt_lab_neural_coopvec_supported(render_context->gpu_context->device);
    auto const & AT = RtLabNeuralLatentH::Info::AT;
    RtLabNeuralLatentPush push = {};
    push.ray_length_bound = ray_length_bound ? 1u : 0u;
    push.detail_normal_bound = detail_normal_bound ? 1u : 0u;
    push.ray_capped_bound = ray_capped_bound ? 1u : 0u;
    // Stateless network (recurrent off): the previous latent is always treated as after a reset.
    push.reset = (render_context->render_data.rt_lab_settings.reset_history != 0 || !render_context->rt_lab.neural.recurrent) ? 1u : 0u;
    auto const size = ti.info(AT.latent).value().size;
    rt_lab_neural_dispatch(ti, render_context, push, size, size, RT_LAB_NEURAL_X, RenderTimes::index<"RT_LAB", "NEURAL_LATENT">(),
        rt_lab_neural_variant(coopvec ? rt_lab_neural_latent_coopvec_compile_info() : rt_lab_neural_latent_compile_info(), render_context->rt_lab.neural.feature_mask,
            render_context->render_data.rt_lab_settings.neural_use_half != 0)
            .name);
}

void rt_lab_neural_add_latent_task(daxa::TaskGraph & tg, RenderContext & render_context, RtLabNeuralLatentViews const & views)
{
    auto & state = render_context.rt_lab.neural;
    tg.register_buffer(state.weights);
    tg.register_buffer(state.weights_f16);
    tg.add_task(daxa::HeadTask<RtLabNeuralLatentH::Info>()
            .head_views(RtLabNeuralLatentH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .lab_depth = views.lab_depth,
                .lab_face_normal = views.lab_face_normal,
                .lab_depth_history = views.lab_depth_history,
                .lab_face_normal_history = views.lab_face_normal_history,
                .accumulated = views.accumulated,
                .lab_raw_diffuse = views.lab_raw_diffuse,
                .lab_variance = views.lab_variance,
                .lab_ray_length = views.lab_ray_length,
                .detail_normal = views.detail_normal,
                .ray_capped = views.ray_capped,
                .latent_history = views.latent_history,
                .latent = views.latent,
                .affinity = views.affinity,
                .weights = state.weights.view(),
                .weights_f16 = state.weights_f16.view(),
            })
            .executes(rt_lab_neural_latent_callback, &render_context, views.lab_ray_length != daxa::NullTaskImage, views.detail_normal != daxa::NullTaskImage,
                views.ray_capped != daxa::NullTaskImage));
}

static void rt_lab_neural_atrous_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 pass, bool split_enabled, bool history_feedback)
{
    auto const & AT = RtLabNeuralAtrousH::Info::AT;
    RtLabNeuralAtrousPush push = {};
    push.step = 1 << pass;
    push.split_enabled = split_enabled ? 1u : 0u;
    push.clamp_input_count = pass == 0 ? 1u : 0u;
    push.history_feedback = history_feedback ? 1u : 0u;
    push.target_samples = render_context->render_data.rt_lab_settings.neural_target_samples;
    u32 const render_time = RenderTimes::index<"RT_LAB", "ATROUS">(); // all passes add up in one timer
    auto const size = ti.info(AT.dst).value().size;
    rt_lab_neural_dispatch(ti, render_context, push, size, size, RT_LAB_NEURAL_X, render_time,
        rt_lab_neural_variant(rt_lab_neural_atrous_compile_info(), 0u, render_context->render_data.rt_lab_settings.neural_use_half != 0).name);
}

void rt_lab_neural_add_atrous_task(daxa::TaskGraph & tg, RenderContext & render_context, RtLabNeuralAtrousViews const & views, u32 pass, bool split_enabled)
{
    tg.add_task(daxa::HeadTask<RtLabNeuralAtrousH::Info>(fmt::format("RtLabNeuralAtrous{}", pass))
            .head_views(RtLabNeuralAtrousH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .lab_depth = views.lab_depth,
                .lab_face_normal = views.lab_face_normal,
                .src = views.src,
                .dst = views.dst,
                .split_src = views.split_src,
                .split_dst = views.split_dst,
                .lab_variance = views.lab_variance,
                .latent = views.latent,
                .affinity = views.affinity,
                .history_feedback = views.history_feedback,
            })
            .executes(rt_lab_neural_atrous_callback, &render_context, pass, split_enabled, views.history_feedback != daxa::NullTaskImage));
}

static void rt_lab_neural_dump_callback(daxa::TaskInterface ti, RenderContext * render_context, bool ray_length_bound, bool detail_normal_bound, bool albedo_bound, bool ray_stats_bound, bool sh_bound)
{
    auto & lab = render_context->rt_lab;
    auto & dump = lab.neural.dump;
    // Only frames of the test's RUN phase from first_frame on, and only until the requested count.
    if (!dump.enabled || dump.flushed || lab.phase != RtLabTestPhase::RUN || dump.written_frames >= dump.frames)
    {
        return;
    }
    bool const due = dump.frame_list.empty() ? lab.phase_frame >= dump.first_frame : lab.phase_frame >= dump.frame_list[static_cast<size_t>(dump.written_frames)];
    if (!due)
    {
        return;
    }
    auto const & AT = RtLabNeuralDumpH::Info::AT;
    auto const lab_size = ti.info(AT.lab_depth).value().size;
    u32 const crop = std::min({static_cast<u32>(dump.crop_size), lab_size.x, lab_size.y, dump.allocated_crop});
    u32 const origin_x = std::min(static_cast<u32>(std::max(dump.crop_x, 0)), lab_size.x - crop);
    u32 const origin_y = std::min(static_cast<u32>(std::max(dump.crop_y, 0)), lab_size.y - crop);
    if (dump.written_frames == 0)
    {
        dump.actual_crop = crop;
        dump.actual_origin_x = origin_x;
        dump.actual_origin_y = origin_y;
        dump.lab_width = lab_size.x;
        dump.lab_height = lab_size.y;
    }
    for (u32 block : {0u, 1u + static_cast<u32>(dump.written_frames)})
    {
        if (block == 0 && dump.written_frames != 0)
        {
            continue; // geometry once (static camera)
        }
        RtLabNeuralDumpPush push = {};
        push.crop_origin = {dump.actual_origin_x, dump.actual_origin_y};
        push.crop_size = dump.actual_crop;
        push.block = block;
        push.ray_length_bound = ray_length_bound ? 1u : 0u;
        push.frame_index = static_cast<u32>(lab.phase_frame);
        push.detail_normal_bound = detail_normal_bound ? 1u : 0u;
        push.albedo_bound = albedo_bound ? 1u : 0u;
        if (albedo_bound)
        {
            auto const albedo_size = ti.info(AT.half_res_albedo).value().size;
            push.albedo_shift = albedo_size.x == lab_size.x ? 0u : 1u;
        }
        push.ray_stats_bound = ray_stats_bound ? 1u : 0u;
        push.sh_bound = sh_bound ? 1u : 0u;
        rt_lab_neural_dispatch(ti, render_context, push, lab_size, daxa::Extent3D{dump.actual_crop, dump.actual_crop, 1}, RT_LAB_NEURAL_X,
            RenderTimes::index<"RT_LAB", "NEURAL_LATENT">(), rt_lab_neural_dump_compile_info().name);
    }
    dump.written_frames += 1;
    dump.frames_since_done = 0;
}

void rt_lab_neural_add_dump_task(daxa::TaskGraph & tg, RenderContext & render_context, RtLabNeuralDumpViews const & views)
{
    auto & dump = render_context.rt_lab.neural.dump;
    tg.register_buffer(dump.buffer);
    tg.add_task(daxa::HeadTask<RtLabNeuralDumpH::Info>()
            .head_views(RtLabNeuralDumpH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .lab_depth = views.lab_depth,
                .lab_face_normal = views.lab_face_normal,
                .accumulated = views.accumulated,
                .lab_raw_diffuse = views.lab_raw_diffuse,
                .lab_variance = views.lab_variance,
                .lab_ray_length = views.lab_ray_length,
                .lab_sh_y = views.lab_sh_y,
                .lab_cocg = views.lab_cocg,
                .lab_sq = views.lab_sq,
                .detail_normal = views.detail_normal,
                .half_res_albedo = views.half_res_albedo,
                .ray_alloc = views.ray_alloc,
                .ray_lengths = views.ray_lengths,
                .dump = dump.buffer.view(),
            })
            .executes(rt_lab_neural_dump_callback, &render_context, views.lab_ray_length != daxa::NullTaskImage, views.detail_normal != daxa::NullTaskImage,
                views.half_res_albedo != daxa::NullTaskImage, views.ray_alloc != daxa::NullTaskImage && views.ray_lengths != daxa::NullTaskBuffer,
                views.lab_sh_y != daxa::NullTaskImage && views.lab_cocg != daxa::NullTaskImage && views.lab_sq != daxa::NullTaskImage));
}
