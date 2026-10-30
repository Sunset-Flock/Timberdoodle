#pragma once

#include <filesystem>
#include <vector>
#include <daxa/utils/task_graph.hpp>

#include "../../../../timberdoodle.hpp"

using namespace tido::types;

// CPU state of the RT Lab neural affinity (embedded in RtLabState::neural). Kept free of RenderContext so rt_lab_state.hpp
// can include it.

// Training data dump: copies a square crop of the lab data (temporal output, raw samples, variance, ray length guide,
// geometry) of `frames` consecutive frames of a test's RUN phase into a host visible buffer and writes it to `path`
// (layout: RtLabNeuralDumpH in rt_lab_neural.inl, file header in rt_lab_neural.cpp, read by train/dataset.py).
// Feature mask of a weights file (header word 3), the compile time features of the network the weights were trained with.
// Bits 0..2 and 4 are shader defines (rt_lab_neural_variant), bit 3 is a runtime behavior.
constexpr u32 RT_LAB_NEURAL_FEATURE_WINDOW = 1;          // RT_LAB_NEURAL_WINDOW: raw 5x5 log luminance window
constexpr u32 RT_LAB_NEURAL_FEATURE_NO_GUIDE_INPUTS = 2; // RT_LAB_NEURAL_NO_GUIDE_INPUTS
constexpr u32 RT_LAB_NEURAL_FEATURE_RAYLENGTH3 = 4;      // RT_LAB_NEURAL_RAYLENGTH3
constexpr u32 RT_LAB_NEURAL_FEATURE_RECURRENT = 8;       // trained with its previous latent as input: the engine runs it recurrent
constexpr u32 RT_LAB_NEURAL_FEATURE_RAW3 = 16;           // RT_LAB_NEURAL_RAW3: raw 3x3 data (needs inputs the engine does not provide yet)
constexpr u32 RT_LAB_NEURAL_FEATURE_LITE = 32;           // RT_LAB_NEURAL_LITE: plus shaped raw data, same missing inputs as RAW3
constexpr u32 RT_LAB_NEURAL_FEATURE_V2 = 64;            // RT_LAB_NEURAL_V2: SH feature set (3x3 footprint, shared pixel encoder, IN 64); trainer only so far, the engine refuses these weights
constexpr u32 RT_LAB_NEURAL_FEATURE_W2 = 128;           // RT_LAB_NEURAL_WEIGHTING_V2: variance guide style weighting on the latent vector (same 16 outputs); trainer only so far, the engine refuses these weights
constexpr u32 RT_LAB_NEURAL_FEATURE_SHADER_BITS = RT_LAB_NEURAL_FEATURE_WINDOW | RT_LAB_NEURAL_FEATURE_NO_GUIDE_INPUTS | RT_LAB_NEURAL_FEATURE_RAYLENGTH3 | RT_LAB_NEURAL_FEATURE_RAW3 | RT_LAB_NEURAL_FEATURE_LITE;

struct RtLabNeuralDump
{
    bool enabled = false;
    std::filesystem::path path = {};
    i32 first_frame = 0; // RUN phase frame of the first dumped frame (consecutive mode)
    i32 frames = 0;
    // Explicit RUN phase frames to dump (snapshots after 1, 4, 8, ... accumulated frames); non empty = frames = its size.
    std::vector<i32> frame_list = {};
    i32 crop_x = 0;
    i32 crop_y = 0;
    i32 crop_size = 192;

    // Runtime.
    daxa::ExternalTaskBuffer buffer = daxa::ExternalTaskBuffer{{.name = "rt_lab_neural_dump"}}; // host visible
    u32 allocated_blocks = 0;
    u32 allocated_crop = 0;
    u32 actual_crop = 0;       // crop edge the passes used (clamped to the lab size)
    u32 actual_origin_x = 0;
    u32 actual_origin_y = 0;
    u32 lab_width = 0;
    u32 lab_height = 0;
    i32 written_frames = 0;    // frame blocks recorded so far
    i32 frames_since_done = 0; // frames since the last block was recorded (GPU latency before the readback)
    bool flushed = false;
};

struct RtLabNeuralState
{
    // The settings (enabled, tensor cores, f16, target samples) live in RtLabSettings (neural_*), per preset.
    // false = the network never sees its previous latent (always as after a reset): stateless, how the snapshot training works.
    // Set from the weights file on load (feature mask bit RECURRENT).
    bool recurrent = false;
    std::filesystem::path weights_path = "src/rendering/rtgi/rt_lab/neural/weights/rt_lab_neural.bin";

    // Runtime.
    bool weights_loaded = false;
    std::filesystem::path loaded_path = {};
    u32 param_count = 0;
    u32 feature_mask = 0; // RT_LAB_NEURAL_FEATURE_* of the loaded weights (weights file header; old files: the 5x5 window detected from the parameter count)
    daxa::ExternalTaskBuffer weights = daxa::ExternalTaskBuffer{{.name = "rt_lab_neural_weights"}};
    daxa::ExternalTaskBuffer weights_f16 = daxa::ExternalTaskBuffer{{.name = "rt_lab_neural_weights_f16"}}; // same weights, half, for CoopVec
    RtLabNeuralDump dump = {};
    // Graph shape the current graph was built with (rebuild triggers, see rt_lab_neural_graph_changed).
    bool built_active = false;
    bool built_dump = false;
};
