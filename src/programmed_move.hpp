#pragma once

#include "timberdoodle.hpp"
#include <array>
#include <filesystem>
#include <string>
#include <nlohmann/json_fwd.hpp>
using namespace tido::types;

// Programmed move: a constant velocity / angular speed sequence over a duration, shared by the camera and the sun panel
// (property viewer) and the RT Lab tests.
//
// A pose is a position plus yaw / pitch in degrees.
//   Camera: CameraController position / yaw / pitch, velocity relative to the camera (x right, y up, z forward).
//   Sun:    yaw = azimuth (sun panel "Angle X"), pitch = zenith angle (sun panel "Angle Y"), no translation.

struct ProgrammedMovePose
{
    f32vec3 position = {};
    f32 yaw = 0.0f;
    f32 pitch = 0.0f;
};

struct ProgrammedMove
{
    // == Setup ==
    bool translation = true;     // false: rotation only (sun), the velocity is hidden and ignored
    bool use_start_pose = false; // false: start from the pose current when the move starts
    ProgrammedMovePose start = {};
    // UI text of the start pose: camera "px py pz fx fy fz" (the View Camera "Copy to clipboard" format: position +
    // forward), sun "angle_x angle_y" (degrees). Empty = not written from start yet.
    std::array<char, 256> start_text = {};
    std::array<char, 64> file_name = {}; // UI: name to save under (PROGRAMMED_MOVES_DIRECTORY/<name>.json)
    f32vec3 velocity = {};       // per second, relative to the current pose orientation (x right, y up, z forward)
    f32 yaw_speed = 0.0f;        // degrees per second
    f32 pitch_speed = 0.0f;      // degrees per second
    f32 duration = 1.0f;         // seconds
    bool return_to_start = true; // free runs: restore the start pose one frame after the move ended
    bool repeat = false;         // free runs: restart whenever the move ends

    // == Free run state (programmed_move_update) ==
    bool active = false;
    bool return_pending = false;
    bool jump_pending = false;      // first frame of a run with an explicit start pose: jump there before moving
    f32 time_remaining = 0.0f;
    ProgrammedMovePose origin = {}; // pose the run started from
};

// Pose after one step of dt seconds from `pose`.
auto programmed_move_integrate(ProgrammedMove const & move, ProgrammedMovePose const & pose, f32 dt) -> ProgrammedMovePose;

// == Fixed step runs (reproducible, RT Lab tests): the pose of step k is a pure function of the origin, k and dt. ==
// Steps until the move ended: ceil(duration / dt), 0 for duration <= 0.
auto programmed_move_step_count(ProgrammedMove const & move, f32 dt) -> i32;
// Pose of step k (step 0 = origin), steps past the end hold the end pose.
auto programmed_move_pose_at(ProgrammedMove const & move, ProgrammedMovePose const & origin, i32 step, f32 dt) -> ProgrammedMovePose;
// Origin of a run starting now: the start pose if set, else `current`.
auto programmed_move_origin(ProgrammedMove const & move, ProgrammedMovePose const & current) -> ProgrammedMovePose;

// == Free runs (property viewer buttons, real frame time) ==
void programmed_move_begin(ProgrammedMove & move, ProgrammedMovePose const & current);
void programmed_move_cancel(ProgrammedMove & move);
enum struct ProgrammedMoveEvent
{
    NONE,
    FINISHED, // this frame was the last moving frame
    RETURNED, // this frame restored the start pose
};
// Advances a free run by dt and writes the pose. No-op when no run is active.
auto programmed_move_update(ProgrammedMove & move, ProgrammedMovePose & pose, f32 dt) -> ProgrammedMoveEvent;

// Sun pose <-> sun direction (z up, the sun panel's angle convention).
auto programmed_move_sun_pose(f32vec3 sun_direction) -> ProgrammedMovePose;
auto programmed_move_sun_direction(ProgrammedMovePose const & pose) -> f32vec3;
// Camera forward from yaw / pitch (CameraController::process_input convention).
auto programmed_move_camera_forward(f32 yaw, f32 pitch) -> f32vec3;
// Start pose text <-> pose (ProgrammedMove::start_text format). Parse returns false on malformed text.
auto programmed_move_pose_to_text(ProgrammedMovePose const & pose, bool translation) -> std::string;
auto programmed_move_pose_from_text(char const * text, bool translation, ProgrammedMovePose & pose) -> bool;

// == Files: PROGRAMMED_MOVES_DIRECTORY/<name>.json (gitignored) ==
// "kind" = "camera" (translation) or "sun" (rotation only); loading only accepts a file of the move's own kind.
#define PROGRAMMED_MOVES_DIRECTORY "programmed_moves"
// Setup fields only (no run state). from_json keeps the fields the json does not have from `move`.
auto programmed_move_to_json(ProgrammedMove const & move) -> nlohmann::json;
auto programmed_move_from_json(nlohmann::json const & json, ProgrammedMove move) -> ProgrammedMove;
auto programmed_move_save(ProgrammedMove const & move, std::string const & name) -> bool;
// false: unreadable, or a file of the other kind. Loads the setup only, a running move keeps running.
auto programmed_move_load(std::filesystem::path const & path, ProgrammedMove & move) -> bool;
// Saved moves of one kind (file stems), sorted.
auto programmed_move_list(bool translation) -> std::vector<std::string>;

// Setup widgets of a move (ImGui, ids scoped by `id`). `current` = pose the "Take Current Position" button copies.
// free_run: also shows the return / repeat options (only used by free runs).
void programmed_move_ui(char const * id, ProgrammedMove & move, ProgrammedMovePose const & current, bool free_run);
