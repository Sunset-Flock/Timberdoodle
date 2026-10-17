#include "programmed_move.hpp"

#include <imgui.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

auto programmed_move_camera_forward(f32 yaw, f32 pitch) -> f32vec3
{
    return f32vec3{
        -glm::cos(glm::radians(yaw - 90.0f)) * glm::cos(glm::radians(pitch)),
        glm::sin(glm::radians(yaw - 90.0f)) * glm::cos(glm::radians(pitch)),
        -glm::sin(glm::radians(pitch)),
    };
}

auto programmed_move_integrate(ProgrammedMove const & move, ProgrammedMovePose const & pose, f32 dt) -> ProgrammedMovePose
{
    ProgrammedMovePose next = pose;
    if (move.translation)
    {
        // Basis of the pose orientation (z up world): x = right, y = up, z = forward.
        f32vec3 const forward = programmed_move_camera_forward(pose.yaw, pose.pitch);
        f32vec3 const right = glm::normalize(glm::cross(forward, f32vec3{0.0f, 0.0f, 1.0f}));
        f32vec3 const up = glm::normalize(glm::cross(right, forward));
        next.position += (move.velocity.x * right + move.velocity.y * up + move.velocity.z * forward) * dt;
    }
    next.yaw += move.yaw_speed * dt;
    next.pitch += move.pitch_speed * dt;
    return next;
}

auto programmed_move_step_count(ProgrammedMove const & move, f32 dt) -> i32
{
    if (!(move.duration > 0.0f) || !(dt > 0.0f))
    {
        return 0;
    }
    // The epsilon keeps durations that are whole multiples of dt from gaining a step through rounding.
    return static_cast<i32>(std::ceil(move.duration / dt - 1e-4f));
}

auto programmed_move_pose_at(ProgrammedMove const & move, ProgrammedMovePose const & origin, i32 step, f32 dt) -> ProgrammedMovePose
{
    // Integrated step by step from the origin (same arithmetic every run -> bit identical poses).
    i32 const steps = std::min(std::max(step, 0), programmed_move_step_count(move, dt));
    ProgrammedMovePose pose = origin;
    for (i32 i = 0; i < steps; ++i)
    {
        pose = programmed_move_integrate(move, pose, dt);
    }
    return pose;
}

auto programmed_move_origin(ProgrammedMove const & move, ProgrammedMovePose const & current) -> ProgrammedMovePose
{
    ProgrammedMovePose origin = move.use_start_pose ? move.start : current;
    if (!move.translation)
    {
        origin.position = current.position;
    }
    return origin;
}

void programmed_move_begin(ProgrammedMove & move, ProgrammedMovePose const & current)
{
    move.origin = programmed_move_origin(move, current);
    move.active = true;
    move.return_pending = false;
    move.jump_pending = move.use_start_pose;
    move.time_remaining = std::max(move.duration, 0.0f);
}

void programmed_move_cancel(ProgrammedMove & move)
{
    move.active = false;
    move.return_pending = false;
    move.jump_pending = false;
    move.time_remaining = 0.0f;
}

auto programmed_move_update(ProgrammedMove & move, ProgrammedMovePose & pose, f32 dt) -> ProgrammedMoveEvent
{
    if (move.return_pending)
    {
        // One frame after the end: the end pose was rendered (and screenshot), now snap back.
        pose = move.origin;
        move.return_pending = false;
        if (move.repeat)
        {
            programmed_move_begin(move, move.origin);
            move.jump_pending = false; // already there
        }
        return ProgrammedMoveEvent::RETURNED;
    }
    if (!move.active)
    {
        return ProgrammedMoveEvent::NONE;
    }
    if (move.jump_pending)
    {
        pose = move.origin;
        move.jump_pending = false;
    }
    pose = programmed_move_integrate(move, pose, dt);
    move.time_remaining -= dt;
    if (move.time_remaining > 0.0f)
    {
        return ProgrammedMoveEvent::NONE;
    }
    move.active = false;
    move.time_remaining = 0.0f;
    move.return_pending = move.return_to_start;
    if (move.repeat && !move.return_to_start)
    {
        // Repeat without returning: continue straight from the end pose.
        move.active = true;
        move.time_remaining = std::max(move.duration, 0.0f);
        move.origin = pose;
    }
    return ProgrammedMoveEvent::FINISHED;
}

auto programmed_move_sun_pose(f32vec3 sun_direction) -> ProgrammedMovePose
{
    f32vec3 const d = glm::normalize(sun_direction);
    f32 azimuth = glm::degrees(std::atan2(d.y, d.x));
    azimuth += azimuth < 0.0f ? 360.0f : 0.0f;
    return ProgrammedMovePose{
        .position = {},
        .yaw = azimuth,
        .pitch = glm::degrees(std::acos(std::clamp(d.z, -1.0f, 1.0f))),
    };
}

auto programmed_move_sun_direction(ProgrammedMovePose const & pose) -> f32vec3
{
    f32 const azimuth = glm::radians(pose.yaw);
    f32 const zenith = glm::radians(pose.pitch);
    return f32vec3{
        glm::cos(azimuth) * glm::sin(zenith),
        glm::sin(azimuth) * glm::sin(zenith),
        glm::cos(zenith),
    };
}

auto programmed_move_pose_to_text(ProgrammedMovePose const & pose, bool translation) -> std::string
{
    if (!translation)
    {
        return fmt::format("{} {}", pose.yaw, pose.pitch);
    }
    f32vec3 const forward = programmed_move_camera_forward(pose.yaw, pose.pitch);
    return fmt::format("{} {} {} {} {} {}", pose.position.x, pose.position.y, pose.position.z, forward.x, forward.y, forward.z);
}

auto programmed_move_pose_from_text(char const * text, bool translation, ProgrammedMovePose & pose) -> bool
{
    if (!translation)
    {
        f32 x = 0.0f;
        f32 y = 0.0f;
        if (std::sscanf(text, "%f %f", &x, &y) != 2)
        {
            return false;
        }
        pose = ProgrammedMovePose{.yaw = x, .pitch = y};
        return true;
    }
    f32 px, py, pz, fx, fy, fz;
    if (std::sscanf(text, "%f %f %f %f %f %f", &px, &py, &pz, &fx, &fy, &fz) != 6)
    {
        return false;
    }
    f32vec3 const forward = glm::normalize(f32vec3{fx, fy, fz});
    if (!std::isfinite(forward.x) || !std::isfinite(forward.y) || !std::isfinite(forward.z))
    {
        return false;
    }
    // Inverse of programmed_move_camera_forward (same as the View Camera "Apply").
    pose.position = {px, py, pz};
    pose.pitch = glm::degrees(-std::asin(glm::clamp(forward.z, -1.0f, 1.0f)));
    pose.yaw = glm::degrees(std::atan2(forward.y, -forward.x)) + 90.0f;
    return true;
}

static auto programmed_move_pose_to_json(ProgrammedMovePose const & p) -> nlohmann::json
{
    return {p.position.x, p.position.y, p.position.z, p.yaw, p.pitch};
}

static auto programmed_move_pose_from_json(nlohmann::json const & j, ProgrammedMovePose p) -> ProgrammedMovePose
{
    if (j.is_array() && j.size() == 5)
    {
        p.position = {j[0].get<f32>(), j[1].get<f32>(), j[2].get<f32>()};
        p.yaw = j[3].get<f32>();
        p.pitch = j[4].get<f32>();
    }
    return p;
}

auto programmed_move_to_json(ProgrammedMove const & move) -> nlohmann::json
{
    return {
        {"kind", move.translation ? "camera" : "sun"},
        {"translation", move.translation},
        {"use_start_pose", move.use_start_pose},
        {"start", programmed_move_pose_to_json(move.start)},
        {"velocity", {move.velocity.x, move.velocity.y, move.velocity.z}},
        {"yaw_speed", move.yaw_speed},
        {"pitch_speed", move.pitch_speed},
        {"duration", move.duration},
        {"return_to_start", move.return_to_start},
        {"repeat", move.repeat},
    };
}

auto programmed_move_from_json(nlohmann::json const & json, ProgrammedMove move) -> ProgrammedMove
{
    move.translation = json.value("translation", move.translation);
    move.use_start_pose = json.value("use_start_pose", move.use_start_pose);
    if (json.contains("start"))
    {
        if (json["start"].is_string())
        {
            // Text form (start_text format): camera "px py pz fx fy fz", sun "angle_x angle_y". Sets the explicit start.
            ProgrammedMovePose pose = move.start;
            if (programmed_move_pose_from_text(json["start"].get<std::string>().c_str(), move.translation, pose))
            {
                move.start = pose;
                move.use_start_pose = json.value("use_start_pose", true);
            }
        }
        else
        {
            move.start = programmed_move_pose_from_json(json["start"], move.start);
        }
        move.start_text[0] = '\0'; // the UI rewrites it from start
    }
    if (json.contains("velocity") && json["velocity"].is_array() && json["velocity"].size() == 3)
    {
        move.velocity = {json["velocity"][0].get<f32>(), json["velocity"][1].get<f32>(), json["velocity"][2].get<f32>()};
    }
    move.yaw_speed = json.value("yaw_speed", move.yaw_speed);
    move.pitch_speed = json.value("pitch_speed", move.pitch_speed);
    move.duration = json.value("duration", move.duration);
    move.return_to_start = json.value("return_to_start", move.return_to_start);
    move.repeat = json.value("repeat", move.repeat);
    return move;
}

auto programmed_move_save(ProgrammedMove const & move, std::string const & name) -> bool
{
    if (name.empty())
    {
        return false;
    }
    std::error_code error = {};
    std::filesystem::create_directories(PROGRAMMED_MOVES_DIRECTORY, error);
    std::ofstream file(std::filesystem::path(PROGRAMMED_MOVES_DIRECTORY) / (name + ".json"), std::ios::trunc);
    if (!file)
    {
        return false;
    }
    file << programmed_move_to_json(move).dump(4);
    return static_cast<bool>(file);
}

auto programmed_move_load(std::filesystem::path const & path, ProgrammedMove & move) -> bool
{
    std::ifstream file(path);
    if (!file)
    {
        return false;
    }
    nlohmann::json const json = nlohmann::json::parse(file, nullptr, false);
    if (json.is_discarded() || !json.is_object() || json.value("translation", !move.translation) != move.translation)
    {
        return false;
    }
    move = programmed_move_from_json(json, move);
    return true;
}

auto programmed_move_list(bool translation) -> std::vector<std::string>
{
    std::vector<std::string> names = {};
    std::error_code error = {};
    if (!std::filesystem::is_directory(PROGRAMMED_MOVES_DIRECTORY, error))
    {
        return names;
    }
    for (auto const & entry : std::filesystem::directory_iterator(PROGRAMMED_MOVES_DIRECTORY, error))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".json")
        {
            continue;
        }
        std::ifstream file(entry.path());
        nlohmann::json const json = nlohmann::json::parse(file, nullptr, false);
        if (!json.is_discarded() && json.is_object() && json.value("translation", !translation) == translation)
        {
            names.push_back(entry.path().stem().string());
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

void programmed_move_ui(char const * id, ProgrammedMove & move, ProgrammedMovePose const & current, bool free_run)
{
    ImGui::PushID(id);
    char const * const yaw_label = move.translation ? "Yaw Speed (deg/s)" : "Angle X Speed (deg/s)";
    char const * const pitch_label = move.translation ? "Pitch Speed (deg/s)" : "Angle Y Speed (deg/s)";
    if (move.translation)
    {
        ImGui::InputFloat3("Velocity (cam rel)", &move.velocity.x);
        ImGui::SetItemTooltip("Meters per second, relative to the camera orientation of each step: x = right, y = up, z = forward.");
    }
    ImGui::InputFloat(yaw_label, &move.yaw_speed);
    ImGui::InputFloat(pitch_label, &move.pitch_speed);
    ImGui::InputFloat("Duration (s)", &move.duration);
    move.duration = std::max(move.duration, 0.0f);
    ImGui::Checkbox("Explicit Start", &move.use_start_pose);
    ImGui::SetItemTooltip("Off: the move starts from the current pose. On: it jumps to the start pose first.");
    if (move.use_start_pose)
    {
        auto const write_text = [&](ProgrammedMovePose const & pose)
        {
            std::string const text = programmed_move_pose_to_text(pose, move.translation);
            std::snprintf(move.start_text.data(), move.start_text.size(), "%s", text.c_str());
        };
        if (move.start_text[0] == '\0')
        {
            write_text(move.start);
        }
        // One text field, pasteable: the camera uses the View Camera "Copy to clipboard" format.
        if (ImGui::InputText("Start", move.start_text.data(), move.start_text.size()))
        {
            ProgrammedMovePose pose = move.start;
            if (programmed_move_pose_from_text(move.start_text.data(), move.translation, pose))
            {
                move.start = pose;
            }
        }
        ImGui::SetItemTooltip(move.translation
            ? "px py pz fx fy fz: position + forward, the View Camera \"Copy to clipboard\" format."
            : "angle_x angle_y in degrees (the sun panel Angle X / Angle Y).");
        ProgrammedMovePose parsed = {};
        if (!programmed_move_pose_from_text(move.start_text.data(), move.translation, parsed))
        {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Malformed start, using the last valid one.");
        }
        if (ImGui::Button("Take Current Position"))
        {
            move.start = programmed_move_origin(ProgrammedMove{.translation = move.translation}, current);
            write_text(move.start);
        }
    }
    // Save / load (PROGRAMMED_MOVES_DIRECTORY, only files of this move's kind are listed).
    ImGui::InputText("Name", move.file_name.data(), move.file_name.size());
    ImGui::SameLine();
    ImGui::BeginDisabled(move.file_name[0] == '\0');
    if (ImGui::Button("Save"))
    {
        programmed_move_save(move, move.file_name.data());
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Saves the setup as " PROGRAMMED_MOVES_DIRECTORY "/<name>.json (overwrites).");
    if (ImGui::BeginCombo("Load", "(saved moves)"))
    {
        for (std::string const & name : programmed_move_list(move.translation))
        {
            if (ImGui::Selectable(name.c_str()))
            {
                if (programmed_move_load(std::filesystem::path(PROGRAMMED_MOVES_DIRECTORY) / (name + ".json"), move))
                {
                    std::snprintf(move.file_name.data(), move.file_name.size(), "%s", name.c_str());
                }
            }
        }
        ImGui::EndCombo();
    }
    if (free_run)
    {
        ImGui::Checkbox("Return To Start After Move", &move.return_to_start);
        ImGui::Checkbox("Repeat", &move.repeat);
        ImGui::SetItemTooltip("Restart the move whenever it ends (from the start pose when returning to start, else from where it ended).");
    }
    ImGui::PopID();
}
