#include "camera.hpp"

void CameraController::process_input(Window & window, f32 dt)
{
    f32 speed = window.key_pressed(GLFW_KEY_LEFT_SHIFT) ? translationSpeed * 4.0f : translationSpeed;
    speed = window.key_pressed(GLFW_KEY_LEFT_CONTROL) ? speed * 0.25f : speed;

    if (window.is_focused())
    {
        if (window.key_just_pressed(GLFW_KEY_ESCAPE))
        {
            if (window.is_cursor_captured()) { window.release_cursor(); }
            else { window.capture_cursor(); }
        }
    }
    else if (window.is_cursor_captured()) { window.release_cursor(); }

    auto cameraSwaySpeed = this->cameraSwaySpeed;
    if (window.key_pressed(GLFW_KEY_C))
    {
        cameraSwaySpeed *= 0.25;
        bZoom = true;
    }
    else { bZoom = false; }

    glm::vec3 right = glm::cross(forward, up);
    glm::vec3 fake_up = glm::normalize(glm::cross(right, forward));
    if (window.is_cursor_captured())
    {
        if (window.key_pressed(GLFW_KEY_W)) { position += forward * speed * dt; }
        if (window.key_pressed(GLFW_KEY_S)) { position -= forward * speed * dt; }
        if (window.key_pressed(GLFW_KEY_A)) { position -= glm::normalize(glm::cross(forward, up)) * speed * dt; }
        if (window.key_pressed(GLFW_KEY_D)) { position += glm::normalize(glm::cross(forward, up)) * speed * dt; }
        if (window.key_pressed(GLFW_KEY_SPACE)) { position += fake_up * speed * dt; }
        if (window.key_pressed(GLFW_KEY_LEFT_ALT)) { position -= fake_up * speed * dt; }
        if (window.key_pressed(GLFW_KEY_Q)) { position -= up * speed * dt; }
        if (window.key_pressed(GLFW_KEY_E)) { position += up * speed * dt; }
        pitch += window.get_cursor_change_y() * cameraSwaySpeed;
        pitch = std::clamp(pitch, -85.0f, 85.0f);
        yaw += window.get_cursor_change_x() * cameraSwaySpeed;
    }
    forward.x = -glm::cos(glm::radians(yaw - 90.0f)) * glm::cos(glm::radians(pitch));
    forward.y = glm::sin(glm::radians(yaw - 90.0f)) * glm::cos(glm::radians(pitch));
    forward.z = -glm::sin(glm::radians(pitch));
}

auto CameraController::make_camera_info(Settings const & settings) const -> CameraInfo
{
    auto fov = this->fov;
    if (bZoom) { fov *= 0.25f; }
    auto inf_depth_reverse_z_perspective = [](auto fov_rads, auto aspect, auto zNear)
    {
        assert(abs(aspect - std::numeric_limits<f32>::epsilon()) > 0.0f);

        f32 const tanHalfFovy = 1.0f / std::tan(fov_rads * 0.5f);

        glm::mat4x4 ret(0.0f);
        ret[0][0] = tanHalfFovy / aspect;
        ret[1][1] = tanHalfFovy;
        ret[2][2] = 0.0f;
        ret[2][3] = -1.0f;
        ret[3][2] = zNear;
        return ret;
    };
    glm::mat4 prespective =
        inf_depth_reverse_z_perspective(glm::radians(fov), f32(settings.render_target_size.x) / f32(settings.render_target_size.y), near);
    prespective[1][1] *= -1.0f;
    CameraInfo ret = {};
    ret.proj = prespective;
    ret.inv_proj = glm::inverse(prespective);
    ret.view = glm::lookAt(position, position + forward, up);
    ret.inv_view = glm::inverse(ret.view);
    ret.view_proj = ret.proj * ret.view;
    ret.inv_view_proj = glm::inverse(ret.view_proj);
    ret.position = this->position;
    ret.up = this->up;
    glm::vec3 ws_ndc_corners[2][2][2];
    glm::mat4 inv_view_proj = glm::inverse(ret.proj * ret.view);
    for (u32 z = 0; z < 2; ++z)
    {
        for (u32 y = 0; y < 2; ++y)
        {
            for (u32 x = 0; x < 2; ++x)
            {
                glm::vec3 corner = glm::vec3((glm::vec2(x, y) - 0.5f) * 2.0f, 1.0f - z * 0.5f);
                glm::vec4 proj_corner = inv_view_proj * glm::vec4(corner, 1);
                ws_ndc_corners[x][y][z] = glm::vec3(proj_corner) / proj_corner.w;
            }
        }
    }
    ret.is_orthogonal = 0u;
    ret.orthogonal_half_ws_width = 0.0f;
    ret.near_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[0][1][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0]));
    ret.right_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[1][1][0] - ws_ndc_corners[1][0][0], ws_ndc_corners[1][0][1] - ws_ndc_corners[1][0][0]));
    ret.left_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][0][1], ws_ndc_corners[0][0][0] - ws_ndc_corners[0][0][1]));
    ret.top_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[0][0][1] - ws_ndc_corners[0][0][0]));
    ret.bottom_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][1][0], ws_ndc_corners[1][1][0] - ws_ndc_corners[0][1][0]));
    int i = 0;
    ret.screen_size = { settings.render_target_size.x, settings.render_target_size.y };
    ret.inv_screen_size = {
        1.0f / static_cast<f32>(settings.render_target_size.x),
        1.0f / static_cast<f32>(settings.render_target_size.y),
    };
    ret.near_plane = this->near;
    return ret;
}

void CinematicCamera::update_keyframes(std::vector<CameraAnimationKeyframe> && keyframes)
{
    current_keyframe_index = 0;
    current_keyframe_time = 0.0f;
    path_keyframes = keyframes;
}

void CinematicCamera::set_keyframe(i32 keyframe, f32 keyframe_progress)
{
    current_keyframe_index = keyframe;
    f32 const current_keyframe_finish_time = path_keyframes.at(current_keyframe_index).transition_time;
    current_keyframe_time = keyframe_progress * current_keyframe_finish_time;
}

void CinematicCamera::process_input(Window &window, f32 dt)
{
    if(override_keyframe) { dt = 0.0f; }
    // TODO(msakmary) Whenever the update position dt is longer than a whole keyframe transition time
    // this code will not properly account for this
    f32 const current_keyframe_finish_time = path_keyframes.at(current_keyframe_index).transition_time;
    bool const keyframe_finished = current_keyframe_finish_time < current_keyframe_time + dt;
    bool const on_last_keyframe = current_keyframe_index == (path_keyframes.size() - 1);
    bool const animation_finished = keyframe_finished && on_last_keyframe;

    if (keyframe_finished)
    {
        auto prev_keyframe_time = current_keyframe_time;
        current_keyframe_time = (current_keyframe_time + dt) - current_keyframe_finish_time;
        current_keyframe_index = animation_finished ? 0 : current_keyframe_index + 1;
    }
    else
    {
        current_keyframe_time = current_keyframe_time + dt;
    }

    auto const & current_keyframe = path_keyframes.at(current_keyframe_index);
    f32 const t = current_keyframe_time / current_keyframe.transition_time;

    f32 w0 = static_cast<f32>(glm::pow(1.0f - t, 3));
    f32 w1 = static_cast<f32>(glm::pow(1.0f - t, 2) * 3.0f * t);
    f32 w2 = static_cast<f32>((1.0f - t) * 3 * t * t);
    f32 w3 = static_cast<f32>(t * t * t);

    position =
        w0 * current_keyframe.start_position +
        w1 * current_keyframe.first_control_point +
        w2 * current_keyframe.second_control_point +
        w3 * current_keyframe.end_position;

    forward = glm::slerp(current_keyframe.start_rotation, current_keyframe.end_rotation, t);
}

auto CinematicCamera::make_camera_info(Settings const & settings) const -> CameraInfo
{
    auto inf_depth_reverse_z_perspective = [](auto fov_rads, auto aspect, auto zNear)
    {
        assert(abs(aspect - std::numeric_limits<f32>::epsilon()) > 0.0f);

        f32 const tanHalfFovy = 1.0f / std::tan(fov_rads * 0.5f);

        glm::mat4x4 ret(0.0f);
        ret[0][0] = tanHalfFovy / aspect;
        ret[1][1] = tanHalfFovy;
        ret[2][2] = 0.0f;
        ret[2][3] = -1.0f;
        ret[3][2] = zNear;
        return ret;
    };
    glm::mat4 prespective =
        inf_depth_reverse_z_perspective(glm::radians(fov), f32(settings.render_target_size.x) / f32(settings.render_target_size.y), near);
    prespective[1][1] *= -1.0f;
    CameraInfo ret = {};
    ret.proj = prespective;
    ret.inv_proj = glm::inverse(prespective);
    ret.view = glm::toMat4(forward) * glm::translate(glm::identity<glm::mat4x4>(), glm::vec3(-position[0], -position[1], -position[2]));
    ret.inv_view = glm::inverse(ret.view);
    ret.view_proj = ret.proj * ret.view;
    ret.inv_view_proj = glm::inverse(ret.view_proj);
    ret.position = this->position;
    ret.up = this->up;
    glm::vec3 ws_ndc_corners[2][2][2];
    glm::mat4 inv_view_proj = glm::inverse(ret.proj * ret.view);
    for (u32 z = 0; z < 2; ++z)
    {
        for (u32 y = 0; y < 2; ++y)
        {
            for (u32 x = 0; x < 2; ++x)
            {
                glm::vec3 corner = glm::vec3((glm::vec2(x, y) - 0.5f) * 2.0f, 1.0f - z * 0.5f);
                glm::vec4 proj_corner = inv_view_proj * glm::vec4(corner, 1);
                ws_ndc_corners[x][y][z] = glm::vec3(proj_corner) / proj_corner.w;
            }
        }
    }
    ret.is_orthogonal = 0u;
    ret.orthogonal_half_ws_width = 0.0f;
    ret.near_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[0][1][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0]));
    ret.right_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[1][1][0] - ws_ndc_corners[1][0][0], ws_ndc_corners[1][0][1] - ws_ndc_corners[1][0][0]));
    ret.left_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][0][1], ws_ndc_corners[0][0][0] - ws_ndc_corners[0][0][1]));
    ret.top_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[0][0][1] - ws_ndc_corners[0][0][0]));
    ret.bottom_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][1][0], ws_ndc_corners[1][1][0] - ws_ndc_corners[0][1][0]));
    int i = 0;
    ret.screen_size = { settings.render_target_size.x, settings.render_target_size.y };
    ret.inv_screen_size = {
        1.0f / static_cast<f32>(settings.render_target_size.x),
        1.0f / static_cast<f32>(settings.render_target_size.y),
    };
    ret.near_plane = this->near;
    return ret;
}
