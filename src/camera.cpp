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
        ret[1][1] = -tanHalfFovy;
        ret[2][2] = 0.0f;
        ret[2][3] = -1.0f;
        ret[3][2] = zNear;
        return ret;
    };
    glm::mat4 prespective = inf_depth_reverse_z_perspective(glm::radians(fov), f32(settings.render_target_size.x) / f32(settings.render_target_size.y), near);
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

glm::fquat quat_exp(f32vec3 v, float eps=1e-8f)
{
    float halfangle = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
	
    if (halfangle < eps)
    {
        return glm::normalize(glm::fquat(1.0f, v.x, v.y, v.z));
    }
    else
    {
        float c = cosf(halfangle);
        float s = sinf(halfangle) / halfangle;
        return glm::fquat(c, s * v.x, s * v.y, s * v.z);
    }
}


f32vec3 quat_log(glm::fquat q, f32 eps=1e-8f)
{
    float length = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z);
	
    if (length < eps)
    {
        return f32vec3(q.x, q.y, q.z);
    }
    else
    {
        float halfangle = acosf(glm::clamp(q.w, -1.0f, 1.0f));
        return halfangle * (f32vec3(q.x, q.y, q.z) / length);
    }
}


f32vec3 quat_to_scaled_angle_axis(glm::fquat q, f32 eps=1e-8f)
{
    return 2.0f * quat_log(q, eps);
}

glm::fquat quat_from_scaled_angle_axis(f32vec3 v, float eps=1e-8f)
{
    return quat_exp(v / 2.0f, eps);
}


glm::fquat quat_abs(glm::fquat x)
{
    return x.w < 0.0 ? -x : x;
}

void quat_hermite(
    glm::fquat& rot,
    f32vec3& vel, 
    float x, 
    glm::fquat r0,
    glm::fquat r1, 
    f32vec3 v0,
    f32vec3 v1)
{
    float w1 = 3*x*x - 2*x*x*x;
    float w2 = x*x*x - 2*x*x + x;
    float w3 = x*x*x - x*x;
    
    float q1 = 6*x - 6*x*x;
    float q2 = 3*x*x - 4*x + 1;
    float q3 = 3*x*x - 2*x;
    
    f32vec3 r1_sub_r0 = quat_to_scaled_angle_axis(quat_abs((r1 * glm::inverse(r0))));   
    
    rot = quat_from_scaled_angle_axis(w1*r1_sub_r0 + w2*v0 + w3*v1) * r0;
    vel = q1*r1_sub_r0 + q2*v0 + q3*v1;
}

void quat_catmull_rom(
    glm::fquat& rot,
    f32vec3& vel,
    f32 x,
    glm::fquat r0,
    glm::fquat r1, 
    glm::fquat r2,
    glm::fquat r3)
{
    f32vec3 r1_sub_r0 = quat_to_scaled_angle_axis(quat_abs((r1 * glm::inverse(r0))));
    f32vec3 r2_sub_r1 = quat_to_scaled_angle_axis(quat_abs((r2 * glm::inverse(r1))));
    f32vec3 r3_sub_r2 = quat_to_scaled_angle_axis(quat_abs((r3 * glm::inverse(r2))));
  
    f32vec3 v1 = (r1_sub_r0 + r2_sub_r1) / 2.0f;
    f32vec3 v2 = (r2_sub_r1 + r3_sub_r2) / 2.0f;
    quat_hermite(rot, vel, x, r1, r2, v1, v2);
}

void hermite(
    f32vec3& pos,
    f32vec3& vel, 
    f32 x, 
    f32vec3 p0,
    f32vec3 p1, 
    f32vec3 v0,
    f32vec3 v1)
{
    f32 w0 = 2*x*x*x - 3*x*x + 1;
    f32 w1 = 3*x*x - 2*x*x*x;
    f32 w2 = x*x*x - 2*x*x + x;
    f32 w3 = x*x*x - x*x;
    
    f32 q0 = 6*x*x - 6*x;
    f32 q1 = 6*x - 6*x*x;
    f32 q2 = 3*x*x - 4*x + 1;
    f32 q3 = 3*x*x - 2*x;
    
    pos = w0*p0 + w1*p1 + w2*v0 + w3*v1;
    vel = q0*p0 + q1*p1 + q2*v0 + q3*v1;
}

void catmull_rom(
    f32vec3& pos,
    f32vec3& vel,
    f32 x,
    f32vec3 p0,
    f32vec3 p1, 
    f32vec3 p2,
    f32vec3 p3)
{
    f32vec3 v1 = ((p1 - p0) + (p2 - p1)) / 2.0f;
    f32vec3 v2 = ((p2 - p1) + (p3 - p2)) / 2.0f;
    hermite(pos, vel, x, p1, p2, v1, v2);
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

    f32vec3 velocity;
    auto last_keyframe_idx = (current_keyframe_index + (path_keyframes.size() - 1)) % path_keyframes.size();
    auto next_keyframe_idx = (current_keyframe_index +  1) % path_keyframes.size();

    catmull_rom(
        position,
        velocity,
        t,
        path_keyframes.at(last_keyframe_idx).start_position,
        current_keyframe.start_position,
        current_keyframe.end_position,
        path_keyframes.at(next_keyframe_idx).end_position
    );

    quat_catmull_rom(
        forward,
        velocity,
        t,
        path_keyframes.at(last_keyframe_idx).start_rotation,
        current_keyframe.start_rotation,
        current_keyframe.end_rotation,
        path_keyframes.at(next_keyframe_idx).end_rotation
    );
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
