#include <daxa/daxa.inl>
#include "shader_lib/sky_util.glsl"
#include "sky.inl"

#if defined(CUBEMAP)
DAXA_DECL_PUSH_CONSTANT(SkyIntoCubemapH, push)
layout(local_size_x = IBL_CUBE_X, local_size_y = IBL_CUBE_Y, local_size_z = IBL_CUBE_RES) in;

float radical_inverse_vdc(uint bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint i, uint n) {
    return vec2(float(i + 1) / n, radical_inverse_vdc(i + 1));
}

mat3 CUBE_MAP_FACE_ROTATION(uint face) 
{
    switch (face) {
    case 0: return mat3(+0, +0, -1, +0, -1, +0, -1, +0, +0);
    case 1: return mat3(+0, +0, +1, +0, -1, +0, +1, +0, +0);
    case 2: return mat3(+1, +0, +0, +0, +0, +1, +0, -1, +0);
    case 3: return mat3(+1, +0, +0, +0, +0, -1, +0, +1, +0);
    case 4: return mat3(+1, +0, +0, +0, -1, +0, +0, +0, -1);
    default: return mat3(-1, +0, +0, +0, -1, +0, +0, +0, +1);
    }
}

uint _rand_state;
void rand_seed(uint seed) {
    _rand_state = seed;
}

float rand() {
    // https://www.pcg-random.org/
    _rand_state = _rand_state * 747796405u + 2891336453u;
    uint result = ((_rand_state >> ((_rand_state >> 28u) + 4u)) ^ _rand_state) * 277803737u;
    result = (result >> 22u) ^ result;
    return result / 4294967295.0;
}

float rand_normal_dist() {
    float theta = 2.0 * PI * rand();
    float rho = sqrt(-2.0 * log(rand()));
    return rho * cos(theta);
}

vec3 rand_dir() {
    return normalize(vec3(
        rand_normal_dist(),
        rand_normal_dist(),
        rand_normal_dist()));
}

vec3 rand_hemi_dir(vec3 nrm) {
    vec3 result = rand_dir();
    return result * sign(dot(nrm, result));
}

void main() {
    const uvec2 wg_base_pix_pos = gl_WorkGroupID.xy * uvec2(IBL_CUBE_X, IBL_CUBE_Y);
    const uvec2 sg_pix_pos = wg_base_pix_pos + uvec2(gl_SubgroupID % IBL_CUBE_X, gl_SubgroupID / IBL_CUBE_X);
    uint face = gl_WorkGroupID.z;
    vec2 uv = (vec2(sg_pix_pos) + vec2(0.5)) / IBL_CUBE_RES;

    SkySettings sky_settings = deref(push.globals).sky_settings;

    vec3 output_dir = normalize(CUBE_MAP_FACE_ROTATION(face) * vec3(uv * 2 - 1, -1.0));
    const mat3 basis = build_orthonormal_basis(output_dir);

    // Because the atmosphere is using km as it's default units and we want one unit in world
    // space to be one meter we need to scale the position by a factor to get from meters -> kilometers
    const vec3 camera_position = deref(push.globals).camera.position * M_TO_KM_SCALE;
    vec3 world_camera_position = camera_position + vec3(0.0, 0.0, sky_settings.atmosphere_bottom + BASE_HEIGHT_OFFSET);
    const float height = length(world_camera_position);

    vec3 accumulated_result = vec3(0);

    // We hardcode the subgroup size to be 32
    const uint sample_count = 128;
    const uint subgroup_size = 32;
    const uint iter_count = sample_count / subgroup_size;
    const uint global_thread_index = (gl_GlobalInvocationID.x * IBL_CUBE_RES * IBL_CUBE_RES + gl_GlobalInvocationID.y * IBL_CUBE_RES + gl_GlobalInvocationID.z);
    const uint seed = global_thread_index + deref(push.globals).frame_index * IBL_CUBE_RES * IBL_CUBE_RES * 6;

    for (uint i = 0; i < iter_count; ++i) {
        rand_seed((i * subgroup_size + gl_SubgroupInvocationID + seed * sample_count));
        vec3 input_dir = rand_hemi_dir(output_dir);
        // TODO: Now that we sample the atmosphere directly, computing this IBL is really slow.
        // We should cache the IBL cubemap, and only re-render it when necessary.
        const vec3 result = get_atmosphere_illuminance_along_ray(
            sky_settings,
            push.transmittance,
            push.sky,
            deref(push.globals).samplers.linear_clamp,
            input_dir,
            world_camera_position
        );
        const vec3 cos_weighed_result = result * dot(output_dir, input_dir);
        accumulated_result += subgroupInclusiveAdd(cos_weighed_result);
    }
    // Only last thread in each subgroup contains the correct accumulated result
    if(gl_SubgroupInvocationID == 31)
    {
        const vec3 this_frame_luminance = accumulated_result / sample_count;
        const vec4 compressed_accumulated_luminance = imageLoad(daxa_image2DArray(push.ibl_cube), ivec3(sg_pix_pos, gl_WorkGroupID.z));
        // Could be nan for some reason
        const vec3 unsafe_accumulated_luminance = compressed_accumulated_luminance.rgb * compressed_accumulated_luminance.a;
        const vec3 accumulated_luminance = isnan(unsafe_accumulated_luminance.x) ? vec3(0.0) : unsafe_accumulated_luminance;
        const vec3 luminance = 0.995 * accumulated_luminance + 0.005 * this_frame_luminance;
        const vec3 inv_luminance = 1.0 / max(luminance, vec3(1.0 / 1048576.0));
        const float inv_mult = min(1048576.0, max(inv_luminance.x, max(inv_luminance.y, inv_luminance.z)));
        imageStore(daxa_image2DArray(push.ibl_cube), ivec3(sg_pix_pos, gl_WorkGroupID.z), vec4(luminance * inv_mult, 1.0/inv_mult));
    }
}
#endif //CUBEMAP