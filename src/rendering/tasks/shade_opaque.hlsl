#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>
#include "shade_opaque.inl"

#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/debug.glsl"
#include "shader_lib/transform.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/sky_util.glsl"
#include "shader_lib/vsm_util.glsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/volumetric.hlsl"
#include "shader_lib/shading.hlsl"
#include "shader_lib/raytracing.hlsl"
#include "shader_lib/transform.hlsl"



[[vk::push_constant]] ShadeOpaquePush push_opaque;

#define AT deref(push_opaque.attachments).attachments

float compute_exposure(float average_luminance) 
{
    const float exposure_bias = AT.globals->postprocess_settings.exposure_bias;
    const float calibration = AT.globals->postprocess_settings.calibration;
    const float sensor_sensitivity = AT.globals->postprocess_settings.sensor_sensitivity;
    const float ev100 = log2(average_luminance * sensor_sensitivity * exposure_bias / calibration);
	const float exposure = 1.0 / (1.2 * exp2(ev100));
	return exposure;
}

static const uint PCF_NUM_SAMPLES = 8;
// https://developer.download.nvidia.com/whitepapers/2008/PCSS_Integration.pdf
static const float2 poisson_disk[16] = {
    float2( -0.94201624, -0.39906216 ),
    float2( 0.94558609, -0.76890725 ),
    float2( -0.094184101, -0.92938870 ),
    float2( 0.34495938, 0.29387760 ),
    float2( -0.91588581, 0.45771432 ),
    float2( -0.81544232, -0.87912464 ),
    float2( -0.38277543, 0.27676845 ),
    float2( 0.97484398, 0.75648379 ),
    float2( 0.44323325, -0.97511554 ),
    float2( 0.53742981, -0.47373420 ),
    float2( -0.26496911, -0.41893023 ),
    float2( 0.79197514, 0.19090188 ),
    float2( -0.24188840, 0.99706507 ),
    float2( -0.81409955, 0.91437590 ),
    float2( 0.19984126, 0.78641367 ),
    float2( 0.14383161, -0.14100790 )
};

// ndc going in needs to be in range [-1, 1]
float3 get_view_direction(float2 ndc_xy)
{
    float3 world_direction; 
    if(AT.globals->settings.draw_from_observer == 1)
    {
        const float3 camera_position = AT.globals->observer_camera.position;
        const float4 unprojected_pos = mul(AT.globals->observer_camera.inv_view_proj, float4(ndc_xy, 1.0, 1.0));
        world_direction = normalize((unprojected_pos.xyz / unprojected_pos.w) - camera_position);
    }
    else 
    {
        const float3 camera_position = AT.globals->camera.position;
        const float4 unprojected_pos = mul(AT.globals->camera.inv_view_proj, float4(ndc_xy, 1.0, 1.0));
        world_direction = normalize((unprojected_pos.xyz / unprojected_pos.w) - camera_position);
    }
    return world_direction;
}

float map(float value, float min1, float max1, float min2, float max2)
{
    // Convert the current value to a percentage
    // 0% - min1, 100% - max1
    float perc = (value - min1) / (max1 - min1);

    // Do the same operation backwards with min2 and max2
    float value = perc * (max2 - min2) + min2;
    return value;
}

float3 get_vsm_point_debug_page_color(float2 uv, float depth, float3 normal, float3 world_pos)
{
    const uint point_light_index = max(AT.globals.vsm_settings.force_point_light_idx, 0);
    PointMipInfo info = project_into_point_light(depth, normal, point_light_index, uv, AT.globals, AT.vsm_point_lights, AT.vsm_globals);
    if(info.mip_level > 5) 
    {
        return float3(0.0f, 0.0f, 1.0f);
    }

    float3 color = hsv2rgb(float3(float(info.cube_face) / 6.0f, float(5 - int(info.mip_level)) / 5.0f, 1.0));
    const uint point_page_array_index = get_vsm_point_page_array_idx(info.cube_face, point_light_index);
    const uint vsm_page_entry = AT.vsm_point_page_table[info.mip_level].get()[int3(info.page_texel_coords.xy, point_page_array_index)];

    const int2 physical_texel_coords = info.page_uvs * (VSM_TEXTURE_RESOLUTION / (1 << int(info.mip_level)));
    const int2 in_page_texel_coords = int2(_mod(physical_texel_coords, float(VSM_PAGE_SIZE)));

    bool texel_near_border = any(greaterThan(in_page_texel_coords, int2(VSM_PAGE_SIZE - 1))) ||
                             any(lessThan(in_page_texel_coords, int2(1)));


    if(!get_is_allocated(vsm_page_entry)) 
    {
        color = float3(1.0, 0.0, 0.0);
    }

    if(texel_near_border)
    {
        color = float3(0.001, 0.001, 0.001);
    }

    return color;
}

float3 get_vsm_debug_page_color(float2 uv, float depth, float3 world_position)
{
    float3 color = float3(1.0, 1.0, 1.0);

    const float4x4 inv_projection_view = AT.globals->camera.inv_view_proj;
    const bool level_forced = AT.globals->vsm_settings.force_clip_level != 0;
    const int force_clip_level = level_forced ? AT.globals->vsm_settings.forced_clip_level : -1;

    ClipInfo clip_info;
    uint2 render_target_size = AT.globals->settings.render_target_size;
    float real_depth = depth;
    float2 real_uv = uv;
    if(AT.globals->settings.draw_from_observer == 1u)
    {
        const float4 main_cam_proj_world = mul(AT.globals->camera.view_proj, float4(world_position, 1.0));
        const float2 ndc = main_cam_proj_world.xy / main_cam_proj_world.w;
        if(main_cam_proj_world.w < 0.0 || abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0)
        {
            return float3(1.0);
        }
        real_uv = (ndc + float2(1.0)) / float2(2.0);
        real_depth = main_cam_proj_world.z / main_cam_proj_world.w;
    }
    clip_info = clip_info_from_uvs(ClipFromUVsInfo(
        real_uv,
        render_target_size,
        real_depth,
        inv_projection_view,
        force_clip_level,
        AT.vsm_clip_projections,
        AT.vsm_globals,
        AT.globals
    ));
    if(clip_info.clip_level >= VSM_CLIP_LEVELS) { return color; }

    const daxa_i32vec3 vsm_page_pix_coords = daxa_i32vec3(daxa_i32vec2(floor(clip_info.clip_depth_uv * VSM_PAGE_TABLE_RESOLUTION)), clip_info.clip_level);
    const uint is_dynamic_invalidated = unwrap_vsm_page_from_mask(vsm_page_pix_coords, AT.vsm_wrapped_pages);
    const int3 vsm_page_texel_coords = vsm_clip_info_to_wrapped_coords(clip_info, AT.vsm_clip_projections);
    const uint page_entry = Texture2DArray<uint>::get(AT.vsm_page_table).Load(int4(vsm_page_texel_coords, 0)).r;

    if(get_is_allocated(page_entry))
    {
        const int2 physical_page_coords = get_meta_coords_from_vsm_entry(page_entry);
        const int2 physical_texel_coords = virtual_uv_to_physical_texel(clip_info.clip_depth_uv, physical_page_coords);
        uint overdraw_amount = 0;
        if (AT.globals->settings.debug_draw_mode == DEBUG_DRAW_MODE_VSM_OVERDRAW)
        {
            overdraw_amount = RWTexture2D<uint>::get(AT.vsm_overdraw_debug)[physical_texel_coords].x;
        }
        const int2 in_page_texel_coords = int2(_mod(physical_texel_coords, float(VSM_PAGE_SIZE)));
        bool texel_near_border = any(greaterThan(in_page_texel_coords, int2(VSM_PAGE_SIZE - 1))) ||
                                 any(lessThan(in_page_texel_coords, int2(1)));
        if(texel_near_border)
        {
            color = float3(0.001, 0.001, 0.001);
        }
        else if(is_dynamic_invalidated != 0)
        {
            color.rgb = float3(1.0, 0.0, 1.0);
        }
        else
        {
            if(get_is_visited_marked(page_entry)) 
            {
                color.rgb = hsv2rgb(float3(pow(float(vsm_page_texel_coords.z) / float(VSM_CLIP_LEVELS - 1), 0.5), 1.0, 1.0));
            }
            else 
            {
                color.rgb = hsv2rgb(float3(pow(float(vsm_page_texel_coords.z) / float(VSM_CLIP_LEVELS - 1), 0.5), 0.8, 0.2));
            }
        }
        if (AT.globals->settings.debug_draw_mode == DEBUG_DRAW_MODE_VSM_OVERDRAW)
        {
            const float3 overdraw_color = 3.0 * TurboColormap(float(overdraw_amount) / 25.0);
            color.rgb = overdraw_color;
        }
    } else {
        color = float3(1.0, 0.0, 0.0);
        if(get_is_dirty(page_entry)) {color = float3(0.0, 0.0, 1.0);}
    }
    return color;
}

float vsm_shadow_test(ClipInfo clip_info, uint page_entry, float3 world_position, float3 page_camera_position, float sun_norm_dot, float2 screen_uv)
{
    const int2 physical_page_coords = get_meta_coords_from_vsm_entry(page_entry);
    const int2 physical_texel_coords = virtual_uv_to_physical_texel(clip_info.clip_depth_uv, physical_page_coords);

    const float vsm_sample = Texture2D<float>::get(AT.vsm_memory_block).Load(int3(physical_texel_coords, 0)).r;

    const float4x4 vsm_shadow_view = deref_i(AT.vsm_clip_projections, clip_info.clip_level).camera.view;

    float4x4 vsm_shifted_shadow_view = vsm_shadow_view;
    vsm_shifted_shadow_view[0][3] = page_camera_position[0]; 
    vsm_shifted_shadow_view[1][3] = page_camera_position[1]; 
    vsm_shifted_shadow_view[2][3] = page_camera_position[2]; 

    const float4x4 vsm_shadow_proj = deref_i(AT.vsm_clip_projections, clip_info.clip_level).camera.proj;

    const float3 view_projected_world_pos = (mul(vsm_shifted_shadow_view, daxa_f32vec4(world_position, 1.0))).xyz;

    const float view_space_offset = 0.04;// / abs(sun_norm_dot);//0.004 * pow(2.0, clip_info.clip_level);// / max(abs(sun_norm_dot), 0.05);
    const float3 offset_view_pos = float3(view_projected_world_pos.xy, view_projected_world_pos.z + view_space_offset);

    const float4 vsm_projected_world = mul(vsm_shadow_proj, float4(offset_view_pos, 1.0));
    const float vsm_projected_depth = vsm_projected_world.z / vsm_projected_world.w;

    const bool is_in_shadow = vsm_sample < vsm_projected_depth;
    return is_in_shadow ? 0.0 : 1.0;
}

float get_vsm_shadow(float2 uv, float depth, float3 world_position, float sun_norm_dot)
{
    const bool level_forced = AT.globals->vsm_settings.force_clip_level != 0;
    const int force_clip_level = level_forced ? AT.globals->vsm_settings.forced_clip_level : -1;

    const float4x4 inv_projection_view = AT.globals->camera.inv_view_proj;
    uint2 render_target_size = AT.globals->settings.render_target_size;
    ClipInfo clip_info;
    float real_depth = depth;
    float2 real_uv = uv;

    if(AT.globals->settings.draw_from_observer == 1u)
    {
        const float4 main_cam_proj_world = mul(AT.globals->camera.view_proj, float4(world_position, 1.0));
        const float2 ndc = main_cam_proj_world.xy / main_cam_proj_world.w;
        real_uv = (ndc + float2(1.0)) / float2(2.0);
        real_depth = main_cam_proj_world.z / main_cam_proj_world.w;
    }

    let base_clip_info = ClipFromUVsInfo(
        real_uv,
        render_target_size,
        real_depth,
        inv_projection_view,
        force_clip_level,
        AT.vsm_clip_projections,
        AT.vsm_globals,
        AT.globals
    );
    clip_info = clip_info_from_uvs(base_clip_info);
    if(clip_info.clip_level >= VSM_CLIP_LEVELS) { return 1.0; }

    const float filter_radius = 0.01;
    float sum = 0.0;

    rand_seed(asuint(uv.x + uv.y * 13136.1235f));
    float rand_angle = rand();

    for(int sample = 0; sample < PCF_NUM_SAMPLES; sample++)
    {
        let filter_rot_offset = float2(
            poisson_disk[sample].x * cos(rand_angle) - poisson_disk[sample].y * sin(rand_angle),
            poisson_disk[sample].y * cos(rand_angle) + poisson_disk[sample].x * sin(rand_angle),
        );

        let level = 0;
        let filter_view_space_offset = float4(filter_rot_offset * filter_radius * pow(1.0, clip_info.clip_level), 0.0, 0.0);
        const daxa_f32vec3 center_world_space = world_space_from_uv( real_uv, real_depth, inv_projection_view);

        let clip_proj = AT.vsm_clip_projections[clip_info.clip_level].camera.proj;
        let clip_view = AT.vsm_clip_projections[clip_info.clip_level].camera.view;

        let view_space_world_pos = mul(clip_view, float4(world_position, 1.0));
        let view_space_offset_world_pos = view_space_world_pos + filter_view_space_offset;
        let proj_filter_offset_world = mul(clip_proj, view_space_offset_world_pos);

        let clip_uv = ((proj_filter_offset_world.xy / proj_filter_offset_world.w) + 1.0) / 2.0;
        let offset_info = ClipInfo(clip_info.clip_level, clip_uv, 0.0f);

        if(all(greaterThanEqual(offset_info.clip_depth_uv, 0.0)) && all(lessThan(offset_info.clip_depth_uv, 1.0)))
        {
            let vsm_page_texel_coords = vsm_clip_info_to_wrapped_coords(offset_info, AT.vsm_clip_projections);
            let page_entry = Texture2DArray<uint>::get(AT.vsm_page_table).Load(int4(vsm_page_texel_coords, 0)).r;
            const float3 page_camera_pos = Texture2DArray<float3>::get(AT.vsm_page_view_pos_row).Load(int4(vsm_page_texel_coords, 0));

            if(get_is_allocated(page_entry))
            {
                sum += vsm_shadow_test(offset_info, page_entry, world_position, page_camera_pos, sun_norm_dot, uv);
            }
        }
    }
    return sum / PCF_NUM_SAMPLES;
}

float vsm_point_shadow_test(PointMipInfo info, uint vsm_page_entry, float3 world_position, int point_light_idx)
{
        let memory_page_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);

        const int2 physical_texel_coords = int2(info.page_uvs * (VSM_TEXTURE_RESOLUTION / (1 << int(info.mip_level))));
        const int2 in_page_texel_coords = int2(_mod(physical_texel_coords, float(VSM_PAGE_SIZE)));

        const uint2 in_memory_offset = memory_page_coords * VSM_PAGE_SIZE;
        const uint2 memory_texel_coord = in_memory_offset + in_page_texel_coords;


        let shadow_view = AT.vsm_point_lights[point_light_idx].face_cameras[info.cube_face].view;
        let shadow_proj = AT.vsm_point_lights[point_light_idx].face_cameras[info.cube_face].proj;
        const float4 world_position_in_shadow_vs = mul(shadow_view, float4(world_position, 1.0f));
        const float shadow_vs_offset = 0.07;
        const float4 offset_world_position_in_shadow_vs = float4(world_position_in_shadow_vs.xy, world_position_in_shadow_vs.z + shadow_vs_offset, 1.0f);
        const float4 world_position_in_shadow_cs = mul(shadow_proj, offset_world_position_in_shadow_vs);

        const float depth = world_position_in_shadow_cs.z / world_position_in_shadow_cs.w;
        const float vsm_depth = Texture2D<float>::get(AT.vsm_memory_block).Load(int3(memory_texel_coord, 0)).r;

        const bool is_in_shadow = depth < vsm_depth;
        return is_in_shadow ? 0.0f : 1.0f;
}

float get_vsm_point_shadow(float2 screen_uv, float depth, float3 world_normal, float3 world_position, int point_light_idx)
{
    PointMipInfo info = project_into_point_light(depth, world_normal, point_light_idx, screen_uv, AT.globals, AT.vsm_point_lights, AT.vsm_globals);
    if(info.cube_face == -1) 
    {
        return float(1.0f);
    }
    info.mip_level = clamp(info.mip_level, 0, 5);

    const float filter_radius = 0.01;
    float sum = 0.0;

    rand_seed(asuint(screen_uv.x + screen_uv.y * 13136.1235f));
    float rand_angle = rand();

    for(int sample = 0; sample < PCF_NUM_SAMPLES; sample++)
    {
        let filter_rot_offset = float2(
            poisson_disk[sample].x * cos(rand_angle) - poisson_disk[sample].y * sin(rand_angle),
            poisson_disk[sample].y * cos(rand_angle) + poisson_disk[sample].x * sin(rand_angle),
        );

        let level = 0;
        let filter_view_space_offset = float4(filter_rot_offset * filter_radius, 0.0, 0.0);

        let mip_proj = AT.vsm_point_lights[point_light_idx].face_cameras[info.cube_face].proj;
        let mip_view = AT.vsm_point_lights[point_light_idx].face_cameras[info.cube_face].view;

        world_position += world_normal * 0.001;
        let view_space_world_pos = mul(mip_view, float4(world_position, 1.0));
        let view_space_offset_world_pos = view_space_world_pos + filter_view_space_offset;
        let clip_filter_offset_world = mul(mip_proj, view_space_offset_world_pos);

        let clip_uv = ((clip_filter_offset_world.xy / clip_filter_offset_world.w) + 1.0) / 2.0;

        if(all(greaterThanEqual(clip_uv, 0.0)) && all(lessThan(clip_uv, 1.0)))
        {
            const int2 texel_coords = int2(clip_uv * (VSM_PAGE_TABLE_RESOLUTION / (1 << info.mip_level)));
            const uint point_page_array_index = get_vsm_point_page_array_idx(info.cube_face, point_light_idx);
            const uint vsm_page_entry = AT.vsm_point_page_table[info.mip_level].get()[int3(texel_coords, point_page_array_index)];
            info.page_uvs = clip_uv;
            info.page_texel_coords = texel_coords;

            if(get_is_allocated(vsm_page_entry))
            {
                sum += vsm_point_shadow_test(info, vsm_page_entry, world_position, point_light_idx);
            }
        }
        else
        {
            sum += 1.0f;
        }
    }
    return sum / PCF_NUM_SAMPLES;
}

float3 point_lights_contribution(float2 screen_uv, float screen_depth, float3 shading_normal, float3 world_normal, float3 world_position, float3 view_direction, GPUPointLight * lights, uint light_count)
{
    float3 total_contribution = float3(0.0);
    for(int light_index = 0; light_index < light_count; light_index++)
    {
        GPUPointLight light = lights[light_index];
        const float3 position_to_light = normalize(light.position - world_position);
        const float diffuse = max(dot(shading_normal, position_to_light), 0.0);

        const float to_light_dist = length(light.position - world_position);

        float win = (to_light_dist / light.cutoff);
        win = win * win * win * win;
        win = max(0.0, 1.0 - win);
        win = win * win;
        float attenuation = win / (to_light_dist * to_light_dist + 0.1);
        float shadowing = 1.0f;
        if(attenuation > 0.0f) {
            shadowing = get_vsm_point_shadow(screen_uv, screen_depth, world_normal, world_position, light_index);
        }

        total_contribution += light.color * diffuse * attenuation * light.intensity * shadowing;
    }
    return total_contribution;
}

struct VsmLightVisibilityTester : LightVisibilityTesterI
{
    RaytracingAccelerationStructure tlas;
    RenderGlobalData* globals;
    float2 screen_uv;
    float depth;
    float sun_light(MaterialPointData material_point, float3 incoming_ray)
    {
        const float3 sun_direction = AT.globals->sky_settings.sun_direction;
        const float sun_norm_dot = clamp(dot(material_point.normal, sun_direction), 0.0, 1.0);
        float shadow = AT.globals->vsm_settings.enable != 0 ? get_vsm_shadow(screen_uv, depth, material_point.position, sun_norm_dot) : 1.0f;
        const float final_shadow = sun_norm_dot * shadow.x;

        return final_shadow;
    }
    float point_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        return 0.0f;
    }
}

[numthreads(SHADE_OPAQUE_WG_X, SHADE_OPAQUE_WG_Y, 1)]
[shader("compute")]
void entry_main_cs(
    uint3 svdtid : SV_DispatchThreadID
)
{
    let push = push_opaque;

    let clk_start = clockARB();

    if (svdtid.x == 0 && svdtid.y == 0)
    {
        push.attachments.attachments.globals.readback.first_pass_meshlet_count_post_cull = push.attachments.attachments.instantiated_meshlets.pass_counts[0];
        push.attachments.attachments.globals.readback.second_pass_meshlet_count_post_cull = push.attachments.attachments.instantiated_meshlets.pass_counts[1];
    }

    const int2 index = svdtid.xy;
    const float2 screen_uv = (float2(svdtid.xy) + 0.5f) * AT.globals->settings.render_target_size_inv;

    uint triangle_id = INVALID_TRIANGLE_ID;
    if(all(lessThan(index, AT.globals->settings.render_target_size)))
    {
        triangle_id = AT.vis_image.get()[index].x;
    }

    const bool is_center_pixel = all(index == AT.globals->settings.render_target_size/2);
    if (is_center_pixel)
    {
        debug_pixel = true;
    }

    float3 atmo_position = get_atmo_position(AT.globals);

    float4 output_value = float4(0);
    float4 debug_value = float4(0);

    bool triangle_id_valid = triangle_id != INVALID_TRIANGLE_ID;

    float4x4 view_proj;
    float3 camera_position;
    CameraInfo camera = {};
    if(AT.globals->settings.draw_from_observer == 1)
    {
        view_proj = AT.globals->observer_camera.view_proj;
        camera_position = AT.globals->observer_camera.position;
        camera = AT.globals->observer_camera;
    }
    else 
    {
        view_proj = AT.globals->camera.view_proj;
        camera_position = AT.globals->camera.position;
        camera = AT.globals->camera;
    }

    float nonlinear_depth = AT.depth.get()[index];

    let primary_ray = normalize(pixel_index_to_world_space(camera, index, nonlinear_depth) - camera.position);
    float world_space_depth = 0.0f;

    float ambient_occlusion = 1.0f;
    if(triangle_id_valid)
    {
        daxa_BufferPtr(MeshletInstancesBufferHead) instantiated_meshlets = AT.instantiated_meshlets;
        daxa_BufferPtr(GPUMesh) meshes = AT.meshes;
        daxa_BufferPtr(daxa_f32mat4x3) combined_transforms = AT.combined_transforms;
        VisbufferTriangleGeometry visbuf_tri = visgeo_triangle_data(
            triangle_id,
            float2(index),
            push.size,
            push.inv_size,
            view_proj,
            instantiated_meshlets,
            meshes,
            combined_transforms
        );        
        TriangleGeometry tri_geo = visbuf_tri.tri_geo;
        TriangleGeometryPoint tri_point = visbuf_tri.tri_geo_point;
        float depth = visbuf_tri.depth;
        uint meshlet_triangle_index = visbuf_tri.meshlet_triangle_index;
        uint meshlet_instance_index = visbuf_tri.meshlet_instance_index;
        uint meshlet_index = visbuf_tri.meshlet_index;
        tri_point.world_normal = flip_normal_to_incoming(tri_point.face_normal, tri_point.world_normal, primary_ray);
        tri_point.face_normal = flip_normal_to_incoming(tri_point.face_normal, tri_point.face_normal, primary_ray);

        if (is_center_pixel)
        {
            AT.globals.readback.hovered_entity = tri_geo.entity_index;
        }

        world_space_depth = length(tri_point.world_position - camera_position);

        GPUMaterial material = GPU_MATERIAL_FALLBACK;
        if(tri_geo.material_index != INVALID_MANIFEST_INDEX)
        {
            material = AT.material_manifest[tri_geo.material_index];
        }

        MaterialPointData material_point = evaluate_material<SHADING_QUALITY_HIGH>(
            AT.globals,
            tri_geo,
            tri_point
        );
        let mapped_normal = material_point.normal;
        let albedo = material_point.albedo;
        
        const float3 sun_direction = AT.globals->sky_settings.sun_direction;
        const float sun_norm_dot = clamp(dot(mapped_normal, sun_direction), 0.0, 1.0);
        float shadow = AT.globals->vsm_settings.enable != 0 ? get_vsm_shadow(screen_uv, depth, tri_point.world_position, sun_norm_dot) : 1.0f;
        if (AT.globals->vsm_settings.shadow_everything == 1)
        {
            shadow = 0.0f;
        }
        const float final_shadow = sun_norm_dot * shadow.x;

        float3 point_lights_direct = point_lights_contribution(screen_uv, depth, mapped_normal, tri_point.world_normal, tri_point.world_position, primary_ray, AT.point_lights, AT.globals.vsm_settings.point_light_count);

        const float3 directional_light_direct = final_shadow * get_sun_direct_lighting(
            AT.globals, AT.transmittance, AT.sky,
            sun_direction, atmo_position);

        float3 indirect_lighting = {};        
        if (AT.globals.pgi_settings.enabled)
        {
            if (AT.globals.settings.draw_from_observer == 1)
            {
                float3 pgi_irradiance = pgi_sample_irradiance(
                    AT.globals,
                    &AT.globals.pgi_settings,
                    tri_point.world_position,
                    tri_point.world_normal,
                    material_point.normal,
                    primary_ray,
                    AT.pgi_radiance.get(),
                    AT.pgi_visibility.get(),
                    AT.pgi_info.get(),
                    AT.pgi_requests.get(),
                    PGI_PROBE_REQUEST_MODE_NONE
                );
                indirect_lighting = pgi_irradiance;
            }
            else
            {
                float3 pgi_irradiance = push.attachments.attachments.pgi_screen_irrdiance.get()[index].rgb;
                indirect_lighting = pgi_irradiance;
            }
        }
        else
        {
            const float4 compressed_indirect_lighting = TextureCube<float4>::get(AT.sky_ibl).SampleLevel(SamplerState::get(AT.globals->samplers.linear_clamp), mapped_normal, 0);
            indirect_lighting = compressed_indirect_lighting.rgb * compressed_indirect_lighting.a;
        }

        ambient_occlusion = 1.0f;
        const bool ao_enabled = (AT.globals.settings.ao_mode != AO_MODE_NONE) && !AT.ao_image.id.is_empty();
        if (ao_enabled && (AT.globals.settings.draw_from_observer == 0))
        {
            ambient_occlusion = AT.ao_image.get().Load(index);
            ambient_occlusion = pow(ambient_occlusion, 1.1f);
        }

        float3 highlight_lighting = {};
        if (AT.globals.hovered_entity_index == tri_geo.entity_index)
        {
            highlight_lighting = float3(0.2,0.2,0.2) * 5;
        }
        if (AT.globals.selected_entity_index == tri_geo.entity_index)
        {
            highlight_lighting = float3(0.4,0.4,0.4) * 10;
        }
        
        const float3 lighting = directional_light_direct + point_lights_direct + (indirect_lighting.rgb * ambient_occlusion) + material.emissive_color + highlight_lighting;

        let shaded_color = albedo.rgb * lighting;


        float3 dummy_color = float3(1,0,1);
        switch(AT.globals->settings.debug_draw_mode)
        {
            case DEBUG_DRAW_MODE_OVERDRAW:
            {
                if (AT.overdraw_image.value != 0)
                {
                    let value = Texture2D<uint>::get(AT.overdraw_image)[index].x;
                    let scaled_value = float(value) * AT.globals->settings.debug_visualization_scale;
                    let color = TurboColormap(scaled_value);
                    output_value.rgb = color;
                }
                break;
            }
            case DEBUG_DRAW_MODE_TRIANGLE_INSTANCE_ID:
            {
                output_value.rgb = hsv2rgb(float3(IdFloatScramble(meshlet_triangle_index), 1, 1)) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_MESHLET_INSTANCE_ID:
            {
                output_value.rgb = hsv2rgb(float3(IdFloatScramble(meshlet_index), 1, IdFloatScramble(meshlet_triangle_index) * 0.2f + 0.8f)) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_ENTITY_ID:
            {
                output_value.rgb = hsv2rgb(float3(IdFloatScramble(tri_geo.entity_index), 1, 1)) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_MESH_ID:
            {
                output_value.rgb = hsv2rgb(float3(IdFloatScramble(tri_geo.mesh_index), 1, 1)) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_MESH_GROUP_ID:
            {
                uint mesh_group_index = AT.mesh_instances.instances[tri_geo.mesh_instance_index].mesh_group_index;
                output_value.rgb = hsv2rgb(float3(IdFloatScramble(mesh_group_index), 1, 1)) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_MESH_LOD:
            {
                uint lod = tri_geo.mesh_index % MAX_MESHES_PER_LOD_GROUP;
                output_value.rgb = TurboColormap(2 * float(lod) / float(MAX_MESHES_PER_LOD_GROUP)) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_VSM_OVERDRAW: 
            {
                let vsm_debug_color = get_vsm_debug_page_color(screen_uv, depth, tri_point.world_position);
                output_value.rgb = vsm_debug_color;
                break;
            }
            case DEBUG_DRAW_MODE_VSM_CLIP_LEVEL: 
            {
                if (AT.globals->vsm_settings.enable != 0)
                {
                    let vsm_debug_color = get_vsm_debug_page_color(screen_uv, depth, tri_point.world_position) * ambient_occlusion;
                    let debug_albedo = albedo.rgb * lighting * vsm_debug_color;
                    output_value.rgb = debug_albedo;
                }
                break;
            }
            case DEBUG_DRAW_MODE_VSM_POINT_LEVEL:
            {
                let vsm_debug_color = get_vsm_point_debug_page_color(screen_uv, depth, material_point.normal, tri_point.world_position) * ambient_occlusion;
                let debug_albedo = albedo.rgb * lighting * vsm_debug_color.rgb;
                output_value.rgb = debug_albedo;
                break;
            }
            case DEBUG_DRAW_MODE_DEPTH:
            {
                float depth = depth;
                let color = unband_z_color(index.x, index.y, linearise_depth(AT.globals.camera.near_plane, depth));
                output_value.rgb = color;
                break;
            }
            case DEBUG_DRAW_MODE_ALBEDO:
            {
                output_value.rgb = albedo;
                break;
            }
            case DEBUG_DRAW_MODE_FACE_NORMAL:
            {
                let color = tri_point.face_normal * 0.5 + 0.5f;
                output_value.rgb = color;
                break;
            }
            case DEBUG_DRAW_MODE_SMOOTH_NORMAL:
            {
                let color = tri_point.world_normal * 0.5 + 0.5f;
                output_value.rgb = color;
                break;
            }
            case DEBUG_DRAW_MODE_MAPPED_NORMAL:
            {
                let color = mapped_normal * 0.5 + 0.5f;
                output_value.rgb = color;
                break;
            }
            case DEBUG_DRAW_MODE_FACE_TANGENT:
            {
                let color = tri_point.world_tangent * 0.5 + 0.5f;
                output_value.rgb = color;
                break;
            }
            case DEBUG_DRAW_MODE_SMOOTH_TANGENT:
            {
                output_value.rgb = float3(frac(tri_point.uv), 0);
                break;
            }
            case DEBUG_DRAW_MODE_DIRECT_DIFFUSE:
            {
                output_value.rgb = directional_light_direct + point_lights_direct;
                break;
            }
            case DEBUG_DRAW_MODE_INDIRECT_DIFFUSE:
            {
                output_value.rgb = indirect_lighting;
                break;
            }
            case DEBUG_DRAW_MODE_AMBIENT_OCCLUSION:
            {
                output_value.rgb = ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_INDIRECT_DIFFUSE_AO:
            {
                output_value.rgb = indirect_lighting * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_ALL_DIFFUSE:
            {
                output_value.rgb = directional_light_direct + point_lights_direct + indirect_lighting * ambient_occlusion + material.emissive_color;
                break;
            }
            case DEBUG_DRAW_PGI_CASCADE_SMOOTH:
            case DEBUG_DRAW_PGI_CASCADE_ABSOLUTE:
            case DEBUG_DRAW_PGI_CASCADE_SMOOTH_ABS_DIFF:
            {
                uint pgi_absolute_cascade = 0;
                bool pgi_is_center_8 = false;
                bool pgi_checker = false;
                float pgi_color_mul = 1.0f;
                PGISettings * pgi = &AT.globals.pgi_settings;
                for (int cascade = 0; cascade < pgi.cascade_count; ++cascade)
                {
                    let in_cascade = pgi_is_pos_in_cascade(pgi, material_point.position, cascade);
                    if (in_cascade)
                    {
                        pgi_absolute_cascade = cascade;
                        float3 grid_coord = pgi_grid_coord_of_position(pgi, material_point.position, cascade);
                        int4 base_probe = int4(floor(grid_coord), cascade);
                        pgi_is_center_8 = any(base_probe.xyz >= pgi.probe_count/2-1 && base_probe.xyz < pgi.probe_count/2);
                        let stable_index = pgi_probe_to_stable_index(pgi, base_probe);
                        pgi_checker = ((uint(stable_index.x + int(~0u >> 2)) & 0x1) != 0) ^ ((uint(stable_index.y + int(~0u >> 2)) & 0x1) != 0) ^ ((uint(stable_index.z + int(~0u >> 2)) & 0x1) != 0);
                        if (pgi_is_center_8)
                        {
                            pgi_color_mul *= 0.25f;
                        }
                        if (pgi_checker)
                        {
                            pgi_color_mul *= 0.75f;
                        }

                        break;
                    }
                }
                float smooth_cascade = pgi_select_cascade_smooth_spherical(pgi, material_point.position - AT.globals.camera.position);
                switch(AT.globals->settings.debug_draw_mode)
                {
                    case DEBUG_DRAW_PGI_CASCADE_SMOOTH:
                    {
                        float3 cascade_color = TurboColormap(float(smooth_cascade) * rcp(12)) * pgi_color_mul;
                        output_value.rgb = lerp(shaded_color * cascade_color, cascade_color, 0.4f);
                        break;
                    }
                    case DEBUG_DRAW_PGI_CASCADE_ABSOLUTE:
                    {
                        float3 cascade_color = TurboColormap(float(pgi_absolute_cascade) * rcp(12)) * pgi_color_mul;
                        output_value.rgb = lerp(shaded_color * cascade_color, cascade_color, 0.4f);
                        break;
                    }
                    case DEBUG_DRAW_PGI_CASCADE_SMOOTH_ABS_DIFF:
                    {
                        if (smooth_cascade < float(pgi_absolute_cascade))
                        {
                            output_value.rgb = lerp(shaded_color * float3(1,0,0), float3(1,0,0), 0.4f) * pgi_color_mul;
                        }
                        else
                        {
                            float3 cascade_color = TurboColormap(float(smooth_cascade) * rcp(12)) * pgi_color_mul;
                            output_value.rgb = lerp(shaded_color * cascade_color, cascade_color, 0.4f);
                        }
                        break;
                    }
                }
                break;
            }           
            case DEBUG_DRAW_MODE_NONE:
            default:
            output_value.rgb = shaded_color;
            break;
        }
    }
    else 
    {
        if (is_center_pixel)
        {
            AT.globals.readback.hovered_entity = ~0u;
        }
        world_space_depth = VOLUMETRIC_SKY_DEPTH;

        const float2 ndc_xy = screen_uv * 2.0 - 1.0;
        const float3 view_direction = get_view_direction(ndc_xy);
        const float3 atmosphere_direct_illuminnace = get_atmosphere_illuminance_along_ray(
            AT.globals->sky_settings,
            AT.transmittance,
            AT.sky,
            AT.globals->samplers.linear_clamp,
            view_direction,
            atmo_position
        );
        const float3 sun_direct_illuminance = get_sun_direct_lighting(
            AT.globals, AT.transmittance, AT.sky,
            view_direction, atmo_position);
        const float3 total_direct_illuminance = sun_direct_illuminance + atmosphere_direct_illuminnace;
        output_value.rgb = total_direct_illuminance;
        debug_value.xyz = atmosphere_direct_illuminnace;
    }
        
    switch(push.attachments.attachments.globals.settings.debug_draw_mode)
    {
        case DEBUG_DRAW_SHADE_OPAQUE_CLOCKS:
        {
            let clk_end = clockARB();
            output_value.rgb = TurboColormap(float(clk_end - clk_start) * 0.0001f * push.attachments.attachments.globals.settings.debug_visualization_scale) * lerp(ambient_occlusion, 1.0f, 0.5f);
            break;
        }
        case DEBUG_DRAW_PGI_EVAL_CLOCKS:
        case DEBUG_DRAW_RTAO_TRACE_CLOCKS:
        {
            let dgb_img_v = RWTexture2D<float4>::get(push.attachments.attachments.debug_image)[index];
            output_value.rgb = TurboColormap(dgb_img_v.x * 0.0001f * push.attachments.attachments.globals.settings.debug_visualization_scale) * lerp(ambient_occlusion, 1.0f, 0.5f);
            break;
        }
    }

    const float exposure = compute_exposure(deref(AT.luminance_average));
    float3 exposed_color = output_value.rgb * exposure;
    
    AT.color_image.get()[index] = exposed_color;
}