#include "scene.hpp"

#include <fstream>
#include <array>
#include <algorithm>

#include <fmt/format.h>
#include <glm/gtx/quaternion.hpp>
#include <thread>
#include <chrono>
#include <ktx.h>
#include <random>
#include "../daxa_helper.hpp"

#include "../shader_shared/raytracing.inl"
#include "../shader_shared/scene.inl"
#include "../rendering/tasks/misc.hpp"

#include "mesh_lod.hpp"

Scene::Scene(daxa::Device device, GPUContext * gpu_context)
    : _device{std::move(device)}, gpu_context{gpu_context}

{
    /// TODO: THIS IS TEMPORARY! Make manifest and entity buffers growable!
    _gpu_entity_meta = tido::make_task_buffer(_device, sizeof(GPUEntityMetaData), "_gpu_entity_meta");
    _gpu_entity_parents = tido::make_task_buffer(_device, sizeof(RenderEntityId) * MAX_ENTITIES, "_gpu_entity_parents");
    _gpu_entity_transforms = tido::make_task_buffer(_device, sizeof(daxa_f32mat4x3) * MAX_ENTITIES, "_gpu_entity_transforms");
    _gpu_entity_combined_transforms = tido::make_task_buffer(_device, sizeof(daxa_f32mat4x3) * MAX_ENTITIES, "_gpu_entity_combined_transforms");
    _gpu_entity_mesh_groups = tido::make_task_buffer(_device, sizeof(u32) * MAX_ENTITIES, "_gpu_entity_mesh_groups");
    _gpu_mesh_manifest = tido::make_task_buffer(_device, sizeof(GPUMesh) * MAX_MESHES, "_gpu_mesh_manifest");
    _gpu_mesh_lod_group_manifest = tido::make_task_buffer(_device, sizeof(GPUMeshLodGroup) * MAX_MESH_LOD_GROUPS, "_gpu_mesh_lod_group_manifest");
    _gpu_mesh_group_manifest = tido::make_task_buffer(_device, sizeof(GPUMeshGroup) * MAX_MESH_LOD_GROUPS, "_gpu_mesh_group_manifest");
    _gpu_material_manifest = tido::make_task_buffer(_device, sizeof(GPUMaterial) * MAX_MATERIALS, "_gpu_material_manifest");
    _gpu_point_lights = tido::make_task_buffer(_device, sizeof(GPUPointLight) * MAX_POINT_LIGHTS, "_gpu_point_lights", daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE);
    _gpu_spot_lights = tido::make_task_buffer(_device, sizeof(GPUSpotLight) * MAX_SPOT_LIGHTS, "_gpu_spot_lights", daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE);
    _gpu_scratch_buffer = tido::make_task_buffer(_device, _gpu_scratch_buffer_size, "_gpu_scratch_buffer");
    _gpu_mesh_acceleration_structure_build_scratch_buffer = tido::make_task_buffer(_device, _gpu_mesh_acceleration_structure_build_scratch_buffer_size, "_gpu_mesh_acceleration_structure_build_scratch_buffer");
    _gpu_tlas_build_scratch_buffer = tido::make_task_buffer(_device, _gpu_tlas_build_scratch_buffer_size, "_gpu_tlas_build_scratch_buffer");
    mesh_instances_buffer = daxa::ExternalTaskBuffer{daxa::ExternalTaskBufferInfo{.name = "mesh_instances"}};
    cloud_volume_instances_buffer = daxa::ExternalTaskBuffer{daxa::ExternalTaskBufferInfo{.name = "cloud_volume_instances"}};
    _scene_as_indirections = tido::make_task_buffer(_device, _indirections_count, "_scene_as_indirections", daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE);
}

Scene::~Scene()
{
    if (!_scene_blas.is_empty()) { _device.destroy_blas(_scene_blas); }

    if (!_gpu_entity_meta.id().is_empty())
    {
        _device.destroy_buffer(_gpu_entity_meta.id());
        _gpu_entity_meta = {};
    }
    if (!_gpu_entity_parents.id().is_empty())
    {
        _device.destroy_buffer(_gpu_entity_parents.id());
        _gpu_entity_parents = {};
    }
    if (!_gpu_entity_transforms.id().is_empty())
    {
        _device.destroy_buffer(_gpu_entity_transforms.id());
        _gpu_entity_transforms = {};
    }
    if (!_gpu_entity_combined_transforms.id().is_empty())
    {
        _device.destroy_buffer(_gpu_entity_combined_transforms.id());
        _gpu_entity_combined_transforms = {};
    }
    if (!_gpu_entity_mesh_groups.id().is_empty())
    {
        _device.destroy_buffer(_gpu_entity_mesh_groups.id());
        _gpu_entity_mesh_groups = {};
    }
    if (!_gpu_mesh_manifest.id().is_empty())
    {
        _device.destroy_buffer(_gpu_mesh_manifest.id());
        _gpu_mesh_manifest = {};
    }
    if (!_gpu_mesh_lod_group_manifest.id().is_empty())
    {
        _device.destroy_buffer(_gpu_mesh_lod_group_manifest.id());
        _gpu_mesh_lod_group_manifest = {};
    }
    if (!_gpu_mesh_group_manifest.id().is_empty())
    {
        _device.destroy_buffer(_gpu_mesh_group_manifest.id());
        _gpu_mesh_group_manifest = {};
    }
    if (!_gpu_material_manifest.id().is_empty())
    {
        _device.destroy_buffer(_gpu_material_manifest.id());
        _gpu_material_manifest = {};
    }
    if (!_gpu_point_lights.id().is_empty())
    {
        _device.destroy_buffer(_gpu_point_lights.id());
        _gpu_point_lights = {};
    }
    if (!_gpu_spot_lights.id().is_empty())
    {
        _device.destroy_buffer(_gpu_spot_lights.id());
        _gpu_spot_lights = {};
    }
    if (!_gpu_scratch_buffer.id().is_empty())
    {
        _device.destroy_buffer(_gpu_scratch_buffer.id());
        _gpu_scratch_buffer = {};
    }
    if (!_gpu_mesh_acceleration_structure_build_scratch_buffer.id().is_empty())
    {
        _device.destroy_buffer(_gpu_mesh_acceleration_structure_build_scratch_buffer.id());
        _gpu_mesh_acceleration_structure_build_scratch_buffer = {};
    }
    if (!_gpu_tlas_build_scratch_buffer.id().is_empty())
    {
        _device.destroy_buffer(_gpu_tlas_build_scratch_buffer.id());
        _gpu_tlas_build_scratch_buffer = {};
    }
    if (!_scene_as_indirections.id().is_empty())
    {
        _device.destroy_buffer(_scene_as_indirections.id());
        _scene_as_indirections = {};
    }

    for (auto & mesh_group : _mesh_group_manifest)
    {
        if (!mesh_group.blas.is_empty())
        {
            _device.destroy_blas(mesh_group.blas);
        }
    }

    for (auto & mesh : _mesh_lod_group_manifest)
    {
        if (mesh.runtime_data.has_value())
        {
            for (daxa_u32 lod = 0; lod < mesh.runtime_data.value().lod_count; ++lod)
            {
                _device.destroy_buffer(std::bit_cast<daxa::BufferId>(mesh.runtime_data.value().lods[lod].mesh_buffer));
                if (!mesh.runtime_data.value().blas_lods[lod].is_empty())
                {
                    _device.destroy_blas(mesh.runtime_data.value().blas_lods[lod]);
                }
            }
        }
    }

    for (auto & texture : _texture_manifest)
    {
        if (texture.runtime_data.image.has_value())
        {
            _device.destroy_image(std::bit_cast<daxa::ImageId>(texture.runtime_data.image.value()));
        }
    }

    if (!mesh_instances_buffer.id().is_empty())
    {
        _device.destroy_buffer(mesh_instances_buffer.id());
    }

    if (!cloud_volume_instances_buffer.id().is_empty())
    {
        _device.destroy_buffer(cloud_volume_instances_buffer.id());
    }
}
void Scene::start_async_loads_of_dirty_cloud_volumes(AssetProcessor * asset_processor, ThreadPool * thread_pool)
{
    struct LoadCloudVolumeTask : Task
    {
        struct TaskInfo
        {
            AssetProcessor * asset_processor = {};
            CloudVolume const * volume;
        };

        TaskInfo info = {};

        LoadCloudVolumeTask(TaskInfo const & info)
            : info{info}
        {
            chunk_count = 1;
        }

        virtual void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override{
            {AssetProcessor::LoadCloudVolumetricDataInfo load_data_info = {
                .volumetric_data_path = std::filesystem::path(info.volume -> cloud_volume_data_path),
                .cloud_data_texture_manifest_index = info.volume->data_texture_manifest_index,
                .cloud_sdf_texture_manifest_index = info.volume->sdf_texture_manifest_index,
            };

        auto const ret_status = info.asset_processor->load_cloud_volumetric_data(load_data_info);
        if (ret_status != AssetProcessor::AssetLoadResultCode::SUCCESS)
        {
            DEBUG_MSG(fmt::format("[ERROR] Failed to load cloud volume {} - error {}",
                info.volume->cloud_volume_data_path, AssetProcessor::to_string(ret_status)));
        }
        else
        {
            DEBUG_MSG(fmt::format("[SUCCESS] Successfuly loaded cloud volume {}", info.volume->cloud_volume_data_path));
        }
    }

    {
        AssetProcessor::LoadCloudVolumetricDataInfo load_data_info = {
            .volumetric_data_path = std::filesystem::path(info.volume->detail_noise_path),
            .cloud_data_texture_manifest_index = info.volume->detail_noise_texture_manifest_index,
            .cloud_sdf_texture_manifest_index = std::numeric_limits<u32>::max(), // Currently should be unused.
        };
        auto const ret_status = info.asset_processor->load_cloud_volumetric_data(load_data_info);
        if (ret_status != AssetProcessor::AssetLoadResultCode::SUCCESS)
        {
            DEBUG_MSG(fmt::format("[ERROR] Failed to load cloud volume {} - error {}",
                info.volume->detail_noise_path, AssetProcessor::to_string(ret_status)));
        }
        else
        {
            DEBUG_MSG(fmt::format("[SUCCESS] Successfuly loaded cloud volume {}", info.volume->detail_noise_path));
        }
    }
};
}
;

for (u32 cloud_volume_manifest_index : _cloud_volumes_requesting_load)
{
    // Launch loading of this cloud volume
    auto task = std::make_shared<LoadCloudVolumeTask>(LoadCloudVolumeTask::TaskInfo{
        .asset_processor = asset_processor,
        .volume = &_cloud_volumes.at(cloud_volume_manifest_index),
    });
    thread_pool->async_dispatch(task, TaskPriority::LOW);
}
_cloud_volumes_requesting_load.clear();
}

void Scene::build_tlas_from_mesh_instances(daxa::CommandRecorder & recorder, daxa::TlasId tlas)
{
    auto & mesh_instances = this->current_frame_mesh_instances;

    std::vector<daxa_BlasInstanceData> blas_instances = {};
    blas_instances.reserve(mesh_instances.mesh_instances.size());
    for (u32 mesh_inst_i = 0; mesh_inst_i < mesh_instances.mesh_instances.size(); ++mesh_inst_i)
    {
        MeshInstance const & mesh_instance = mesh_instances.mesh_instances[mesh_inst_i];
        auto const lod = mesh_instance.mesh_index % MAX_MESHES_PER_LOD_GROUP;
        auto const lod_group = mesh_instance.mesh_index / MAX_MESHES_PER_LOD_GROUP;

        if (!_mesh_lod_group_manifest[lod_group].runtime_data.has_value()) { continue; }
        if (_mesh_lod_group_manifest[lod_group].runtime_data.value().blas_lods[lod].is_empty()) { continue; }

        RenderEntity const * render_entity = _render_entities.slot_by_index(mesh_instance.entity_index);
        auto const & t = render_entity->combined_transform;
        blas_instances.push_back(daxa_BlasInstanceData{
            .transform = {
                {t[0][0], t[1][0], t[2][0], t[3][0]},
                {t[0][1], t[1][1], t[2][1], t[3][1]},
                {t[0][2], t[1][2], t[2][2], t[3][2]},
            },
            .instance_custom_index = mesh_inst_i,
            .mask = 0xFF,
            .instance_shader_binding_table_record_offset = ((mesh_instance.flags & MESH_INSTANCE_FLAG_MASKED) != 0) ? 1u : 0u,
            .flags = 0,
            .blas_device_address = _device.blas_device_address(_mesh_lod_group_manifest[lod_group].runtime_data.value().blas_lods[lod]).value(),
        });
    }

    daxa::BufferId blas_instances_buffer = {};
    if (!blas_instances.empty())
    {
        blas_instances_buffer = _device.create_buffer({.size = sizeof(daxa_BlasInstanceData) * blas_instances.size(),
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
            .name = "blas instances buffer"});
        recorder.destroy_buffer_deferred(blas_instances_buffer);

        std::memcpy(_device.buffer_host_address_as<daxa_BlasInstanceData>(blas_instances_buffer).value(),
            blas_instances.data(),
            blas_instances.size() * sizeof(daxa_BlasInstanceData));
    }
    else
    {
        return;
    }

    auto tlas_blas_instances_infos = std::array{daxa::TlasInstanceInfo{
        .data = blas_instances.empty() ? daxa::DeviceAddress{0ull} : _device.buffer_device_address(blas_instances_buffer).value(),
        .count = s_cast<u32>(blas_instances.size()),
        .is_data_array_of_pointers = false,
        .flags = daxa::GeometryFlagBits::NONE,
    }};

    auto tlas_build_info = daxa::TlasBuildInfo{
        .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE,
        .instances = tlas_blas_instances_infos,
    };

    daxa::AccelerationStructureBuildSizesInfo const tlas_build_sizes = _device.tlas_build_sizes(tlas_build_info);

    DAXA_DBG_ASSERT_TRUE_M(tlas_build_sizes.acceleration_structure_size <= _device.info(tlas).value().size, "Tlas Size Overflow");

    DBG_ASSERT_TRUE_M(tlas_build_sizes.build_scratch_size < _gpu_tlas_build_scratch_buffer_size,
        "[ERROR][Scene::build_tlas_from_mesh_instances] Tlas too big for scratch buffer - create bigger scratch buffer");

    daxa::DeviceAddress scratch_device_address = _device.buffer_device_address(_gpu_tlas_build_scratch_buffer.id()).value();
    tlas_build_info.dst_tlas = tlas;
    tlas_build_info.scratch_data = scratch_device_address;

    recorder.build_acceleration_structures({.tlas_build_infos = std::array{tlas_build_info}});
    recorder.pipeline_barrier({
        .src_access = daxa::AccessConsts::ACCELERATION_STRUCTURE_BUILD_READ_WRITE,
        .dst_access = daxa::AccessConsts::READ_WRITE,
    });
}

auto Scene::process_entities(RenderGlobalData & render_data) -> CPUSceneInstances
{
    CPUSceneInstances ret = {};

    auto * const gpu_point_lights_write_ptr = _device.buffer_host_address_as<GPUPointLight>(_gpu_point_lights.id()).value();
    auto * const gpu_spot_lights_write_ptr = _device.buffer_host_address_as<GPUSpotLight>(_gpu_spot_lights.id()).value();

    for (u32 entity_i = 0; entity_i < _render_entities.capacity(); ++entity_i)
    {
        RenderEntity * r_ent = _render_entities.slot_by_index(entity_i);
        if (r_ent == nullptr) { continue; }
        bool const is_entity_dirty = r_ent->dirty;
        r_ent->dirty = false;

        if (r_ent->cloud_volume_index.has_value())
        {
            DBG_ASSERT_TRUE_M(r_ent->type == EntityType::CLOUD_VOLUME, "IMPOSSIBLE CASE! Only cloud volume entities can have cloud volume index");
            auto const & cloud_volume = _cloud_volumes.at(r_ent->cloud_volume_index.value());

            // ===================================== Calculate instance AABB =======================================
            {
                const f32vec3 cloud_bottom_left_corner = s_cast<f32vec3>((mat_4x3_to_4x4(r_ent->combined_transform) * f32vec4(0.0f, 0.0f, 0.0f, 1.0f)));
                const f32vec3 cloud_top_right_corner = s_cast<f32vec3>((mat_4x3_to_4x4(r_ent->combined_transform) * f32vec4(1.0f, 1.0f, 1.0f, 1.0f)));

                f32vec3 const instance_aabb_size = cloud_top_right_corner - cloud_bottom_left_corner;
                f32vec3 const instance_aabb_center = cloud_bottom_left_corner + (instance_aabb_size * f32vec3(0.5f));
                ret.cloud_volume_instances.instance_aabbs.push_back({
                    .center = std::bit_cast<daxa_f32vec3>(instance_aabb_center),
                    .size = std::bit_cast<daxa_f32vec3>(instance_aabb_size),
                });
            }

            // ===================================== Fill out cloud instance data =======================================
            {
                CloudVolumeInstance cloud_volume_instance = {};
                cloud_volume_instance.transform = std::bit_cast<daxa_f32mat4x3>(r_ent->combined_transform);
                cloud_volume_instance.albedo = 1.0f;
                cloud_volume_instance.density_scale = 0.1f;

                cloud_volume_instance.cloud_data_texture = _texture_manifest.at(cloud_volume.data_texture_manifest_index).runtime_data.image.value_or(daxa::ImageId{}).default_view();
                cloud_volume_instance.cloud_sdf_texture = _texture_manifest.at(cloud_volume.sdf_texture_manifest_index).runtime_data.image.value_or(daxa::ImageId{}).default_view();
                cloud_volume_instance.detail_noise_texture = _texture_manifest.at(cloud_volume.detail_noise_texture_manifest_index).runtime_data.image.value_or(daxa::ImageId{}).default_view();

                cloud_volume_instance.texture_size = {0u, 0u, 0u};
                if(_texture_manifest.at(cloud_volume.data_texture_manifest_index).loaded())
                {
                    daxa::ImageId cloud_data_texture = _texture_manifest.at(cloud_volume.data_texture_manifest_index).runtime_data.image.value();
                    daxa::ImageInfo const & cloud_data_texture_info = _device.image_info(cloud_data_texture).value();
                    cloud_volume_instance.texture_size = {cloud_data_texture_info.size.x, cloud_data_texture_info.size.y, cloud_data_texture_info.size.z};
                }
                ret.cloud_volume_instances.instances.push_back(cloud_volume_instance);
            }

        }

        if (r_ent->light_index.has_value())
        {
            if (r_ent->type == EntityType::POINT_LIGHT)
            {
                PointLight & point_light = _point_lights.at(r_ent->light_index.value());
                point_light.position = r_ent->combined_transform[3];

                gpu_point_lights_write_ptr[r_ent->light_index.value()] = GPUPointLight{
                    .position = std::bit_cast<daxa_f32vec3>(point_light.position),
                    .color = std::bit_cast<daxa_f32vec3>(point_light.color),
                    .intensity = point_light.intensity,
                    .cutoff = point_light.cutoff,
                };
            }
            else if (r_ent->type == EntityType::SPOT_LIGHT)
            {
                SpotLight & spot_light = _spot_lights.at(r_ent->light_index.value());
                spot_light.transform = r_ent->combined_transform;

                glm::mat4 transform4 = glm::mat4(
                    glm::vec4(spot_light.transform[0], 0.0f),
                    glm::vec4(spot_light.transform[1], 0.0f),
                    glm::vec4(spot_light.transform[2], 0.0f),
                    glm::vec4(spot_light.transform[3], 1.0f));
                f32vec3 const spot_direction = transform4 * f32vec4(0.0f, 0.0f, -1.0f, 0.0f);

                gpu_spot_lights_write_ptr[r_ent->light_index.value()] = GPUSpotLight{
                    .transform = std::bit_cast<daxa_f32mat4x3>(spot_light.transform),
                    .position = std::bit_cast<daxa_f32vec3>(spot_light.transform[3]),
                    .direction = std::bit_cast<daxa_f32vec3>(spot_direction),
                    .color = std::bit_cast<daxa_f32vec3>(spot_light.color),
                    .intensity = spot_light.intensity,
                    .cutoff = spot_light.cutoff,
                    .inner_cone_angle = spot_light.inner_cone_angle,
                    .outer_cone_angle = spot_light.outer_cone_angle,
                };
            }
        }

        if (r_ent->mesh_group_manifest_index.has_value())
        {
            usize mesh_group_index = r_ent->mesh_group_manifest_index.value();
            MeshGroupManifestEntry & mesh_group = _mesh_group_manifest.at(mesh_group_index);
            bool const is_mesh_group_loaded = (mesh_group.loaded_mesh_lod_groups == mesh_group.mesh_lod_group_count);
            bool const is_mesh_group_just_loaded = !mesh_group.fully_loaded_last_frame && is_mesh_group_loaded;
            mesh_group.fully_loaded_last_frame = is_mesh_group_loaded;

            // Process all fully loaded mesh groups
            if (is_mesh_group_loaded)
            {
                auto const mesh_lod_group_indices_meshgroup_offset = mesh_group.mesh_lod_group_manifest_indices_array_offset;
                for (u32 in_mesh_group_index = 0; in_mesh_group_index < mesh_group.mesh_lod_group_count; in_mesh_group_index++)
                {
                    u32 const mesh_lod_group_manifest_index = _mesh_lod_group_manifest_indices.at(mesh_lod_group_indices_meshgroup_offset + in_mesh_group_index);
                    auto const & mesh_lod_group = _mesh_lod_group_manifest.at(mesh_lod_group_manifest_index);
                    bool is_alpha_discard = false;
                    bool is_blend = false;
                    if (mesh_lod_group.material_index.has_value())
                    {
                        auto const & material = _material_manifest.at(mesh_lod_group.material_index.value());
                        is_alpha_discard = material.alpha_discard_enabled;
                        is_blend = material.blend_enabled;
                    }

                    // Put this mesh into appropriate drawlist for prepass
                    u32 const draw_list_type = is_alpha_discard ? PREPASS_DRAW_LIST_MASKED : PREPASS_DRAW_LIST_OPAQUE;

                    ret.mesh_instances.prepass_draw_lists[draw_list_type].push_back(static_cast<u32>(ret.mesh_instances.mesh_instances.size()));

                    // If the mesh loaded for the first time, it needs to invalidate VSM
                    if (is_mesh_group_just_loaded || is_entity_dirty)
                    {
                        ret.mesh_instances.vsm_invalidate_draw_list.push_back(static_cast<u32>(ret.mesh_instances.mesh_instances.size()));
                    }

                    u32 mesh_index = select_lod(render_data, mesh_lod_group, mesh_lod_group_manifest_index, r_ent);

                    // Because this mesh will be referenced by the prepass drawlist, we need also need it's appropriate mesh instance data
                    ret.mesh_instances.mesh_instances.push_back({
                        .entity_index = entity_i,
                        .mesh_index = mesh_index,
                        .in_mesh_group_index = in_mesh_group_index,
                        .mesh_group_index = static_cast<u32>(mesh_group_index),
                        .flags = s_cast<daxa_u32>(is_alpha_discard ? MESH_INSTANCE_FLAG_MASKED : MESH_INSTANCE_FLAG_OPAQUE),
                    });
                }
            }
        }
    }
    return ret;
}

void Scene::write_gpu_mesh_instances_buffer(CPUMeshInstances const & cpu_mesh_instances)
{
    // Calculate offsets into buffer and required size:
    usize offset = {};
    MeshInstancesBufferHead buffer_head = {};
    offset += sizeof(MeshInstancesBufferHead);
    buffer_head.instances = offset;
    buffer_head.count = static_cast<u32>(cpu_mesh_instances.mesh_instances.size());
    offset += sizeof(MeshInstance) * cpu_mesh_instances.mesh_instances.size();
    for (int i = 0; i < PREPASS_DRAW_LIST_TYPE_COUNT; ++i)
    {
        buffer_head.prepass_draw_lists[i].instances = offset;
        buffer_head.prepass_draw_lists[i].count = static_cast<u32>(cpu_mesh_instances.prepass_draw_lists[i].size());
        offset += sizeof(daxa_u32) * cpu_mesh_instances.prepass_draw_lists[i].size();
    }
    buffer_head.vsm_invalidate_draw_list.instances = offset;
    buffer_head.vsm_invalidate_draw_list.count = static_cast<u32>(cpu_mesh_instances.vsm_invalidate_draw_list.size());
    offset += sizeof(daxa_u32) * cpu_mesh_instances.vsm_invalidate_draw_list.size();
    usize const required_size = offset;

    // TODO: Allocate this into a ring buffer.
    // Allocate buffer
    if (!mesh_instances_buffer.id().is_empty())
    {
        _device.destroy_buffer(mesh_instances_buffer.id());
    }
    mesh_instances_buffer.set_buffer(_device.create_buffer({
        .size = required_size,
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "cpu_mesh_instances_buffer",
    }));
    daxa::DeviceAddress device_address = _device.buffer_device_address(mesh_instances_buffer.id()).value();
    std::byte * host_address = _device.buffer_host_address(mesh_instances_buffer.id()).value();

    // Write Buffer, add address on offsets and counts:
    cpu_mesh_instance_counts = {};
    usize const mesh_instances_size = sizeof(MeshInstance) * cpu_mesh_instances.mesh_instances.size();
    std::memcpy(host_address + buffer_head.instances, cpu_mesh_instances.mesh_instances.data(), mesh_instances_size);

    buffer_head.instances += device_address;
    cpu_mesh_instance_counts.mesh_instance_count = s_cast<u32>(cpu_mesh_instances.mesh_instances.size());
    for (int draw_list_type = 0; draw_list_type < PREPASS_DRAW_LIST_TYPE_COUNT; ++draw_list_type)
    {
        std::memcpy(
            host_address + buffer_head.prepass_draw_lists[draw_list_type].instances,
            cpu_mesh_instances.prepass_draw_lists[draw_list_type].data(),
            sizeof(u32) * cpu_mesh_instances.prepass_draw_lists[draw_list_type].size());
        buffer_head.prepass_draw_lists[draw_list_type].instances += device_address;
        cpu_mesh_instance_counts.prepass_instance_counts[draw_list_type] = s_cast<u32>(cpu_mesh_instances.prepass_draw_lists[draw_list_type].size());
    }
    std::memcpy(
        host_address + buffer_head.vsm_invalidate_draw_list.instances,
        cpu_mesh_instances.vsm_invalidate_draw_list.data(),
        sizeof(u32) * cpu_mesh_instances.vsm_invalidate_draw_list.size());
    buffer_head.vsm_invalidate_draw_list.instances += device_address;

    std::memcpy(host_address, &buffer_head, sizeof(MeshInstancesBufferHead));

    cpu_mesh_instance_counts.vsm_invalidate_instance_count = s_cast<u32>(cpu_mesh_instances.vsm_invalidate_draw_list.size());
}

void Scene::write_gpu_cloud_volume_instances_buffer(CPUCloudVolumeInstaces const & cpu_cloud_volume_instances)
{
    DBG_ASSERT_TRUE_M(cpu_cloud_volume_instances.instance_aabbs.size() == cpu_cloud_volume_instances.instances.size(),
                      "Each cloud volume instance must have a corresponding AABB");

    u32 const instance_count = s_cast<u32>(cpu_cloud_volume_instances.instances.size());

    // Calculate offsets into buffer and required size:
    // The offsets are relative to the start of the buffer.
    // Later they are added to the device address of the buffer to form the actual location.
    usize offset = {};
    CloudVolumeInstancesBufferHead buffer_head = {};
    offset += sizeof(CloudVolumeInstancesBufferHead);

    buffer_head.count = instance_count;

    buffer_head.instance_aabbs = offset;
    offset += sizeof(AABB) * instance_count;

    buffer_head.instances = offset;
    offset += sizeof(CloudVolumeInstance) * instance_count;
    usize const required_size = offset;

    // TODO: Allocate this into a ring buffer.
    // Allocate buffer
    if (!cloud_volume_instances_buffer.id().is_empty())
    {
        _device.destroy_buffer(cloud_volume_instances_buffer.id());
    }
    cloud_volume_instances_buffer.set_buffer(_device.create_buffer({
        .size = required_size,
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "cpu_cloud_volume_instances_buffer",
    }));
    daxa::DeviceAddress device_address = _device.buffer_device_address(cloud_volume_instances_buffer.id()).value();
    std::byte * host_address = _device.buffer_host_address(cloud_volume_instances_buffer.id()).value();

    // Write Buffer, add address on offsets and counts:
    usize const cloud_volume_aabbs_size = sizeof(AABB) * instance_count;
    std::memcpy(host_address + buffer_head.instance_aabbs, cpu_cloud_volume_instances.instance_aabbs.data(), cloud_volume_aabbs_size);

    usize const cloud_volume_instances_size = sizeof(CloudVolumeInstance) * instance_count;
    std::memcpy(host_address + buffer_head.instances, cpu_cloud_volume_instances.instances.data(), cloud_volume_instances_size);

    buffer_head.instance_aabbs += device_address;
    buffer_head.instances += device_address;
    std::memcpy(host_address, &buffer_head, sizeof(CloudVolumeInstancesBufferHead));
}

void Scene::clear(std::unique_ptr<ThreadPool> & thread_pool, std::unique_ptr<AssetProcessor> & asset_processor)
{
    // WARNING: Currently unused (no call site anywhere). Before wiring this up (e.g. an "unload
    // scene" button), it must refuse to run - or wait - while a scene import (SceneRuntime's pending
    // import) is in flight: SceneRuntime applying an import batch and any in-flight stream task would
    // publish into manifest indices this wipes.

    // for (auto &task : scene_load_tasks)
    // {
    //     thread_pool->block_on(task);
    // }
    asset_processor->clear();

    // NOTE(grundlett): Destroy all GPU resources (from destructor)
    {
        if (!_gpu_mesh_group_indices_array_buffer.is_empty()) { _device.destroy_buffer(_gpu_mesh_group_indices_array_buffer); }
        if (!_scene_blas.is_empty()) { _device.destroy_blas(_scene_blas); }

        for (auto & mesh_group : _mesh_group_manifest)
        {
            if (!mesh_group.blas.is_empty())
            {
                _device.destroy_blas(mesh_group.blas);
            }
        }

        for (auto & mesh : _mesh_lod_group_manifest)
        {
            if (mesh.runtime_data.has_value())
            {
                for (daxa_u32 lod = 0; lod < mesh.runtime_data.value().lod_count; ++lod)
                {
                    _device.destroy_buffer(std::bit_cast<daxa::BufferId>(mesh.runtime_data.value().lods[lod].mesh_buffer));
                    if (!mesh.runtime_data.value().blas_lods[lod].is_empty())
                    {
                        _device.destroy_blas(mesh.runtime_data.value().blas_lods[lod]);
                    }
                }
            }
        }

        for (auto & texture : _texture_manifest)
        {
            if (texture.runtime_data.image.has_value())
            {
                _device.destroy_image(std::bit_cast<daxa::ImageId>(texture.runtime_data.image.value()));
            }
            // if (texture.secondary_runtime_texture.has_value())
            // {
            //     _device.destroy_image(std::bit_cast<daxa::ImageId>(texture.secondary_runtime_texture.value()));
            // }
        }

        if (!mesh_instances_buffer.id().is_empty())
        {
            _device.destroy_buffer(mesh_instances_buffer.id());
        }
        mesh_instances_buffer.set_buffer({});
    }

    // NOTE(grundlett): clear all state
    {
        _render_entities.clear();
        _dirty_render_entities.clear();
        _newly_completed_mesh_groups.clear();

        _root_render_entities.clear();
        _texture_manifest.clear();
        _material_manifest.clear();
        _mesh_lod_group_manifest.clear();
        _mesh_lod_group_manifest_indices.clear();
        _mesh_group_manifest.clear();
        _point_lights.clear();
        _spot_lights.clear();

        // Discard any pending dirty indices; the manifests they referred to are gone.
        _dirty_material_indices.clear();
        _dirty_mesh_lod_group_indices.clear();
        _dirty_mesh_group_indices.clear();
        _dirty_texture_indices.clear();
        _dirty_mesh_lod_group_streaming_indices.clear();
    }
}
