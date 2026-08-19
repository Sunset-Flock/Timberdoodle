#include "scene_write.hpp"

#include <fmt/format.h>

auto create_images(Scene & scene, u32 count) -> u32
{
    DBG_ASSERT_TRUE_M(scene._image_manifest.size() + count <= MAX_TEXTURES, "Exceeded MAX_TEXTURES");
    u32 const base = s_cast<u32>(scene._image_manifest.size());
    scene._image_manifest.resize(scene._image_manifest.size() + count);
    return base;
}

auto create_materials(Scene & scene, std::span<MaterialWrite const> materials) -> u32
{
    DBG_ASSERT_TRUE_M(scene._material_manifest.size() + materials.size() <= MAX_MATERIALS, "Exceeded MAX_MATERIALS");
    u32 const base = s_cast<u32>(scene._material_manifest.size());
    for (u32 local_index = 0; local_index < s_cast<u32>(materials.size()); ++local_index)
    {
        MaterialWrite const & material = materials[local_index];
        // Texture bindings are left alone: they are set one slot at a time, and `is_metal` is the engine's.
        MaterialManifestEntry & entry = scene._material_manifest.emplace_back();
        std::visit([&](auto const & write)
        {
            using T = std::decay_t<decltype(write)>;
            if constexpr (std::is_same_v<T, SurfaceMaterialWrite>)
            {
                entry.payload = MaterialManifestEntry::Surface{
                    .alpha_discard_enabled = write.alpha_discard_enabled,
                    .double_sided = write.double_sided,
                    .blend_enabled = write.blend_enabled,
                    .base_color = write.base_color,
                    .emissive_color = write.emissive_color,
                };
            }
            else if constexpr (std::is_same_v<T, CloudMaterialWrite>)
            {
                entry.payload = MaterialManifestEntry::Cloud{
                    .albedo = write.albedo,
                    .density_scale = write.density_scale,
                };
            }
            else
            {
                DBG_ASSERT_TRUE_M(false, "create_materials: material write has no manifest payload");
            }
        }, material.payload);
        entry.name = material.name;
        scene._dirty_material_indices.push_back(base + local_index);
    }
    return base;
}

auto create_mesh_lod_groups(Scene & scene, u32 count) -> u32
{
    DBG_ASSERT_TRUE_M(scene._mesh_lod_group_manifest.size() + count <= MAX_MESH_LOD_GROUPS, "Exceeded MAX_MESH_LOD_GROUPS");
    u32 const base = s_cast<u32>(scene._mesh_lod_group_manifest.size());
    scene._mesh_lod_group_manifest.resize(scene._mesh_lod_group_manifest.size() + count);
    return base;
}

void write_image_name(Scene & scene, u32 image_manifest_index, std::string name)
{
    scene._image_manifest.at(image_manifest_index).name = std::move(name);
}

void set_image_artifact(Scene & scene, u32 image_manifest_index, ImageStreamerData streamer_data)
{
    scene._image_manifest.at(image_manifest_index).streamer_data = std::move(streamer_data);
    scene._dirty_texture_indices.push_back(image_manifest_index);
}

void set_material_texture(Scene & scene, u32 material_manifest_index, MaterialTextureSlot slot,
    std::optional<MaterialManifestEntry::ImageInfo> binding)
{
    if (binding.has_value())
    {
        DBG_ASSERT_TRUE_M(binding->image_manifest_index < scene._image_manifest.size(),
            "Texture binding references an invalid manifest index");
    }

    MaterialManifestEntry & entry = scene._material_manifest.at(material_manifest_index);
    // A slot belongs to exactly one material type, so binding one on a material of another type is a producer
    // bug. Nothing is written in that case, which keeps a rejected bind from leaving a back-reference behind.
    auto expect_surface = [&]() -> MaterialManifestEntry::Surface *
    {
        auto * const surface = std::get_if<MaterialManifestEntry::Surface>(&entry.payload);
        DBG_ASSERT_TRUE_M(surface != nullptr, "set_material_texture: surface slot bound on a material that is not a surface");
        return surface;
    };
    auto expect_cloud = [&]() -> MaterialManifestEntry::Cloud *
    {
        auto * const cloud = std::get_if<MaterialManifestEntry::Cloud>(&entry.payload);
        DBG_ASSERT_TRUE_M(cloud != nullptr, "set_material_texture: cloud slot bound on a material that is not a cloud");
        return cloud;
    };

    bool bound = false;
    // The slot's binding as it was before this call, taken by value: the pointer form would be read after the
    // field it points at has already been overwritten below.
    std::optional<MaterialManifestEntry::ImageInfo> previous_binding = {};
    switch (slot)
    {
        case MaterialTextureSlot::DIFFUSE:             if (auto * m = expect_surface()) { previous_binding = m->diffuse_info; m->diffuse_info = binding;                           bound = true; } break;
        case MaterialTextureSlot::OPACITY:             if (auto * m = expect_surface()) { previous_binding = m->opacity_mask_info; m->opacity_mask_info = binding;                 bound = true; } break;
        case MaterialTextureSlot::NORMAL:              if (auto * m = expect_surface()) { previous_binding = m->normal_info; m->normal_info = binding;                             bound = true; } break;
        case MaterialTextureSlot::ROUGHNESS_METALNESS: if (auto * m = expect_surface()) { previous_binding = m->roughness_metalness_info; m->roughness_metalness_info = binding;   bound = true; } break;
        case MaterialTextureSlot::CLOUD_DATA:          if (auto * m = expect_cloud())   { previous_binding = m->data_info; m->data_info = binding;                                 bound = true; } break;
        case MaterialTextureSlot::CLOUD_SDF:           if (auto * m = expect_cloud())   { previous_binding = m->sdf_info; m->sdf_info = binding;                                   bound = true; } break;
        case MaterialTextureSlot::CLOUD_DETAIL_NOISE:  if (auto * m = expect_cloud())   { previous_binding = m->detail_noise_info; m->detail_noise_info = binding;                 bound = true; } break;
        case MaterialTextureSlot::COUNT:
        default:
            DBG_ASSERT_TRUE_M(false, "set_material_texture: unhandled material texture slot");
            return;
    }
    if (!bound) { return; }

    // Remove the previous binding if there was one.
    if (previous_binding.has_value())
    {
        auto & prev_image_material_bindings = scene._image_manifest.at(previous_binding.value().image_manifest_index).material_manifest_indices;
        auto const found_it = std::find(prev_image_material_bindings.begin(), prev_image_material_bindings.end(), material_manifest_index);
        DBG_ASSERT_TRUE_M(found_it != prev_image_material_bindings.end(), "set_material_texture: previous binding not found in image manifest");
        if(found_it != prev_image_material_bindings.end())
        {
            prev_image_material_bindings.erase(found_it);
        }
    }

    // Let the new image know it is now bound to this material.
    if (binding.has_value())
    {
        scene._image_manifest.at(binding->image_manifest_index).material_manifest_indices.push_back(material_manifest_index);
    }
    scene._dirty_material_indices.push_back(material_manifest_index);
}

void write_mesh_lod_group(Scene & scene, u32 mesh_manifest_index, MeshLodGroupWrite mesh)
{
    // `mesh_group_manifest_index` is the engine's - a group claims its meshes - so it is left alone.
    MeshLodGroupManifestEntry & entry = scene._mesh_lod_group_manifest.at(mesh_manifest_index);
    entry.material_index = mesh.material_manifest_index;
    entry.name = std::move(mesh.name);
    // Dirtied for the GPU manifest sync even with no artifact yet; that uploads a zeroed slot until resident.
    scene._dirty_mesh_lod_group_indices.push_back(mesh_manifest_index);
}

void set_mesh_artifact(Scene & scene, u32 mesh_manifest_index, MeshStreamerData streamer_data)
{
    scene._mesh_lod_group_manifest.at(mesh_manifest_index).streamer_data = std::move(streamer_data);
    scene._dirty_mesh_lod_group_streaming_indices.push_back(mesh_manifest_index);
}

auto create_mesh_groups(Scene & scene, std::span<MeshGroupWrite const> mesh_groups) -> u32
{
    u32 const base = s_cast<u32>(scene._mesh_group_manifest.size());
    for (MeshGroupWrite const & mesh_group : mesh_groups)
    {
        MeshGroupManifestEntry entry = {};
        entry.name = mesh_group.name;
        u32 const mesh_group_manifest_index = s_cast<u32>(scene._mesh_group_manifest.size());

        // Mesh group points to meshes through a contiguous range of indices.
        entry.mesh_lod_group_manifest_indices_array_offset = s_cast<u32>(scene._mesh_lod_group_manifest_indices.size());
        entry.mesh_lod_group_count = s_cast<u32>(mesh_group.mesh_manifest_indices.size());
        for (u32 const mesh_manifest_index : mesh_group.mesh_manifest_indices)
        {
            DBG_ASSERT_TRUE_M(mesh_manifest_index < scene._mesh_lod_group_manifest.size(), "Mesh group references an invalid mesh manifest index");
            scene._mesh_lod_group_manifest_indices.push_back(mesh_manifest_index);
            MeshLodGroupManifestEntry & mesh_lod_group = scene._mesh_lod_group_manifest.at(mesh_manifest_index);
            mesh_lod_group.mesh_group_manifest_index = mesh_group_manifest_index;
            // A mesh can finish streaming before this group exists to claim it; count it as already loaded
            // rather than waiting for a residency event that already happened.
            if (mesh_lod_group.loaded())
            {
                entry.loaded_mesh_lod_groups += 1;
            }
        }
        bool const is_completely_loaded = entry.loaded_mesh_lod_groups == entry.mesh_lod_group_count;
        if (is_completely_loaded)
        {
            scene._newly_completed_mesh_groups.push_back(mesh_group_manifest_index);
        }
        scene._mesh_group_manifest.push_back(std::move(entry));
        scene._dirty_mesh_group_indices.push_back(mesh_group_manifest_index);
    }
    return base;
}

auto create_point_lights(Scene & scene, std::span<PointLightWrite const> lights) -> u32
{
    DBG_ASSERT_TRUE_M(scene._point_lights.size() + lights.size() <= MAX_POINT_LIGHTS, "Maximum point light limit is currently hardcoded");
    u32 const base = s_cast<u32>(scene._point_lights.size());
    for (PointLightWrite const & light : lights)
    {
        u32 const point_light_index = s_cast<u32>(scene._point_lights.size());
        PointLight point_light = {
            .position = light.position,
            .color = light.color,
            .intensity = light.intensity,
            .cutoff = light.cutoff,
            .point_light_ptr = {},
        };
        // point_light_ptr is keyed on the light's global index.
        point_light.point_light_ptr = scene._device.buffer_device_address(scene._gpu_point_lights.id()).value() + point_light_index * sizeof(GPUPointLight);
        scene._point_lights.push_back(point_light);
    }
    return base;
}

auto create_spot_lights(Scene & scene, std::span<SpotLightWrite const> lights) -> u32
{
    DBG_ASSERT_TRUE_M(scene._spot_lights.size() + lights.size() <= MAX_SPOT_LIGHTS, "Maximum spot light limit is currently hardcoded");
    u32 const base = s_cast<u32>(scene._spot_lights.size());
    for (SpotLightWrite const & light : lights)
    {
        u32 const spot_light_index = s_cast<u32>(scene._spot_lights.size());
        SpotLight spot_light = {
            .transform = light.transform,
            .color = light.color,
            .intensity = light.intensity,
            .cutoff = light.cutoff,
            .inner_cone_angle = light.inner_cone_angle,
            .outer_cone_angle = light.outer_cone_angle,
            .spot_light_ptr = {},
        };
        // spot_light_ptr is keyed on the light's global index.
        spot_light.spot_light_ptr = scene._device.buffer_device_address(scene._gpu_spot_lights.id()).value() + spot_light_index * sizeof(GPUSpotLight);
        scene._spot_lights.push_back(spot_light);
    }
    return base;
}

auto create_entity_subtree(Scene & scene, EntitySubtreeWrite subtree) -> std::vector<RenderEntityId>
{
    // The synthetic subtree root is named after the source file by the parse; append the running import count
    // so two sources sharing a stem stay distinguishable.
    subtree.entities.at(subtree.root_entity_index).name += fmt::format("_{}", scene._root_render_entities.size());

    // Entity ids must all exist before the tree's parent/child/sibling links (below) can reference them, so
    // every local entity gets an empty slot up front.
    std::vector<RenderEntityId> entity_local_to_global = {};
    entity_local_to_global.reserve(subtree.entities.size());
    for (u32 local_index = 0; local_index < s_cast<u32>(subtree.entities.size()); ++local_index)
    {
        RenderEntityId const entity_id = scene._render_entities.create_slot({});
        scene._dirty_render_entities.push_back(entity_id);
        entity_local_to_global.push_back(entity_id);
    }
    for (u32 local_index = 0; local_index < s_cast<u32>(subtree.entities.size()); ++local_index)
    {
        EntitySubtreeWrite::Entity & local_entity = subtree.entities[local_index];
        RenderEntity entity = {
            .transform = local_entity.transform,
            .type = local_entity.type,
            .name = std::move(local_entity.name),
        };
        entity.parent = local_entity.parent_index.has_value()
            ? std::optional{entity_local_to_global.at(local_entity.parent_index.value())} : std::nullopt;

        entity.first_child = local_entity.first_child_index.has_value()
            ? std::optional{entity_local_to_global.at(local_entity.first_child_index.value())} : std::nullopt;

        entity.next_sibling = local_entity.next_sibling_index.has_value()
            ? std::optional{entity_local_to_global.at(local_entity.next_sibling_index.value())} : std::nullopt;

        entity.mesh_group_manifest_index = local_entity.mesh_group_manifest_index;
        entity.material_manifest_index = local_entity.material_manifest_index;
        entity.light_index = local_entity.light_index;

        RenderEntityId const entity_id = entity_local_to_global.at(local_index);
        RenderEntity * entity_slot = scene._render_entities.slot(entity_id);
        DBG_ASSERT_TRUE_M(entity_slot != nullptr, "create_entity_subtree: invalid entity id");
        *entity_slot = std::move(entity);
        scene._dirty_render_entities.push_back(entity_id);
    }
    scene._root_render_entities.push_back(entity_local_to_global.at(subtree.root_entity_index));
    return entity_local_to_global;
}
