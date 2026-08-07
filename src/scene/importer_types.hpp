#pragma once

#include <optional>
#include <string>
#include <vector>

#include <daxa/daxa.hpp>

#include "../io/file_io.hpp"
#include "optimizers/image_processor.hpp"
using namespace tido::types;

/// --- Importer types ---
/// What a backend resolves a slot down to: where its bytes are and how to cook them. Editor-side only - a
/// manifest entry holds none of this, and the engine never sees it.

enum struct ComponentType
{
    F32,
    U16,
    U32,
};

// Which of a material's four texture bindings is meant - the role a cook recipe is picked for, and what
// decides which stand-in a slot is bound to until its own image is cooked.
enum struct MaterialTextureSlot
{
    DIFFUSE,
    OPACITY,
    NORMAL,
    ROUGHNESS_METALNESS,
    COUNT,
};

struct MeshAttribSource
{
    SourceLocation location = {};
    ComponentType component_type = {};
};

struct ImageImporterData
{
    // The URI image file, or a bufferView slice (even into a .glb).
    SourceLocation source_location = {};
    ImageFileFormat container_format = {};   // PNG | KTX2
    std::vector<u8> channel_mapping = {};    // cook recipe
    daxa::Format target_format = {};         // cook recipe
};

struct VdbImporterData
{
    // The whole .vdb file: a location with no slice.
    SourceLocation source_location = {};
    std::vector<std::string> grid_names = {};
    std::vector<u8> channel_mapping = {};
    daxa::Format target_format = {};
};

struct MeshImporterData
{
    // Tightly-packed byte ranges for each vertex/index stream.
    MeshAttribSource indices = {};
    MeshAttribSource positions = {};
    MeshAttribSource normals = {};
    std::optional<MeshAttribSource> uvs = {};
    u32 vertex_count = {};   // shared by positions/normals/uvs
    u32 index_count = {};
};
