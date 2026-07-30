#pragma once

#include "../io/file_io.hpp"
using namespace tido::types;

enum struct ComponentType
{
    F32,
    U16,
    U32,
};

struct MeshAttribSource
{
    SourceLocation location = {};
    ComponentType component_type = {};
};
