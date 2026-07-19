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
    FileByteRange range = {};
    ComponentType component_type = {};
};
