#include <daxa/daxa.inl>
#include "shared.inl"

// Describes how a Mesh is shaded.
struct GPUMaterial
{
    daxa_ImageViewId diffuse;
};
DAXA_DECL_BUFFER_PTR(GPUMaterial)

