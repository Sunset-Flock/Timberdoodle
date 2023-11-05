#pragma once

// #include <assimp/Importer.hpp>
// #include <assimp/mesh.h>
// #include <assimp/scene.h>
// #include <assimp/postprocess.h>
// #include <assimp/IOStream.hpp>
// #include <assimp/IOSystem.hpp>

#include <meshoptimizer.h>

#include "../sandbox.hpp"
#include "../rendering/gpu_context.hpp"
#include "../../shader_shared/asset.inl"
#include "scene.hpp"

using MeshIndex = size_t;
using ImageIndex = size_t;

#define MAX_MESHES 10000

// inline std::string generate_mesh_name(aiMesh* mesh)
// {
//     return
//         std::string(mesh->mName.C_Str()) + std::string(" m:") + std::to_string(mesh->mMaterialIndex);
// }
//
// inline std::string generate_texture_name(aiMaterial * material, aiTextureType type)
// {
//     return std::string(material->GetName().C_Str()) + aiTextureTypeToString(type);
// }

struct AssetProcessor
{
    enum struct AssetLoadResultCode
    {
        SUCCESS,
        ERROR_MISSING_INDEX_BUFFER,
        ERROR_FAULTY_INDEX_BUFFER_GLTF_ACCESSOR,
        ERROR_FAULTY_BUFFER_VIEW,
        ERROR_COULD_NOT_OPEN_GLTF,
        ERROR_COULD_NOT_READ_BUFFER_IN_GLTF,
        ERROR_COULD_NOT_OPEN_TEXTURE_FILE,
        ERROR_COULD_NOT_READ_TEXTURE_FILE,
        ERROR_COULD_NOT_READ_TEXTURE_FILE_FROM_MEMSTREAM,
        ERROR_UNSUPPORTED_TEXTURE_PIXEL_FORMAT,
        ERROR_UNKNOWN_FILETYPE_FORMAT,
        ERROR_UNSUPPORTED_READ_FOR_FILEFORMAT,
        ERROR_URI_FILE_OFFSET_NOT_SUPPORTED,
        ERROR_UNSUPPORTED_ABSOLUTE_PATH,
        ERROR_MISSING_VERTEX_POSITIONS,
        ERROR_FAULTY_GLTF_VERTEX_POSITIONS,
    };
    static auto to_string(AssetLoadResultCode code) -> std::string_view
    {
        switch(code)
        {
            case AssetLoadResultCode::SUCCESS:                                          return "SUCCESS";
            case AssetLoadResultCode::ERROR_MISSING_INDEX_BUFFER:                       return "ERROR_MISSING_INDEX_BUFFER";
            case AssetLoadResultCode::ERROR_FAULTY_INDEX_BUFFER_GLTF_ACCESSOR:          return "ERROR_FAULTY_INDEX_BUFFER_GLTF_ACCESSOR";
            case AssetLoadResultCode::ERROR_FAULTY_BUFFER_VIEW:                         return "ERROR_FAULTY_BUFFER_VIEW";
            case AssetLoadResultCode::ERROR_COULD_NOT_OPEN_GLTF:                        return "ERROR_COULD_NOT_OPEN_GLTF";
            case AssetLoadResultCode::ERROR_COULD_NOT_READ_BUFFER_IN_GLTF:              return "ERROR_COULD_NOT_READ_BUFFER_IN_GLTF";
            case AssetLoadResultCode::ERROR_COULD_NOT_OPEN_TEXTURE_FILE:                return "ERROR_COULD_NOT_OPEN_TEXTURE_FILE";
            case AssetLoadResultCode::ERROR_COULD_NOT_READ_TEXTURE_FILE:                return "ERROR_COULD_NOT_READ_TEXTURE_FILE";
            case AssetLoadResultCode::ERROR_COULD_NOT_READ_TEXTURE_FILE_FROM_MEMSTREAM: return "ERROR_COULD_NOT_READ_TEXTURE_FILE_FROM_MEMSTREAM";
            case AssetLoadResultCode::ERROR_UNSUPPORTED_TEXTURE_PIXEL_FORMAT:           return "ERROR_UNSUPPORTED_TEXTURE_PIXEL_FORMAT";
            case AssetLoadResultCode::ERROR_UNKNOWN_FILETYPE_FORMAT:                    return "ERROR_UNKNOWN_FILETYPE_FORMAT";
            case AssetLoadResultCode::ERROR_UNSUPPORTED_READ_FOR_FILEFORMAT:            return "ERROR_UNSUPPORTED_READ_FOR_FILEFORMAT";
            case AssetLoadResultCode::ERROR_URI_FILE_OFFSET_NOT_SUPPORTED:              return "ERROR_URI_FILE_OFFSET_NOT_SUPPORTED";
            case AssetLoadResultCode::ERROR_UNSUPPORTED_ABSOLUTE_PATH:                  return "ERROR_UNSUPPORTED_ABSOLUTE_PATH";
            case AssetLoadResultCode::ERROR_MISSING_VERTEX_POSITIONS:                   return "ERROR_MISSING_VERTEX_POSITIONS";
            case AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_POSITIONS:               return "ERROR_FAULTY_GLTF_VERTEX_POSITIONS";
            default: return "UNKNOWN";
        }
    }
    AssetProcessor(daxa::Device device);
    AssetProcessor(AssetProcessor &&) = default;
    ~AssetProcessor();

    /**
     * THREADSAFETY:
     * * internally synchronized, can be called on multiple threads in parallel.
     */
    auto load_texture(Scene &scene, u32 texture_manifest_index) -> AssetLoadResultCode;

    /**
     * THREADSAFETY:
     * * internally synchronized, can be called on multiple threads in parallel.
     */
    auto load_mesh(Scene &scene, u32 mesh_manifest_index) -> AssetLoadResultCode;

    /**
     * Loads all unloded meshes and material textures for the given scene.
     * THREADSAFETY:
     * * internally synchronized, can be called on multiple threads in parallel.
     */
    auto load_all(Scene& scene) -> AssetLoadResultCode;

    /**
     * NOTE:
     * After loading meshes and textures they are NOT on the gpu yet!
     * They also lack some processing that will be done on the gpu!
     * This function records gpu commands that will:
     * 1. upload cpu processed mesh and texture data
     * 2. process the mesh and texture data
     * 3. upadte the mesh and texture manifest on the gpu
     * 4. memory barrier all following read commands on the queue
     * THREADSAFETY:
     * * internally synchronized, can be called on multiple threads in parallel
     * * fully blocks, it makes no sense to parallelize this function
     * * optimally called once a frame
     * * should not be called in parallel with load_texture and load_mesh
     */
    auto record_gpu_load_processing_commands() -> daxa::ExecutableCommandList;

private:
    static inline const std::string VERT_ATTRIB_POSITION_NAME = "POSITION";
    static inline const std::string VERT_ATTRIB_NORMAL_NAME = "NORMAL";
    static inline const std::string VERT_ATTRIB_TEXCOORD0_NAME = "TEXCOORD_0";

    struct TextureUpload
    {
        Scene *scene = {};
        daxa::BufferId staging_buffer = {};
        daxa::ImageId dst_image = {};
        u32 texture_manifest_index = {};
    };
    struct MeshUpload
    {
        // TODO: replace with buffer offset into staging memory.
        Scene *scene = {};
        daxa::BufferId staging_buffer = {};
        u32 mesh_manifest_index = {};
    };
    daxa::Device _device = {};
    // TODO: Replace with lockless queue.
    std::vector<MeshUpload> _upload_mesh_queue = {};
    std::vector<TextureUpload> _upload_texture_queue = {};
    std::unique_ptr<std::mutex> _mtx = std::make_unique<std::mutex>();
};