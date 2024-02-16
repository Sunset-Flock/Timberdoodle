#pragma once

#include <filesystem>
#include <meshoptimizer.h>
#include <fastgltf/types.hpp>
#include <mutex>

#include "../timberdoodle.hpp"
#include "../gpu_context.hpp"
#include "../shader_shared/asset.inl"

using namespace tido::types;

using MeshIndex = size_t;
using ImageIndex = size_t;

#define MAX_MESHES 10000

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
        ERROR_MISSING_VERTEX_TEXCOORD_0,
        ERROR_FAULTY_GLTF_VERTEX_TEXCOORD_0,
        ERROR_MISSING_VERTEX_NORMALS,
        ERROR_FAULTY_GLTF_VERTEX_NORMALS,
        ERROR_MISSING_VERTEX_TANGENTS,
        ERROR_FAULTY_GLTF_VERTEX_TANGENTS,
    };
    static auto to_string(AssetLoadResultCode code) -> std::string_view
    {
        switch (code)
        {
            case AssetLoadResultCode::SUCCESS: return "SUCCESS"; 
            case AssetLoadResultCode::ERROR_MISSING_INDEX_BUFFER: return "ERROR_MISSING_INDEX_BUFFER"; 
            case AssetLoadResultCode::ERROR_FAULTY_INDEX_BUFFER_GLTF_ACCESSOR: return "ERROR_FAULTY_INDEX_BUFFER_GLTF_ACCESSOR"; 
            case AssetLoadResultCode::ERROR_FAULTY_BUFFER_VIEW: return "ERROR_FAULTY_BUFFER_VIEW"; 
            case AssetLoadResultCode::ERROR_COULD_NOT_OPEN_GLTF: return "ERROR_COULD_NOT_OPEN_GLTF"; 
            case AssetLoadResultCode::ERROR_COULD_NOT_READ_BUFFER_IN_GLTF: return "ERROR_COULD_NOT_READ_BUFFER_IN_GLTF"; 
            case AssetLoadResultCode::ERROR_COULD_NOT_OPEN_TEXTURE_FILE: return "ERROR_COULD_NOT_OPEN_TEXTURE_FILE"; 
            case AssetLoadResultCode::ERROR_COULD_NOT_READ_TEXTURE_FILE: return "ERROR_COULD_NOT_READ_TEXTURE_FILE"; 
            case AssetLoadResultCode::ERROR_COULD_NOT_READ_TEXTURE_FILE_FROM_MEMSTREAM: return "ERROR_COULD_NOT_READ_TEXTURE_FILE_FROM_MEMSTREAM"; 
            case AssetLoadResultCode::ERROR_UNSUPPORTED_TEXTURE_PIXEL_FORMAT: return "ERROR_UNSUPPORTED_TEXTURE_PIXEL_FORMAT"; 
            case AssetLoadResultCode::ERROR_UNKNOWN_FILETYPE_FORMAT: return "ERROR_UNKNOWN_FILETYPE_FORMAT"; 
            case AssetLoadResultCode::ERROR_UNSUPPORTED_READ_FOR_FILEFORMAT: return "ERROR_UNSUPPORTED_READ_FOR_FILEFORMAT"; 
            case AssetLoadResultCode::ERROR_URI_FILE_OFFSET_NOT_SUPPORTED: return "ERROR_URI_FILE_OFFSET_NOT_SUPPORTED"; 
            case AssetLoadResultCode::ERROR_UNSUPPORTED_ABSOLUTE_PATH: return "ERROR_UNSUPPORTED_ABSOLUTE_PATH"; 
            case AssetLoadResultCode::ERROR_MISSING_VERTEX_POSITIONS: return "ERROR_MISSING_VERTEX_POSITIONS"; 
            case AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_POSITIONS: return "ERROR_FAULTY_GLTF_VERTEX_POSITIONS"; 
            case AssetLoadResultCode::ERROR_MISSING_VERTEX_TEXCOORD_0: return "ERROR_MISSING_VERTEX_TEXCOORD_0"; 
            case AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_TEXCOORD_0: return "ERROR_FAULTY_GLTF_VERTEX_TEXCOORD_0"; 
            case AssetLoadResultCode::ERROR_MISSING_VERTEX_NORMALS: return "ERROR_MISSING_VERTEX_NORMALS"; 
            case AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_NORMALS: return "ERROR_FAULTY_GLTF_VERTEX_NORMALS"; 
            case AssetLoadResultCode::ERROR_MISSING_VERTEX_TANGENTS: return "ERROR_MISSING_VERTEX_TANGENTS"; 
            case AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_TANGENTS: return "ERROR_FAULTY_GLTF_VERTEX_TANGENTS"; 
            default: return "UNKNOWN";
        }
    }
    AssetProcessor(daxa::Device device);
    AssetProcessor(AssetProcessor &&) = default;
    ~AssetProcessor();

    using NonmanifestLoadRet = std::variant<AssetProcessor::AssetLoadResultCode, daxa::ImageId>;
    auto load_nonmanifest_texture(std::filesystem::path const & filepath, bool const load_as_srgb = true) -> NonmanifestLoadRet;

    /**
     * THREADSAFETY:
     * * internally synchronized, can be called on multiple threads in parallel.
     */
    struct TextureUploadInfo
    {
        daxa::BufferId staging_buffer = {};
        daxa::ImageId dst_image = {};

        u32 texture_manifest_index = {};
    };
    struct LoadTextureInfo
    {
        std::filesystem::path asset_path = {};
        fastgltf::Asset * asset;
        u32 gltf_texture_index = {};
        u32 texture_manifest_index = {};
        bool load_as_srgb = {};
    };
    auto load_texture(LoadTextureInfo const & info) -> AssetLoadResultCode;

    struct MeshUploadInfo
    {
        // TODO: replace with buffer offset into staging memory.
        daxa::BufferId staging_buffer = {};
        daxa::BufferId mesh_buffer = {};

        GPUMesh mesh = {};
        u32 manifest_index = {};
    };
    struct LoadMeshInfo
    {
        std::filesystem::path asset_path = {};
        fastgltf::Asset * asset;
        u32 gltf_mesh_index = {};
        u32 gltf_primitive_index = {};
        u32 global_material_manifest_offset = {};
        u32 manifest_index = {};
    };
    /**
     * THREADSAFETY:
     * * internally synchronized, can be called on multiple threads in parallel.
     */
    auto load_mesh(LoadMeshInfo const & info) -> AssetLoadResultCode;

    /**
     * Loads all unloded meshes and material textures for the given scene.
     * THREADSAFETY:
     * * internally synchronized, can be called on multiple threads in parallel.
     */
    // auto load_all(Scene & scene) -> AssetLoadResultCode;

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
    struct RecordCommandsRet
    {
        daxa::ExecutableCommandList upload_commands = {};
        std::vector<MeshUploadInfo> uploaded_meshes = {};
        std::vector<TextureUploadInfo> uploaded_textures = {};
    };
    auto record_gpu_load_processing_commands() -> RecordCommandsRet;

  private:
    static inline std::string const VERT_ATTRIB_POSITION_NAME = "POSITION";
    static inline std::string const VERT_ATTRIB_TEXCOORD0_NAME = "TEXCOORD_0";
    static inline std::string const VERT_ATTRIB_NORMAL_NAME = "NORMAL";
    static inline std::string const VERT_ATTRIB_TANGENT_NAME = "TANGENT";

    daxa::Device _device = {};
    // TODO: Replace with lockless queue.
    std::vector<MeshUploadInfo> _upload_mesh_queue = {};
    std::vector<TextureUploadInfo> _upload_texture_queue = {};
    std::unique_ptr<std::mutex> _mesh_upload_mutex = std::make_unique<std::mutex>();
    std::unique_ptr<std::mutex> _texture_upload_mutex = std::make_unique<std::mutex>();
};