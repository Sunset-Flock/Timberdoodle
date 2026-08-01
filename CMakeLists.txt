cmake_minimum_required (VERSION 3.21)

project(TimberDoodle)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")

set(TIDO_ENABLE_UTILS_VDB_LOADER true)

if("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Windows")
    # /MT for Release, /MTd for Debug. Force static CRT
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

include(cmake/deps.cmake)

set(DAXA_ENABLE_UTILS_IMGUI true)
set(DAXA_ENABLE_UTILS_MEM true)
set(DAXA_ENABLE_UTILS_PIPELINE_MANAGER_GLSLANG true)
set(DAXA_ENABLE_UTILS_PIPELINE_MANAGER_SLANG true)
set(DAXA_ENABLE_UTILS_TASK_GRAPH true)
set(DAXA_ENABLE_TESTS false)
add_subdirectory(${PROJECT_SOURCE_DIR}/deps/daxa)

add_executable(${PROJECT_NAME}
    "src/main.cpp"
    "src/window.cpp"
    "src/camera.cpp"
    "src/application.cpp"
    "src/gpu_context.cpp"
    "src/json_utils/camera_animation.cpp"
    "src/json_utils/sky_settings.cpp"
    "src/json_utils/tido_format.cpp"
    "src/ui/ui.cpp"
    "src/ui/widgets/scene_graph.cpp"
    "src/ui/widgets/property_viewer.cpp"
    "src/ui/widgets/camera_path_editor.cpp"
    "src/rendering/renderer.cpp"
    "src/rendering/pgi/pgi.cpp"
    "src/multithreading/thread_pool.cpp"
    "src/daxa_helper.cpp"
    "src/scene/scene.cpp"
    "src/scene/scene_runtime.cpp"
    "src/scene/nonmanifest_texture.cpp"
    "src/scene/importers/importer.cpp"
    "src/scene/importers/gltf_importer.cpp"
    "src/scene/importers/standalone_importer.cpp"
    "src/scene/streamer.cpp"
    "src/scene/optimizers/image_compressor.cpp"
    "src/scene/optimizers/image_processor.cpp"
    "src/scene/optimizers/vdb_processor.cpp"
    "src/scene/optimizers/sdf_bc1_compressor.cpp"
    "src/scene/optimizers/geometry_optimizer.cpp"
    "src/io/file_io.cpp"
    "src/scene/tido_format/tido_util.cpp"
    "src/scene/tido_format/tido_format.cpp"
    "src/scene/optimizers/test/test.cpp"
    "src/rendering/rasterize_visbuffer/rasterize_visbuffer.cpp"
    "src/rendering/rtgi/rtgi.cpp"
)

if(TIDO_ENABLE_UTILS_VDB_LOADER)
    target_compile_definitions(${PROJECT_NAME}
        PRIVATE
        TIDO_BUILT_WITH_UTILS_VDB_LOADER=true
    )
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # NOTE(grundlett): Really hacky. It appears there's no way to create plugin dll dependencies...
    # see https://gitlab.kitware.com/cmake/cmake/-/issues/22993
    add_custom_command(TARGET TimberDoodle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${Slang_DLL} $<TARGET_FILE_DIR:TimberDoodle>
        COMMAND ${CMAKE_COMMAND} -E copy ${Slang_Glslang_DLL} $<TARGET_FILE_DIR:TimberDoodle>
    )

    # clangd matches the editor's lowercase drive letter case-sensitively, so lowercase the
    # drive in compile_commands.json's "file" entries and publish the fixed file to the repo root.
    # The script is developer-local and gitignored, so skip the step when it is absent.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tools/fix_compile_commands_drive.ps1")
        add_custom_command(TARGET TimberDoodle POST_BUILD
            COMMAND powershell -NoProfile -ExecutionPolicy Bypass -File "${CMAKE_SOURCE_DIR}/tools/fix_compile_commands_drive.ps1" -Paths "${CMAKE_BINARY_DIR}/compile_commands.json" -CopyTo "${CMAKE_SOURCE_DIR}/compile_commands.json"
            VERBATIM
        )
    endif()
endif()

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE
    fmt::fmt
    daxa::daxa
    glfw
    meshoptimizer
    fastgltf::fastgltf
    KTX::ktx
    png_static
    nlohmann_json::nlohmann_json
    simdjson::simdjson
    implot::implot
    glm::glm
    CMP_Core
)

if(TIDO_ENABLE_UTILS_VDB_LOADER)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        openvdb_static
    )
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(${PROJECT_NAME} PRIVATE Dwmapi)
endif()

# Standalone converter recovering a .vdb from a cooked .cloudbin. Developer-local and gitignored, so the
# target only exists where the source does.
if(TIDO_ENABLE_UTILS_VDB_LOADER AND EXISTS "${CMAKE_SOURCE_DIR}/tools/cloudbin_to_vdb/main.cpp")
    add_executable(cloudbin_to_vdb
        "tools/cloudbin_to_vdb/main.cpp"
        "src/scene/optimizers/sdf_bc1_compressor.cpp"
        "src/scene/optimizers/vdb_processor.cpp"
        "src/scene/optimizers/image_processor.cpp"
        "src/scene/optimizers/image_compressor.cpp"
        "src/multithreading/thread_pool.cpp"
        "src/scene/tido_format/tido_format.cpp"
        "src/scene/tido_format/tido_util.cpp"
        "src/json_utils/tido_format.cpp"
        "src/io/file_io.cpp"
    )
    target_compile_features(cloudbin_to_vdb PRIVATE cxx_std_20)
    # The tool verifies its output by running the importer's own vdb_parse over it.
    target_compile_definitions(cloudbin_to_vdb PRIVATE TIDO_BUILT_WITH_UTILS_VDB_LOADER=true)
    target_link_libraries(cloudbin_to_vdb PRIVATE
        fmt::fmt
        glm::glm
        daxa::daxa
        nlohmann_json::nlohmann_json
        simdjson::simdjson
        KTX::ktx
        png_static
        CMP_Core
        openvdb_static
    )
endif()
