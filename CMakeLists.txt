cmake_minimum_required (VERSION 3.21)

project(TimberDoodle)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")

set(TIDO_ENABLE_UTILS_VDB_LOADER false)

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
    "src/json_handler.cpp"
    "src/ui/ui.cpp"
    "src/ui/widgets/scene_graph.cpp"
    "src/ui/widgets/property_viewer.cpp"
    "src/ui/widgets/camera_path_editor.cpp"
    "src/rendering/renderer.cpp"
    "src/rendering/pgi/pgi.cpp"
    "src/multithreading/thread_pool.cpp"
    "src/daxa_helper.cpp"
    "src/scene/scene.cpp"
    "src/scene/asset_processor.cpp"
    "src/scene/openvdb_loader.cpp"
    "src/scene/tex_compression/tex_compression.cpp"
    "src/scene/tex_compression/sdf_bc1_compressor.cpp"
    "src/scene/tex_compression/test.cpp"
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
