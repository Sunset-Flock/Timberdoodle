find_package(Vulkan REQUIRED)

include(FetchContent)

# Clone daxa if it's not present in the deps directory
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/daxa/CMakeLists.txt")
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
        COMMAND_ERROR_IS_FATAL ANY)
endif()

if (NOT TARGET meshoptimizer)
    FetchContent_Declare(
        meshoptimizer
        GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
        GIT_TAG        v1.0.1
    )
    FetchContent_MakeAvailable(meshoptimizer)
endif()

if (NOT TARGET fmt::fmt)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG        12.1.0
    )
    FetchContent_MakeAvailable(fmt)
endif()

if (NOT TARGET fastgltf::fastgltf)
    FetchContent_Declare(
        fastgltf
        GIT_REPOSITORY https://github.com/spnda/fastgltf
        GIT_TAG        v0.9.0
    )
    FetchContent_MakeAvailable(fastgltf)
endif()

if (NOT TARGET KTX::ktx)
    FetchContent_Declare(
        ktx
        GIT_REPOSITORY https://github.com/KhronosGroup/KTX-Software
        GIT_TAG        v4.4.2
    )
    FetchContent_MakeAvailable(ktx)
    add_library(KTX::ktx ALIAS ktx)
endif()

if (NOT TARGET freeimage::FreeImage)
    FetchContent_Declare(
        freeimage
        GIT_REPOSITORY https://github.com/swm8023/FreeImage-Cmake
        GIT_TAG        47d485039119e4cbf844f934ffc548c27d2cdc5a
    )
    FetchContent_MakeAvailable(freeimage)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CXX_COMPILER_ID MATCHES Clang)
        target_compile_definitions(FreeImage PRIVATE WIN32) # for some reason it looks for this define, but Clang doesn't provide it
    endif()
    target_include_directories(FreeImage PUBLIC ${freeimage_SOURCE_DIR}/Source)
    add_library(freeimage::FreeImage ALIAS FreeImage)
endif()

if (NOT TARGET nlohmann_json::nlohmann_json)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json
        GIT_TAG        v3.12.0
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

if (NOT TARGET glm::glm)
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm
        GIT_TAG        1.0.3
    )
    FetchContent_MakeAvailable(glm)
endif()

if (NOT TARGET glfw)
    option(GLFW_BUILD_TESTS "" OFF)
    option(GLFW_BUILD_DOCS "" OFF)
    option(GLFW_INSTALL "" OFF)
    option(GLFW_BUILD_EXAMPLES "" OFF)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw
        GIT_TAG        3.4
    )
    FetchContent_MakeAvailable(glfw)
endif()

if (NOT TARGET imgui::imgui)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui
        GIT_TAG        fdc084f532189fda8474079f79e74fa5e3541c9f
    )

    FetchContent_GetProperties(imgui)
    if(NOT imgui_POPULATED)
        FetchContent_MakeAvailable(imgui)

        add_library(lib_imgui
            ${imgui_SOURCE_DIR}/imgui.cpp
            ${imgui_SOURCE_DIR}/imgui_demo.cpp
            ${imgui_SOURCE_DIR}/imgui_draw.cpp
            ${imgui_SOURCE_DIR}/imgui_widgets.cpp
            ${imgui_SOURCE_DIR}/imgui_tables.cpp)

        target_include_directories(lib_imgui PUBLIC
            ${imgui_SOURCE_DIR}
            ${imgui_SOURCE_DIR}/backends
            ${Vulkan_INCLUDE_DIRS})

        if(TARGET glfw)
            target_sources(lib_imgui PRIVATE ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp)
            target_link_libraries(lib_imgui PRIVATE glfw)
            target_include_directories(lib_imgui PUBLIC ${glfw_SOURCE_DIR}/include)
        endif()

        add_library(imgui::imgui ALIAS lib_imgui)
    endif()
endif()

if (NOT TARGET implot::implot)
    FetchContent_Declare(
        implot
        GIT_REPOSITORY https://github.com/epezent/implot
        GIT_TAG        v0.17
    )

    FetchContent_GetProperties(implot)
    if(NOT implot_POPULATED)
        FetchContent_MakeAvailable(implot)

        add_library(lib_implot
            ${implot_SOURCE_DIR}/implot.cpp
            ${implot_SOURCE_DIR}/implot_items.cpp
            ${implot_SOURCE_DIR}/implot_demo.cpp)

        target_include_directories(lib_implot PUBLIC ${implot_SOURCE_DIR})
        target_link_libraries(lib_implot PRIVATE imgui::imgui)

        add_library(implot::implot ALIAS lib_implot)
    endif()
endif()
