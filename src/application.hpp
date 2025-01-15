#pragma once

// Standart headers:
#include <chrono>
#include <memory>
// Library headers:
// Project headers:
#include "timberdoodle.hpp"
using namespace tido::types;

#include "window.hpp"

#include "scene/scene.hpp"
#include "scene/asset_processor.hpp"
#include "ui/ui.hpp"
#include "rendering/renderer.hpp"
#include "gpu_context.hpp"
#include "multithreading/thread_pool.hpp"
#include "application_state.hpp"

struct Application
{
public:
    Application();
    ~Application();

    auto run() -> i32;
private:
    void update();
    /**
        * EXPLANATION: Why do we use unique pointers here?
        * Many of these members are non-movable.
        * They can NOT be made movable easily!
        * They require dependency injection between each other!
        * A pattern that solves these problems (non-movable + dep injection), is to wrap these structs in heap allocations and refer to them only with pointers.
        * We do NOT need shared_ptr here, as we know the lifetime of these members beforehand! The Application controls their lifetime!
        * If member B referes to member A, it will be below it in the struct delaration. This means it will be destroyed before A. This way we wont have dangling pointers.
        * There are no performance implications of this as these structs are VERY low frequency creation/deletion and are never copied.
        * This allows the construction of Application to be much simpler, less bug prone and makes Application movable!
        * A good rule of thumb is to have structs movable. If you need members that are not movable, simply wrap them in pointers.
        * WARNING: THIS CAN ONLY BE APPLIED LIKE THIS FOR LOW FREQUENCY TYPES, AS IT MIGHT INCUR PERFORMANCE PROBLEMS OTHERWISE!
        */
    std::unique_ptr<Window> _window = {};
    std::unique_ptr<GPUContext> _gpu_context = {};
    std::unique_ptr<Scene> _scene = {};
    std::unique_ptr<AssetProcessor> _asset_manager = {};
    std::unique_ptr<UIEngine> _ui_engine = {};
    std::unique_ptr<Renderer> _renderer = {};
    std::unique_ptr<ThreadPool> _threadpool = {};
    std::unique_ptr<AsteroidSimulation> _simulation = {};
    ApplicationState app_state = {};
};