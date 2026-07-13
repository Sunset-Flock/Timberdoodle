#pragma once

#include <condition_variable>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "../../timberdoodle.hpp"
#include "../../multithreading/thread_pool.hpp"
#include "gltf_importer.hpp"
#include "importer_task.hpp"
#include "importer_task_result.hpp"
using namespace tido::types;

/// --- Importer ---
/// Owns the dedicated importer orchestration thread. SceneRuntime pushes ImporterTasks in; finished
/// ImporterTaskResults come back out through pop_results. The thread itself only orchestrates - it
/// drains the task queue, hands the tasks to the per-format importers (which group them, dispatch
/// parse/cook work to the ThreadPool) and runs cache upkeep. It is paced by a condition variable,
/// woken by task pushes and by notify() from finishing worker-side tasks.
struct Importer
{
    explicit Importer(ThreadPool * thread_pool);
    ~Importer();

    // Signals the orchestration thread to exit and joins it; still-queued tasks are dropped. Idempotent.
    // Must run before ~ThreadPool so nothing new is dispatched into a joining pool; the pool's own join
    // then completes the in-flight parse/cook/cache-write tasks while this Importer is still alive.
    void stop();

    // Pushes a group of tasks under one lock, so a single drain of the orchestration loop sees them
    // together (a source's asset tasks then batch into a single parse). Callable from any thread.
    void push_tasks(std::span<ImporterTask> tasks);
    // Moves out every result produced so far; SceneRuntime drains this once per frame.
    auto pop_results() -> std::vector<ImporterTaskResult>;

    // Worker-side API, called from the ThreadPool tasks the importers dispatch:
    void push_result(ImporterTaskResult result);
    // Wakes the orchestration loop so upkeep can act on finished work (cook chunk / cache write done).
    void notify();

    ThreadPool * thread_pool = {};

  private:
    std::mutex _queue_mutex = {};
    std::condition_variable _wake_signal = {};
    // All three guarded by _queue_mutex.
    std::vector<ImporterTask> _task_queue = {};
    bool _stop_requested = false;
    bool _upkeep_requested = false;

    std::mutex _result_queue_mutex = {};
    std::vector<ImporterTaskResult> _result_queue = {};

    GltfImporter _gltf_importer;

    std::thread _thread = {};
    void thread_main();
};
