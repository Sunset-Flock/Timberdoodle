#include "importer.hpp"

Importer::Importer(ThreadPool * thread_pool)
    : thread_pool{thread_pool},
      _gltf_importer{this},
      _raw_importer{this}
{
    _thread = std::thread([this]() { thread_main(); });
}

Importer::~Importer()
{
    stop();
}

void Importer::stop()
{
    {
        std::lock_guard<std::mutex> lock{_queue_mutex};
        _stop_requested = true;
    }
    _wake_signal.notify_all();
    if (_thread.joinable())
    {
        _thread.join();
    }
}

void Importer::push_tasks(std::span<ImporterTask> tasks)
{
    {
        std::lock_guard<std::mutex> lock{_queue_mutex};
        if (_stop_requested)
        {
            return;
        }
        _task_queue.insert(_task_queue.end(), std::make_move_iterator(tasks.begin()), std::make_move_iterator(tasks.end()));
    }
    _wake_signal.notify_one();
}

auto Importer::pop_results() -> std::vector<ImporterTaskResult>
{
    std::lock_guard<std::mutex> lock{_result_queue_mutex};
    std::vector<ImporterTaskResult> results = std::move(_result_queue);
    _result_queue.clear();
    return results;
}

void Importer::push_result(ImporterTaskResult result)
{
    std::lock_guard<std::mutex> lock{_result_queue_mutex};
    _result_queue.push_back(std::move(result));
}

void Importer::notify()
{
    {
        std::lock_guard<std::mutex> lock{_queue_mutex};
        _upkeep_requested = true;
    }
    _wake_signal.notify_one();
}

void Importer::thread_main()
{
    for (;;)
    {
        std::vector<ImporterTask> tasks = {};
        {
            std::unique_lock<std::mutex> lock{_queue_mutex};
            _wake_signal.wait(lock, [&]() { return _stop_requested || !_task_queue.empty() || _upkeep_requested; });
            if (_stop_requested)
            {
                return; // Still-queued tasks are deliberately dropped on shutdown.
            }
            tasks = std::move(_task_queue);
            _task_queue.clear();
            _upkeep_requested = false;
        }
        // An upkeep-only wake passes an empty task list; the importers still run their cache upkeep.
        _gltf_importer.update(tasks);
        _raw_importer.update(tasks);
        DBG_ASSERT_TRUE_M(tasks.empty(), "An ImporterTask was left unconsumed - no importer handles its provenance");
    }
}
