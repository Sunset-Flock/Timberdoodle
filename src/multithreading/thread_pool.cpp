#include "thread_pool.hpp"
using namespace tido::types;

ThreadPool::~ThreadPool()
{
}

void ThreadPool::worker(std::shared_ptr<ThreadPool::SharedData> shared_data, u32 thread_index)
{
    TaskChunk current_chunk = {};
    while (true)
    {
        {
            std::unique_lock lock{shared_data->task_queues_mutex};
            /// TODO: Use separate mutex for high priority and low priority?
            shared_data->work_available.wait(
                lock,
                [&]
                {
                    return !shared_data->high_priority_tasks.empty() ||
                           !shared_data->low_priority_tasks.empty();
                });

            bool const high_priority_work_available = !shared_data->high_priority_tasks.empty();
            auto & selected_queue = high_priority_work_available ? shared_data->high_priority_tasks : shared_data->low_priority_tasks;
            current_chunk = std::move(selected_queue.front());
            selected_queue.pop_front();
        }
        /// NOTE: Lock is no longer needed for the execution of the callback

        // If we receive an invalid chunk index in the chunk code this is a code end this thread
        if(current_chunk.chunk_index == EXIT_CHUNK_CODE) { return; }
        current_chunk.task->callback(current_chunk.chunk_index, thread_index);
    }
}

ThreadPool::ThreadPool(std::optional<u32> thread_count)
{
    u32 const real_thread_count = thread_count.value_or(std::thread::hardware_concurrency());
    for (u32 thread_index = 0; thread_index < real_thread_count; thread_index++)
    {
        worker_threads.push_back({
            std::thread([=]()
                        { ThreadPool::worker(shared_data, thread_index); }),
        });
    }
}

void ThreadPool::blocking_dispatch(std::shared_ptr<Task> task, TaskPriority priority)
{
}

void ThreadPool::async_dispatch(std::shared_ptr<Task> task, TaskPriority priority)
{
}