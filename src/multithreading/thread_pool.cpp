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
                lock, [&] { return !shared_data->high_priority_tasks.empty() || !shared_data->low_priority_tasks.empty(); });

            bool const high_priority_work_available = !shared_data->high_priority_tasks.empty();
            auto & selected_queue =
                high_priority_work_available ? shared_data->high_priority_tasks : shared_data->low_priority_tasks;
            current_chunk = std::move(selected_queue.front());
            selected_queue.pop_front();
            /// NOTE: Make sure to mark this chunk as started before we release the lock!
            //        If this task was added by calling blocking_dispatch the calling thread checks the started count to decide if
            //        there are still chunks of this task left in the queue. Not doing this while the queue is locked will result
            //        in race condition where the dispatching thread pops a chunk belonging to wrong task from the queue
            current_chunk.task->started += 1;
        }

        // Received invalid chunk code in the chunk index which exits this thread
        if (current_chunk.chunk_index == EXIT_CHUNK_CODE) { return; }
        current_chunk.task->callback(current_chunk.chunk_index, thread_index);
        const u32 prev_finished_count = current_chunk.task->finished.fetch_add(1);
        // Working on last chunk of a task, notify in case there is a thread waiting for this task to be done
        if(prev_finished_count == (current_chunk.task->chunk_count - 1))
        {
            shared_data->work_done.notify_all();
        }
    }
}

ThreadPool::ThreadPool(std::optional<u32> thread_count)
{
    u32 const real_thread_count = thread_count.value_or(std::thread::hardware_concurrency());
    shared_data = std::make_shared<SharedData>();
    for (u32 thread_index = 0; thread_index < real_thread_count; thread_index++)
    {
        worker_threads.push_back({
            std::thread([=, this]() { ThreadPool::worker(shared_data, thread_index); }),
        });
    }
}

void ThreadPool::blocking_dispatch(std::shared_ptr<Task> task, TaskPriority priority)
{
    auto & selected_queue = priority == TaskPriority::HIGH ? shared_data->high_priority_tasks : shared_data->low_priority_tasks;
    {
        std::lock_guard<std::mutex> lock(shared_data->task_queues_mutex);
        for (u32 chunk_index = 1; chunk_index < task->chunk_count; chunk_index++)
        {
            selected_queue.push_back({task, chunk_index});
        }
    }
    shared_data->work_available.notify_all();

    // Since this is a blocking dispatch this thread should also contribute to executing the task
    task->started += 1;
    task->callback(0, EXTERNAL_THREAD_INDEX);
    task->finished += 1;

    bool worked_on_last_chunk = {};
    // If there is still unassigned work for this task we attempt to acquire it
    while(task->started != task->chunk_count)
    {
        u32 current_chunk_index = {};
        {
            // We need to recheck if the task was not started before we locked the queue. 
            // This relies on the fact that noone increments the started count after releasing a lock on the queue
            std::lock_guard<std::mutex> lock(shared_data->task_queues_mutex);
            // Someone started the last bit of work on this task while we were acquiring the lock, no more work to be done 
            if(task->started == task->chunk_count) { break; }
            current_chunk_index = selected_queue.front().chunk_index;
            selected_queue.pop_front();
            task->started += 1;
        }
        task->callback(current_chunk_index, EXTERNAL_THREAD_INDEX);
        const u32 prev_finished_count = task->finished.fetch_add(1);
        worked_on_last_chunk = prev_finished_count == (task->chunk_count - 1);
    }

    if(!worked_on_last_chunk)
    {
        // This thread was not the last one working on this task, therefore we wait here to be notified once
        // the last worker thread processing this task is done
        std::unique_lock work_done_lock {shared_data->task_queues_mutex};
        shared_data->work_done.wait( work_done_lock, [&]{return task->finished == task->chunk_count;});
    }
}

void ThreadPool::async_dispatch(std::shared_ptr<Task> task, TaskPriority priority)
{
    auto & selected_queue = priority == TaskPriority::HIGH ? shared_data->high_priority_tasks : shared_data->low_priority_tasks;
    {
        std::lock_guard<std::mutex> lock(shared_data->task_queues_mutex);
        for (u32 chunk_index = 0; chunk_index < task->chunk_count; chunk_index++)
        {
            selected_queue.push_back({task, chunk_index});
        }
    }
    shared_data->work_available.notify_all();
}