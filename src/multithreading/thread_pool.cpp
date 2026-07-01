#include "thread_pool.hpp"
using namespace tido::types;

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock lock{shared_data->threadpool_mutex};
        shared_data->kill = true;
        shared_data->work_available.notify_all();
    }
    for (auto & worker : worker_threads)
    {
        worker.join();
    }
}

void ThreadPool::worker(std::shared_ptr<ThreadPool::SharedData> shared_data, u32 thread_index)
{
    std::unique_lock lock{shared_data->threadpool_mutex};
    while (true)
    {
        shared_data->work_available.wait(
            lock, [&]
            { return !shared_data->high_priority_tasks.empty() || !shared_data->low_priority_tasks.empty() || shared_data->kill; });
        if (shared_data->kill)
        {
            return;
        }

        bool const high_priority_work_available = !shared_data->high_priority_tasks.empty();
        auto & selected_queue = high_priority_work_available ? shared_data->high_priority_tasks : shared_data->low_priority_tasks;
        TaskChunk current_chunk = std::move(selected_queue.front());
        selected_queue.pop_front();
        current_chunk.task->started += 1;
        lock.unlock();

        current_chunk.task->callback(current_chunk.chunk_index, thread_index);

        lock.lock();
        current_chunk.task->not_finished -= 1;
        // Working on last chunk of a task, notify in case there is a thread waiting for this task to be done
        if (current_chunk.task->not_finished == 0) { shared_data->work_done.notify_all(); }
    }
}

ThreadPool::ThreadPool(std::optional<u32> thread_count)
{
    u32 const real_thread_count = thread_count.value_or(std::thread::hardware_concurrency());
    shared_data = std::make_shared<SharedData>();
    for (u32 thread_index = 0; thread_index < real_thread_count; thread_index++)
    {
        worker_threads.push_back({
            std::thread([=, this]()
                { ThreadPool::worker(shared_data, thread_index); }),
        });
    }
}

void ThreadPool::blocking_dispatch(std::shared_ptr<Task> task, TaskPriority priority)
{
    // Don't need mutex here as no thread is working on this task yet
    task->not_finished = task->chunk_count;
    auto & selected_queue = priority == TaskPriority::HIGH ? shared_data->high_priority_tasks : shared_data->low_priority_tasks;

    // chunk_index 0 will be worked on by this thread
    std::unique_lock lock{shared_data->threadpool_mutex};

    for (u32 chunk_index = 1; chunk_index < task->chunk_count; chunk_index++)
    {
        selected_queue.push_back({task, chunk_index});
    }

    shared_data->work_available.notify_all();
    // Process chunk 0 on this thread.
    task->started += 1;
    lock.unlock();
    task->callback(0, EXTERNAL_THREAD_INDEX);
    lock.lock();
    task->not_finished -= 1;
    if (task->not_finished == 0) { shared_data->work_done.notify_all(); return; }

    // Our task isn't done yet. Keep stealing and processing any available work from the
    // queue until it finishes. Processing any chunk (not just ours) prevents deadlock when
    // nested blocking_dispatch calls are in flight — all threads stay productive instead of
    // sleeping while their sub-tasks are stuck behind other work in the queue.
    while (task->not_finished != 0)
    {
        bool const has_high = !shared_data->high_priority_tasks.empty();
        bool const has_low  = !shared_data->low_priority_tasks.empty();
        if (!has_high && !has_low)
        {
            shared_data->work_done.wait(lock, [&]{ return task->not_finished == 0; });
            return;
        }
        auto & steal_queue = has_high ? shared_data->high_priority_tasks : shared_data->low_priority_tasks;
        TaskChunk chunk = std::move(steal_queue.front());
        steal_queue.pop_front();
        chunk.task->started += 1;
        lock.unlock();
        chunk.task->callback(chunk.chunk_index, EXTERNAL_THREAD_INDEX);
        lock.lock();
        chunk.task->not_finished -= 1;
        if (chunk.task->not_finished == 0) { shared_data->work_done.notify_all(); }
    }
}

void ThreadPool::async_dispatch(std::shared_ptr<Task> task, TaskPriority priority)
{
    task->not_finished = task->chunk_count;
    auto & selected_queue = priority == TaskPriority::HIGH ? shared_data->high_priority_tasks : shared_data->low_priority_tasks;
    {
        std::lock_guard lock(shared_data->threadpool_mutex);
        for (u32 chunk_index = 0; chunk_index < task->chunk_count; chunk_index++)
        {
            selected_queue.push_back({task, chunk_index});
        }
    }
    shared_data->work_available.notify_all();
}

void ThreadPool::block_on(std::shared_ptr<Task> task)
{
    std::unique_lock lock{shared_data->threadpool_mutex};
    shared_data->work_done.wait(lock, [&]
        { return task->not_finished == 0; });
}