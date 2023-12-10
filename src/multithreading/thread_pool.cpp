#include "thread_pool.hpp"
using namespace tido::types;

ThreadPool::ThreadPool(ThreadPool &&)
{

}
ThreadPool& ThreadPool::operator=(ThreadPool &&)
{

}

ThreadPool::~ThreadPool()
{

}

thread_local static i32 this_thread_index = INVALID_THREAD_INDEX;
thread_local static TaskChunk current_task_chunk = {};

ThreadPool::ThreadPool(std::optional<u32> thread_count)
{
    const u32 real_thread_count = thread_count.value_or(std::thread::hardware_concurrency());
    for(u32 thread_index = 0; thread_index < real_thread_count; thread_index++)
    {
        worker_threads.push_back(std::thread(
            [&](){
                this_thread_index = thread_index; 
                while(true)
                {
                    std::unique_lock lock{task_queues_mutex};
                    work_available.wait(lock, [&]{ return !high_priority_tasks.empty() || !low_priority_tasks.empty(); });

                }
            }
        ));
    }
}

void ThreadPool::blocking_dispatch(std::shared_ptr<Task> task, TaskPriority priority)
{

}

void ThreadPool::async_dispatch(std::shared_ptr<Task> task, TaskPriority priority)
{

}