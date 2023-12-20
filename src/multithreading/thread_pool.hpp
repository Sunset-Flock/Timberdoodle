#pragma once
#include <thread>
#include <cstddef>
#include <optional>
#include <atomic>
#include <deque>
#include <condition_variable>
#include <mutex>

#include "../timberdoodle.hpp"
using namespace tido::types;

static constexpr u32 EXIT_CHUNK_CODE = std::numeric_limits<u32>::max();
static constexpr u32 NO_MORE_CHUNKS_CODE = std::numeric_limits<u32>::max();
static constexpr u32 EXTERNAL_THREAD_INDEX = std::numeric_limits<u32>::max();

enum struct TaskPriority
{
    LOW,
    HIGH
};

struct Task
{
    virtual ~Task() = default;
    virtual void callback(u32 chunk_index, u32 thread_index) = 0;

    u32 chunk_count = {};
    u32 not_finished = {};
    u32 started = {};
};

struct TaskChunk
{
    std::shared_ptr<Task> task = {};
    u32 chunk_index = {};
};

struct ThreadPool
{
  public:
    ThreadPool(std::optional<u32> thread_count = std::nullopt);
    ThreadPool(ThreadPool &&) = default;
    ThreadPool & operator=(ThreadPool &&) = default;
    ThreadPool(ThreadPool const &) = delete;
    ThreadPool & operator=(ThreadPool const &) = delete;
    ~ThreadPool();
    void blocking_dispatch(std::shared_ptr<Task> task, TaskPriority priority = TaskPriority::LOW);
    void async_dispatch(std::shared_ptr<Task> task, TaskPriority priority = TaskPriority::LOW);

  private:
    struct SharedData
    {
        // Signaled whenever there is work added to one of the work queues
        std::condition_variable work_available = {};
        // Signaled whenever a worker thread detects that it is finishing the last chunk of a task 
        std::condition_variable work_done = {};

        std::mutex threadpool_mutex = {};
        std::deque<TaskChunk> high_priority_tasks = {};
        std::deque<TaskChunk> low_priority_tasks = {};
    };
    static void worker(std::shared_ptr<ThreadPool::SharedData> shared_data, u32 thread_id);
    std::shared_ptr<SharedData> shared_data = {};
    std::vector<std::thread> worker_threads = {};
};
