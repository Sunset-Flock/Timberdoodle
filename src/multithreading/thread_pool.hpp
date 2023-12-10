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

static constexpr u32 TASK_STORAGE_SIZE = 64;

enum struct TaskPriority
{
    LOW,
    HIGH
};

struct Task
{
    virtual ~Task();
    virtual void callback(u32 chunk_index, u32 thread_index) = 0;

    u32 chunk_count = {};
    std::atomic_uint32_t finished = {};
};

struct TaskChunk
{
    std::shared_ptr<Task> task = {};
    u32 chunk_index = {};
};

constexpr static i32 INVALID_THREAD_INDEX = -1;
struct ThreadPool
{
  public:
    ThreadPool(std::optional<u32> thread_count);
	ThreadPool(ThreadPool &&);
	ThreadPool& operator=(ThreadPool &&);
	ThreadPool(const ThreadPool &) = delete;
	ThreadPool& operator=(const ThreadPool &) = delete;
    ~ThreadPool();
	void blocking_dispatch(std::shared_ptr<Task> task, TaskPriority priority = TaskPriority::LOW);
	void async_dispatch(std::shared_ptr<Task> task, TaskPriority priority = TaskPriority::LOW);

  private:
	ThreadPool(u32 thread_count);
    std::vector<std::thread> worker_threads = {};
	std::condition_variable work_available = {};
	std::mutex task_queues_mutex = {};
    std::deque<TaskChunk> high_priority_tasks = {};
    std::deque<TaskChunk> low_priority_tasks = {};
};
