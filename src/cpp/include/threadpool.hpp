#ifndef __THREADPOOL_HPP__
#define __THREADPOOL_HPP__

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "port_utils.hpp"
#include "threadsafequeue.hpp"

class ThreadPool {
private:
  class IThreadTask {
  public:
    IThreadTask(void) = default;
    virtual ~IThreadTask(void) = default;
    IThreadTask(const IThreadTask& rhs) = delete;
    IThreadTask& operator=(const IThreadTask& rhs) = delete;
    IThreadTask(IThreadTask&& other) = default;
    IThreadTask& operator=(IThreadTask&& other) = default;

    /**
     * Run the task.
     */
    virtual void execute() = 0;
  };

  template <typename Func>
  class ThreadTask : public IThreadTask {
  public:
    ThreadTask(Func&& func) : m_func{std::move(func)} {}

    ~ThreadTask(void) override = default;
    ThreadTask(const ThreadTask& rhs) = delete;
    ThreadTask& operator=(const ThreadTask& rhs) = delete;
    ThreadTask(ThreadTask&& other) = default;
    ThreadTask& operator=(ThreadTask&& other) = default;

    /**
     * Run the task.
     */
    void execute() override { m_func(); }

  private:
    Func m_func;
  };

public:
  /**
   * A wrapper around a std::future that adds the behavior of futures returned from std::async.
   * Specifically, this object will block and wait for execution to finish before going out of
   * scope.
   */
  template <typename T>
  class TaskFuture {
  public:
    TaskFuture() = default;
    TaskFuture(std::future<T>&& future) : m_future{std::move(future)} {}

    TaskFuture(const TaskFuture&) = delete;
    TaskFuture& operator=(const TaskFuture&) = delete;
    TaskFuture(TaskFuture&&) = default;
    TaskFuture& operator=(TaskFuture&&) = default;
    ~TaskFuture(void) {
      // Not threadsafe, just make sure accessed from single thread
      if (m_future.valid()) {
        m_future.get();
      }
    }

    auto get(void) -> T { return m_future.get(); }

  private:
    std::future<T> m_future;
  };

public:
  /**
   * Constructor.
   */
  ThreadPool(void) : ThreadPool{std::max(std::thread::hardware_concurrency(), 2u) - 1u} {
    /*
     * Always create at least one thread.  If hardware_concurrency() returns 0,
     * subtracting one would turn it to UINT_MAX, so get the maximum of
     * hardware_concurrency() and 2 before subtracting 1.
     */
  }

  /**
   * Constructor.
   */
  explicit ThreadPool(const std::uint32_t numThreads) : m_done{false}, m_workQueue{}, m_threads{} {
    try {
      for (std::uint32_t i = 0u; i < numThreads; ++i) {
        m_threads.emplace_back(&ThreadPool::worker, this);
      }
    } catch (...) {
      destroy();
      throw;
    }
  }

  /**
   * Non-copyable.
   */
  ThreadPool(const ThreadPool& rhs) = delete;

  /**
   * Non-assignable.
   */
  ThreadPool& operator=(const ThreadPool& rhs) = delete;

  /**
   * Destructor.
   */
  ~ThreadPool(void) { destroy(); }

  /**
   * Submit a job to be run by the thread pool.
   */
  template <typename Func, typename... Args>
  auto submit(Func&& func, Args&&... args)
      -> TaskFuture<std::invoke_result_t<std::decay_t<Func>, std::decay_t<Args>...>>

  {
    auto boundTask = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
    using ResultType = std::invoke_result_t<decltype(boundTask)>;
    using PackagedTask = std::packaged_task<ResultType()>;
    using TaskType = ThreadTask<PackagedTask>;

    PackagedTask task{std::move(boundTask)};
    TaskFuture<ResultType> result{task.get_future()};
    m_workQueue.push(std::make_unique<TaskType>(std::move(task)));
    return result;
  }

private:
  /**
   * Constantly running function each thread uses to acquire work items from the queue.
   */
  void worker(void) {
    while (!m_done) {
      std::unique_ptr<IThreadTask> pTask{nullptr};
      if (m_workQueue.waitPop(pTask)) {
        pTask->execute();
      }
    }
  }

  /**
   * Invalidates the queue and joins all running threads.
   */
  void destroy(void) {
    m_done = true;
    m_workQueue.invalidate();
    for (auto& thread : m_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

private:
  std::atomic_bool m_done;
  ThreadsafeQueue<std::unique_ptr<IThreadTask> > m_workQueue;
  std::vector<std::thread> m_threads;
};

namespace DefaultThreadPool {
/**
 * Get the default thread pool for the application.
 * This pool is created with std::thread::hardware_concurrency() - 1 threads.
 */
inline ThreadPool& getThreadPool() {
  static ThreadPool defaultPool;
  return defaultPool;
}

inline ThreadPool& getThreadPool_n(std::uint32_t numThreads) {
  static ThreadPool defaultPool{numThreads};
  return defaultPool;
}

/**
 * Submit a job to the default thread pool.
 */
template <typename Func, typename... Args>
inline auto submitJob(Func&& func, Args&&... args)
    -> ThreadPool::TaskFuture<std::invoke_result_t<Func, Args...>> {
  return getThreadPool().submit(std::forward<Func>(func), std::forward<Args>(args)...);
}
template <std::uint32_t n, typename Func, typename... Args>
inline auto submitJob_n(Func&& func, Args&&... args)
    -> ThreadPool::TaskFuture<std::invoke_result_t<Func, Args...>> {
  return getThreadPool_n(n).submit(std::forward<Func>(func), std::forward<Args>(args)...);
}
}  // namespace DefaultThreadPool

#endif
