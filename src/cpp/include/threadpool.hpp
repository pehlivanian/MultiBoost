#ifndef __THREADPOOL_HPP__
#define __THREADPOOL_HPP__

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "port_utils.hpp"
#include "threadsafequeue.hpp"

// C++20 concepts for better type safety
template <typename F, typename... Args>
concept Invocable = std::invocable<F, Args...>;

template <typename F, typename... Args>
concept InvocableWithResult =
    Invocable<F, Args...> && !std::is_void_v<std::invoke_result_t<F, Args...>>;

class ThreadPool {
private:
  class IThreadTask {
  public:
    IThreadTask() = default;
    virtual ~IThreadTask() = default;
    IThreadTask(const IThreadTask&) = delete;
    IThreadTask& operator=(const IThreadTask&) = delete;
    IThreadTask(IThreadTask&&) = default;
    IThreadTask& operator=(IThreadTask&&) = default;

    virtual void execute() = 0;
  };

  template <typename Func>
  class ThreadTask : public IThreadTask {
  public:
    explicit ThreadTask(Func&& func) : m_func{std::move(func)} {}

    ~ThreadTask() override = default;
    ThreadTask(const ThreadTask&) = delete;
    ThreadTask& operator=(const ThreadTask&) = delete;
    ThreadTask(ThreadTask&&) = default;
    ThreadTask& operator=(ThreadTask&&) = default;

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
   * Constructor - uses optimal thread count.
   */
  ThreadPool() : ThreadPool{std::max(std::jthread::hardware_concurrency(), 2u) - 1u} {
    /*
     * Always create at least one thread. If hardware_concurrency() returns 0,
     * subtracting one would turn it to UINT_MAX, so get the maximum of
     * hardware_concurrency() and 2 before subtracting 1.
     */
  }

  /**
   * Constructor with explicit thread count.
   */
  explicit ThreadPool(std::uint32_t numThreads) : m_workQueue{}, m_threads{} {
    m_threads.reserve(numThreads);
    try {
      for (std::uint32_t i = 0u; i < numThreads; ++i) {
        m_threads.emplace_back([this](std::stop_token stoken) { worker(stoken); });
      }
    } catch (...) {
      destroy();
      throw;
    }
  }

  /**
   * Non-copyable.
   */
  ThreadPool(const ThreadPool&) = delete;

  /**
   * Non-assignable.
   */
  ThreadPool& operator=(const ThreadPool&) = delete;

  /**
   * Destructor - automatically stops all threads.
   */
  ~ThreadPool() { destroy(); }

  /**
   * Submit a job to be run by the thread pool.
   * C++20 optimized version using perfect forwarding and lambda capture.
   */
  template <typename Func, typename... Args>
  requires Invocable<Func, Args...>
  auto submit(Func&& func, Args&&... args)
      -> TaskFuture<std::invoke_result_t<std::decay_t<Func>, std::decay_t<Args>...>> {
    using ResultType = std::invoke_result_t<std::decay_t<Func>, std::decay_t<Args>...>;
    using PackagedTask = std::packaged_task<ResultType()>;
    using TaskType = ThreadTask<PackagedTask>;

    // C++20: Replace std::bind with lambda capture for better performance
    auto task_lambda = [func = std::forward<Func>(func),
                        ... args = std::forward<Args>(args)]() mutable -> ResultType {
      if constexpr (std::is_void_v<ResultType>) {
        std::invoke(std::move(func), std::move(args)...);
      } else {
        return std::invoke(std::move(func), std::move(args)...);
      }
    };

    PackagedTask task{std::move(task_lambda)};
    TaskFuture<ResultType> result{task.get_future()};
    m_workQueue.push(std::make_unique<TaskType>(std::move(task)));
    return result;
  }

private:
  /**
   * Worker function using C++20 std::stop_token for cooperative cancellation.
   */
  void worker(std::stop_token stoken) {
    while (!stoken.stop_requested()) {
      std::unique_ptr<IThreadTask> pTask{nullptr};
      if (m_workQueue.waitPop(pTask)) {
        if (!stoken.stop_requested()) {
          pTask->execute();
        }
      }
    }
  }

  /**
   * Gracefully stops all threads using C++20 cooperative cancellation.
   */
  void destroy() {
    // Request stop for all threads
    for (auto& thread : m_threads) {
      thread.request_stop();
    }

    // Invalidate the queue to wake up waiting threads
    m_workQueue.invalidate();

    // Join all threads (jthread handles this automatically in destructor, but explicit is clearer)
    for (auto& thread : m_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

private:
  ThreadsafeQueue<std::unique_ptr<IThreadTask>> m_workQueue;
  std::vector<std::jthread> m_threads;
};

namespace DefaultThreadPool {
/**
 * Get the default thread pool for the application.
 * This pool is created with std::jthread::hardware_concurrency() - 1 threads.
 */
inline ThreadPool& getThreadPool() {
  static ThreadPool defaultPool;
  return defaultPool;
}

/**
 * Get a thread pool with a specific number of threads.
 * Uses C++20 template parameter for compile-time optimization.
 */
template <std::uint32_t NumThreads>
inline ThreadPool& getThreadPool_n() {
  static ThreadPool defaultPool{NumThreads};
  return defaultPool;
}

// Backward compatibility overload
inline ThreadPool& getThreadPool_n(std::uint32_t numThreads) {
  static ThreadPool defaultPool{numThreads};
  return defaultPool;
}

/**
 * Submit a job to the default thread pool.
 * C++20 optimized with concepts.
 */
template <typename Func, typename... Args>
requires Invocable<Func, Args...>
inline auto submitJob(Func&& func, Args&&... args)
    -> ThreadPool::TaskFuture<std::invoke_result_t<Func, Args...>> {
  return getThreadPool().submit(std::forward<Func>(func), std::forward<Args>(args)...);
}

/**
 * Submit a job to a fixed-size thread pool.
 * C++20 template parameter version for better performance.
 */
template <std::uint32_t NumThreads, typename Func, typename... Args>
requires Invocable<Func, Args...>
inline auto submitJob_n(Func&& func, Args&&... args)
    -> ThreadPool::TaskFuture<std::invoke_result_t<Func, Args...>> {
  return getThreadPool_n<NumThreads>().submit(
      std::forward<Func>(func), std::forward<Args>(args)...);
}

// Backward compatibility overload
template <typename Func, typename... Args>
requires Invocable<Func, Args...>
inline auto submitJob_n(std::uint32_t numThreads, Func&& func, Args&&... args)
    -> ThreadPool::TaskFuture<std::invoke_result_t<Func, Args...>> {
  return getThreadPool_n(numThreads).submit(std::forward<Func>(func), std::forward<Args>(args)...);
}
}  // namespace DefaultThreadPool

#endif
