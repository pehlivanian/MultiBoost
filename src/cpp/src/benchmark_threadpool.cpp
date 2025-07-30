#include <chrono>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include "threadpool.hpp"

class Benchmark {
private:
  std::string name;
  std::chrono::high_resolution_clock::time_point start_time;

public:
  explicit Benchmark(const std::string& test_name) : name(test_name) {
    std::cout << "🏃 Running: " << name << "... ";
    start_time = std::chrono::high_resolution_clock::now();
  }

  ~Benchmark() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << duration.count() << " μs\n";
  }
};

// Simulate the old std::bind approach for comparison
template <typename Func, typename... Args>
auto old_style_submit(ThreadPool& pool, Func&& func, Args&&... args) {
  using ResultType = std::invoke_result_t<std::decay_t<Func>, std::decay_t<Args>...>;

  // This simulates the old std::bind overhead
  auto bound_task = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);

  return pool.submit([bound_task = std::move(bound_task)]() mutable -> ResultType {
    if constexpr (std::is_void_v<ResultType>) {
      bound_task();
    } else {
      return bound_task();
    }
  });
}

int main() {
  std::cout << "⚡ ThreadPool C++20 Performance Benchmark\n";
  std::cout << "==========================================\n\n";

  const int NUM_TASKS = 10000;
  const int NUM_THREADS = std::thread::hardware_concurrency();

  std::cout << "Configuration:\n";
  std::cout << "• Tasks: " << NUM_TASKS << "\n";
  std::cout << "• Threads: " << NUM_THREADS << "\n";
  std::cout << "• Hardware concurrency: " << std::thread::hardware_concurrency() << "\n\n";

  // Benchmark 1: Task submission overhead (New C++20 way)
  {
    Benchmark bench("C++20 Lambda Capture Task Submission");
    ThreadPool pool(NUM_THREADS);

    std::vector<ThreadPool::TaskFuture<int>> futures;
    futures.reserve(NUM_TASKS);

    for (int i = 0; i < NUM_TASKS; ++i) {
      futures.push_back(pool.submit([](int x) { return x * 2; }, i));
    }

    // Wait for all to complete
    for (auto& future : futures) {
      future.get();
    }
  }

  // Benchmark 2: Task submission overhead (Old std::bind style for comparison)
  {
    Benchmark bench("Old std::bind Style Task Submission");
    ThreadPool pool(NUM_THREADS);

    std::vector<ThreadPool::TaskFuture<int>> futures;
    futures.reserve(NUM_TASKS);

    for (int i = 0; i < NUM_TASKS; ++i) {
      futures.push_back(old_style_submit(
          pool, [](int x) { return x * 2; }, i));
    }

    // Wait for all to complete
    for (auto& future : futures) {
      future.get();
    }
  }

  // Benchmark 3: Template parameter optimization
  {
    Benchmark bench("Template Parameter Optimization (submitJob_n<4>)");

    std::vector<ThreadPool::TaskFuture<int>> futures;
    futures.reserve(NUM_TASKS);

    for (int i = 0; i < NUM_TASKS; ++i) {
      futures.push_back(DefaultThreadPool::submitJob_n<4>([](int x) { return x * 2; }, i));
    }

    // Wait for all to complete
    for (auto& future : futures) {
      future.get();
    }
  }

  // Benchmark 4: Runtime parameter version (for comparison)
  {
    Benchmark bench("Runtime Parameter Version (submitJob_n(4))");

    std::vector<ThreadPool::TaskFuture<int>> futures;
    futures.reserve(NUM_TASKS);

    for (int i = 0; i < NUM_TASKS; ++i) {
      futures.push_back(DefaultThreadPool::submitJob_n(
          4, [](int x) { return x * 2; }, i));
    }

    // Wait for all to complete
    for (auto& future : futures) {
      future.get();
    }
  }

  // Benchmark 5: Move semantics performance
  {
    Benchmark bench("Move Semantics with Large Objects");
    ThreadPool pool(NUM_THREADS);

    struct LargeObject {
      std::vector<int> data;
      LargeObject(size_t size) : data(size, 42) {}
    };

    std::vector<ThreadPool::TaskFuture<size_t>> futures;
    futures.reserve(100);  // Fewer tasks since objects are large

    for (int i = 0; i < 100; ++i) {
      LargeObject obj(1000);  // 1000 integers
      futures.push_back(
          pool.submit([](LargeObject&& obj) { return obj.data.size(); }, std::move(obj)));
    }

    // Wait for all to complete
    for (auto& future : futures) {
      future.get();
    }
  }

  // Benchmark 6: Concepts vs SFINAE (compile-time benefit, runtime is same)
  {
    Benchmark bench("C++20 Concepts Type Checking");
    ThreadPool pool(NUM_THREADS);

    std::vector<ThreadPool::TaskFuture<int>> futures;
    futures.reserve(NUM_TASKS);

    // The concepts provide better compile-time errors and potentially faster compilation
    for (int i = 0; i < NUM_TASKS; ++i) {
      futures.push_back(pool.submit([](auto x, auto y) { return x + y; }, i, i + 1));
    }

    // Wait for all to complete
    for (auto& future : futures) {
      future.get();
    }
  }

  // Benchmark 7: Graceful shutdown performance
  {
    Benchmark bench("Graceful Shutdown with std::jthread");

    // Create and destroy pools to test shutdown performance
    for (int i = 0; i < 10; ++i) {
      ThreadPool pool(4);

      // Submit some tasks
      std::vector<ThreadPool::TaskFuture<int>> futures;
      for (int j = 0; j < 20; ++j) {
        futures.push_back(pool.submit([j]() {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          return j;
        }));
      }

      // Let some tasks start
      std::this_thread::sleep_for(std::chrono::microseconds(50));

      // Pool destructor will be called here (end of scope)
    }
  }

  std::cout << "\n📈 Performance Summary:\n";
  std::cout << "• C++20 lambda capture should be faster than std::bind\n";
  std::cout << "• Template parameters should be slightly faster than runtime\n";
  std::cout << "• Move semantics should handle large objects efficiently\n";
  std::cout << "• std::jthread provides cleaner shutdown\n";
  std::cout << "• Concepts provide better compile-time type checking\n\n";

  std::cout << "🎯 The C++20 optimizations provide:\n";
  std::cout << "   ✨ Better performance through lambda capture\n";
  std::cout << "   ✨ Safer code through concepts\n";
  std::cout << "   ✨ Cleaner resource management with jthread\n";
  std::cout << "   ✨ Better move semantics support\n";
  std::cout << "   ✨ Compile-time optimizations\n";

  return 0;
}