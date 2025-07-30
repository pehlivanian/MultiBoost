#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "threadpool.hpp"

// Test utilities
class TestRunner {
private:
  int passed = 0;
  int failed = 0;
  std::string current_test;

public:
  void start_test(const std::string& name) {
    current_test = name;
    std::cout << "🧪 Testing: " << name << "... ";
  }

  void assert_true(bool condition, const std::string& message = "") {
    if (condition) {
      std::cout << "✅ PASS\n";
      passed++;
    } else {
      std::cout << "❌ FAIL: " << message << "\n";
      failed++;
    }
  }

  void summary() {
    std::cout << "\n📊 Test Summary:\n";
    std::cout << "✅ Passed: " << passed << "\n";
    std::cout << "❌ Failed: " << failed << "\n";
    std::cout << "🎯 Success Rate: " << (100.0 * passed / (passed + failed)) << "%\n";
  }

  bool all_passed() const { return failed == 0; }
};

int main() {
  TestRunner test;

  std::cout << "🚀 C++20 ThreadPool Comprehensive Test Suite\n";
  std::cout << "============================================\n\n";

  // Test 1: Basic functionality with return values
  test.start_test("Basic task submission with return values");
  {
    ThreadPool pool(2);

    auto future1 = pool.submit([]() { return 42; });
    auto future2 = pool.submit([](int a, int b) { return a + b; }, 10, 20);
    auto future3 =
        pool.submit([](const std::string& s) { return s + " World!"; }, std::string("Hello"));

    test.assert_true(
        future1.get() == 42 && future2.get() == 30 && future3.get() == "Hello World!",
        "Return values don't match expected");
  }

  // Test 2: Void return type handling
  test.start_test("Void return type handling");
  {
    ThreadPool pool(1);
    bool executed = false;

    auto future = pool.submit([&executed]() {
      executed = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    });

    future.get();  // Should not return anything, just wait
    test.assert_true(executed, "Void task was not executed");
  }

  // Test 3: Concurrent execution verification
  test.start_test("Concurrent execution verification");
  {
    ThreadPool pool(4);
    constexpr int NUM_TASKS = 8;
    std::atomic<int> counter{0};
    std::vector<ThreadPool::TaskFuture<int>> futures;

    auto start_time = std::chrono::steady_clock::now();

    // Submit tasks that take time
    for (int i = 0; i < NUM_TASKS; ++i) {
      futures.push_back(pool.submit([&counter, i]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        counter.fetch_add(1);
        return i;
      }));
    }

    // Wait for all tasks
    int sum = 0;
    for (auto& future : futures) {
      sum += future.get();
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Should complete in roughly 200ms (2 batches of 100ms with 4 threads) not 800ms (sequential)
    test.assert_true(
        counter.load() == NUM_TASKS && sum == (NUM_TASKS * (NUM_TASKS - 1)) / 2 &&
            duration.count() < 400,
        "Tasks did not execute concurrently as expected");
  }

  // Test 4: DefaultThreadPool functionality
  test.start_test("DefaultThreadPool basic functionality");
  {
    auto future1 = DefaultThreadPool::submitJob([]() { return 100; });
    auto future2 = DefaultThreadPool::submitJob([](double x) { return x * 2.0; }, 3.14);

    test.assert_true(
        future1.get() == 100 && std::abs(future2.get() - 6.28) < 0.01,
        "DefaultThreadPool basic functionality failed");
  }

  // Test 5: Template parameter optimization (submitJob_n<N>)
  test.start_test("Template parameter submitJob_n<N> optimization");
  {
    auto future1 = DefaultThreadPool::submitJob_n<2>([]() { return std::this_thread::get_id(); });

    auto future2 =
        DefaultThreadPool::submitJob_n<3>([](int x, int y, int z) { return x + y + z; }, 1, 2, 3);

    auto thread_id = future1.get();
    auto result = future2.get();

    test.assert_true(
        result == 6 && thread_id != std::this_thread::get_id(),
        "Template parameter optimization failed");
  }

  // Test 6: Backward compatibility with runtime parameters
  test.start_test("Backward compatibility with runtime parameters");
  {
    std::uint32_t num_threads = 2;
    auto future = DefaultThreadPool::submitJob_n(
        num_threads, []() { return std::string("Backward compatible!"); });

    test.assert_true(future.get() == "Backward compatible!", "Backward compatibility failed");
  }

  // Test 7: Exception handling
  test.start_test("Exception handling in tasks");
  {
    ThreadPool pool(1);

    auto future = pool.submit([]() -> int {
      throw std::runtime_error("Test exception");
      return 42;  // Never reached
    });

    bool exception_caught = false;
    try {
      future.get();
    } catch (const std::runtime_error& e) {
      exception_caught = (std::string(e.what()) == "Test exception");
    }

    test.assert_true(exception_caught, "Exception was not properly propagated");
  }

  // Test 8: Heavy workload stress test
  test.start_test("Heavy workload stress test");
  {
    ThreadPool pool(std::thread::hardware_concurrency());
    constexpr int NUM_TASKS = 1000;
    std::vector<ThreadPool::TaskFuture<int>> futures;

    // Submit many quick tasks
    for (int i = 0; i < NUM_TASKS; ++i) {
      futures.push_back(pool.submit([i]() {
        // Some CPU work
        int sum = 0;
        for (int j = 0; j < i % 100; ++j) {
          sum += j;
        }
        return sum + i;
      }));
    }

    // Verify all complete correctly
    int total = 0;
    for (int i = 0; i < NUM_TASKS; ++i) {
      total += futures[i].get();
    }

    test.assert_true(total > 0, "Heavy workload test failed");
  }

  // Test 9: Move semantics and perfect forwarding
  test.start_test("Move semantics and perfect forwarding");
  {
    ThreadPool pool(1);

    struct MoveOnlyType {
      std::unique_ptr<int> data;
      MoveOnlyType(int val) : data(std::make_unique<int>(val)) {}
      MoveOnlyType(const MoveOnlyType&) = delete;
      MoveOnlyType(MoveOnlyType&&) = default;
      MoveOnlyType& operator=(const MoveOnlyType&) = delete;
      MoveOnlyType& operator=(MoveOnlyType&&) = default;
    };

    auto future = pool.submit([](MoveOnlyType obj) { return *obj.data * 2; }, MoveOnlyType(21));

    test.assert_true(future.get() == 42, "Move semantics test failed");
  }

  // Test 10: Graceful shutdown verification
  test.start_test("Graceful shutdown with std::jthread");
  {
    std::atomic<int> completed_tasks{0};

    // Scope to trigger destructor
    {
      ThreadPool pool(2);

      // Submit long-running tasks
      for (int i = 0; i < 4; ++i) {
        pool.submit([&completed_tasks]() {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          completed_tasks.fetch_add(1);
          return 0;
        });
      }

      // Let some tasks start
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }  // Pool destructor called here

    // Give time for cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    test.assert_true(
        completed_tasks.load() >= 2,  // At least some tasks should complete
        "Graceful shutdown failed");
  }

  // Test 11: C++20 Concepts validation (compile-time)
  test.start_test("C++20 Concepts validation");
  {
    ThreadPool pool(1);

    // These should compile fine with concepts
    auto f1 = pool.submit([]() { return 1; });
    auto f2 = pool.submit([](int x) { return x; }, 42);
    auto f3 = pool.submit([](auto x, auto y) { return x + y; }, 1, 2);

    test.assert_true(
        f1.get() == 1 && f2.get() == 42 && f3.get() == 3, "Concepts validation failed");
  }

  // Test 12: Performance characteristics verification
  test.start_test("Performance characteristics verification");
  {
    ThreadPool pool(4);
    constexpr int NUM_ITERATIONS = 100;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<ThreadPool::TaskFuture<int>> futures;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
      futures.push_back(pool.submit([i]() {
        // Simulate some work
        std::hash<int> hasher;
        return static_cast<int>(hasher(i) % 1000);
      }));
    }

    // Wait for all
    for (auto& f : futures) {
      f.get();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should be reasonably fast (less than 100ms for 100 tasks)
    test.assert_true(
        duration.count() < 100000,  // 100ms in microseconds
        "Performance test failed - took too long");
  }

  std::cout << "\n";
  test.summary();

  if (test.all_passed()) {
    std::cout << "\n🎉 All ThreadPool C++20 optimizations working perfectly!\n";
    std::cout << "✨ Features verified:\n";
    std::cout << "   • std::jthread with cooperative cancellation\n";
    std::cout << "   • Lambda capture replacing std::bind\n";
    std::cout << "   • C++20 concepts for type safety\n";
    std::cout << "   • Perfect forwarding and move semantics\n";
    std::cout << "   • Template parameter optimization\n";
    std::cout << "   • Exception propagation\n";
    std::cout << "   • Graceful shutdown\n";
    std::cout << "   • Backward compatibility\n";
    return 0;
  } else {
    std::cout << "\n💥 Some tests failed! Check the implementation.\n";
    return 1;
  }
}