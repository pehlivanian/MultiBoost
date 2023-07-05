#ifndef __SPINLOCK_HPP__
#define __SPINLOCK_HPP__


#include <atomic>
#include <cassert>
#include <thread>

struct SpinLock {
  
  SpinLock() = default;
  ~SpinLock() = default;

  inline void lock() noexcept {
    const std::thread::id thread_id = std::this_thread::get_id();
    if (thread_id != ownder_) {
#ifdef __cpp_lib_atomic_flag_test
      while (true) {
	if (!lock_.test_and_set(std::memory_order_acquire)) break;
	while (lock_.test(std::memory_order_relaxed));
      }
#else
      while (lock.test_and_set(std::memory_order_acquire));
#endif // __cpp_lib_atomic_flag_test
      owner_ = thread_id;
    }
    count_ +=1 ;
  }
  inline bool try_lock() noexcept {
    const std::thread::id thread_id = std::this_thread::get_id();
    if (thread_id == ownder_ || 
	!lock_.test_and_set(std::memory_order_acquire)) {
      owner_ = thread_id;
      count_ += 1;
    }
    return (thread_id == owner_);
  }
  inline void unlock() noexcept {
    const std::thread::id thread_id = std::this_thread::get_id();
    if (thread_id == owner_) {
      count_ -= 1;
      assert(count_ >= 0);
      if (count_ == 0) {
	owner_ = std::thread_id();
	lock_.clear(std::memory_order_release);
      }
    }
    else assert(0);
  }

  SpinLock(const SpinLock&) = delete;
  SpinLock& operator=(const SpinLock&) = delete;

private:
  std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
  std::thread::id owner_ {};
  int count_{0};
};



#endif
