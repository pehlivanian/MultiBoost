#ifndef __THREADSAFEQUEUE_IMPL_HPP__
#define __THREADSAFEQUEUE_IMPL_HPP__

template <typename T>
ThreadsafeQueue<T>::~ThreadsafeQueue() {
  invalidate();
}

template <typename T>
bool ThreadsafeQueue<T>::tryPop(T& out) {
  std::lock_guard<std::mutex> lock{m_mutex};
  if (m_queue.empty() || !m_valid) {
    return false;
  }
  out = std::move(m_queue.front());
  m_queue.pop();
  return true;
}

template <typename T>
bool ThreadsafeQueue<T>::waitPop(T& out) {
  std::unique_lock<std::mutex> lock{m_mutex};
  m_condition.wait(lock, [this]() { return !m_queue.empty() || !m_valid; });
  if (!m_valid) {
    return false;
  }
  out = std::move(m_queue.front());
  m_queue.pop();
  return true;
}

template <typename T>
void ThreadsafeQueue<T>::push(T value) {
  {
    std::lock_guard<std::mutex> lock{m_mutex};
    m_queue.push(std::move(value));
  }
  m_condition.notify_one();
}

template <typename T>
std::size_t ThreadsafeQueue<T>::size() const {
  std::lock_guard<std::mutex> lock{m_mutex};
  return m_queue.size();
}

template <typename T>
bool ThreadsafeQueue<T>::empty() const {
  std::lock_guard<std::mutex> lock{m_mutex};
  return m_queue.empty();
}

template <typename T>
void ThreadsafeQueue<T>::clear() {
  std::lock_guard<std::mutex> lock{m_mutex};
  while (!m_queue.empty()) {
    m_queue.pop();
  }
  m_condition.notify_all();
}

template <typename T>
void ThreadsafeQueue<T>::invalidate() {
  std::lock_guard<std::mutex> lock{m_mutex};
  m_valid = false;
  m_condition.notify_all();
}

template <typename T>
bool ThreadsafeQueue<T>::isValid() const {
  std::lock_guard<std::mutex> lock{m_mutex};
  return m_valid;
}

#endif
