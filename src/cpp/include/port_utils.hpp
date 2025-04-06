#ifndef __PORT_UTILS__
#define __PORT_UTILS__

#if IS_CXX_11 || (__cplusplus == 201103L)
namespace std {
template <typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... params) {
  return unique_ptr<T>(new T(std::forward<Args>(params)...));
}
}  // namespace std
#endif
#endif
