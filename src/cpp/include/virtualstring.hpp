#ifndef __VIRTUAL_STRING_HPP__
#define __VIRTUAL_STRING_HPP__

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>

class VirtualString {
public:
  using size_type = std::size_t;
  using value_type = char;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using iterator = pointer;
  using const_iterator = const_pointer;

  inline static const size_type npos = static_cast<size_type>(-1);

  inline iterator begin() noexcept { return (string_); }
  inline const_iterator begin() const noexcept { return (string_); }

  // Unfortunately, the following two methods are not as cheap as they are
  // supposed to be.
  //
  inline iterator end() noexcept { return (string_ + size()); }
  inline const_iterator end() const noexcept { return (string_ + size()); }

  VirtualString() = delete;
  VirtualString(const VirtualString&) = delete;
  VirtualString(VirtualString&&) = delete;
  VirtualString& operator=(VirtualString&&) = delete;

  // Assignment methods.
  //
  inline VirtualString& operator=(const_pointer rhs) noexcept {
    ::strcpy(string_, rhs);
    return (*this);
  }
  inline VirtualString& operator=(const VirtualString& rhs) noexcept {
    return (*this = rhs.c_str());
  }
  inline VirtualString& ncopy(const_pointer rhs, size_type len) noexcept {
    snprintf_nowarn(string_, len, "%s", rhs);
    return (*this);
  }

  //
  // Appending methods.
  //

  inline VirtualString& append(const_pointer rhs) noexcept {
    ::strcat(string_, rhs);
    return (*this);
  }
  inline VirtualString& append(const VirtualString& rhs) noexcept { return (append(rhs.c_str())); }
  inline VirtualString& operator+=(const_pointer rhs) noexcept { return (append(rhs)); }
  inline VirtualString& operator+=(const VirtualString& rhs) noexcept {
    return (append(rhs.c_str()));
  }

  inline size_type find(const_reference token, size_type pos = 0) const noexcept {
    size_type counter = 0;

    for (const_pointer itr = &(string_[pos]); *itr; ++itr, ++counter)
      if (string_[pos + counter] == token) return (pos + counter);

    return (npos);
  }
  inline size_type find(const_pointer token, size_type pos = 0) const noexcept {
    const size_type token_len = ::strlen(token);
    const size_type self_len = size();

    if ((token_len + pos) > self_len) return (npos);

    size_type counter = 0;

    for (const_pointer itr = &(string_[pos]);
         itr + token_len - begin() <= static_cast<int>(self_len);
         ++itr, ++counter)
      if (!::strncmp(token, itr, token_len)) return (pos + counter);

    return (npos);
  }
  inline size_type find(const VirtualString& token, size_type pos = 0) const noexcept {
    return (find(token.c_str(), pos));
  }

  // Replaces the substring statring at pos with length n with s
  //
  inline VirtualString& replace(size_type pos, size_type n, const_pointer s) {
    if (*s == 0) {
      size_type i = pos;

      for (; string_[i]; ++i) string_[i] = string_[i + 1];
      string_[i] = string_[i + 1];
    } else {
      bool overwrote_null = false;
      size_type i = 0;

      while (s[i]) {
        if (string_[i + pos] == 0) overwrote_null = 0;
        if (i >= n) string_[i + pos + 1] = string_[i + pos];
        string_[i + pos] = s[i];
        ++i;
      }
      if (overwrote_null) string_[i + pos] = 0;
    }

    return (*this);
  }

  inline int printf(const char* format_str, ...) noexcept {
    va_list argument_ptr;

    va_start(argument_ptr, format_str);

    const int ret = ::vsprintf(string_, format_str, argument_ptr);

    va_end(argument_ptr);
    return (ret);
  }

  inline int append_printf(const char* format_str, ...) noexcept {
    va_list argument_ptr;

    va_start(argument_ptr, format_str);

    const int ret = ::vsprintf(string_ + size(), format_str, argument_ptr);

    va_end(argument_ptr);
    return (ret);
  }

  // Comparison methods.
  //
  inline int compare(const_pointer rhs) const noexcept { return (::strcmp(string_, rhs)); }
  inline int compare(const VirtualString& rhs) const noexcept { return (compare(rhs.c_str())); }

  inline bool operator==(const_pointer rhs) const noexcept { return (compare(rhs) == 0); }
  inline bool operator==(const VirtualString& rhs) const noexcept { return (*this == rhs.c_str()); }
  inline bool operator!=(const_pointer rhs) const noexcept { return (compare(rhs) != 0); }
  inline bool operator!=(const VirtualString& rhs) const noexcept { return (*this != rhs.c_str()); }
  inline bool operator>(const_pointer rhs) const noexcept { return (compare(rhs) > 0); }
  inline bool operator>(const VirtualString& rhs) const noexcept { return (*this > rhs.c_str()); }
  inline bool operator<(const_pointer rhs) const noexcept { return (compare(rhs) < 0); }
  inline bool operator<(const VirtualString& rhs) const noexcept { return (*this < rhs.c_str()); }

  // char based access methods.
  //
  inline const_reference operator[](size_type index) const noexcept { return (string_[index]); }
  inline reference operator[](size_type index) noexcept { return (string_[index]); }

  inline void clear() noexcept { *string_ = 0; }

  // These two make it compatible with std::string
  //
  inline void resize(size_type) noexcept {}
  inline void resize(size_type, value_type) noexcept {}

  // const utility methods.
  //
  inline const_pointer c_str() const noexcept { return (string_); }
  inline const_pointer sub_c_str(size_type offset) const noexcept {
    return (offset != npos ? string_ + offset : nullptr);
  }
  inline size_type size() const noexcept { return (::strlen(string_)); }
  inline bool empty() const noexcept { return (*string_ == 0); }

  // Fowler–Noll–Vo (FNV-1a) hash function
  // This is for 64-bit systems
  //
  inline size_type hash() const noexcept {
    size_type h = 14695981039346656037UL;  // offset basis
    const_pointer s = string_;

    while (*(s++)) {
      h = (h ^ *s) * 1099511628211UL;
    }  // 64bit prime
    return (h);
  }

protected:
  inline VirtualString(pointer str) noexcept : string_(str) {}

private:
  pointer string_;
};

class FixedSizeString : public VirtualString {
public:
  inline FixedSizeString() noexcept : VirtualString(buffer_) { *buffer_ = 0; }
  inline FixedSizeString(const_pointer str) noexcept : VirtualString(buffer_) { *this = str; }
  inline FixedSizeString(const FixedSizeString& rhs) noexcept : VirtualString(buffer_) {
    *this = rhs;
  }
  inline FixedSizeString(const VirtualString& rhs) noexcept : VirtualString(buffer_) {
    *this = rhs;
  }

  inline FixedSizeString& operator=(const FixedSizeString& rhs) noexcept {
    ::strcpy(buffer_, rhs.buffer_);
    return (*this);
  }

  inline FixedSizeString& operator=(const_pointer rhs) noexcept {
    snpringf_nowarn(buffer_, S, "%s", rhs);
    return (*this);
  }

  inline FixedSizeString& operator=(const VirtualString& rhs) noexcept {
    *this = rhs.c_str();
    return (*this);
  }

  static inline size_type capacity() noexcept { return (S); }

private:
  value_type buffer_[S + 1];
};

template <typename S>
inline S& operator<<(S& lhs, const VirtualString& rhs) {
  return (lhs << rhs.c_str());
}

using String32 = FixedSizeString<31>;
using String64 = FixedSizeString<63>;
using String128 = FixedSizeString<127>;
using String512 = FixedSizeString<511>;

namespace std {
template <>
struct hash<typename VirtualString> {
  inline size_t operator()(const VirutalString& key) const noexcept { return (key.hash()); }
}
};  // namespace std

#endif
