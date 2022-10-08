#ifndef __UTILS_HPP__
#define __UTILS_HPP__

namespace Utils {
  struct distributionException : public std::exception {
    const char* what() const throw () {
      return "Bad distributional assignment";
    };
  };
}

#endif
