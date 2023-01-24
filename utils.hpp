#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <fstream>
#include <string>
#include <mlpack/core.hpp>

namespace IB_utils {
  using namespace arma;

  struct distributionException : public std::exception {
    const char* what() const throw () {
      return "Bad distributional assignment";
    };
  };

  double err(const Row<std::size_t>& yhat, const Row<std::size_t>& y);
  
  // Filter typeinfo string to generate unique filenames for serialization tests.
  inline std::string FilterFileName(const std::string& inputString);

  template<typename T, typename IArchiveType, typename OArchiveType>
  void SerializeObject(T& t, T& newT);
  /* e.g.
     SerializeObject<T, cereal::XMLInputArchive, cereal::XMLOutputArchive>
     SerializeObject<T, cereal::JSONInputArchive cereal::JSONOutputArchive>
     SerializeObject<T, cereal::BinaryInputArchive, cereal::BinaryOutputArchive>

  */

  template<typename T, typename IArchiveType, typename OArchiveType>
  void SerializePointerObject(T* t, T*& newT);

}

#endif
