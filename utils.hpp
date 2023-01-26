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
  inline std::string FilterFileName(const std::string& inputString)
  {
    // Take the last valid 32 characters for the filename.
    std::string fileName;
    for (auto it = inputString.rbegin(); it != inputString.rend() &&
	   fileName.size() != 32; ++it)
      {
	if (std::isalnum(*it))
	  fileName.push_back(*it);
      }
    
    return fileName;
  }
    
  template<typename T, typename IArchiveType, typename OArchiveType>
  std::string dumps(T& t) {
    std::string fileName = FilterFileName(typeid(T).name());
    std::ofstream ofs(fileName, std::ios::binary);
    {
      OArchiveType o(ofs);

      T& x(t);
      o(CEREAL_NVP(x));      
    }
    ofs.close();

    return fileName;
  }

  template<typename T, typename IArchiveType, typename OArchiveType>
  void loads(T& t) {
    std::string fileName = FilterFileName(typeid(T).name());
    std::ifstream ifs{fileName, std::ios::binary};
    
    {
      IArchiveType i(ifs);
      T& x(t);
      i(CEREAL_NVP(x));
    }
    ifs.close();

  }

  template<typename T, typename IArchiveType, typename OArchiveType>
  void SerializeObject(T& t, T& newT)
  {
    std::string fileName = FilterFileName(typeid(T).name());
    std::ofstream ofs(fileName, std::ios::binary);
    
    {
      OArchiveType o(ofs);
      
      T& x(t);
      o(CEREAL_NVP(x));
    }
    ofs.close();

    std::ifstream ifs(fileName, std::ios::binary);

    {
      IArchiveType i(ifs);
      T& x(newT);
      i(CEREAL_NVP(x));
    }
    ifs.close();
	
    // remove(fileName.c_str());
  }

  template<typename T, typename IArchiveType, typename OArchiveType>
  void SerializePointerObject(T* t, T*& newT)
  {
    std::string fileName = FilterFileName(typeid(T).name());
    std::ofstream ofs(fileName, std::ios::binary);

    {
      OArchiveType o(ofs);
      o(CEREAL_POINTER(t));
    }
    ofs.close();

    std::ifstream ifs(fileName, std::ios::binary);

    {
      IArchiveType i(ifs);
      i(CEREAL_POINTER(newT));
    }
    ifs.close();
    // remove(fileName.c_str());
  }

  /* e.g.
     SerializeObject<T, cereal::XMLInputArchive, cereal::XMLOutputArchive>
     SerializeObject<T, cereal::JSONInputArchive cereal::JSONOutputArchive>
     SerializeObject<T, cereal::BinaryInputArchive, cereal::BinaryOutputArchive>

  */

  template<typename T, typename IArchiveType, typename OArchiveType>
  void SerializePointerObject(T* t, T*& newT);

}

#endif
