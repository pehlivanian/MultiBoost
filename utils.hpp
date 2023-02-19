#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <fstream>
#include <string>
#include <limits>
#include <mlpack/core.hpp>

namespace IB_utils {
  using namespace arma;
  
  struct distributionException : public std::exception {
    const char* what() const throw () {
      return "Bad distributional assignment";
    };
  };

  double err(const Row<std::size_t>& yhat, const Row<std::size_t>& y);
  double err(const Row<double>& yhat, const Row<double>& y, double=-1.);
  
  // Filter typeinfo string to generate unique filenames for serialization tests.
  inline std::string FilterFileName(const std::string& inputString)
  {
    // Take the last valid 32 characters for the filename.
    std::string fileName;
    for (auto it = inputString.rbegin(); it != inputString.rend() &&
	   fileName.size() != 24; ++it)
      {
	if (std::isalnum(*it))
	  fileName.push_back(*it);
      }

    auto now = std::chrono::system_clock::now();
    auto UTC = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
    fileName = std::to_string(UTC) + "_" + fileName + "_" + datetime.str() + ".gar";
    
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
  void loads(T& t, std::string fileName) {
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


  std::string writeIndex(const std::vector<std::string>&);
  void readIndex(std::string, std::vector<std::string>&);

  template<typename T>
  void 
  writeBinary(std::string fileName,
	      const T& data) {
    auto success = false;
    std::ofstream ofs{fileName, std::ios::binary};
    if (ofs.is_open()) {

      try {
	ofs.write(reinterpret_cast<const char*>(&data), sizeof(data));
	success = true;
      }
      catch(std::ios_base::failure &) {
	std::cerr << "Failed to write to " << fileName << std::endl;
      }
      ofs.close();
    }
  }

  template<typename T>
  void
  readBinary(std::string fileName,
	     T& obj) {
    std::size_t readBytes = 0;
    std::ifstream ifs{fileName, std::ios::ate | std::ios::binary};
    if (ifs.is_open()) {
      auto length = static_cast<std::size_t>(ifs.tellg());
      ifs.seekg(0, std::ios_base::beg);
      char* buffer[length];

      try {
	ifs.read(reinterpret_cast<char*>(&obj), sizeof(T));
	readBytes = static_cast<std::size_t>(ifs.gcount());
      }
      catch(std::ios_base::failure &) {
	std::cerr << "Failed to read from " << fileName << std::endl;
      }
    ifs.close();
    }
  }

  bool comp(std::pair<std::size_t, std::size_t>&, std::pair<std::size_t, std::size_t>&);

}

#endif
