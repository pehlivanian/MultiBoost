#include "utils.hpp"

namespace IB_utils {

  std::string writeColMask(const uvec& colMask, boost::filesystem::path fldr) {
    
    ColMaskArchive cma{colMask};
    std::string fileName = dumps<ColMaskArchive, CerealIArch, CerealOArch>(cma, SerializedType::COLMASK, fldr);
    return fileName;
    
  }

  std::string writeDatasetIS(const mat& dataset, boost::filesystem::path fldr) {
    DatasetArchive da{dataset};
    std::string fileName = dumps<DatasetArchive, CerealIArch, CerealOArch>(da, SerializedType::DATASET_IS, fldr);
    return fileName;
  }

  std::string writeDatasetOOS(const mat& dataset, boost::filesystem::path fldr) {
    DatasetArchive da{dataset};
    std::string fileName = dumps<DatasetArchive, CerealIArch, CerealOArch>(da, SerializedType::DATASET_OOS, fldr);
    return fileName;    
  }

  double err(const Row<std::size_t>& yhat, const Row<std::size_t>& y) {
    return accu(yhat != y) * 100. / y.n_elem;
  }
  
  double err(const Row<double>& yhat, const Row<double>& y, double resolution) {
    if (resolution < 0.)
      resolution = 100. * std::numeric_limits<double>::epsilon();
    uvec ind = find( abs(yhat - y) > resolution);
    return static_cast<double>(ind.n_elem) * 100. / static_cast<double>(y.n_elem);
  }

  bool comp(std::pair<std::size_t, std::size_t>& a, std::pair<std::size_t, std::size_t>& b) {
    return a.second > b.second;
  }

  std::string writeIndex(const std::vector<std::string>& fileNames, std::string path, boost::filesystem::path fldr) {
    
    std::string abs_path;
    
    if (fldr.string().size()) {
      fldr /= path;
      abs_path = fldr.string();
    } else {
      abs_path = path;
    }
    
    std::ofstream ofs{abs_path, std::ios::binary|std::ios::out};
    if (ofs.is_open()) {
      for (auto &fileName : fileNames) {
	ofs << fileName << "\n";
      }
    }
    ofs.close();

    return path;
  }

  std::string writeIndex(const std::vector<std::string>& fileNames, boost::filesystem::path fldr) {

    auto now = std::chrono::system_clock::now();
    // auto UTC = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    auto UTC = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
    std::string path = std::to_string(UTC) + "_index_" + datetime.str() + ".gnd";
    
    return writeIndex(fileNames, path, fldr);
  }

  void readIndex(std::string path, std::vector<std::string>& fileNames, boost::filesystem::path fldr) {

    std::string abs_path;

    if (fldr.string().size()) {
      fldr /= path;
      abs_path = fldr.string();
    } else {
      abs_path = path;
    }

    std::ifstream ifs{abs_path, std::ios::binary|std::ios::in};
    std::vector<std::string>().swap(fileNames);
    std::string fileName;

    if (ifs.is_open()) {
      while (ifs >> fileName) {
	fileNames.push_back(fileName);
      }
    }
    ifs.close();
  }

  void mergeIndices(std::string pathOld, std::string pathNew, boost::filesystem::path fldr, bool removeOld) {

    std::string pathOut = pathNew;

    if (fldr.string().size()) {
      boost::filesystem::path fldrOld{fldr}, fldrNew{fldr};

      fldrOld /= pathOld;
      pathOld = fldrOld.string();

      fldrNew /= pathNew;
      pathNew = fldrNew.string();      
    }

    // Prepend information in old file to new file, save in updated 
    // new file
    std::ifstream ifsOld{pathOld, std::ios::binary|std::ios::in};
    std::ifstream ifsNew{pathNew, std::ios::binary|std::ios::in};
    std::vector<std::string> fileNames;

    std::string fileName;
    
    if (ifsOld.is_open()) {
      while (ifsOld >> fileName) {
	fileNames.push_back(fileName);
      }
    }
    ifsOld.close();
    
    if (ifsNew.is_open()) {
      while (ifsNew >> fileName) {
	fileNames.push_back(fileName);
      }
    }
    ifsNew.close();

    if (removeOld)
      boost::filesystem::remove(pathOld);

    writeIndex(fileNames, pathOut, fldr);
    
  }

}
