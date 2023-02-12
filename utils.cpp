#include "utils.hpp"

namespace IB_utils {
  
  double err(const Row<std::size_t>& yhat, const Row<std::size_t>& y) {
    return accu(yhat != y) * 100. / y.n_elem;
  }
  
  double err(const Row<double>& yhat, const Row<double>& y, double resolution) {
    if (resolution < 0.)
      resolution = 100. * std::numeric_limits<double>::epsilon();
    uvec ind = find( abs(yhat - y) > resolution);
    return static_cast<double>(ind.n_elem) * 100. / static_cast<double>(y.n_elem);
  }

  std::string writeIndex(const std::vector<std::string>& fileNames) {
    
    auto now = std::chrono::system_clock::now();
    auto UTC = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
    std::string path = std::to_string(UTC) + "_index_" + datetime.str() + ".gnd";
    
    std::ofstream ofs{path, std::ios::binary|std::ios::out};
    if (ofs.is_open()) {
      for (auto &fileName : fileNames) {
	ofs << fileName << "\n";
      }
    }
    ofs.close();

    return path;
  }

  void readIndex(std::string path, std::vector<std::string>& fileNames) {
    std::ifstream ifs{path, std::ios::binary|std::ios::in};
    std::vector<std::string>().swap(fileNames);
    std::string fileName;

    if (ifs.is_open()) {
      while (ifs >> fileName) {
      fileNames.push_back(fileName);
      }
    }
    ifs.close();
  }

}
