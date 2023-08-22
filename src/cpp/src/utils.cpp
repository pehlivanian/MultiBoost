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

  std::tuple<double,double> Perlich_rank_scores(const Row<double>& yhat, const Row<double>& y) {
    uvec yhat_index = sort_index(yhat);
    double T=0., R=0., tau, rho;
    int n = yhat.n_elem;
    for(int i=0; i<n; ++i) {
      for(int j=i+1; j<n; ++j) {
	if (y[yhat_index[i]] > y[yhat_index[j]]) {
	  T += 1.;
	  R += static_cast<double>(j-i);
	}
      }
    }

    double n_ = static_cast<double>(n);
    tau = 1. - (4.*T)/(n_*(n_-1));
    rho = 1. - (12.*R)/(n_*(n_-1)*(n_+1));
    return std::make_tuple(0.5 + tau/2., 0.5 + rho/2.);
  }

  std::tuple<double, double, double> precision(const Row<int>& y, const Row<int>& yhat) {
    // assume y has values in {+-1}
    double TP=0.,TN=0.,FP=0.,FN=0.;
    for (std::size_t i=0; i<y.n_cols; ++i) {
      if (y[i] == 1) {
	if (yhat[i] == 1)
	  TP += 1.;
	else
	  FN += 1.;
      } else {
	if (yhat[i] == -1)
	  TN += 1.;
	else
	  FP += 1.;
      }
    }
    // precision, recall, F1
    return std::make_tuple(TP/(TP+FP), TP/(TP+FN), 2*TP/(2*TP+FP+FN));
  }

  double imbalance(const Row<int>& y) {
    // assume y has values in {+-1}
    double n = static_cast<double>(y.n_cols);
    uvec ind0 = find(y == -1);
    uvec ind1 = find(y == 1);
    double num0 = static_cast<double>(ind0.n_cols);
    double num1 = static_cast<double>(ind1.n_cols);
    return 2.0 * (std::pow(num0/n - .5, 2) + std::pow(num1/n - .5, 2));
  }

  double imbalance(const Row<double>& y) {
    return imbalance(conv_to<Row<int>>::from(y));
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

  std::string fit_prefix(std::size_t depth) {
    std::string pref = "  ";
    if (depth) {
      pref += "|";
      for (std::size_t i=0; i<depth; ++i)
	pref += "____";
      pref += " ";
    }
    return pref;
  }

}
