#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <vector>
#include <limits>
#include <chrono>
#include <memory>
#include <type_traits>
#include <mlpack/core.hpp>

#include <boost/filesystem.hpp>

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/access.hpp>

#include "loss.hpp"

// using fs = CXX_FILESYSTEM_NAMESPACE;
using namespace boost::filesystem;
using namespace LossMeasures;

namespace LearningRate {
  enum class LearningRateMethod {
    FIXED = 0,
      INCREASING = 1,
      DECREASING = 2,
      };

} // namespace LearningRate

namespace PartitionSize {
  enum class PartitionSizeMethod { 
    FIXED = 0,
      FIXED_PROPORTION = 1,
      DECREASING = 2,
      INCREASING = 3,
      RANDOM = 4,
      MULTISCALE = 5
      };
  
} // namespace PartitionSize

namespace StepSize {
  enum class StepSizeMethod {
      LOG = 0,
      PROPORTION = 1
  };
} // namespace StepSize

namespace ModelContext{

  struct Context {

    Context() :
      baseSteps{-1},
      removeRedundantLabels{false},
      quietRun{false},
      recursiveFit{false},
      partitionSizeMethod{PartitionSize::PartitionSizeMethod::FIXED},
      learningRateMethod{LearningRate::LearningRateMethod::FIXED},
      stepSizeMethod{StepSize::StepSizeMethod::LOG},
      childPartitionSize{std::vector<std::size_t>()},
      childNumSteps{std::vector<std::size_t>()},
      childLearningRate{std::vector<double>()},
      childMinLeafSize{std::vector<std::size_t>()},
      childMaxDepth{std::vector<std::size_t>()},
      childMinimumGainSplit{std::vector<double>()},
      serializeModel{false},
      serializePrediction{false},
      serializeColMask{false},
      serializeDataset{false},
      serializeLabels{false},
      serializationWindow{500},
      depth{0}
    {}

    Context(const Context& rhs) {
      loss = rhs.loss;
      clamp_gradient = rhs.clamp_gradient;
      upper_val = rhs.upper_val;
      lower_val = rhs.lower_val;
      partitionSize = rhs.partitionSize;
      partitionRatio = rhs.partitionRatio;
      learningRate = rhs.learningRate;
      steps = rhs.steps;
      if (rhs.baseSteps > 0) {
	baseSteps = rhs.baseSteps;
      } else {
	baseSteps = rhs.steps;
      }
      symmetrizeLabels = rhs.symmetrizeLabels;
      removeRedundantLabels = rhs.removeRedundantLabels;
      quietRun = rhs.quietRun;
      rowSubsampleRatio = rhs.rowSubsampleRatio;
      colSubsampleRatio = rhs.colSubsampleRatio;
      recursiveFit = rhs.recursiveFit;
      partitionSizeMethod = rhs.partitionSizeMethod;
      learningRateMethod = rhs.learningRateMethod;
      stepSizeMethod = rhs.stepSizeMethod;
      childPartitionSize = rhs.childPartitionSize;
      childNumSteps = rhs.childNumSteps;
      childLearningRate = rhs.childLearningRate;
      childMinLeafSize = rhs.childMinLeafSize;
      childMaxDepth = rhs.childMaxDepth;
      childMinimumGainSplit = rhs.childMinimumGainSplit;
      minLeafSize = rhs.minLeafSize;
      maxDepth = rhs.maxDepth;
      minimumGainSplit = rhs.minimumGainSplit;
      numTrees = rhs.numTrees;
      serializeModel = rhs.serializeModel;
      serializePrediction = rhs.serializePrediction;
      serializeColMask = rhs.serializeColMask;
      serializeDataset = rhs.serializeDataset;
      serializeLabels = rhs.serializeLabels;
      serializationWindow = rhs.serializationWindow;
      depth = rhs.depth;

    }

    template<class Archive>
    void serialize(Archive &ar) {
      ar(loss);
      ar(clamp_gradient);
      ar(upper_val);
      ar(lower_val);
      ar(partitionSize);
      ar(partitionRatio);
      ar(learningRate);
      ar(steps);
      ar(baseSteps);
      ar(symmetrizeLabels);
      ar(removeRedundantLabels);
      ar(quietRun);
      ar(rowSubsampleRatio);
      ar(colSubsampleRatio);
      ar(recursiveFit);
      ar(partitionSizeMethod);
      ar(learningRateMethod);
      ar(stepSizeMethod);
      ar(childPartitionSize);
      ar(childNumSteps);
      ar(childLearningRate);
      ar(childMinLeafSize);
      ar(childMinimumGainSplit);
      ar(childMaxDepth);
      ar(numTrees);
      ar(serializeModel);
      ar(serializePrediction);
      ar(serializeColMask);
      ar(serializeDataset);
      ar(serializeLabels);
      ar(serializationWindow);
      ar(depth);
    }
      
    lossFunction loss;
    bool clamp_gradient;
    double upper_val;
    double lower_val;
    std::size_t partitionSize;
    double partitionRatio;
    double learningRate;
    std::size_t minLeafSize;
    std::size_t maxDepth;
    double minimumGainSplit;
    std::size_t numTrees;
    int steps;
    int baseSteps;
    bool symmetrizeLabels;
    bool removeRedundantLabels;
    bool quietRun;
    double rowSubsampleRatio;
    double colSubsampleRatio;
    bool recursiveFit;
    PartitionSize::PartitionSizeMethod partitionSizeMethod;
    LearningRate::LearningRateMethod learningRateMethod;
    StepSize::StepSizeMethod stepSizeMethod;
    std::vector<std::size_t> childPartitionSize;
    std::vector<std::size_t> childNumSteps;
    std::vector<double> childLearningRate;
    std::vector<std::size_t> childMinLeafSize;
    std::vector<std::size_t> childMaxDepth;
    std::vector<double> childMinimumGainSplit;
    bool serializeModel;
    bool serializePrediction;
    bool serializeColMask;
    bool serializeDataset;
    bool serializeLabels;
    std::size_t serializationWindow;
    std::size_t depth;
  };

} // namespace ModelContext

class PartitionUtils {
public:
  static std::vector<int> _shuffle(int sz) {

    std::vector<int> ind(sz), r(sz);
    std::iota(ind.begin(), ind.end(), 0);
    
    std::vector<std::vector<int>::iterator> v(static_cast<int>(ind.size()));
    std::iota(v.begin(), v.end(), ind.begin());
    
    std::shuffle(v.begin(), v.end(), std::mt19937{std::random_device{}()});
    
    for (std::size_t i=0; i<v.size(); ++i) {
      r[i] = *(v[i]);
    }

    return r;
  }

  static std::vector<std::vector<int>> _fullPartition(int sz) {
    
    std::vector<int> subset(sz);
    std::iota(subset.begin(), subset.end(), 0);
    std::vector<std::vector<int>> p{1, subset};
    return p;
  }

  static uvec sortedSubsample1(std::size_t n, std::size_t numCols) {
    float p = static_cast<float>(numCols)/static_cast<float>(n);     
    uvec r(numCols);
    int i=0, j=0;
    float max_ = (float)(RAND_MAX);

    while (numCols > 0) {
      float s = (float)rand()/max_;
      if (s < p) {
	r[i] = j;
	i += 1;
	numCols -= 1.; n -= 1.;
	p = static_cast<float>(numCols)/static_cast<float>(n);
      }
      j+=1;
    }
    return r;
  }

  static uvec sortedSubsample2(std::size_t n, std::size_t numCols) {
    uvec r(numCols);

    std::size_t i=0, j=0;
    while (numCols > 0) {
      std::size_t s = rand() % n;
      if (s < numCols) {
	r[i] = j;
	i += 1;numCols -= 1;
      }
      j += 1;n -= 1;
    }
    return r;
  }

  static uvec sortedSubsample(std::size_t n, std::size_t numCols) {
    // Faster option; see benchmarks.cpp
    return sortedSubsample2(n, numCols);
  }

};

namespace IB_utils {
  using namespace arma;

  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;
  
  struct distributionException : public std::exception {
    const char* what() const throw () {
      return "Bad distributional assignment";
    };
  };

  class __debug {
  public:
    __debug(const char* fl, const char* fn, int ln) :
      fl_{fl},
      fn_{fn},
      ln_{ln} 
    {
      std::cerr << "===> ENTER FILE: " << fl_
		<< " FUNCTION: " << fn_
		<<" LINE: " << ln_ << std::endl;
    }
    ~__debug() {
      std::cerr << "===< EXIT FILE:  " << fl_
		<< " FUNCTION: " << fn_
		<<" LINE: " << ln_ << std::endl;
    }
  private:
    const char* fl_;
    const char* fn_;
    int ln_;
  };

  template<typename Clock=std::chrono::high_resolution_clock,
	   typename Units=typename Clock::duration>
  class __timer {
  public:
    __timer() : start_point_(Clock::now()) {}
    __timer(std::string& msg) : msg_{msg}, start_point_{Clock::now()} {}
    __timer(std::string&& msg) : msg_{std::move(msg)}, start_point_{Clock::now()} {}

    ~__timer() { 
      unsigned int elapsed = elapsed_time(); 
      if (!msg_.empty()) {
	std::cerr << msg_ << " :: ";
      }
      std::cerr << "ELAPSED: " << elapsed 
		<< std::endl; 
    }

    unsigned int elapsed_time() const {
      std::atomic_thread_fence(std::memory_order_relaxed);
      auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point_).count();
      std::atomic_thread_fence(std::memory_order_relaxed);
      return static_cast<unsigned int>(counted_time);
    }
    
  private:
    std::string msg_;
    const typename Clock::time_point start_point_;
    
  };

  using precise_timer = __timer<std::chrono::high_resolution_clock,
			      std::chrono::microseconds>;
  using system_timer = __timer<std::chrono::system_clock,
			     std::chrono::microseconds>;
  using monotonic_timer = __timer<std::chrono::steady_clock,
				std::chrono::microseconds>;
  
  enum class SerializedType {
      CLASSIFIER = 0,
      PREDICTION = 1,
      COLMASK = 2,
      DATASET_IS = 3,
      DATASET_OOS = 4,
      LABELS_IS = 5,
      LABELS_OOS = 6,
      REGRESSOR = 7,
      CONTEXT = 8
      };

  class DatasetArchive {
  public:
    DatasetArchive() = default;
    DatasetArchive(mat dataset) : dataset_{dataset} {}
    DatasetArchive(mat&& dataset) : dataset_{std::move(dataset)} {}
  
    template<class Archive>
    void serialize(Archive &ar) {
      ar(dataset_);
    }

    // public
    mat dataset_;
  };

  template<typename DataType>
  class PredictionArchive {
  public:
    PredictionArchive() = default;
    PredictionArchive(Row<DataType> prediction) : prediction_{prediction} {}
    PredictionArchive(Row<DataType>&& prediction) : prediction_{std::move(prediction)} {}

    template<class Archive>  
    void serialize(Archive &ar) {
      ar(prediction_);
    }

    // public
    Row<DataType> prediction_;
  };

  template<typename DataType>
  class LabelsArchive {
  public:
    LabelsArchive() = default;
    LabelsArchive(Row<DataType> labels) : labels_{labels} {}
    LabelsArchive(Row<DataType>&& labels) : labels_{labels} {}

    template<class Archive>
    void serialize(Archive &ar) {
      ar(labels_);
    }

    // public
    Row<DataType> labels_;
  };

  class ColMaskArchive {
  public:
    ColMaskArchive() = default;
    ColMaskArchive(uvec colMask) : colMask_{colMask} {}
    ColMaskArchive(uvec&& colMask) : colMask_{std::move(colMask)} {}

    template<class Archive>
    void serialize(Archive &ar) {
      ar(colMask_);
    }

    // public
    uvec colMask_;
  };

  // For directory digest
  inline boost::filesystem::path FilterDigestLocation(boost::filesystem::path path=boost::filesystem::current_path()) {
    auto now = std::chrono::system_clock::now();
    auto UTC = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
    std::string suff = "_digest__" + std::to_string(UTC) + "_" + datetime.str();

    path /= suff;

    return path;
    
  }

  // Filter typeinfo string to generate unique filenames for serialization tests.
  inline std::string FilterFileName(const std::string& inputString)
  {
    // Take the last valid 24 characters for the filename.
    std::string fileName;
    for (auto it = inputString.rbegin(); it != inputString.rend() &&
	   fileName.size() != 24; ++it)
      {
	if (std::isalnum(*it))
	  fileName.push_back(*it);
      }

    auto now = std::chrono::system_clock::now();
    // auto UTC = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    auto UTC = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
    fileName = std::to_string(UTC) + "_" + fileName + "_" + datetime.str() + ".gar";
    
    return fileName;
  }

  template<typename T, typename IARchiveType, typename OArchiveType>
  void dumps(T& t, std::string fileName, boost::filesystem::path fldr=boost::filesystem::path{}) {
    std::string abs_path;

    if (fldr.string().size()) {
      fldr /= fileName;
      abs_path = fldr.string();
    } else {
      abs_path = fileName;
    }

    std::ofstream ofs(abs_path, std::ios::binary);
    {
      OArchiveType o(ofs);

      T& x(t);
      o(CEREAL_NVP(x));      
    }
    ofs.close();
  }

  template<typename T, typename IArchiveType, typename OArchiveType>
  std::string dumps(T& t, SerializedType typ, boost::filesystem::path fldr=boost::filesystem::path{}) {

    std::map<int, std::string> SerializedTypeMap = 
      {
	{0, "__CLS_"},
	{1, "__PRED_"},
	{2, "__CMASK_"},
	{3, "__DIS_"},
	{4, "__DOOS_"},
	{5, "__LIS_"},
	{6, "__LOOS_"},
	{7, "__REG_"},
	{8, "__CXT_"}
      };
    
    std::string pref = SerializedTypeMap[static_cast<std::underlying_type_t<SerializedType>>(typ)];

    std::string fileName = FilterFileName(typeid(T).name());

    dumps<T, IArchiveType, OArchiveType>(t, fileName, fldr);

    return pref + fileName;
  }

  template<typename T, typename IArchiveType, typename OArchiveType>
  void loads(T& t, std::string fileName, boost::filesystem::path fldr=boost::filesystem::path{}) {

    std::string abs_path;

    if (fldr.string().size()) {
      fldr /= fileName;
      abs_path = fldr.string();
    } else {
      abs_path = fileName;
    }

    std::ifstream ifs{abs_path, std::ios::binary};
    
    {
      IArchiveType i(ifs);
      T& x(t);
      i(CEREAL_NVP(x));
    }
    ifs.close();

  }

  template<typename T>
  void read(T& rhs, std::string fileName, boost::filesystem::path fldr=boost::filesystem::path{}) {
    using CerealT = T;    
    loads<CerealT, CerealIArch, CerealOArch>(rhs, fileName, fldr);
  }
    
  template<typename DataType>
  void writePrediction(const Row<DataType>& prediction, std::string fileName, boost::filesystem::path fldr=boost::filesystem::path{}) {
    PredictionArchive pa{prediction};
    dumps<PredictionArchive<DataType>, CerealIArch, CerealOArch>(pa, fileName, fldr);
  }

  template<typename DataType>
  std::string writePrediction(const Row<DataType>& prediction, boost::filesystem::path fldr=boost::filesystem::path{}) {

    PredictionArchive pa{prediction};
    std::string fileName = dumps<PredictionArchive<DataType>, CerealIArch, CerealOArch>(pa, SerializedType::PREDICTION, fldr);
    return fileName;
  }

  template<typename DataType>
  std::string writeLabelsIS(const Row<DataType>& labels, boost::filesystem::path fldr=boost::filesystem::path{}) {
    LabelsArchive la{labels};
    std::string fileName = dumps<LabelsArchive<DataType>, CerealIArch, CerealOArch>(la, SerializedType::LABELS_IS, fldr);
    return fileName;
  }

  template<typename DataType>
  std::string writeLabelsOOS(const Row<DataType>& labels, boost::filesystem::path fldr=boost::filesystem::path{}) {
    LabelsArchive la{labels};
    std::string fileName = dumps<LabelsArchive<DataType>, CerealIArch, CerealOArch>(la, SerializedType::LABELS_OOS, fldr);
    return fileName;
  }

  std::string writeColMask(const uvec&, boost::filesystem::path fldr=boost::filesystem::path{});
  std::string writeDatasetIS(const mat&, boost::filesystem::path fldr=boost::filesystem::path{});
  std::string writeDatasetOOS(const mat&, boost::filesystem::path fldr=boost::filesystem::path{});

  double err(const Row<std::size_t>&, const Row<std::size_t>&);
  double err(const Row<double>&, const Row<double>&, double=-1.);

  std::tuple<double, double, double> precision(const Row<int>&, const Row<int>&);
  std::tuple<double, double> Perlich_rank_scores(const Row<double>&, const Row<double>&);

  double imbalance(const Row<int>&);
  double imbalance(const Row<double>&);

  template<typename CharT>
  using tstring = std::basic_string<CharT, std::char_traits<CharT>, std::allocator<CharT>>;
  template<typename CharT>
  using tstringstream = std::basic_stringstream<CharT, std::char_traits<CharT>, std::allocator<CharT>>;

  template<typename CharT>
  inline std::vector<tstring<CharT>> strSplit(tstring<CharT> text, CharT const delimiter) {
    auto sstr = tstringstream<CharT>{text};
    auto tokens = std::vector<tstring<CharT>>{};
    auto token = tstring<CharT>{};
    while (std::getline(sstr, token, delimiter)) {
      if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
  }

  template<typename CharT>
  tstring<CharT> strJoin(const std::vector<tstring<CharT>> &tokens, char delim, int firstInd) {
    tstring<CharT> r;
    int size_ = static_cast<int>(tokens.size());
    for (int i=firstInd; i<size_; ++i) {
      if (!r.size())
	r = tokens[i];
      else
	r = r + delim + tokens[i];
    }
    return r;
  }

  template<typename T, typename IArchiveType, typename OArchiveType>
  void SerializeObject(T& t, T& newT, boost::filesystem::path fldr=boost::filesystem::path{})
  {

    std::string fileName = FilterFileName(typeid(T).name());

    std::string abs_path;
    if (fldr.string().size()) {
      fldr /= fileName;
      abs_path = fldr.string();
    } else {
      abs_path = fileName;
    }

    std::ofstream ofs(abs_path, std::ios::binary);
    
    {
      OArchiveType o(ofs);
      
      T& x(t);
      o(CEREAL_NVP(x));
    }
    ofs.close();

    std::ifstream ifs(abs_path, std::ios::binary);

    {
      IArchiveType i(ifs);
      T& x(newT);
      i(CEREAL_NVP(x));
    }
    ifs.close();
	
    // remove(abs_path.c_str());
  }

  template<typename T, typename IArchiveType, typename OArchiveType>
  void SerializePointerObject(T* t, T*& newT, boost::filesystem::path fldr=boost::filesystem::path{})
  {
    std::string fileName = FilterFileName(typeid(T).name());
    std::string abs_path;
    if (fldr.string().size()) {
      fldr /= fileName;
      abs_path = fldr.string();
    } else {
      abs_path = fileName;
    }

    std::ofstream ofs(abs_path, std::ios::binary);

    {
      OArchiveType o(ofs);
      o(CEREAL_POINTER(t));
    }
    ofs.close();

    std::ifstream ifs(abs_path, std::ios::binary);

    {
      IArchiveType i(ifs);
      i(CEREAL_POINTER(newT));
    }
    ifs.close();
    // remove(abs_path.c_str());
  }

  /* e.g.
     SerializeObject<T, cereal::XMLInputArchive, cereal::XMLOutputArchive>
     SerializeObject<T, cereal::JSONInputArchive cereal::JSONOutputArchive>
     SerializeObject<T, cereal::BinaryInputArchive, cereal::BinaryOutputArchive>

  */

  template<typename T, typename IArchiveType, typename OArchiveType>
  void SerializePointerObject(T* t, T*& newT, boost::filesystem::path fldr);

  std::string writeIndex(const std::vector<std::string>&, boost::filesystem::path fldr=boost::filesystem::path{});
  std::string writeIndex(const std::vector<std::string>&, std::string, boost::filesystem::path fldr=boost::filesystem::path{});
  void readIndex(std::string, std::vector<std::string>&, boost::filesystem::path fldr=boost::filesystem::path{});
  void mergeIndices(std::string, std::string, boost::filesystem::path fldr=boost::filesystem::path{}, bool=false);

  template<typename T>
  void 
  writeBinary(std::string fileName,
	      const T& data,
	      boost::filesystem::path fldr=boost::filesystem::path{}) {
    auto success = false;

    std::string abs_path;
    if (fldr.string().size()) {
      fldr /= fileName;
      abs_path = fldr.string();
    } else {
      abs_path = fileName;
    }

    std::ofstream ofs{abs_path, std::ios::binary};
    if (ofs.is_open()) {

      try {
	ofs.write(reinterpret_cast<const char*>(&data), sizeof(data));
	success = true;
      }
      catch(std::ios_base::failure &) {
	std::cerr << "Failed to write to " << abs_path << std::endl;
      }
      ofs.close();
    }
  }

  template<typename T>
  void
  readBinary(std::string fileName,
	     T& obj,
	     boost::filesystem::path fldr=boost::filesystem::path{}) {

    std::string abs_path;
    if (fldr.string().size()) {
      fldr /= fileName;
      abs_path = fldr.string();
    } else {
      abs_path = fileName;
    }    

    std::size_t readBytes = 0;
    std::ifstream ifs{abs_path, std::ios::ate | std::ios::binary};
    if (ifs.is_open()) {

      ifs.seekg(0, std::ios_base::beg);

      try {
	ifs.read(reinterpret_cast<char*>(&obj), sizeof(T));
	readBytes = static_cast<std::size_t>(ifs.gcount());
      }
      catch(std::ios_base::failure &) {
	std::cerr << "Failed to read from " << abs_path << std::endl;
      }
      ifs.close();
    }
  }

  template<typename DataType>
  void readPrediction(std::string indexName, Row<DataType>& prediction, boost::filesystem::path fldr=boost::filesystem::path{}) {
    Row<DataType> predictionNew;  
    std::vector<std::string> fileNames;
    readIndex(indexName, fileNames, fldr);
    
    for (auto &fileName : fileNames) {
      auto tokens = strSplit(fileName, '_');
      if (tokens[0] == "PRED") {
	fileName = strJoin(tokens, '_', 1);
	read(predictionNew, fileName, fldr);
	prediction = predictionNew;
      }
    }

  }

  bool comp(std::pair<std::size_t, std::size_t>&, std::pair<std::size_t, std::size_t>&);
  std::string fit_prefix(std::size_t);


} // namespace IB_utils

#endif
