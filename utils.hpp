#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <limits>
#include <type_traits>
#include <mlpack/core.hpp>

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

#include "utils.hpp"
#include "loss.hpp"

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

namespace ClassifierContext{
  struct Context {
    Context(std::size_t minLeafSize=1,
	    double minimumGainSplit=.0,
	    std::size_t maxDepth=100,
	    std::size_t numTrees=10,
	    bool recursiveFit=false) : 
      minLeafSize{minLeafSize},
      minimumGainSplit{minimumGainSplit},
      maxDepth{maxDepth},
      numTrees{numTrees},
      removeRedundantLabels{false},
      stepSizeMethod{StepSize::StepSizeMethod::LOG},
      partitionSizeMethod{PartitionSize::PartitionSizeMethod::FIXED},
      learningRateMethod{LearningRate::LearningRateMethod::FIXED},
      quietRun{false},
      recursiveFit{recursiveFit},
      serialize{false},
      serializePrediction{false},
      serializeDataset{false},
      serializeLabels{false},
      serializeColMask{false},
      serializationWindow{500},
      baseSteps{-1}
    {}

    Context(const Context& rhs) {
      loss = rhs.loss;
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
      minLeafSize = rhs.minLeafSize;
      minimumGainSplit = rhs.minimumGainSplit;
      maxDepth = rhs.maxDepth;
      numTrees = rhs.numTrees;
      serialize = rhs.serialize;
      serializePrediction = rhs.serializePrediction;
      serializeColMask = rhs.serializeColMask;
      serializeDataset = rhs.serializeDataset;
      serializeLabels = rhs.serializeLabels;
      serializationWindow = rhs.serializationWindow;

    }
      
    lossFunction loss;
    std::size_t partitionSize;
    double partitionRatio;
    double learningRate;
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
    std::size_t minLeafSize;
    double minimumGainSplit;
    std::size_t maxDepth;
    std::size_t numTrees;
    bool serialize;
    bool serializePrediction;
    bool serializeColMask;
    bool serializeDataset;
    bool serializeLabels;
    std::size_t serializationWindow;
  };
} // namespace ClassifierContext

namespace IB_utils {
  using namespace arma;

  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;
  
  struct distributionException : public std::exception {
    const char* what() const throw () {
      return "Bad distributional assignment";
    };
  };

  enum class SerializedType {
    CLASSIFIER = 0,
      PREDICTION = 1,
      COLMASK = 2,
      DATASET_IS = 3,
      DATASET_OOS = 4,
      LABELS_IS = 5,
      LABELS_OOS = 6
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

  template<typename T, typename IARchiveType, typename OArchiveType>
  void dumps(T& t, std::string fileName) {
    std::ofstream ofs(fileName, std::ios::binary);
    {
      OArchiveType o(ofs);

      T& x(t);
      o(CEREAL_NVP(x));      
    }
    ofs.close();
  }

  template<typename T, typename IArchiveType, typename OArchiveType>
  std::string dumps(T& t, SerializedType typ) {

    std::map<int, std::string> SerializedTypeMap = 
      {
	{0, "__CLS_"},
	{1, "__PRED_"},
	{2, "__CMASK_"},
	{3, "__DIS_"},
	{4, "__DOOS_"},
	{5, "__LIS_"},
	{6, "__LOOS_"}
      };
    
    std::string pref = SerializedTypeMap[static_cast<std::underlying_type_t<SerializedType>>(typ)];

    std::string fileName = FilterFileName(typeid(T).name());

    dumps<T, IArchiveType, OArchiveType>(t, fileName);

    return pref + fileName;
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

  template<typename T>
  void read(T& rhs, std::string fileName) {
    using CerealT = T;
    loads<CerealT, CerealIArch, CerealOArch>(rhs, fileName);
  }
    
  template<typename DataType>
  void writePrediction(const Row<DataType>& prediction, std::string fileName) {
    PredictionArchive pa{prediction};
    dumps<PredictionArchive<DataType>, CerealIArch, CerealOArch>(pa, fileName);
  }

  template<typename DataType>
  std::string writePrediction(const Row<DataType>& prediction) {

    PredictionArchive pa{prediction};
    std::string fileName = dumps<PredictionArchive<DataType>, CerealIArch, CerealOArch>(pa, SerializedType::PREDICTION);
    return fileName;
  }

  template<typename DataType>
  std::string writeLabelsIS(const Row<DataType>& labels) {
    LabelsArchive la{labels};
    std::string fileName = dumps<LabelsArchive<DataType>, CerealIArch, CerealOArch>(la, SerializedType::LABELS_IS);
    return fileName;
  }

  template<typename DataType>
  std::string writeLabelsOOS(const Row<DataType>& labels) {
    LabelsArchive la{labels};
    std::string fileName = dumps<LabelsArchive<DataType>, CerealIArch, CerealOArch>(la, SerializedType::LABELS_OOS);
    return fileName;
  }

  std::string writeColMask(const uvec&);
  std::string writeDatasetIS(const mat&);
  std::string writeDatasetOOS(const mat&);

  double err(const Row<std::size_t>& yhat, const Row<std::size_t>& y);
  double err(const Row<double>& yhat, const Row<double>& y, double=-1.);

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
    for (int i=firstInd; i<tokens.size(); ++i) {
      if (!r.size())
	r = tokens[i];
      else
	r = r + delim + tokens[i];
    }
    return r;
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
  std::string writeIndex(const std::vector<std::string>&, std::string);
  void readIndex(std::string, std::vector<std::string>&);
  void mergeIndices(std::string, std::string);

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

  template<typename DataType>
  void readPrediction(std::string indexName, Row<DataType>& prediction) {
    Row<DataType> predictionNew;  
    std::vector<std::string> fileNames;
    readIndex(indexName, fileNames);
    
    for (auto &fileName : fileNames) {
      auto tokens = strSplit(fileName, '_');
      if (tokens[0] == "PRED") {
	fileName = strJoin(tokens, '_', 1);
	read(predictionNew, fileName);
	prediction = predictionNew;
      }
    }

  }

  bool comp(std::pair<std::size_t, std::size_t>&, std::pair<std::size_t, std::size_t>&);

}

#endif
