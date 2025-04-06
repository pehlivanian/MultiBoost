#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <boost/filesystem.hpp>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mlpack/core.hpp>
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "classifier_loss.hpp"
#include "regressor_loss.hpp"

#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)

// using fs = CXX_FILESYSTEM_NAMESPACE;
using namespace boost::filesystem;
using namespace LossMeasures;

namespace LearningRate {
enum class LearningRateMethod {
  FIXED = 0,
  INCREASING = 1,
  DECREASING = 2,
};

}  // namespace LearningRate

namespace PartitionSize {
enum class PartitionSizeMethod {
  FIXED = 0,
  FIXED_PROPORTION = 1,
  DECREASING = 2,
  INCREASING = 3,
  RANDOM = 4,
  MULTISCALE = 5
};

}  // namespace PartitionSize

namespace StepSize {
enum class StepSizeMethod { LOG = 0, PROPORTION = 1 };
}  // namespace StepSize

namespace ModelContext {

struct Context {
  Context()
      : steps{-1},
        removeRedundantLabels{false},
        quietRun{false},
        useWeights{false},
        recursiveFit{false},
        childPartitionSize{std::vector<std::size_t>()},
        childNumSteps{std::vector<std::size_t>()},
        childLearningRate{std::vector<double>()},
        childActivePartitionRatio{std::vector<double>()},
        childMinLeafSize{std::vector<std::size_t>()},
        childMaxDepth{std::vector<std::size_t>()},
        childMinimumGainSplit{std::vector<double>()},
        serializeModel{false},
        serializePrediction{false},
        serializeColMask{false},
        serializeDataset{false},
        serializeLabels{false},
        serializationWindow{500},
        depth{0} {}

  Context(const Context& rhs) {
    loss = rhs.loss;
    lossPower = rhs.lossPower;
    clamp_gradient = rhs.clamp_gradient;
    upper_val = rhs.upper_val;
    lower_val = rhs.lower_val;
    partitionSize = rhs.partitionSize;
    learningRate = rhs.learningRate;
    activePartitionRatio = rhs.activePartitionRatio;
    maxDepth = rhs.maxDepth;
    minLeafSize = rhs.minLeafSize;
    minimumGainSplit = rhs.minimumGainSplit;
    steps = rhs.steps;
    symmetrizeLabels = rhs.symmetrizeLabels;
    removeRedundantLabels = rhs.removeRedundantLabels;
    quietRun = rhs.quietRun;
    useWeights = rhs.useWeights;
    rowSubsampleRatio = rhs.rowSubsampleRatio;
    colSubsampleRatio = rhs.colSubsampleRatio;
    recursiveFit = rhs.recursiveFit;
    childPartitionSize = rhs.childPartitionSize;
    childNumSteps = rhs.childNumSteps;
    childLearningRate = rhs.childLearningRate;
    childActivePartitionRatio = rhs.childActivePartitionRatio;
    childMinLeafSize = rhs.childMinLeafSize;
    childMaxDepth = rhs.childMaxDepth;
    childMinimumGainSplit = rhs.childMinimumGainSplit;
    numTrees = rhs.numTrees;
    serializeModel = rhs.serializeModel;
    serializePrediction = rhs.serializePrediction;
    serializeColMask = rhs.serializeColMask;
    serializeDataset = rhs.serializeDataset;
    serializeLabels = rhs.serializeLabels;
    serializationWindow = rhs.serializationWindow;
    depth = rhs.depth;
  }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(CEREAL_NVP(steps),
       CEREAL_NVP(recursiveFit),
       CEREAL_NVP(useWeights),
       CEREAL_NVP(rowSubsampleRatio),
       CEREAL_NVP(colSubsampleRatio),
       CEREAL_NVP(removeRedundantLabels),
       CEREAL_NVP(symmetrizeLabels),

       // Loss
       CEREAL_NVP(loss),
       CEREAL_NVP(lossPower),
       CEREAL_NVP(clamp_gradient),
       CEREAL_NVP(upper_val),
       CEREAL_NVP(lower_val),

       // Unused
       CEREAL_NVP(numTrees),
       CEREAL_NVP(depth),

       // Array types
       CEREAL_NVP(childPartitionSize),
       CEREAL_NVP(childNumSteps),
       CEREAL_NVP(childLearningRate),
       CEREAL_NVP(childActivePartitionRatio),
       CEREAL_NVP(childMinLeafSize),
       CEREAL_NVP(childMinimumGainSplit),
       CEREAL_NVP(childMaxDepth),

       // Serialization
       CEREAL_NVP(serializeModel),
       CEREAL_NVP(serializePrediction),
       CEREAL_NVP(serializeColMask),
       CEREAL_NVP(serializeDataset),
       CEREAL_NVP(serializeLabels),
       CEREAL_NVP(serializationWindow),

       CEREAL_NVP(quietRun)

    );
  }

  std::variant<classifierLossFunction, regressorLossFunction> loss;
  float lossPower;
  bool clamp_gradient;
  double upper_val;
  double lower_val;
  std::size_t partitionSize;
  double learningRate;
  double activePartitionRatio;
  std::size_t maxDepth;
  std::size_t minLeafSize;
  double minimumGainSplit;
  std::size_t numTrees;
  int steps;
  bool symmetrizeLabels;
  bool removeRedundantLabels;
  bool quietRun;
  bool useWeights;
  double rowSubsampleRatio;
  double colSubsampleRatio;
  bool recursiveFit;
  std::vector<std::size_t> childPartitionSize;
  std::vector<std::size_t> childNumSteps;
  std::vector<double> childLearningRate;
  std::vector<double> childActivePartitionRatio;
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

}  // namespace ModelContext

namespace TupleUtils {
template <typename... Ts>
decltype(auto) to_tuple(Ts&&... ts) {
  auto tup = std::make_tuple(std::forward<Ts>(ts)...);
  return tup;
}

template <typename T, std::size_t... I>
void print_tuple(const T& tup, std::index_sequence<I...>) {
  std::cout << "(";
  (..., (std::cout << (I == 0 ? "" : ", ") << std::get<I>(tup)));
}

template <typename... T>
void print_tuple(const std::tuple<T...>& tup) {
  print_tuple(tup, std::make_index_sequence<sizeof...(T)>());
}

template <std::size_t Ofst, class Tuple, std::size_t... I>
constexpr auto slice_impl(Tuple&& t, std::index_sequence<I...>) {
  return std::forward_as_tuple(std::get<I + Ofst>(std::forward<Tuple>(t))...);
}

template <std::size_t I1, std::size_t I2, class Cont>
constexpr auto tuple_slice(Cont&& t) {
  static_assert(I2 >= I1, "invalid slice");
  static_assert(std::tuple_size<std::decay_t<Cont>>::value >= I2, "slice index out of bounds");
  return slice_impl<I1>(std::forward<Cont>(t), std::make_index_sequence<I2 - I1>{});
}

template <std::size_t N, typename... Types>
auto remove_element_from_tuple(const std::tuple<Types...>& t) {
  return std::tuple_cat(tuple_slice<0, N>(t), tuple_slice<N + 1, sizeof...(Types)>(t));
}

template <std::size_t N, typename... T, std::size_t... I>
std::tuple<std::tuple_element_t<N + I, std::tuple<T...>>...> sub(std::index_sequence<I...>);

template <std::size_t N, typename... T>
using subpack = decltype(sub<N, T...>(std::make_index_sequence<sizeof...(T) - N>{}));

template <typename T, typename... Ts>
struct remove_first {
  using type = std::tuple<Ts...>;
};

}  // namespace TupleUtils

class PartitionUtils {
public:
  static std::vector<int> _shuffle(int sz) {
    std::vector<int> ind(sz), r(sz);
    std::iota(ind.begin(), ind.end(), 0);

    std::vector<std::vector<int>::iterator> v(static_cast<int>(ind.size()));
    std::iota(v.begin(), v.end(), ind.begin());

    std::shuffle(v.begin(), v.end(), std::mt19937{std::random_device{}()});

    for (std::size_t i = 0; i < v.size(); ++i) {
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
    float p = static_cast<float>(numCols) / static_cast<float>(n);
    uvec r(numCols);
    int i = 0, j = 0;
    float max_ = (float)(RAND_MAX);

    while (numCols > 0) {
      float s = (float)rand() / max_;
      if (s < p) {
        r[i] = j;
        i += 1;
        numCols -= 1.;
        n -= 1.;
        p = static_cast<float>(numCols) / static_cast<float>(n);
      }
      j += 1;
    }
    return r;
  }

  static uvec sortedSubsample2(std::size_t n, std::size_t numCols) {
    uvec r(numCols);

    std::size_t i = 0, j = 0;
    while (numCols > 0) {
      std::size_t s = rand() % n;
      if (s < numCols) {
        r[i] = j;
        i += 1;
        numCols -= 1;
      }
      j += 1;
      n -= 1;
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
  const char* what() const throw() { return "Bad distributional assignment"; };
};

class __debug {
public:
  __debug(const char* fl, const char* fn, int ln) : fl_{fl}, fn_{fn}, ln_{ln} {
    std::cerr << "===> ENTER FILE: " << fl_ << " FUNCTION: " << fn_ << " LINE: " << ln_
              << std::endl;
  }
  ~__debug() {
    std::cerr << "===< EXIT FILE:  " << fl_ << " FUNCTION: " << fn_ << " LINE: " << ln_
              << std::endl;
  }

private:
  const char* fl_;
  const char* fn_;
  int ln_;
};

template <
    typename Clock = std::chrono::high_resolution_clock,
    typename Units = typename Clock::duration>
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
    std::cerr << "ELAPSED: " << elapsed << std::endl;
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

using precise_timer = __timer<std::chrono::high_resolution_clock, std::chrono::microseconds>;
using system_timer = __timer<std::chrono::system_clock, std::chrono::microseconds>;
using monotonic_timer = __timer<std::chrono::steady_clock, std::chrono::microseconds>;

enum class SerializedType {
  CLASSIFIER = 0,
  PREDICTION = 1,
  COLMASK = 2,
  DATASET_IS = 3,
  DATASET_OOS = 4,
  LABELS_IS = 5,
  LABELS_OOS = 6,
  WEIGHTS_IS = 7,
  REGRESSOR = 8,
  CONTEXT = 9
};

template <typename DataType>
class DatasetArchive {
public:
  DatasetArchive() = default;
  DatasetArchive(Mat<DataType> dataset) : dataset_{dataset} {}
  DatasetArchive(Mat<DataType>&& dataset) : dataset_{std::move(dataset)} {}

  template <class Archive>
  void serialize(Archive& ar) {
    ar(dataset_);
  }

  // public
  Mat<DataType> dataset_;
};

template <typename DataType>
class PredictionArchive {
public:
  PredictionArchive() = default;
  PredictionArchive(Row<DataType> prediction) : prediction_{prediction} {}
  PredictionArchive(Row<DataType>&& prediction) : prediction_{std::move(prediction)} {}

  template <class Archive>
  void serialize(Archive& ar) {
    ar(prediction_);
  }

  // public
  Row<DataType> prediction_;
};

template <typename DataType>
class LabelsArchive {
public:
  LabelsArchive() = default;
  LabelsArchive(Row<DataType> labels) : labels_{labels} {}
  LabelsArchive(Row<DataType>&& labels) : labels_{labels} {}

  template <class Archive>
  void serialize(Archive& ar) {
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

  template <class Archive>
  void serialize(Archive& ar) {
    ar(colMask_);
  }

  // public
  uvec colMask_;
};

// For directory digest
inline boost::filesystem::path FilterDigestLocation(
    boost::filesystem::path path = boost::filesystem::current_path()) {
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
inline std::string FilterFileName(const std::string& inputString) {
  // Take the last valid 24 characters for the filename.
  std::string fileName;
  for (auto it = inputString.rbegin(); it != inputString.rend() && fileName.size() != 24; ++it) {
    if (std::isalnum(*it)) fileName.push_back(*it);
  }

  auto now = std::chrono::system_clock::now();
  // auto UTC =
  // std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  auto UTC = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream datetime;
  datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
  fileName = std::to_string(UTC) + "_" + fileName + "_" + datetime.str() + ".gar";

  return fileName;
}

template <typename T, typename IARchiveType, typename OArchiveType>
void dumps(T& t, std::string fileName, boost::filesystem::path fldr = boost::filesystem::path{}) {
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

template <typename T, typename IArchiveType, typename OArchiveType>
std::string dumps(
    T& t, SerializedType typ, boost::filesystem::path fldr = boost::filesystem::path{}) {
  std::map<int, std::string> SerializedTypeMap = {
      {0, "__CLS_"},
      {1, "__PRED_"},
      {2, "__CMASK_"},
      {3, "__DIS_"},
      {4, "__DOOS_"},
      {5, "__LIS_"},
      {6, "__LOOS_"},
      {7, "__WIS_"},
      {8, "__REG_"},
      {9, "__CXT_"}};

  std::string pref = SerializedTypeMap[static_cast<std::underlying_type_t<SerializedType>>(typ)];

  std::string fileName = FilterFileName(typeid(T).name());

  dumps<T, IArchiveType, OArchiveType>(t, fileName, fldr);

  return pref + fileName;
}

template <typename T, typename IArchiveType, typename OArchiveType>
void loads(T& t, std::string fileName, boost::filesystem::path fldr = boost::filesystem::path{}) {
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

template <typename T>
void read(T& rhs, std::string fileName, boost::filesystem::path fldr = boost::filesystem::path{}) {
  using CerealT = T;
  loads<CerealT, CerealIArch, CerealOArch>(rhs, fileName, fldr);
}

template <typename DataType>
void writePrediction(
    const Row<DataType>& prediction,
    std::string fileName,
    boost::filesystem::path fldr = boost::filesystem::path{}) {
  PredictionArchive<DataType> pa{prediction};
  dumps<PredictionArchive<DataType>, CerealIArch, CerealOArch>(pa, fileName, fldr);
}

template <typename DataType>
std::string writePrediction(
    const Row<DataType>& prediction, boost::filesystem::path fldr = boost::filesystem::path{}) {
  PredictionArchive<DataType> pa{prediction};
  std::string fileName = dumps<PredictionArchive<DataType>, CerealIArch, CerealOArch>(
      pa, SerializedType::PREDICTION, fldr);
  return fileName;
}

template <typename DataType>
std::string writeLabelsIS(
    const Row<DataType>& labels, boost::filesystem::path fldr = boost::filesystem::path{}) {
  LabelsArchive<DataType> la{labels};
  std::string fileName =
      dumps<LabelsArchive<DataType>, CerealIArch, CerealOArch>(la, SerializedType::LABELS_IS, fldr);
  return fileName;
}

template <typename DataType>
std::string writeWeightsIS(
    const Row<DataType>& weights, boost::filesystem::path fldr = boost::filesystem::path{}) {
  LabelsArchive<DataType> wg{weights};
  std::string fileName = dumps<LabelsArchive<DataType>, CerealIArch, CerealOArch>(
      wg, SerializedType::WEIGHTS_IS, fldr);
  return fileName;
}

template <typename DataType>
std::string writeLabelsOOS(
    const Row<DataType>& labels, boost::filesystem::path fldr = boost::filesystem::path{}) {
  LabelsArchive<DataType> la{labels};
  std::string fileName = dumps<LabelsArchive<DataType>, CerealIArch, CerealOArch>(
      la, SerializedType::LABELS_OOS, fldr);
  return fileName;
}

std::string writeColMask(const uvec&, boost::filesystem::path fldr = boost::filesystem::path{});

template <typename DataType>
std::string writeDatasetIS(
    const Mat<DataType>& dataset, boost::filesystem::path fldr = boost::filesystem::path{}) {
  DatasetArchive<DataType> da{dataset};
  std::string fileName = dumps<DatasetArchive<DataType>, CerealIArch, CerealOArch>(
      da, SerializedType::DATASET_IS, fldr);
  return fileName;
}

template <typename DataType>
std::string writeDatasetOOS(
    const Mat<DataType>& dataset, boost::filesystem::path fldr = boost::filesystem::path{}) {
  DatasetArchive<DataType> da{dataset};
  std::string fileName = dumps<DatasetArchive<DataType>, CerealIArch, CerealOArch>(
      da, SerializedType::DATASET_OOS, fldr);
  return fileName;
}

double err(const Row<double>&, const Row<double>&, double = -1.);
double err(const Row<float>&, const Row<float>&, double = -1.);
double err(const Row<std::size_t>&, const Row<std::size_t>&);

std::tuple<double, double, double> precision(const Row<int>&, const Row<int>&);
std::tuple<double, double> Perlich_rank_scores(const Row<double>&, const Row<double>&);

double imbalance(const Row<int>&);
double imbalance(const Row<double>&);
double imbalance(const Row<float>&);

template <typename CharT>
using tstring = std::basic_string<CharT, std::char_traits<CharT>, std::allocator<CharT>>;
template <typename CharT>
using tstringstream =
    std::basic_stringstream<CharT, std::char_traits<CharT>, std::allocator<CharT>>;

template <typename CharT>
inline void strReplace(tstring<CharT>& text, tstring<CharT> from, tstring<CharT> to) {
  text.replace(text.find(from), from.length(), to);
}

template <typename CharT>
inline std::vector<tstring<CharT>> strSplit(tstring<CharT> text, CharT const delimiter) {
  auto sstr = tstringstream<CharT>{text};
  auto tokens = std::vector<tstring<CharT>>{};
  auto token = tstring<CharT>{};
  while (std::getline(sstr, token, delimiter)) {
    if (!token.empty()) tokens.push_back(token);
  }
  return tokens;
}

template <typename CharT>
tstring<CharT> strJoin(const std::vector<tstring<CharT>>& tokens, char delim, int firstInd) {
  tstring<CharT> r;
  int size_ = static_cast<int>(tokens.size());
  for (int i = firstInd; i < size_; ++i) {
    if (!r.size())
      r = tokens[i];
    else
      r = r + delim + tokens[i];
  }
  return r;
}

template <typename T, typename IArchiveType, typename OArchiveType>
void SerializeObject(T& t, T& newT, boost::filesystem::path fldr = boost::filesystem::path{}) {
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

template <typename T, typename IArchiveType, typename OArchiveType>
void SerializePointerObject(
    T* t, T*& newT, boost::filesystem::path fldr = boost::filesystem::path{}) {
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

template <typename T, typename IArchiveType, typename OArchiveType>
void SerializePointerObject(T* t, T*& newT, boost::filesystem::path fldr);

std::string writeIndex(
    const std::vector<std::string>&, boost::filesystem::path fldr = boost::filesystem::path{});
std::string writeIndex(
    const std::vector<std::string>&,
    std::string,
    boost::filesystem::path fldr = boost::filesystem::path{});
void readIndex(
    std::string,
    std::vector<std::string>&,
    boost::filesystem::path fldr = boost::filesystem::path{});
void mergeIndices(
    std::string,
    std::string,
    boost::filesystem::path fldr = boost::filesystem::path{},
    bool = false);

template <typename DataType>
std::vector<DataType> sort_not_in_place(std::vector<DataType> v) {
  std::sort(v.begin(), v.end());
  return v;
}

template <typename DataType>
void printSubsets(
    std::vector<std::vector<int>>& subsets,
    const std::vector<DataType>& targets,
    const std::vector<DataType>& preds,
    const std::vector<DataType>& a,
    const std::vector<DataType>& b,
    const std::vector<DataType>& y,
    const std::vector<DataType>& yhat,
    const uvec& colMask) {
  DataType abs_stddev_max = 0.;

  std::vector<DataType> sorted_targets = sort_not_in_place(targets);
  std::vector<DataType> sorted_preds = sort_not_in_place(preds);

  auto it = std::unique(sorted_targets.begin(), sorted_targets.end());
  sorted_targets.resize(std::distance(sorted_targets.begin(), it));

  it = std::unique(sorted_preds.begin(), sorted_preds.end());
  sorted_preds.resize(std::distance(sorted_preds.begin(), it));

  std::cerr << "[ Unique targets: ";
  std::copy(
      sorted_targets.begin(),
      sorted_targets.end(),
      std::ostream_iterator<DataType>(std::cerr, " "));
  std::cerr << "]";

  std::cerr << "[ Unique preds: ";
  std::copy(
      sorted_preds.begin(), sorted_preds.end(), std::ostream_iterator<DataType>(std::cerr, " "));
  std::cerr << "]\n";

  std::cerr << "SUBSETS\n";
  std::cerr << "[\n";
  std::for_each(
      subsets.begin(),
      subsets.end(),
      [&targets, &preds, &y, &yhat, &a, &b, &colMask, &abs_stddev_max](std::vector<int>& subset) {
        std::cerr << "  [size: " << subset.size() << "] ";
        std::cerr << "[";
        std::sort(subset.begin(), subset.end());
        for (const int& ind : subset) {
          std::cerr << colMask[ind] << " ";
        }
        std::cerr << "] ";

        std::cerr << "[ a ";
        for (const int& ind : subset) {
          std::cerr << a[ind] << " ";
        }
        std::cerr << "]";

        std::cerr << "[ b ";
        for (const int& ind : subset) {
          std::cerr << b[ind] << " ";
        }
        std::cerr << "]";

        std::cerr << "[ targets ";
        for (const int& ind : subset) {
          std::cerr << targets[ind] << " ";
        }
        std::cerr << "]";

        std::cerr << "[ preds ";
        for (const int& ind : subset) {
          std::cerr << preds[ind] << " ";
        }
        std::cerr << "]";

        std::cerr << "[ y ";
        for (const int& ind : subset) {
          std::cerr << y[colMask[ind]] << " ";
        }
        std::cerr << "]";

        std::cerr << "[ yhat ";
        for (const int& ind : subset) {
          std::cerr << yhat[colMask[ind]] << " ";
        }
        std::cerr << "]";

        DataType sum_a = 0., sum_b = 0., ab = 0., ab2 = 0.;
        std::cerr << "[a/b: ";
        for (const int& ind : subset) {
          sum_a += a[ind];
          sum_b += b[ind];
          ab += a[ind] / b[ind];
          ab2 += std::pow(a[ind] / b[ind], 2);
          std::cerr << a[ind] / b[ind] << " ";
        }
        std::cerr << "]";

        DataType ab_mean = ab / static_cast<DataType>(subset.size());
        DataType ab2_mean = ab2 / static_cast<DataType>(subset.size());
        DataType stddev = ab2_mean - std::pow(ab_mean, 2);
        std::cerr << "[agg priority: " << sum_a / sum_b << " avg priority: " << ab_mean
                  << " stddev priority: " << stddev << " cum ratl score: " << sum_a * sum_a / sum_b
                  << "]\n"
                  << "]";
        if (fabs(stddev) > abs_stddev_max) {
          abs_stddev_max = fabs(stddev);
        }
      });
  std::cerr << "MAX STDDEV: " << abs_stddev_max << std::endl;
}

template <typename T>
void writeBinary(
    std::string fileName, const T& data, boost::filesystem::path fldr = boost::filesystem::path{}) {
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
    } catch (std::ios_base::failure&) {
      std::cerr << "Failed to write to " << abs_path << std::endl;
    }
    ofs.close();
  }
}

template <typename T>
void readBinary(
    std::string fileName, T& obj, boost::filesystem::path fldr = boost::filesystem::path{}) {
  std::string abs_path;
  if (fldr.string().size()) {
    fldr /= fileName;
    abs_path = fldr.string();
  } else {
    abs_path = fileName;
  }

  std::ifstream ifs{abs_path, std::ios::ate | std::ios::binary};
  if (ifs.is_open()) {
    ifs.seekg(0, std::ios_base::beg);

    try {
      ifs.read(reinterpret_cast<char*>(&obj), sizeof(T));
    } catch (std::ios_base::failure&) {
      std::cerr << "Failed to read from " << abs_path << std::endl;
    }
    ifs.close();
  }
}

template <typename DataType>
void readPrediction(
    std::string indexName,
    Row<DataType>& prediction,
    boost::filesystem::path fldr = boost::filesystem::path{}) {
  Row<DataType> predictionNew;
  std::vector<std::string> fileNames;
  readIndex(indexName, fileNames, fldr);

  for (auto& fileName : fileNames) {
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

}  // namespace IB_utils

#endif
