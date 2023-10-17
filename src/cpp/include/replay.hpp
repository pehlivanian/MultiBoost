#ifndef __REPLAY_HPP__
#define __REPLAY_HPP__

#include <numeric>
#include <string>
#include <vector>
#include <algorithm>
#include <optional>
#include <memory>
#include <cmath>

#include <mlpack/core.hpp>
#include <boost/process.hpp>
#include <boost/process/child.hpp>
#include <boost/filesystem.hpp>

#include "utils.hpp"
#include "threadsafequeue.hpp"
#include "threadpool.hpp"
#include "gradientboostclassifier.hpp"
#include "gradientboostregressor.hpp"
#include "model_traits.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

using namespace boost::process;

using namespace Model_Traits;
using namespace IB_utils;
using namespace LossMeasures;


template<typename DataType, typename ModelType>
class Replay {

  using optRV = std::tuple<std::optional<double>, 
			   std::optional<double>,
			   std::optional<double>,
			   std::optional<double>,
			   std::optional<double>,
			   std::optional<double>,
			   std::optional<double>,
			   std::optional<double>>;
  using optCV = std::tuple<std::optional<double>,
			   std::optional<double>,
			   std::optional<double>,
			   std::optional<double>,
			   std::optional<double>,
			   std::optional<double>,
			   std::optional<double>,
			   std::optional<double>,
			   std::optional<double>,
			   std::optional<double>>;
			   

public:
  Replay()  = default;
  ~Replay() = default;

  // Classify methods are for classifiers
  // Predict  methods are for regressors

  //
  // For predicting OOS on given test dataset /
  /////////////////////////////////////////////
  static void Classify(std::string, const Mat<DataType>&, Row<DataType>&, boost::filesystem::path=boost::filesystem::path{}, bool=false);
  static void Predict(std::string, const Mat<DataType>&, Row<DataType>&, boost::filesystem::path=boost::filesystem::path{});
  /////////////////////////////////////////////

  // 
  // For predicting IS for given model /
  //////////////////////////////////////
  static void Classify(std::string, Row<DataType>&, boost::filesystem::path=boost::filesystem::path{});
  static void Predict(std::string, Row<DataType>&, boost::filesystem::path=boost::filesystem::path{});
  //////////////////////////////////////

  // Helpers for incremental trainers /
  /////////////////////////////////////
  static optCV ClassifyStepwise(std::string, Row<DataType>&, Row<DataType>&, bool=false, bool=false, bool=false, boost::filesystem::path=boost::filesystem::path{});
  static optRV PredictStepwise(std::string, Row<DataType>&, Row<DataType>&, bool=false, bool=false, boost::filesystem::path=boost::filesystem::path{});

  // Helpers for incremental trainers
  static void ClassifyStep(std::string, std::string, Row<double>&, bool=false, boost::filesystem::path=boost::filesystem::path{});
  static void PredictStep(std::string, std::string, Row<double>&, boost::filesystem::path=boost::filesystem::path{});

  // Helpers for incremental trainers
  static void ClassifyStep(std::string, std::string, std::string, bool=false, boost::filesystem::path=boost::filesystem::path{});
  static void PredictStep(std::string, std::string, std::string, boost::filesystem::path=boost::filesystem::path{});  
  /////////////////////////////////////

private:
  static void desymmetrize(Row<DataType>&, double, double);

};

#include "replay_impl.hpp"

#endif
