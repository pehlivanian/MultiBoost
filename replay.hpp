#ifndef __REPLAY_HPP__
#define __REPLAY_HPP__

#include <numeric>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include <mlpack/core.hpp>
#include <boost/process.hpp>
#include <boost/process/child.hpp>

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


template<typename DataType, typename ClassifierType>
class Replay {
public:
  Replay()  = default;
  ~Replay() = default;

  // Classify methods are for classifiers
  // Predict  methods are for regressors
  static void Classify(std::string, const mat&, Row<DataType>&, bool=false);
  static void Predict(std::string, const mat&, Row<DataType>&);

  static void Classify(std::string, Row<DataType>&);
  static void Predict(std::string, Row<DataType>&);

  static void ClassifyStepwise(std::string, Row<DataType>&, Row<DataType>&, bool=false, bool=false);
  static void PredictStepwise(std::string, Row<DataType>&, Row<DataType>&, bool);

  static void ClassifyStep(std::string, std::string, Row<double>&, bool=false);
  static void PredictStep(std::string, std::string, Row<double>&);

  static void ClassifyStep(std::string, std::string, std::string, bool=false);
  static void PredictStep(std::string, std::string, std::string);  


private:
  static void desymmetrize(Row<DataType>&, double, double);

};

#include "replay_impl.hpp"

#endif
