#ifndef __REPLAY_HPP__
#define __REPLAY_HPP__

#include <numeric>
#include <string>
#include <algorithm>
#include <memory>

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

#include "utils.hpp"
#include "gradientboostclassifier.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

using namespace IB_utils;

template<typename DataType, typename ClassifierType>
class Replay {
public:
  Replay() = default;
  ~Replay() = default;

  static void Predict(std::string, const mat&, Row<DataType>&);
  static void Classify(std::string, const mat&, Row<DataType>&);
  static void read(GradientBoostClassifier<ClassifierType>&, std::string);

};

#include "replay_impl.hpp"

#endif
