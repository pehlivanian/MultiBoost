#ifndef __MODEL_TRAITS_HPP__
#define __MODEL_TRAITS_HPP__

#include <utility>

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

namespace Model_Traits {

  class DecisionTreeClassifier;
  class RandomForestClassifier;
  class DecisionTreeRegressorRegressor;

  namespace ClassifierTypes {
    using RandomForestClassifierType = RandomForest<>;
    using DecisionTreeClassifierType = DecisionTree<>;

    // using DecisionTreeClassifierType = DecisionTree<GiniGain, BestBinaryNumericSplit>;
    // using DecisionTreeClassifierType = DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<MADGain>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<MSEGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<InformationGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  
  };


  namespace RegressorTypes {
    using DecisionTreeRegressorRegressorType  = DecisionTreeRegressor<MADGain, BestBinaryNumericSplit>;

    // using DecisionTreeClassifierType = DecisionTreeRegressor<MADGain>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<MSEGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<InformationGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  
  };

  template<typename T>
  struct model_traits {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = ClassifierTypes::DecisionTreeClassifierType;
    using modelArgs = std::tuple<std::size_t, std::size_t, double, std::size_t>;
  };

  template<>
  struct model_traits<DecisionTreeClassifier> {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = ClassifierTypes::DecisionTreeClassifierType;
    using modelArgs = std::tuple<std::size_t, std::size_t, double, std::size_t>;
  };

  template<>
  struct model_traits<DecisionTreeRegressorRegressor> {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = RegressorTypes::DecisionTreeRegressorRegressorType;
    using modelArgs = std::tuple<std::size_t, double, std::size_t>;
  };

  template<>
  struct model_traits<RandomForestClassifier> {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = ClassifierTypes::RandomForestClassifierType;
    using modelArgs = std::tuple<std::size_t, std::size_t, std::size_t>;
  };
} // namespace Model_Traits

#endif
