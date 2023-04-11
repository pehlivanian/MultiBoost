#ifndef __CLASSIFIERS_HPP__
#define __CLASSIFIERS_HPP__

#include <utility>

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

#include "classifier.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

namespace Model_Traits {

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
} // namespace Model_Traits

template<typename... Args>
class RandomForestClassifierBase : 
  public DiscreteClassifierBase<double,
				Model_Traits::ClassifierTypes::RandomForestClassifierType,
				Args...> {
public:
  RandomForestClassifierBase() = default;
  
  RandomForestClassifierBase(const mat& dataset,
			     rowvec& labels,
			     Args&&... args) :
    DiscreteClassifierBase<double, Model_Traits::ClassifierTypes::RandomForestClassifierType, Args...>(dataset, labels, std::forward<Args>(args)...) {}
  
};

class RandomForestClassifier : 
    public RandomForestClassifierBase<std::size_t, std::size_t, std::size_t> {

public:
  RandomForestClassifier() = default;

  RandomForestClassifier(const mat& dataset,
			 rowvec& labels,
			 std::size_t numClasses=1,
			 std::size_t numTrees=10,
			 std::size_t minLeafSize=2) :
    RandomForestClassifierBase<std::size_t, std::size_t, std::size_t>(dataset, 
								      labels, 
								      std::move(numClasses),
								      std::move(numTrees),
								      std::move(minLeafSize)) {}
};

template<typename... Args>
class DecisionTreeClassifierBase : 
  public DiscreteClassifierBase<double,
				Model_Traits::ClassifierTypes::DecisionTreeClassifierType,
				Args...> {
public:
  DecisionTreeClassifierBase() = default;
  
  DecisionTreeClassifierBase(const mat& dataset,
			     rowvec& labels,
			     Args&&... args) :
    DiscreteClassifierBase<double, Model_Traits::ClassifierTypes::DecisionTreeClassifierType, Args...>(dataset, labels, std::forward<Args>(args)...) {}
};

class DecisionTreeClassifier : 
  public DecisionTreeClassifierBase<std::size_t, std::size_t, double, std::size_t> {

public:
  DecisionTreeClassifier() = default;
  
  DecisionTreeClassifier(const mat& dataset,
			 rowvec& labels,
			 std::size_t numClasses,
			 std::size_t minLeafSize,
			 double minGainSplit,
			 std::size_t maxDepth) :
    DecisionTreeClassifierBase<std::size_t, std::size_t, double, std::size_t>(dataset,
									      labels,
									      std::move(numClasses),
									      std::move(minLeafSize),
									      std::move(minGainSplit),
									      std::move(maxDepth))
  {}
};

namespace Model_Traits {

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
  struct model_traits<RandomForestClassifier> {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = ClassifierTypes::RandomForestClassifierType;
    using modelArgs = std::tuple<std::size_t, std::size_t, std::size_t>;
  };
} // namespace Model_Traits

#endif
