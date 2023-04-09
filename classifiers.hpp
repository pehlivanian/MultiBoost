#ifndef __CLASSIFIERS_HPP__
#define __CLASSIFIERS_HPP__

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

#include "model.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;


namespace ClassifierTypes {
  using DecisionTreeRegressorType  = DecisionTreeRegressor<MADGain, BestBinaryNumericSplit>;
  using RandomForestClassifierType = RandomForest<>;
  using DecisionTreeClassifierType = DecisionTree<>;

  // using DecisionTreeClassifierType = DecisionTree<GiniGain, BestBinaryNumericSplit>;
  // using DecisionTreeClassifierType = DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  // using DecisionTreeClassifierType = DecisionTreeRegressor<MADGain>;
  // using DecisionTreeClassifierType = DecisionTreeRegressor<>;
  // using DecisionTreeClassifierType = DecisionTreeRegressor<MSEGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  // using DecisionTreeClassifierType = DecisionTreeRegressor<InformationGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  
};



template<typename... Args>
class DecisionTreeRegressorClassifierBase : 
  public ContinuousClassifierBase<double, 
				  ClassifierTypes::DecisionTreeRegressorType,
				  Args...> {
public:
  DecisionTreeRegressorClassifierBase() = default;

  DecisionTreeRegressorClassifierBase(const mat& dataset,
				      rowvec& labels,
				      Args&&... args) :
    ContinuousClassifierBase<double, ClassifierTypes::DecisionTreeRegressorType, Args...>(dataset, labels, std::forward<Args>(args)...) {}
  
};

class DecisionTreeRegressorClassifier : 
  public DecisionTreeRegressorClassifierBase<std::size_t, double, std::size_t> {
  
public:
  DecisionTreeRegressorClassifier() = default;
  DecisionTreeRegressorClassifier(const mat& dataset,
				  rowvec& labels,
				  std::size_t minLeafSize=1,
				  double minGainSplit=0.,
				  std::size_t maxDepth=100) :
    DecisionTreeRegressorClassifierBase<std::size_t, double, std::size_t>(dataset, 
									  labels, 
									  std::move(minLeafSize),
									  std::move(minGainSplit),
									  std::move(maxDepth))
  {}
};

template<typename... Args>
class RandomForestClassifierBase : 
  public DiscreteClassifierBase<double,
				ClassifierTypes::RandomForestClassifierType,
				Args...> {
public:
  RandomForestClassifierBase() = default;
  
  RandomForestClassifierBase(const mat& dataset,
			     rowvec& labels,
			     Args&&... args) :
    DiscreteClassifierBase<double, ClassifierTypes::RandomForestClassifierType, Args...>(dataset, labels, std::forward<Args>(args)...) {}
  
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
				ClassifierTypes::DecisionTreeClassifierType,
				Args...> {
public:
  DecisionTreeClassifierBase() = default;
  
  DecisionTreeClassifierBase(const mat& dataset,
			     rowvec& labels,
			     Args&&... args) :
    DiscreteClassifierBase<double, ClassifierTypes::DecisionTreeClassifierType, Args...>(dataset, labels, std::forward<Args>(args)...) {}
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


#endif
