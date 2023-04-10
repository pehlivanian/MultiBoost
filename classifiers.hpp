#ifndef __CLASSIFIERS_HPP__
#define __CLASSIFIERS_HPP__

#include "classifier.hpp"
#include "model_traits.hpp"

using namespace Model_Traits;

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
