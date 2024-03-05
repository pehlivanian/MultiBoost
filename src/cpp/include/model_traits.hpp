#ifndef __MODEL_TRAITS_HPP__
#define __MODEL_TRAITS_HPP__

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

#include "constantclassifier.hpp"
#include "constantregressor.hpp"

class DecisionTreeRegressorRegressor;
class ConstantTreeRegressorRegressor;
class DecisionTreeClassifier;
class RandomForestClassifier;
class ConstantTreeClassifier;

template<typename DecoratedType, typename... Args>
class DecoratorClassifier;
template<typename DecoratedType, typename... Args>
class NegativeFeedback;

using namespace mlpack;

namespace Model_Traits {
    
  using AllClassifierArgs = std::tuple<std::size_t,	// (0) numClasses
				       std::size_t,	// (1) minLeafSize
				       double,		// (2) minGainSplit
				       std::size_t,	// (3) numTrees
				       std::size_t>;	// (4) maxDepth

  namespace ClassifierTypes {
    using DecisionTreeClassifierType = DecisionTree<>;
    using RandomForestClassifierType = RandomForest<>;
    using ConstantTreeClassifierType = ConstantTree;
    using NegativeFeedbackDecisionTreeClassifierType = DecisionTree<>;
    using NegativeFeedbackRandomForestClassifierType = RandomForest<>;
    // using NegativeFeedbackDecisionTreeClassifierType = NegativeFeedback<DecisionTreeClassifier, 
    // 									std::size_t, std::size_t, double, std::size_t>;
    // using NegativeFeedbackRandomForestClassifierType = NegativeFeedback<RandomForestClassifier,
    // 								std::size_t, std::size_t, double, std::size_t>;

    // [==========--===========]
    // [============--=========]
    // [==============--=======]
    // Possible options
    // [==========--===========]
    // [========--=============]
    // [======--===============]
    // using DecisionTreeClassifierType = DecisionTree<GiniGain, BestBinaryNumericSplit>;
    // using DecisionTreeClassifierType = DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
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
  struct model_traits<RandomForestClassifier> {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = ClassifierTypes::RandomForestClassifierType;
    using modelArgs = std::tuple<std::size_t, std::size_t, std::size_t>;
  };

  template<>
  struct model_traits<ConstantTreeClassifier> {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = ClassifierTypes::ConstantTreeClassifierType;
    using modelArgs = std::tuple<std::size_t, std::size_t>;
  };

  template<>
  struct model_traits<DecoratorClassifier<DecisionTreeClassifier, std::size_t, std::size_t, double, std::size_t>> {
    using datatype = model_traits<DecisionTreeClassifier>::datatype;
    using integrallabeltype = model_traits<DecisionTreeClassifier>::integrallabeltype;
    using model = model_traits<DecisionTreeClassifier>::model;
    using modelArgs = model_traits<DecisionTreeClassifier>::modelArgs;
  };

  template<>
  struct model_traits<NegativeFeedback<DecisionTreeClassifier, std::size_t, std::size_t, double, std::size_t>> {
    using datatype = model_traits<DecisionTreeClassifier>::datatype;
    using integrallabeltype = model_traits<DecisionTreeClassifier>::integrallabeltype;
    using model = model_traits<DecisionTreeClassifier>::model;
    using modelArgs = model_traits<DecisionTreeClassifier>::modelArgs;    
  };

  using AllRegressorArgs = std::tuple<std::size_t,	// (0) minLeafSize
				      double,		// (1) minGainSplit
				      std::size_t>;	// (2) maxDepth
  
  namespace RegressorTypes {

    // XXX
    using DecisionTreeRegressorRegressorType = DecisionTreeRegressor<MADGain>;
    using ConstantTreeRegressorRegressorType = ConstantTreeRegressor;

    // [==========--===========]
    // [============--=========]
    // [==============--=======]
    // Possible options
    // [==========--===========]
    // [========--=============]
    // [======--===============]
    // using DecisionTreeRegressorRegressorType  = DecisionTreeRegressor<MADGain, BestBinaryNumericSplit>;
    // using DecisionTreeRegressorRegressorType = DecisionTreeRegressor<>;
    // using DecisionTreeRegressorRegressorType = DecisionTreeRegressor<MSEGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
    // using DecisionTreeRegressorRegressorType = DecisionTreeRegressor<InformationGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
    // using DecisionTreeRegressorRegressorType = DecisionTreeRegressor<MADGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  
  };

  template<typename T>
  struct is_classifier {
    bool operator()() { return true; }
  };

  template<>
  struct is_classifier<DecisionTreeRegressorRegressor> {
    bool operator()() { return false; }
  };

  template<>
  struct is_classifier<ConstantTreeRegressorRegressor> {
    bool operator()() { return false; }
  };

  template<>
  struct model_traits<DecisionTreeRegressorRegressor> {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = RegressorTypes::DecisionTreeRegressorRegressorType;
    using modelArgs = std::tuple<std::size_t, double, std::size_t>;
  };

  template<>
  struct model_traits<ConstantTreeRegressorRegressor> {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = RegressorTypes::ConstantTreeRegressorRegressorType;
    using modelArgs = std::tuple<double>;
  };

} // namespace Model_Traits


#endif
