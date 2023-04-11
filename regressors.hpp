#ifndef __REGRESSORS_HPP__
#define __REGRESSORS_HPP__

#include <utility>

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

#include "regressor.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

namespace Model_Traits {

  namespace RegressorTypes {
    using DecisionTreeRegressorRegressorType  = DecisionTreeRegressor<MADGain, BestBinaryNumericSplit>;

    // using DecisionTreeRegressorRegressorType = DecisionTreeRegressor<MADGain>;
    // using DecisionTreeRegressorRegressorType = DecisionTreeRegressor<>;
    // using DecisionTreeRegressorRegressorType = DecisionTreeRegressor<MSEGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
    // using DecisionTreeRegressorRegressorType = DecisionTreeRegressor<InformationGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  
  };
} // namespace Model_Traits

template<typename... Args>
class DecisionTreeRegressorBase : 
  public ContinuousRegressorBase<double,
				 Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType,
				 Args...> {
public:
  DecisionTreeRegressorBase() = default;
  
  DecisionTreeRegressorBase(const mat& dataset,
				      rowvec& labels,
				      Args&&... args) :
    ContinuousRegressorBase<double, Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType, Args...>(dataset, labels, std::forward<Args>(args)...) {}

};

class DecisionTreeRegressorRegressor : 
  public DecisionTreeRegressorBase<std::size_t, double, std::size_t> {
  
public:
  DecisionTreeRegressorRegressor() = default;
  DecisionTreeRegressorRegressor(const mat& dataset,
				 rowvec& labels,
				 std::size_t minLeafSize=1,
				 double minGainSplit=0.,
				 std::size_t maxDepth=100) :
    DecisionTreeRegressorBase<std::size_t, double, std::size_t>(dataset, 
								labels, 
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
  struct model_traits<DecisionTreeRegressorRegressor> {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = RegressorTypes::DecisionTreeRegressorRegressorType;
    using modelArgs = std::tuple<std::size_t, double, std::size_t>;
  };

} // namespace Model_Traits

#endif
