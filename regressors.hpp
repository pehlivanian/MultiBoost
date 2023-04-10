#ifndef __REGRESSORS_HPP__
#define __REGRESSORS_HPP__

#include "regressor.hpp"
#include "model_traits.hpp"

using namespace Model_Traits;

template<typename... Args>
class DecisionTreeRegressorBase : 
  public ContinuousRegressorBase<double,
				 RegressorTypes::DecisionTreeRegressorType,
				 Args...> {
public:
  DecisionTreeRegressorClassifierBase() = default;
  
  DecisionTreeRegressorClassifierBase(const mat& dataset,
				      rowvec& labels,
				      Args&&... args) :
    ContinuousRegressorBase<double, ClassifierTypes::DecisionTreeRegressorType, Args...>(dataset, labels, std::forward<Args>(args)...) {}

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

#endif
