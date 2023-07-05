#ifndef __REGRESSORS_HPP__
#define __REGRESSORS_HPP__

#include <utility>

#include <mlpack/core.hpp>
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

  using AllRegressorArgs = std::tuple<std::size_t,	// (0) minLeafSize
				      double,		// (1) minGainSplit
				      std::size_t>;	// (2) maxDepth
  
  namespace RegressorTypes {

    using DecisionTreeRegressorRegressorType  = DecisionTreeRegressor<MADGain, BestBinaryNumericSplit>;

    // using DecisionTreeRegressorRegressorType = DecisionTreeRegressor<MADGain>;
    // using DecisionTreeRegressorRegressorType = DecisionTreeRegressor<>;
    // using DecisionTreeRegressorRegressorType = DecisionTreeRegressor<MSEGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
    // using DecisionTreeRegressorRegressorType = DecisionTreeRegressor<InformationGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  
  };
} // namespace Model_Traits

template<typename... Args>
class DecisionTreeRegressorRegressorBase : 
  public ContinuousRegressorBase<double,
				 Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType,
				 Args...> {
public:
  DecisionTreeRegressorRegressorBase() = default;
  
  DecisionTreeRegressorRegressorBase(const mat& dataset,
				      rowvec& labels,
				      Args&&... args) :
    ContinuousRegressorBase<double, Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType, Args...>(dataset, labels, std::forward<Args>(args)...) {}

};

class DecisionTreeRegressorRegressor : 
  public DecisionTreeRegressorRegressorBase<std::size_t, double, std::size_t> {
  
public:

  using Args = std::tuple<std::size_t, double, std::size_t>;

  DecisionTreeRegressorRegressor() = default;
  DecisionTreeRegressorRegressor(const mat& dataset,
				 rowvec& labels,
				 std::size_t minLeafSize=1,
				 double minGainSplit=0.,
				 std::size_t maxDepth=100) :
    DecisionTreeRegressorRegressorBase<std::size_t, double, std::size_t>(dataset, 
								labels, 
								std::move(minLeafSize),
								std::move(minGainSplit),
								std::move(maxDepth))
  {}

  static Args _args(const Model_Traits::AllRegressorArgs& p) {
    return std::make_tuple(std::get<0>(p),
			   std::get<1>(p),
			   std::get<2>(p));
  }

};

namespace Model_Traits {

  template<>
  struct is_classifier<DecisionTreeRegressorRegressor> {
    bool operator()() { return false; }
  };

  template<typename RegressorType>
  struct regressor_traits {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = RegressorTypes::DecisionTreeRegressorRegressorType;
    using modelArgs = std::tuple<std::size_t, double, std::size_t>;
  };
  
  template<>
  struct regressor_traits<DecisionTreeRegressorRegressor> {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = RegressorTypes::DecisionTreeRegressorRegressorType;
    using modelArgs = std::tuple<std::size_t, double, std::size_t>;
  };

} // namespace Model_Traits

using DTRRB = DecisionTreeRegressorRegressorBase<std::size_t, double, std::size_t>;

using ContinuousRegressorBaseDTRRB = ContinuousRegressorBase<double,
							     Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType,
							     std::size_t,
							     double,
							     std::size_t>;

using RegressorBaseDTRR = RegressorBase<double, Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType>;
							     
using ModelDTRR = Model<double, Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType>;

CEREAL_REGISTER_TYPE(RegressorBaseDTRR);

CEREAL_REGISTER_TYPE(ContinuousRegressorBaseDTRRB);

CEREAL_REGISTER_TYPE(DTRRB);

CEREAL_REGISTER_TYPE(DecisionTreeRegressorRegressor);

CEREAL_REGISTER_TYPE(ModelDTRR);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTRR, DecisionTreeRegressorRegressor);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTRR, DTRRB);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTRR, ContinuousRegressorBaseDTRRB);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTRR, RegressorBaseDTRR);

CEREAL_REGISTER_POLYMORPHIC_RELATION(RegressorBaseDTRR, DecisionTreeRegressorRegressor);

#endif
