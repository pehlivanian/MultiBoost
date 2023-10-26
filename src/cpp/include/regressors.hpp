#ifndef __REGRESSORS_HPP__
#define __REGRESSORS_HPP__

#include <utility>

#include "model_traits.hpp"
#include "regressor.hpp"

template<typename... Args>
class DecisionTreeRegressorRegressorBase : 
  public ContinuousRegressorBase<Model_Traits::model_traits<DecisionTreeRegressorRegressor>::datatype,
				 Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType,
				 Args...> {
public:
  using DataType = Model_Traits::model_traits<DecisionTreeRegressorRegressor>::datatype;
  using RegressorType = Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType;

  DecisionTreeRegressorRegressorBase() = default;
  
  DecisionTreeRegressorRegressorBase(const Mat<DataType>& dataset,
				      Row<DataType>& labels,
				      Args&&... args) :
    ContinuousRegressorBase<DataType, 
			    RegressorType, 
			    Args...>(dataset, labels, std::forward<Args>(args)...) {}

};

class DecisionTreeRegressorRegressor : 
  public DecisionTreeRegressorRegressorBase<std::size_t, double, std::size_t> {
  
public:

  using Args = std::tuple<std::size_t, double, std::size_t>;
  using DataType = Model_Traits::model_traits<DecisionTreeRegressorRegressor>::datatype;

  DecisionTreeRegressorRegressor() = default;
  DecisionTreeRegressorRegressor(const Mat<DataType>& dataset,
				 Row<DataType>& labels,
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

template<typename... Args>
class ConstantTreeRegressorRegressorBase :
  public ContinuousRegressorBase<Model_Traits::model_traits<ConstantTreeRegressorRegressor>::datatype,
				 Model_Traits::RegressorTypes::ConstantTreeRegressorRegressorType,
				 Args...> {
public:
  using DataType = Model_Traits::model_traits<ConstantTreeRegressorRegressor>::datatype;
  using RegressorType = Model_Traits::RegressorTypes::ConstantTreeRegressorRegressorType;

  ConstantTreeRegressorRegressorBase() = default;
  ConstantTreeRegressorRegressorBase(const ConstantTreeRegressorRegressorBase&) = default;
  
  ConstantTreeRegressorRegressorBase(const Mat<DataType>& dataset,
			    Row<DataType>& labels,
			    Args&&... args) :
    ContinuousRegressorBase<DataType,
			    RegressorType,
			    Args...>(dataset, labels, std::forward<Args>(args)...) {}
};

class ConstantTreeRegressorRegressor :
  public ConstantTreeRegressorRegressorBase<double> {
public:
  using Args = std::tuple<double>;
  using DataType = Model_Traits::model_traits<ConstantTreeRegressorRegressor>::datatype;
  
  ConstantTreeRegressorRegressor() = default;
  ConstantTreeRegressorRegressor(ConstantTreeRegressorRegressor&) = default;

  ConstantTreeRegressorRegressor(const Mat<DataType>& dataset,
				 Row<DataType>& labels,
				 double leafValue) :
    ConstantTreeRegressorRegressorBase<double>(dataset,
					       labels,
					       std::move(leafValue))
  {}

};


using DTRRB = DecisionTreeRegressorRegressorBase<std::size_t, double, std::size_t>;

using ContinuousRegressorBaseDTRRBD = ContinuousRegressorBase<double,
							     Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType,
							     std::size_t,
							     double,
							     std::size_t>;
// using ContinuousRegressorBaseDTRRBF = ContinuousRegressorBase<float,
// 							      Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType,
// 							      std::size_t,
// 							      double,
// 							      std::size_t>;

using RegressorBaseDTRRD = RegressorBase<double, Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType>;
// using RegressorBaseDTRRF = RegressorBase<float,  Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType>;
							     
using ModelD = Model<double>;
// using ModelF = Model<float>;

CEREAL_REGISTER_TYPE(RegressorBaseDTRRD);
// CEREAL_REGISTER_TYPE(RegressorBaseDTRRF);

CEREAL_REGISTER_TYPE(ContinuousRegressorBaseDTRRBD);
// CEREAL_REGISTER_TYPE(ContinuousRegressorBaseDTRRBF);

CEREAL_REGISTER_TYPE(DTRRB);

CEREAL_REGISTER_TYPE(DecisionTreeRegressorRegressor);

// Redefinition
// CEREAL_REGISTER_TYPE(ModelD);
// CEREAL_REGISTER_TYPE(ModelF);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DecisionTreeRegressorRegressor);
// CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DecisionTreeRegressorRegressor);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DTRRB);
// CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DTRRB);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ContinuousRegressorBaseDTRRBD);
// CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, ContinuousRegressorBaseDTRRBF);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, RegressorBaseDTRRD)
// CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, RegressorBaseDTRRF);

CEREAL_REGISTER_POLYMORPHIC_RELATION(RegressorBaseDTRRD, DecisionTreeRegressorRegressor);
// CEREAL_REGISTER_POLYMORPHIC_RELATION(RegressorBaseDTRRF, DecisionTreeRegressorRegressor);

#endif
