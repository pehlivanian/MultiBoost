#ifndef __REGRESSORS_HPP__
#define __REGRESSORS_HPP__

#include <utility>

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/access.hpp>


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

  DecisionTreeRegressorRegressorBase(const Mat<DataType>& dataset,
				     Row<DataType>& labels,
				     Row<DataType>& weights,
				     Args&&... args) :
    ContinuousRegressorBase<DataType,
			    RegressorType,
			    Args...>(dataset, labels, weights, std::forward<Args>(args)...) {}

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

  DecisionTreeRegressorRegressor(const Mat<DataType>& dataset,
				 Row<DataType>& labels,
				 Row<DataType>& weights,
				 std::size_t minLeafSize=1,
				 double minGainSplit=0.,
				 std::size_t maxDepth=100) :
    DecisionTreeRegressorRegressorBase<std::size_t, double, std::size_t>(dataset,
									 labels,
									 weights,
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
  
  ConstantTreeRegressorRegressorBase(const Mat<DataType>& dataset,
				     Row<DataType>& labels,
				     Args&&... args) :
    ContinuousRegressorBase<DataType,
			    RegressorType,
			    Args...>(dataset, labels, std::forward<Args>(args)...) {}

  ConstantTreeRegressorRegressorBase(const Mat<DataType>& dataset,
				     Row<DataType>& labels,
				     Row<DataType>& weights,
				     Args&&... args) :
    ContinuousRegressorBase<DataType,
			    RegressorType,
			    Args...>(dataset, labels, weights, std::forward<Args>(args)...) {}
};

class ConstantTreeRegressorRegressor :
  public ConstantTreeRegressorRegressorBase<> {
public:
  using Args = std::tuple<>;
  using DataType = Model_Traits::model_traits<ConstantTreeRegressorRegressor>::datatype;

  ConstantTreeRegressorRegressor() = default;

  ConstantTreeRegressorRegressor(const Mat<DataType>& dataset,
				 Row<DataType>& labels) :
    ConstantTreeRegressorRegressorBase<>(dataset, 
					 labels)
  {}
  
  ConstantTreeRegressorRegressor(const Mat<DataType>& dataset,
				 Row<DataType>& labels,
				 Row<DataType>& weights) :
    ConstantTreeRegressorRegressorBase<>(dataset, labels, weights) 
  {}
};

using DTRRB = DecisionTreeRegressorRegressorBase<std::size_t, double, std::size_t>;
using CTRRB = ConstantTreeRegressorRegressorBase<>;

using ContinuousRegressorBaseDTRRBD = ContinuousRegressorBase<double,
							     Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType,
							     std::size_t,
							     double,
							     std::size_t>;
using ContinuousRegressorBaseCTRRBD = ContinuousRegressorBase<double,
							      Model_Traits::RegressorTypes::ConstantTreeRegressorRegressorType>;


using RegressorBaseDTRRD = RegressorBase<double, Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType>;
using RegressorBaseCTRRD = RegressorBase<double, Model_Traits::RegressorTypes::ConstantTreeRegressorRegressorType>;

							     
using ModelD = Model<double>;

CEREAL_REGISTER_TYPE(RegressorBaseDTRRD);
CEREAL_REGISTER_TYPE(RegressorBaseCTRRD);

CEREAL_REGISTER_TYPE(ContinuousRegressorBaseDTRRBD);
CEREAL_REGISTER_TYPE(ContinuousRegressorBaseCTRRBD);

CEREAL_REGISTER_TYPE(DTRRB);
CEREAL_REGISTER_TYPE(CTRRB);

CEREAL_REGISTER_TYPE(DecisionTreeRegressorRegressor);
CEREAL_REGISTER_TYPE(ConstantTreeRegressorRegressor);

// Redefinition
// CEREAL_REGISTER_TYPE(ModelD);
// CEREAL_REGISTER_TYPE(ModelF);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DecisionTreeRegressorRegressor);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ConstantTreeRegressorRegressor);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DTRRB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, CTRRB);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ContinuousRegressorBaseDTRRBD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ContinuousRegressorBaseCTRRBD);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, RegressorBaseDTRRD)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, RegressorBaseCTRRD)

CEREAL_REGISTER_POLYMORPHIC_RELATION(RegressorBaseDTRRD, DecisionTreeRegressorRegressor);
CEREAL_REGISTER_POLYMORPHIC_RELATION(RegressorBaseDTRRD, ConstantTreeRegressorRegressor);

#endif
