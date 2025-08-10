#ifndef __REGRESSORS_HPP__
#define __REGRESSORS_HPP__

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <utility>

#include "model_traits.hpp"
#include "regressor.hpp"

template <typename DecoratedType, typename... Args>
class DecoratorRegressorBase : public ContinuousRegressorBase<
                                    typename Model_Traits::model_traits<DecoratedType>::datatype,
                                    typename Model_Traits::model_traits<DecoratedType>::model,
                                    Args...> {
public:
  using DataType = typename Model_Traits::model_traits<DecisionTreeRegressorRegressor>::datatype;
  using RegressorType = typename Model_Traits::model_traits<DecoratedType>::model;

  DecoratorRegressorBase() = default;

  DecoratorRegressorBase(const Mat<DataType>& dataset, Row<DataType>& labels, Args&&... args)
      : ContinuousRegressorBase<DataType, RegressorType, Args...>(
            dataset, labels, std::forward<Args>(args)...) {}

  DecoratorRegressorBase(const Mat<DataType>& dataset, Row<DataType>& labels, Args const&... args)
      : ContinuousRegressorBase<DataType, RegressorType, Args...>(dataset, labels, const_cast<Args&&>(args)...) {}

  DecoratorRegressorBase(
      const Mat<DataType>& dataset, Row<DataType>& labels, Row<DataType>& weights, Args&&... args)
      : ContinuousRegressorBase<DataType, RegressorType, Args...>(
            dataset, labels, weights, std::forward<Args>(args)...) {}

  DecoratorRegressorBase(
      const Mat<DataType>& dataset,
      Row<DataType>& labels,
      Row<DataType>& weights,
      Args const&... args)
      : ContinuousRegressorBase<DataType, RegressorType, Args...>(
            dataset, labels, weights, const_cast<Args&&>(args)...) {}
};

template <typename DecoratedType, typename... Args>
class DecoratorRegressor : public DecoratorRegressorBase<DecoratedType, Args...> {
public:
  using tupArgs = typename Model_Traits::model_traits<DecisionTreeRegressorRegressor>::modelArgs;
  using DataType = typename Model_Traits::model_traits<DecisionTreeRegressorRegressor>::datatype;

  DecoratorRegressor() = default;

  DecoratorRegressor(const Mat<DataType>& dataset, Row<DataType>& labels, Args&&... args)
      : DecoratorRegressorBase<DecoratedType, Args...>(
            dataset, labels, std::forward<Args>(args)...) {}

  DecoratorRegressor(const Mat<DataType>& dataset, Row<DataType>& labels, Args const&... args)
      : DecoratorRegressorBase<DecoratedType, Args...>(dataset, labels, args...) {}

  DecoratorRegressor(
      const Mat<DataType>& dataset, Row<DataType>& labels, Row<DataType>& weights, Args&&... args)
      : DecoratorRegressorBase<DecoratedType, Args...>(
            dataset, labels, weights, std::forward<Args>(args)...) {}

  DecoratorRegressor(
      const Mat<DataType>& dataset,
      Row<DataType>& labels,
      Row<DataType>& weights,
      Args const&... args)
      : DecoratorRegressorBase<DecoratedType, Args...>(dataset, labels, weights, args...) {}

  static tupArgs _args(const Model_Traits::AllRegressorArgs& p) { return DecoratedType::_args(p); }
};

template <typename DecoratedType, typename... DecoratedArgs>
class NegativeFeedbackRegressor : public DecoratorRegressor<DecoratedType, DecoratedArgs...> {
public:
  using Args = typename Model_Traits::model_traits<DecoratedType>::modelArgs;
  using DataType = typename Model_Traits::model_traits<DecoratedType>::datatype;

  NegativeFeedbackRegressor() : beta_{.001}, iterations_{3} {}

  NegativeFeedbackRegressor(float beta, std::size_t iterations) : beta_{beta}, iterations_{iterations} {}

  NegativeFeedbackRegressor(const Mat<DataType>& dataset, Row<DataType>& labels, DecoratedArgs&&... args)
      : DecoratorRegressor<DecoratedType, DecoratedArgs...>(
            dataset, labels, std::forward<DecoratedArgs>(args)...), beta_{.001}, iterations_{3} {}

  NegativeFeedbackRegressor(
      const Mat<DataType>& dataset, Row<DataType>& labels, DecoratedArgs const&... args)
      : DecoratorRegressor<DecoratedType, DecoratedArgs...>(dataset, labels, args...), beta_{.001}, iterations_{3} {}

  NegativeFeedbackRegressor(
      const Mat<DataType>& dataset,
      Row<DataType>& labels,
      Row<DataType>& weights,
      DecoratedArgs&&... args)
      : DecoratorRegressor<DecoratedType, DecoratedArgs...>(
            dataset, labels, weights, std::forward<DecoratedArgs>(args)...), beta_{.001}, iterations_{3} {}

  NegativeFeedbackRegressor(
      const Mat<DataType>& dataset,
      Row<DataType>& labels,
      Row<DataType>& weights,
      DecoratedArgs const&... args)
      : DecoratorRegressor<DecoratedType, DecoratedArgs...>(dataset, labels, weights, args...), beta_{.001}, iterations_{3} {}

  template <typename... Ts>
  void setRootRegressor(
      std::unique_ptr<DecoratedType>& regressor,
      const Mat<DataType>& dataset,
      Row<DataType>& labels,
      std::tuple<Ts...> const& args) {
    // Create the base regressor for the caller
    std::apply([&regressor, &dataset, &labels](auto const&... regressorArgs) {
      regressor = std::make_unique<DecoratedType>(dataset, labels, regressorArgs...);
    }, args);
    
    // Initialize the decorator's internal state by setting up its regressor_ member
    // This mimics what would happen if the decorator was constructed with dataset parameters
    std::apply([this, &dataset, &labels](auto... regressorArgs) mutable {
      this->setRegressor(dataset, labels, std::move(regressorArgs)...);
    }, args);
  }

  template <typename... Ts>
  void setRootRegressor(
      std::unique_ptr<DecoratedType>& regressor,
      const Mat<DataType>& dataset,
      Row<DataType>& labels,
      Row<DataType>& weights,
      std::tuple<Ts...> const& args) {
    // Create the base regressor with weights for the caller
    std::apply([&regressor, &dataset, &labels, &weights](auto const&... regressorArgs) {
      regressor = std::make_unique<DecoratedType>(dataset, labels, weights, regressorArgs...);
    }, args);
    
    // Initialize the decorator's internal state by setting up its regressor_ member  
    // This mimics what would happen if the decorator was constructed with dataset parameters
    this->weights_ = weights;
    std::apply([this, &dataset, &labels](auto... regressorArgs) mutable {
      this->setRegressor(dataset, labels, std::move(regressorArgs)...);
    }, args);
  }

  static Args _args(const Model_Traits::AllRegressorArgs& p) { return DecoratedType::_args(p); }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(beta_);
    ar(iterations_);
  }

private:
  void Predict_(const Mat<DataType>& data, Row<DataType>& pred) override {
    // Apply the negative feedback algorithm during prediction
    if (!this->regressor_) {
      throw std::runtime_error("NegativeFeedbackRegressor: regressor_ is null in Predict_");
    }
    
    // Get initial prediction from the underlying regressor
    Row<DataType> initial_pred;
    this->regressor_->Predict(data, initial_pred);
    
    // Apply negative feedback iterations
    pred = initial_pred;
    for (std::size_t i = 0; i < this->iterations_; ++i) {
      // Apply negative feedback transformation
      pred = pred - this->beta_ * initial_pred;
    }
  }
  
  void Predict_(Mat<DataType>&& data, Row<DataType>& pred) override {
    // Forward to const reference version
    Predict_(data, pred);
  }

protected:
  float beta_;
  std::size_t iterations_;
};

template <typename... Args>
class DecisionTreeRegressorRegressorBase
    : public ContinuousRegressorBase<
          Model_Traits::model_traits<DecisionTreeRegressorRegressor>::datatype,
          Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType,
          Args...> {
public:
  using DataType = Model_Traits::model_traits<DecisionTreeRegressorRegressor>::datatype;
  using RegressorType = Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType;

  DecisionTreeRegressorRegressorBase() = default;

  DecisionTreeRegressorRegressorBase(
      const Mat<DataType>& dataset, Row<DataType>& labels, Args&&... args)
      : ContinuousRegressorBase<DataType, RegressorType, Args...>(
            dataset, labels, std::forward<Args>(args)...) {}

  DecisionTreeRegressorRegressorBase(
      const Mat<DataType>& dataset, Row<DataType>& labels, Row<DataType>& weights, Args&&... args)
      : ContinuousRegressorBase<DataType, RegressorType, Args...>(
            dataset, labels, weights, std::forward<Args>(args)...) {}
};

class DecisionTreeRegressorRegressor
    : public DecisionTreeRegressorRegressorBase<std::size_t, double, std::size_t> {
public:
  using Args = std::tuple<std::size_t, double, std::size_t>;
  using DataType = Model_Traits::model_traits<DecisionTreeRegressorRegressor>::datatype;

  DecisionTreeRegressorRegressor() = default;
  DecisionTreeRegressorRegressor(
      const Mat<DataType>& dataset,
      Row<DataType>& labels,
      std::size_t minLeafSize = 1,
      double minGainSplit = 0.,
      std::size_t maxDepth = 100)
      : DecisionTreeRegressorRegressorBase<std::size_t, double, std::size_t>(
            dataset, labels, std::move(minLeafSize), std::move(minGainSplit), std::move(maxDepth)) {
  }

  DecisionTreeRegressorRegressor(
      const Mat<DataType>& dataset,
      Row<DataType>& labels,
      Row<DataType>& weights,
      std::size_t minLeafSize = 1,
      double minGainSplit = 0.,
      std::size_t maxDepth = 100)
      : DecisionTreeRegressorRegressorBase<std::size_t, double, std::size_t>(
            dataset,
            labels,
            weights,
            std::move(minLeafSize),
            std::move(minGainSplit),
            std::move(maxDepth)) {}

  static Args _args(const Model_Traits::AllRegressorArgs& p) {
    return std::make_tuple(std::get<0>(p), std::get<1>(p), std::get<2>(p));
  }
};

template <typename... Args>
class ConstantTreeRegressorRegressorBase
    : public ContinuousRegressorBase<
          Model_Traits::model_traits<ConstantTreeRegressorRegressor>::datatype,
          Model_Traits::RegressorTypes::ConstantTreeRegressorRegressorType,
          Args...> {
public:
  using DataType = Model_Traits::model_traits<ConstantTreeRegressorRegressor>::datatype;
  using RegressorType = Model_Traits::RegressorTypes::ConstantTreeRegressorRegressorType;

  ConstantTreeRegressorRegressorBase() = default;

  ConstantTreeRegressorRegressorBase(
      const Mat<DataType>& dataset, Row<DataType>& labels, Args&&... args)
      : ContinuousRegressorBase<DataType, RegressorType, Args...>(
            dataset, labels, std::forward<Args>(args)...) {}

  ConstantTreeRegressorRegressorBase(
      const Mat<DataType>& dataset, Row<DataType>& labels, Row<DataType>& weights, Args&&... args)
      : ContinuousRegressorBase<DataType, RegressorType, Args...>(
            dataset, labels, weights, std::forward<Args>(args)...) {}
};

class ConstantTreeRegressorRegressor : public ConstantTreeRegressorRegressorBase<> {
public:
  using Args = std::tuple<>;
  using DataType = Model_Traits::model_traits<ConstantTreeRegressorRegressor>::datatype;

  ConstantTreeRegressorRegressor() = default;

  ConstantTreeRegressorRegressor(const Mat<DataType>& dataset, Row<DataType>& labels)
      : ConstantTreeRegressorRegressorBase<>(dataset, labels) {}

  ConstantTreeRegressorRegressor(
      const Mat<DataType>& dataset, Row<DataType>& labels, Row<DataType>& weights)
      : ConstantTreeRegressorRegressorBase<>(dataset, labels, weights) {}
};

// Decorator class using directives
using DR5  = NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t>; 
using DR4  = DecoratorRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t>; 
using DR3  = DecoratorRegressorBase<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t>; 

using DTRRB = DecisionTreeRegressorRegressorBase<std::size_t, double, std::size_t>;
using CTRRB = ConstantTreeRegressorRegressorBase<>;

using ContinuousRegressorBaseDTRRBD = ContinuousRegressorBase<
    double,
    Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType,
    std::size_t,
    double,
    std::size_t>;
using ContinuousRegressorBaseCTRRBD = ContinuousRegressorBase<
    double,
    Model_Traits::RegressorTypes::ConstantTreeRegressorRegressorType>;

using RegressorBaseDTRRD =
    RegressorBase<double, Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType>;
using RegressorBaseCTRRD =
    RegressorBase<double, Model_Traits::RegressorTypes::ConstantTreeRegressorRegressorType>;

using ModelD = Model<double>;

CEREAL_REGISTER_TYPE(RegressorBaseDTRRD);
CEREAL_REGISTER_TYPE(RegressorBaseCTRRD);

CEREAL_REGISTER_TYPE(ContinuousRegressorBaseDTRRBD);
CEREAL_REGISTER_TYPE(ContinuousRegressorBaseCTRRBD);

CEREAL_REGISTER_TYPE(DTRRB);
CEREAL_REGISTER_TYPE(CTRRB);

// Decorator registrations
CEREAL_REGISTER_TYPE(DR3);
CEREAL_REGISTER_TYPE(DR4);
CEREAL_REGISTER_TYPE(DR5);

CEREAL_REGISTER_TYPE(DecisionTreeRegressorRegressor);
CEREAL_REGISTER_TYPE(ConstantTreeRegressorRegressor);

// Redefinition
// CEREAL_REGISTER_TYPE(ModelD);
// CEREAL_REGISTER_TYPE(ModelF);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DecisionTreeRegressorRegressor);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ConstantTreeRegressorRegressor);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DTRRB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, CTRRB);

// Decorator polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DR3);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DR4);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DR5);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ContinuousRegressorBaseDTRRBD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ContinuousRegressorBaseCTRRBD);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, RegressorBaseDTRRD)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, RegressorBaseCTRRD)

CEREAL_REGISTER_POLYMORPHIC_RELATION(RegressorBaseDTRRD, DecisionTreeRegressorRegressor);
CEREAL_REGISTER_POLYMORPHIC_RELATION(RegressorBaseDTRRD, ConstantTreeRegressorRegressor);

#endif
