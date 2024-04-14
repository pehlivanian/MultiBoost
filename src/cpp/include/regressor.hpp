#ifndef __REGRESSOR_HPP__
#define __REGRESSOR_HPP__

#include <memory>

#include <mlpack/core.hpp>

#include "utils.hpp"
#include "model.hpp"

using namespace arma;
using namespace TupleUtils;

template<typename DataType, typename RegressorType>
class RegressorBase : public Model<DataType> {
public:
  RegressorBase() = default;
  RegressorBase(const RegressorBase&) = default;
  RegressorBase(std::string id) : Model<DataType>(id) {}

  virtual ~RegressorBase() = default;

  virtual void Predict(const Mat<DataType>& data, Row<DataType>& pred) { Predict_(data, pred); }
  virtual void Predict(Mat<DataType>&& data, Row<DataType>& pred) { Predict_(data, pred); }

private:
  virtual void Predict_(const Mat<DataType>&, Row<DataType>&) = 0;
  virtual void Predict_(Mat<DataType>&&, Row<DataType>&) = 0;

  void Project_(const Mat<DataType>& data, Row<DataType>& pred) override { Predict_(data, pred); }
  void Project_(Mat<DataType>&& data, Row<DataType>& pred) override { Predict_(std::move(data), pred); }
};

template<typename DataType, typename RegressorType, typename... Args>
class ContinuousRegressorBase : public RegressorBase<DataType, RegressorType> {
public:  
  ContinuousRegressorBase(const Mat<DataType>& dataset, Row<DataType>& labels, Args&&... args) : 
    RegressorBase<DataType, RegressorType>(typeid(*this).name()) 
  {
    init_(dataset, labels, false, std::forward<Args>(args)...);
  }


  ContinuousRegressorBase(const Mat<DataType>& dataset, Row<DataType>& labels, Row<DataType>& weights, Args&&... args) :
    RegressorBase<DataType, RegressorType>(typeid(*this).name())
  {
    weights_ = weights;
    // XXX
    init_(dataset, labels, false, std::forward<Args>(args)...);
  }

  ContinuousRegressorBase(std::unique_ptr<RegressorType> regressor) : regressor_{std::move(regressor)} {}
  ContinuousRegressorBase() = default;

  virtual ~ContinuousRegressorBase() = default;
  
  void setRegressor(const Mat<DataType>&, Row<DataType>&, Args&&...);

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::base_class<RegressorBase<DataType, RegressorType>>(this), CEREAL_NVP(regressor_));
    // Don't serialize args_
    // ar(cereal::base_class<RegressorBase<DataType, RegressorType>>(this), CEREAL_NVP(args_));
  }

private:
  void init_(const Mat<DataType>&, Row<DataType>&, bool, Args&&...);

  Row<DataType> labels_;
  Row<DataType> weights_;
  std::unique_ptr<RegressorType> regressor_;
  std::tuple<Args...> args_;

  void Predict_(const Mat<DataType>&, Row<DataType>&) override;
  void Predict_(Mat<DataType>&&, Row<DataType>&) override;

  void purge_() override {};

};

#include "regressor_impl.hpp"

#endif
