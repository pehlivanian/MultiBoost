#ifndef __REGRESSOR_HPP__
#define __REGRESSOR_HPP__

#include <memory>

#include <mlpack/core.hpp>

#include "model.hpp"

using namespace arma;

template<typename DataType, typename RegressorType>
class RegressorBase : public Model<DataType, RegressorType> {
public:
  RegressorBase() = default;
  RegressorBase(std::string id) : Model<DataType, RegressorType>(id) {}

  virtual ~RegressorBase() = default;

  virtual void Predict(const mat& data, Row<DataType>& pred) { Predict_(data, pred); }
  virtual void Predict(mat&& data, Row<DataType>& pred) { Predict_(data, pred); }

  virtual void purge() { purge_(); }

private:
  virtual void purge_() = 0;
  virtual void Predict_(const mat&, Row<DataType>&) = 0;
  virtual void Predict_(mat&&, Row<DataType>&) = 0;

  void Project_(const mat& data, Row<DataType>& pred) override { Predict_(data, pred); }
  void Project_(mat&& data, Row<DataType>& pred) override { Predict_(std::move(data), pred); }
};

template<typename DataType, typename RegressorType, typename... Args>
class ContinuousRegressorBase : public RegressorBase<DataType, RegressorType> {
public:  
  ContinuousRegressorBase(const mat& dataset, Row<DataType>& labels, Args&&... args) : 
    RegressorBase<DataType, RegressorType>(typeid(*this).name()) 
  {

    setRegressor(dataset, labels, std::forward<Args>(args)...);
    args_ = std::tuple<Args...>(args...);
  }

  ContinuousRegressorBase(std::unique_ptr<RegressorType> regressor) : regressor_{std::move(regressor)} {}
  ContinuousRegressorBase() = default;

  virtual ~ContinuousRegressorBase() = default;
  
  void setRegressor(const mat&, Row<DataType>&, Args&&...);

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::base_class<RegressorBase<DataType, RegressorType>>(this), CEREAL_NVP(regressor_));
    // Don't serialize args_
    ar(cereal::base_class<RegressorBase<DataType, RegressorType>>(this), CEREAL_NVP(args_));
  }

private:
  std::unique_ptr<RegressorType> regressor_;
  std::tuple<Args...> args_;

  void Predict_(const mat&, Row<DataType>&) override;
  void Predict_(mat&& mat, Row<DataType>&) override;

  void purge_() override {};

};

#include "regressor_impl.hpp"

#endif
