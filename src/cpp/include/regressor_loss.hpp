#ifndef __REGRESSOR_LOSS_HPP__
#define __REGRESSOR_LOSS_HPP__

#include <unordered_map>

#include "loss2.hpp"

#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)

enum class regressorLossFunction {

  MSE = 0,
  SyntheticRegLoss = 1,
  LogLoss = 2,
  RegressorPowerLoss = 3,

};

namespace RegressorLossMeasures {

template <typename DataType>
class RegressorLossFunction : public LossFunction<DataType> {};

template <typename DataType>
class MSELoss : public RegressorLossFunction<DataType> {
private:
  using vtype = typename LossFunction<DataType>::vtype;
  using mtype = typename LossFunction<DataType>::mtype;

public:
  MSELoss() = default;
  MSELoss<DataType>* create() override { return new MSELoss(); }

private:
#ifdef AUTODIFF
  autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
  DataType loss_reverse_arma(const vtype&, const vtype&) override;
  DataType loss_arma(const vtype&, const vtype&) override;
  DataType gradient_(const vtype&, const vtype&, vtype*) override;
  void hessian_(const vtype&, const vtype&, vtype*) override;
};

template <typename DataType>
class SyntheticRegLoss : public RegressorLossFunction<DataType> {
private:
  using vtype = typename LossFunction<DataType>::vtype;
  using mtype = typename LossFunction<DataType>::mtype;

public:
  SyntheticRegLoss() = default;
  SyntheticRegLoss<DataType>* create() override { return new SyntheticRegLoss<DataType>(); }

private:
#ifdef AUTODIFF
  autodiff::real loss_Reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
  DataType loss_reverse_arma(const vtype&, const vtype&) override;
  DataType loss_arma(const vtype&, const vtype&) override;
  DataType gradient_(const vtype&, const vtype&, vtype*) override;
  void hessian_(const vtype&, const vtype&, vtype*) override;
};

template <typename DataType>
class LogLoss : public RegressorLossFunction<DataType> {
private:
  using vtype = typename LossFunction<DataType>::vtype;
  using mtype = typename LossFunction<DataType>::mtype;

public:
  LogLoss() = default;
  LogLoss<DataType>* create() override { return new LogLoss<DataType>(); }

private:
#ifdef AUTODIFF
  autodiff::real loss_reverse(const ArrayXreal&, , const ArrayXreal&) override;
#endif
  DataType loss_reverse_arma(const vtype&, const vtype&) override;
  DataType loss_arma(const vtype&, const vtype&) override;
  DataType gradient_(const vtype&, const vtype&, vtype*) override;
  void hessian_(const vtype&, const vtype&, vtype*) override;
};

template <typename DataType>
class RegressorPowerLoss : public RegressorLossFunction<DataType> {
private:
  using vtype = typename LossFunction<DataType>::vtype;
  using mtype = typename LossFunction<DataType>::mtype;

public:
  RegressorPowerLoss(float p) : p_{p} {}
  RegressorPowerLoss<DataType>* create() override { return new RegressorPowerLoss<DataType>{3.}; }
  RegressorPowerLoss<DataType>* create(float p) { return new RegressorPowerLoss<DataType>{p}; }

private:
#ifdef AUTODIFF
  autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
  DataType loss_reverse_arma(const vtype&, const vtype&) override;
  DataType loss_arma(const vtype&, const vtype&) override;
  DataType gradient_(const vtype&, const vtype&, vtype*) override;
  void hessian_(const vtype&, const vtype&, vtype*) override;

  float p_;
};

struct regressorLossHashMap {
  std::size_t operator()(regressorLossFunction l) const { return static_cast<std::size_t>(l); }
};

template <typename T>
std::unordered_map<regressorLossFunction, RegressorLossFunction<T>*> regressorLossMap = {
    {regressorLossFunction::MSE, new MSELoss<T>()},
    {regressorLossFunction::SyntheticRegLoss, new SyntheticRegLoss<T>()},
    {regressorLossFunction::LogLoss, new LogLoss<T>()}};

template <typename T>
RegressorLossFunction<T>* createLoss(regressorLossFunction loss, float p) {
  if (loss == regressorLossFunction::RegressorPowerLoss) {
    return new RegressorPowerLoss<T>{p};
  } else {
    return regressorLossMap<T>[loss];
  }
}

}  // namespace RegressorLossMeasures

#include "regressor_loss_impl.hpp"

#endif
