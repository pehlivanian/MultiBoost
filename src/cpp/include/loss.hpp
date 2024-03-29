#ifndef __LOSS_HPP__
#define __LOSS_HPP__

#undef AUTODIFF

#include <iostream>
#include <functional>
#include <cstring>
#include <exception>
#include <cmath>
#include <mlpack/core.hpp>

// I orginally coded this to use automatic differentiation to compute
// gradients, hessian of the loss function, but it is much faster to 
// precompute and use closed-forms. In general this should always be
// done in a gradient boosting setting - the gradient descent is on
// a simple loss function with no cross-terms, not some deep multi-layer
// network for which a closed-form is hopeless.
// To enable auto differentiation and slow everything down define AUTODIFF
// accordingly.

#ifdef AUTODIFF
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;
#endif

#include "analytic_utils.hpp"

using namespace arma;
using namespace std::placeholders;
using namespace ANALYTIC_utils;

#define UNUSED(expr) do { (void)(expr); } while (0)

enum class lossFunction {      MSE = 0,
			       BinomialDeviance = 1,
			       Savage = 2,
			       Exp = 3,
			       Arctan = 4,
			       Synthetic = 5,
			       SyntheticVar1 = 6,
			       SyntheticVar2 = 7,
			       SquareLoss = 8,
			       SyntheticRegLoss = 9,
			       LogLoss = 10,
			       SyntheticVar3 = 11,
			       PowerLoss = 12,
			       CrossEntropyLoss = 13
			       };


template<typename DataType>
typename arma_types<DataType>::vectype sigmoid(const typename arma_types<DataType>::vectype& v) {
  return 1.0 / ( 1 + exp(-v));
}

template<typename DataType>
typename arma_types<DataType>::vectype normalize(const typename arma_types<DataType>::vectype& v) {
  return 0.5 * ( v + 1);
}

#ifdef AUTODIFF
template<typename DataType>
class LossUtils {
public:
  static std::enable_if<_is_double<DataType>::value, Eigen::VectorXd> static_cast_eigen(const arma_dvec_type& rhs) {
    Eigen::VectorXd lhs = Eigen::Map<Eigen::VectorXd>(const_cast<double*>(rhs.memptr()), rhs.n_cols);
    return lhs;
  }
  
  static std::enable_if<!_is_double<DataType>::value, Eigen::VectorXf> static_cast_eigen(const arma_fvec_type& rhs) {
    Eigen::Vectorfd lhs = Eigen::Map<Eigen::VectorXf>(const_cast<float*>(rhs.memptr()), rhs.n_cols);
  }
  
  static std::enable_if<_is_double<DataType>::value, arma_dvec_type> static_cast_arma(const Eigen::VectorXd& rhs) {
    arma_dvec_type lhs = arma_dvec_type{const_cast<double*>(rhs.data()), rhs.size(), false, false};
    return lhs;
  }

  static std::enable_if<!_is_double<DataType>::value, arma_fvec_type> static_cast_arma(const Eigen::VectorXf& rhs) {
    arma_fvec_type lhs = arma_fvec_type{const_cast<float*>(rhs.data()), rhs.size(), false, false};
    return lhs;
  }

};
#endif

namespace LossMeasures { 

  struct lossFunctionException : public std::exception {
    lossFunctionException(char *s) : s_{s} {
      msg_ = (char*)malloc(strlen(pref_)+strlen(s_)+1);
      memcpy(msg_, pref_, strlen(pref_));
      memcpy(msg_+strlen(pref_), s_, strlen(s_));
    }
    const char* what() const throw() {
      return msg_;
    };

    char *s_;
    char *msg_;
    const char* pref_ = "General LossFunction exception: ";
  };

  template<typename DataType>
  class LossFunction {
 
  public:
    using vtype = typename arma_types<DataType>::vectype;
    using mtype = typename arma_types<DataType>::mattype;

    DataType loss(const vtype&, const vtype&, vtype*, vtype*, bool=false, DataType=DataType{}, DataType=DataType{});
    DataType loss(const vtype& yhat, const vtype& y) { return loss_arma(yhat, y); }
    virtual LossFunction* create() = 0;
  private:
#ifdef AUTODIFF
    virtual autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) = 0;
#endif
    virtual DataType loss_reverse_arma(const vtype&, const vtype&) = 0;
    virtual DataType loss_arma(const vtype&, const vtype&) = 0;
    virtual DataType gradient_(const vtype&, const vtype&, vtype*) = 0;
    virtual void hessian_(const vtype&, const vtype&, vtype*) = 0;
  };

  template<typename DataType>
  class BinomialDevianceLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    BinomialDevianceLoss() = default;
    BinomialDevianceLoss<DataType>* create() override { return new BinomialDevianceLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class MSELoss : public LossFunction<DataType> {
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

  template<typename DataType>
  class ExpLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    ExpLoss() = default;
    ExpLoss<DataType>* create() override { return new ExpLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class SavageLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    SavageLoss() = default;
    SavageLoss<DataType>* create() override { return new SavageLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class ArctanLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    ArctanLoss() = default;
    ArctanLoss<DataType>* create() override { return new ArctanLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class SyntheticLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    SyntheticLoss() = default;
    SyntheticLoss<DataType>* create() override { return new SyntheticLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class SyntheticRegLoss : public LossFunction<DataType> {
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


  template<typename DataType>
  class SyntheticLossVar1 : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    SyntheticLossVar1() = default;
    SyntheticLossVar1<DataType>* create() override { return new SyntheticLossVar1<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;  
  };

  template<typename DataType>
  class SyntheticLossVar2 : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    SyntheticLossVar2() = default;
    SyntheticLossVar2<DataType>* create() override { return new SyntheticLossVar2<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class SyntheticLossVar3 : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    SyntheticLossVar3() = default;
    SyntheticLossVar3<DataType>* create() override { return new SyntheticLossVar3<DataType>(); }
  private:
#ifdef AUTODIFF
    audodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };
  
  template<typename DataType>
  class SquareLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    SquareLoss() = default;
    SquareLoss<DataType>* create() override { return new SquareLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class LogLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    LogLoss() = default;
    LogLoss<DataType>* create() override { return new LogLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&,, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class PowerLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    PowerLoss(float p) : p_{p} {}
    PowerLoss<DataType>* create() override { return new PowerLoss<DataType>{0.}; }
    PowerLoss<DataType>* create(float p) { return new PowerLoss<DataType>{p}; }
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

  template<typename DataType>
  class CrossEntropyLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;
    
  public:
    CrossEntropyLoss() = default;
    CrossEntropyLoss<DataType>* create() override { return new CrossEntropyLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };
  
  struct lossMapHash {
    std::size_t operator()(lossFunction l) const { return static_cast<std::size_t>(l); }
  };
  
  template<typename T>
  std::unordered_map<lossFunction, LossFunction<T>*, lossMapHash> lossMap = 
    {
      {lossFunction::MSE,		new MSELoss<T>() },
      {lossFunction::BinomialDeviance,	new BinomialDevianceLoss<T>() },
      {lossFunction::Savage,		new SavageLoss<T>() },
      {lossFunction::Exp,		new ExpLoss<T>() },
      {lossFunction::Arctan,		new ArctanLoss<T>() },
      {lossFunction::Synthetic,		new SyntheticLoss<T>() },
      {lossFunction::SyntheticVar1,	new SyntheticLossVar1<T>() },
      {lossFunction::SyntheticVar2,	new SyntheticLossVar2<T>() },
      {lossFunction::SquareLoss,	new SquareLoss<T>() },
      {lossFunction::SyntheticRegLoss,	new SyntheticRegLoss<T>() },
      {lossFunction::LogLoss,		new LogLoss<T>() },
      {lossFunction::SyntheticVar3,	new SyntheticLossVar3<T>() },
      {lossFunction::CrossEntropyLoss,  new CrossEntropyLoss<T>() }
    };

  template<typename T>
  LossFunction<T>* createLoss(lossFunction loss, float p) {
    if (loss == lossFunction::PowerLoss) {
      return new PowerLoss<T>{p};
    } else {
      return lossMap<T>[loss];
    }
  }

} // namespace LossMeasures


#include "loss_impl.hpp"

#endif
