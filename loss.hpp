#ifndef __LOSS_HPP__
#define __LOSS_HPP__

#undef AUTODIFF

#include <iostream>
#include <functional>
#include <cstring>
#include <exception>
#include <mlpack/core.hpp>
#ifdef AUTODIFF
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;
#endif

using namespace arma;
using namespace std::placeholders;

#define UNUSED(expr) do { (void)(expr); } while (0)


enum class lossFunction {    MSE = 0,
			       BinomialDeviance = 1,
			       Savage = 2,
			       Exp = 3,
			       Arctan = 4,
			       Synthetic = 5,
			       SyntheticVar1 = 6,
			       SyntheticVar2 = 7
			       };


#ifdef AUTODIFF
class LossUtils {
public:
  static Eigen::VectorXd static_cast_eigen(const rowvec& rhs) {
    Eigen::VectorXd lhs = Eigen::Map<Eigen::VectorXd>(const_cast<double*>(rhs.memptr()), rhs.n_cols);
    return lhs;
  }
  
  static rowvec static_cast_arma(const Eigen::VectorXd& rhs) {
    rowvec lhs = rowvec(const_cast<double*>(rhs.data()), rhs.size(), false, false);
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
    DataType loss(const rowvec&, const rowvec&, rowvec*, rowvec*);
    DataType loss(const rowvec& yhat, const rowvec& y) { return loss_reverse_arma(yhat, y); }
    virtual LossFunction* create() = 0;
  private:
#ifdef AUTODIFF
    virtual autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) = 0;
#endif
    virtual DataType loss_reverse_arma(const rowvec&, const rowvec&) = 0;
    virtual DataType gradient_(const rowvec&, const rowvec&, rowvec*) = 0;
    virtual void hessian_(const rowvec&, const rowvec&, rowvec*) = 0;
  };

  template<typename DataType>
  class BinomialDevianceLoss : public LossFunction<DataType> {
  public:
    BinomialDevianceLoss() = default;
    BinomialDevianceLoss<DataType>* create() { return new BinomialDevianceLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
#endif
    DataType loss_reverse_arma(const rowvec&, const rowvec&) override;
    DataType gradient_(const rowvec&, const rowvec&, rowvec*) override;
    void hessian_(const rowvec&, const rowvec&, rowvec*) override;
  };

  template<typename DataType>
  class MSELoss : public LossFunction<DataType> {
  public:
    MSELoss() = default;
    MSELoss<DataType>* create() { return new MSELoss(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const rowvec&, const rowvec&) override;
    DataType gradient_(const rowvec&, const rowvec&, rowvec*) override;
    void hessian_(const rowvec&, const rowvec&, rowvec*) override;
  };

  template<typename DataType>
  class ExpLoss : public LossFunction<DataType> {
  public:
    ExpLoss() = default;
    ExpLoss<DataType>* create() { return new ExpLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const rowvec&, const rowvec&) override;
    DataType gradient_(const rowvec&, const rowvec&, rowvec*) override;
    void hessian_(const rowvec&, const rowvec&, rowvec*) override;
  };

  template<typename DataType>
  class SavageLoss : public LossFunction<DataType> {
  public:
    SavageLoss() = default;
    SavageLoss<DataType>* create() { return new SavageLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
#endif
    DataType loss_reverse_arma(const rowvec&, const rowvec&) override;
    DataType gradient_(const rowvec&, const rowvec&, rowvec*) override;
    void hessian_(const rowvec&, const rowvec&, rowvec*) override;
  };

  template<typename DataType>
  class ArctanLoss : public LossFunction<DataType> {
  public:
    ArctanLoss() = default;
    ArctanLoss<DataType>* create() { return new ArctanLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
#endif
    DataType loss_reverse_arma(const rowvec&, const rowvec&) override;
    DataType gradient_(const rowvec&, const rowvec&, rowvec*) override;
    void hessian_(const rowvec&, const rowvec&, rowvec*) override;
  };

  template<typename DataType>
  class SyntheticLoss : public LossFunction<DataType> {
  public:
    SyntheticLoss() = default;
    SyntheticLoss<DataType>* create() { return new SyntheticLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
#endif
    DataType loss_reverse_arma(const rowvec&, const rowvec&) override;
    DataType gradient_(const rowvec&, const rowvec&, rowvec*) override;
    void hessian_(const rowvec&, const rowvec&, rowvec*) override;
  };

  template<typename DataType>
  class SyntheticLossVar1 : public LossFunction<DataType> {
  public:
    SyntheticLossVar1() = default;
    SyntheticLossVar1<DataType>* create() { return new SyntheticLossVar1<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const rowvec&, const rowvec&) override;
    DataType gradient_(const rowvec&, const rowvec&, rowvec*) override;
    void hessian_(const rowvec&, const rowvec&, rowvec*) override;  
  };

  template<typename DataType>
  class SyntheticLossVar2 : public LossFunction<DataType> {
  public:
    SyntheticLossVar2() = default;
    SyntheticLossVar2<DataType>* create() { return new SyntheticLossVar2<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const rowvec&, const rowvec&) override;
    DataType gradient_(const rowvec&, const rowvec&, rowvec*) override;
    void hessian_(const rowvec&, const rowvec&, rowvec*) override;
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
      {lossFunction::SyntheticVar2,	new SyntheticLossVar2<T>() }
    };

} // namespace LossMeasures


#include "loss_impl.hpp"

#endif
