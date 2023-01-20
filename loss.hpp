#ifndef __LOSS_HPP__
#define __LOSS_HPP__

#include <iostream>
#include <functional>
#include <exception>
#include <mlpack/core.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

using namespace arma;
using namespace autodiff;
using namespace std::placeholders;

#define UNUSED(expr) do { (void)(expr); } while (0)


enum class lossFunction {    MSE = 0,
			     BinomialDeviance = 1,
			     Savage = 2,
			};


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

namespace LossMeasures { 

  struct lossFunctionException : public std::exception {
    const char* what() const throw() {
      return "General LossFunction exception";
    };
  };

template<typename DataType>
class LossFunction {
public:
  DataType loss(const rowvec&, const rowvec&, rowvec*, rowvec*);
  DataType loss(const rowvec& yhat, const rowvec& y) { return loss_reverse_arma(yhat, y); }
  virtual LossFunction* create() = 0;
private:
  virtual autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) = 0;
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
  autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
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
  autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
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
  autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
  DataType loss_reverse_arma(const rowvec&, const rowvec&) override;
  DataType gradient_(const rowvec&, const rowvec&, rowvec*) override;
  void hessian_(const rowvec&, const rowvec&, rowvec*) override;
};

} // namespace LossMeasures

#include "loss_impl.hpp"

#endif
