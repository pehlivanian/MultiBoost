#ifndef __LOSS2_HPP__
#define __LOSS2_HPP__

#undef AUTODIFF

#include <iostream>
#include <functional>
#include <cstring>
#include <exception>
#include <cmath>
#include <mlpack/core.hpp>

#undef AUTODIFF

#ifdef AUTODIFF
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;
#endif

#include "analytic_utils.hpp"

using namespace arma;
using namespace std::placeholders;
using namespace ANALYTIC_utils;


#ifdef AUTODIFF
template<typename DataType>
class LossUtils {
public:
  static std::enable_if<_is_double<DataType>::value, Eigen::VectorXd> static_cast_eigen(const arma_dvec_type& rhs) {
    Eigen::VectorXd lhs = Eigen::Map<Eigen::VectorXd>(const_cast<double*>(rhs.memptr()), rhs.n_cols);
    return lhs;
  }
  
  static std::enable_if<!_is_double<DataType>::value, Eigen::VectorXf> static_cast_eigen(const arma_fvec_type& rhs) {
    Eigen::VectorXf lhs = Eigen::Map<Eigen::VectorXf>(const_cast<float*>(rhs.memptr()), rhs.n_cols);
  }>
  
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

} // namespace LossMeasures

#include "loss2_impl.hpp"

#endif
