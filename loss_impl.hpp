#ifndef __LOSS_IMPL_HPP__
#define __LOSS_IMPL_HPP__

using namespace LossMeasures;

template<typename DataType>
DataType
LossFunction<DataType>::loss(rowvec& yhat, rowvec* grad) {
  autodiff::real u;
  ArrayXreal yhatr = static_cast_eigen(yhat).eval();
  
  std::function<autodiff::real(const ArrayXreal&)> loss_ = std::bind(&LossFunction<DataType>::loss_reverse, yhatr, _1);
  Eigen::VectorXd grad_tmp = gradient(loss_, wrt(yhatr), at(yhatr), u);
  *grad = static_cast_arma(grad_tmp);

  return static_cast<DataType>(u.val());
  
}

template<typename DataType>
autodiff::real
BinomialDevianceLoss<DataType>::loss_reverse(const ArrayXreal& y, const ArrayXreal& yhat) {
  return ((1 + (-y * yhat).exp()).log()).sum();
}

template<typename DataType>
autodiff::real
MSELoss<DataType>::loss_reverse(const ArrayXreal& y, const ArrayXreal& yhat) {
  return sqrt((y - yhat).sum());
}

#endif
