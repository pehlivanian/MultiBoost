#ifndef __LOSS_IMPL_HPP__
#define __LOSS_IMPL_HPP__

using namespace LossMeasures;

template<typename DataType>
DataType
BinomialDevianceLoss<DataType>::loss(const rowvec& yhat, const rowvec& y, rowvec* grad) {
  autodiff::real u;
  ArrayXreal yhatr = static_cast_eigen(yhat).eval();
  ArrayXreal yr = static_cast_eigen(y).eval();

  std::function<autodiff::real(const ArrayXreal&)> loss_ = std::bind(&BinomialDevianceLoss<DataType>::loss_reverse, this, yr, _1);
  Eigen::VectorXd grad_tmp = gradient(loss_, wrt(yhatr), at(yhatr), u);
  
  *grad = static_cast_arma(grad_tmp);

  return static_cast<DataType>(u.val());
}

template<typename DataType>
DataType
MSELoss<DataType>::loss(const rowvec& yhat, const rowvec& y, rowvec* grad) {
// LossFunction<DataType>::loss(Eigen::VectorXd& yhat, Eigen::VectorXd& y, rowvec* grad) {
  autodiff::real u;
  ArrayXreal yhatr = static_cast_eigen(yhat).eval();
  ArrayXreal yr = static_cast_eigen(y).eval();

  std::function<autodiff::real(const ArrayXreal&)> loss_ = std::bind(&MSELoss<DataType>::loss_reverse, this, yr, _1);
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
