#ifndef __LOSS_IMPL_HPP__
#define __LOSS_IMPL_HPP__

using namespace LossMeasures;

const bool AUTODIFF_ON = false;

template<typename DataType>
DataType
LossFunction<DataType>::loss(const rowvec& yhat, const rowvec& y, rowvec* grad, rowvec* hess) {
#ifdef AUTODIFFBAD
  {
    // Slooooooowwww...
    // Since we only need gradient, hessian of quadratic approximation to loss and not
    // some deep neural network composition, we can usually compute and 
    // hard-code these for a given loss function, the approach we take here.
    autodiff::real u;
    ArrayXreal yhatr = LossUtils::static_cast_eigen(yhat).eval();
    ArrayXreal yr = LossUtils::static_cast_eigen(y).eval();
    
    std::function<autodiff::real(const ArrayXreal&)> loss_ = std::bind(&LossFunction<DataType>::loss_reverse, this, yr, _1);
    Eigen::VectorXd grad_tmp = gradient(loss_, wrt(yhatr), at(yhatr), u);
    *grad = LossUtils::static_cast_arma(grad_tmp);
    return static_cast<DataType>(u.val());
  }
#else
  {
    DataType r = gradient_(yhat, y, grad);
    hessian_(yhat, y, hess);
    return r;
  }
#endif
}

////////////////
// BEGIN MSELoss
////////////////

template<typename DataType>
DataType
MSELoss<DataType>::gradient_(const rowvec& yhat, const rowvec& y, rowvec* grad) {
  *grad = -2 * (y - yhat);

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template<typename DataType>
void
MSELoss<DataType>::hessian_(const rowvec& yhat, const rowvec& y, rowvec* hess) {
  rowvec f(y.n_cols, arma::fill::ones);
  *hess = 2 * f;
}

template<typename DataType>
DataType
MSELoss<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  return sum(pow((y - yhat), 2));
}

#ifdef AUTODIFF
template<typename DataType>
autodiff::real
MSELoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

////////////////
// END MSELoss
////////////////


////////////////
// BEGIN BinomialDevianceLoss
////////////////
template<typename DataType>
DataType
BinomialDevianceLoss<DataType>::gradient_(const rowvec& yhat, const rowvec& y, rowvec* grad) {
  rowvec f = exp(-y % yhat);
  *grad = (-y % f)/(1 + f);
  
#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template<typename DataType>
void
BinomialDevianceLoss<DataType>::hessian_(const rowvec& yhat, const rowvec& y, rowvec* hess) {
  rowvec a = y % yhat;
  rowvec b = exp(a);
  rowvec c = pow(y, 2) % b;
  rowvec d = 1 + b;
  rowvec e = pow(1 + b, 2);
  rowvec f = exp(y % yhat);
  rowvec ee = (pow(y, 2) % f)/pow(1 + f, 2);
  *hess = (pow(y, 2) % f)/pow(1 + f, 2);
}

template<typename DataType>
DataType
BinomialDevianceLoss<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  return sum(log(1 + exp(-y % yhat)));
}

#ifdef AUTODIFF
template<typename DataType>
autodiff::real
BinomialDevianceLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return ((1 + (-y * yhat).exp()).log()).sum();
}
#endif

////////////////
// END BinomialDevianceLoss
////////////////


////////////////
// BEGIN SavageLoss
////////////////

template<typename DataType>
DataType
SavageLoss<DataType>::gradient_(const rowvec& yhat, const rowvec& y, rowvec* grad) {
  rowvec f = exp(y % yhat);
  rowvec g = exp(2 * y % yhat);
  rowvec h = pow(f, 2);
  rowvec j = 1 / f;
  rowvec k = 2 * pow(f, 2);
  rowvec l = (2 * pow(y, 2) % g % (2 - f));
  rowvec m = pow(1 + f, 4);
  rowvec n = l/m;

  *grad = (2 * y % f) / pow(1 + f, 3);

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template<typename DataType>
void
SavageLoss<DataType>::hessian_(const rowvec& yhat, const rowvec& y, rowvec* hess) {
  rowvec f = exp(y % yhat);
  rowvec g = exp(2 * y % yhat);
  rowvec h = pow(f, 2);
  rowvec j = 1 / f;
  rowvec k = 2 * pow(f, 2);
  rowvec l = (2 * pow(y, 2) % g % (2 - f));
  rowvec m = pow(1 + f, 4);
  rowvec n = l/m;

  *hess = (2 * pow(y, 2) % f % (2*f - 1)) / pow(1 + f, 4);
}

template<typename DataType>
DataType
SavageLoss<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  return sum(1 / pow(1 + exp(y % yhat), 2));
}


#ifdef AUTODIFF
template<typename DataType>
autodiff::real
SavageLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return (1 / pow(1 + (-y * yhat).exp(), 2)).sum();
}
#endif

////////////////
// END SavageLoss
////////////////



#endif
