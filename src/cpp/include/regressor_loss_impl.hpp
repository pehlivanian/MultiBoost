#ifndef __REGRESSOR_LOSS_IMPL_HPP__
#define __REGRESSOR_LOSS_IMPL_HPP__

using namespace RegressorLossMeasures;

////////////////
// BEGIN MSELoss
////////////////

template <typename DataType>
DataType MSELoss<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  *grad = -2 * (y - yhat);

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void MSELoss<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  UNUSED(yhat);
  vtype f(y.n_cols, arma::fill::ones);
  *hess = 2 * f;
}

template <typename DataType>
DataType MSELoss<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType MSELoss<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return sum(pow(y - yhat, 2));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real MSELoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

////////////////
// END MSELoss
////////////////

/////////////////////////
// BEGIN SyntheticRegLoss
/////////////////////////

template <typename DataType>
DataType SyntheticRegLoss<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  vtype f = -2. * sign(y - yhat) % pow(abs(y - yhat), .3333);
  f.transform([](DataType val) { return (std::isnan(val) ? 0. : val); });
  *grad = f;

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void SyntheticRegLoss<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  UNUSED(yhat);

  vtype f(y.n_cols, arma::fill::ones);
  *hess = 2 * f;
}

template <typename DataType>
DataType SyntheticRegLoss<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType SyntheticRegLoss<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return sum(pow(y - yhat, 2));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real SyntheticRegLoss<DataType>::loss_reverse(
    const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

///////////////////////
// END SyntheticRegLoss
///////////////////////

////////////////
// BEGIN LogLoss
////////////////
template <typename DataType>
DataType LogLoss<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  // The usual log loss function assumes that $y \in \left\lbrace 0, 1\right\rbrace$
  // We maintain the consistency by assuming $y \in \left\brace -1, 1\right\rbrace$
  // here, and do a pre-transformation

  vtype yhat_norm = 0.5 * yhat + 0.5;
  vtype y_norm = 0.5 * y + 0.5;
  *grad = -1 * (y_norm - yhat_norm);

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void LogLoss<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  // The usual log loss function assumes that $y \in \left\lbrace 0, 1\right\rbrace$
  // We maintain the consistency by assuming $y \in \left\brace -1, 1\right\rbrace$
  // here, and do a pre-transformation

  vtype yhat_norm = 0.5 * yhat + 0.5 + y + y;
  *hess = yhat_norm % (1 - yhat_norm);
}

template <typename DataType>
DataType LogLoss<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;

  /*
    vtype yhat_norm = 0.5 * yhat + 0.5;
    vtype y_norm = 0.5 * y + 0.5;

    vtype log_odds = log(yhat_norm / (1 - yhat_norm));
    return -1 * sum(y_norm % log_odds - log(1 + exp(log_odds)));
  */
}

template <typename DataType>
DataType LogLoss<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  vtype yhat_norm = 0.5 * yhat + 0.5;
  vtype y_norm = 0.5 * y + 0.5;

  vtype log_odds = log(yhat_norm / (1 - yhat_norm));
  return -1 * sum(y_norm % log_odds - log(1 + exp(log_odds)));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real LogLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow(1 - y * yhat, 2).sum();
}
#endif

////////////////
// END LogLoss
////////////////

////////////////////
// BEGIN RegressorPowerLoss
////////////////////

template <typename DataType>
DataType RegressorPowerLoss<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  // Proper decomposition of \Phi \left( y,\hat{y}\right) = y\left( 1-y\hat{y}\right)^p
  if (p_ != 1) {
    vtype f = exp(1.0 / ((-p_ + 1) * pow(y, 2)) % pow(1 - y % yhat, -p_ + 1));
    vtype g = exp(-1.0 / ((-p_ + 1) * pow(y, 2)));
    *grad = -y % f % g;
  } else {
    vtype f = exp(log(1 - y % yhat) / pow(y, 2));
    *grad = -y % f;
  }

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void RegressorPowerLoss<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  if (p_ != 1) {
    vtype f = exp(1.0 / ((-p_ + 1) * pow(y, 2)) % pow(1 - y % yhat, -p_ + 1));
    vtype g = exp(-1.0 / ((-p_ + 1) * pow(y, 2)));
    *hess = pow(1 - y % yhat, -p_) % f % g;
  } else {
    vtype f = exp(log(1 - y % yhat) / pow(y, 2));
    *hess = (1.0 / (1 - y % yhat)) % f;
  }
}

template <typename DataType>
DataType RegressorPowerLoss<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType RegressorPowerLoss<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return sum(pow(y - yhat, 2));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real RegressorPowerLoss<DataType>::loss_reverse(
    const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

////////////////////
// END RegressorPowerLoss
////////////////////

#endif
