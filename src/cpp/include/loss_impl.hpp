#ifndef __LOSS_IMPL_HPP__
#define __LOSS_IMPL_HPP__

using namespace LossMeasures;

const bool AUTODIFF_ON = false;

template <typename DataType>
DataType LossFunction<DataType>::loss(
    const vtype& yhat,
    const vtype& y,
    vtype* grad,
    vtype* hess,
    bool clamp_gradient,
    DataType upper_val,
    DataType lower_val) {
#ifdef AUTODIFFBAD
  {
    // Slooooooowwww...
    // Since we only need gradient, hessian of quadratic approximation to loss and not
    // some deep neural network composition, we can usually compute and
    // hard-code these for a given loss function, the approach we take here.
    autodiff::real u;
    ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
    ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

    std::function<autodiff::real(const ArrayXreal&)> loss_ =
        std::bind(&LossFunction<DataType>::loss_reverse, this, yr, _1);
    Eigen::VectorXd grad_tmp = gradient(loss_, wrt(yhatr), at(yhatr), u);
    *grad = LossUtils::static_cast_arma<DataType>(gradtmp);
    return static_cast<DataType>(u.val());
  }
#else
  {
    DataType r = gradient_(yhat, y, grad);

    if (clamp_gradient) {
      // Clamp gradient, leave hessian as is
      for (std::size_t i = 0; i < grad->n_elem; ++i) {
        if ((y[i] > 0 && yhat[i] > upper_val) || (y[i] < 0 && yhat[i] < lower_val)) {
          (*grad)[i] = 0.;
        }
      }
    }

    hessian_(yhat, y, hess);
    return r;
  }
#endif
}

///////////////////
// BEGIN SquareLoss
///////////////////

template <typename DataType>
DataType SquareLoss<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  *grad = 2 * y % (y % yhat - 1);
#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void SquareLoss<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  UNUSED(yhat);
  *hess = 2 * pow(y, 2);
}

template <typename DataType>
DataType SquareLoss<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType SquareLoss<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return sum(pow(1 - y % yhat, 2));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real SquareLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow(1 - y * yhat, 2).sum();
}
#endif

/////////////////
// END SquareLoss
/////////////////

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

////////////////
// BEGIN BinomialDevianceLoss
////////////////
template <typename DataType>
DataType BinomialDevianceLoss<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  vtype f = exp(y % yhat);
  *grad = -y / (1 + f);
  // vtype f = exp(-y % yhat);
  // *grad = (1 + f)/-y;

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void BinomialDevianceLoss<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  vtype f = exp(y % yhat);
  *hess = (pow(y, 2) % f) / pow(1 + f, 2);
  // vtype f(y.n_cols, arma::fill::ones);
  // *hess = f;
}

template <typename DataType>
DataType BinomialDevianceLoss<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType BinomialDevianceLoss<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return sum(log(1 + exp(-y % yhat)));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real BinomialDevianceLoss<DataType>::loss_reverse(
    const ArrayXreal& yhat, const ArrayXreal& y) {
  return ((1 + (-y * yhat).exp()).log()).sum();
}
#endif

////////////////
// END BinomialDevianceLoss
////////////////

////////////////
// BEGIN ExpLoss
////////////////

template <typename DataType>
DataType ExpLoss<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  *grad = -y % exp(-y % yhat);
#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void ExpLoss<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  *hess = pow(y, 2) % exp(-y % yhat);
}

template <typename DataType>
DataType ExpLoss<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType ExpLoss<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return sum(exp(-y % yhat));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real ExpLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return ((-y * yhat).exp()).sum();
}
#endif

////////////////
// END ExpLoss
////////////////

////////////////
// BEGIN SavageLoss
////////////////

template <typename DataType>
DataType SavageLoss<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  vtype f = exp(2 * y % yhat);
  *grad = (4 * y % f) / pow(1 + f, 3);

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void SavageLoss<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  vtype f = exp(2 * y % yhat);
  *hess = (8 * pow(y, 2) % f % (2 * f - 1)) / pow(1 + f, 4);
}

template <typename DataType>
DataType SavageLoss<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType SavageLoss<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return 1. / sum(pow(1 + exp(-y % yhat), 2));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real SavageLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return (1 / pow(1 + (-y * yhat).exp(), 2)).sum();
}
#endif

////////////////
// END SavageLoss
////////////////

////////////////
// BEGIN ArctanLoss
////////////////

template <typename DataType>
DataType ArctanLoss<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  *grad = 2 * atan(y - yhat) / (1 + pow(y - yhat, 2));

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void ArctanLoss<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  vtype f = pow(y, 2);
  vtype g = pow(yhat, 2);
  *hess = (2 - 4 * ((y - yhat) % atan(y - yhat))) / pow(f - (2 * y % yhat) + g + 1, 2);
}

template <typename DataType>
DataType ArctanLoss<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType ArctanLoss<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return sum(pow(atan(y - yhat), 2));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real ArctanLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow(atan(y - yhat, 2), 2).sum();
}
#endif

////////////////
// END ArctanLoss
////////////////

////////////////
// BEGIN SyntheticLoss
////////////////

template <typename DataType>
DataType SyntheticLoss<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  *grad = -y % pow(y - yhat, 2);

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void SyntheticLoss<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  UNUSED(yhat);

  // OLD
  vtype f(y.n_cols, arma::fill::ones);
  *hess = f;

  // NEW
  // *hess = (sign(y) % exp(yhat / (pow(y, 2) % (y - yhat)))) / (y % pow(y - yhat, 2));
}

template <typename DataType>
DataType SyntheticLoss<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType SyntheticLoss<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return sum(pow(y - yhat, 2));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real SyntheticLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

////////////////
// END SyntheticLoss
////////////////

//////////////////////////
// BEGIN SyntheticLossVar1
//////////////////////////

template <typename DataType>
DataType SyntheticLossVar1<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  // Recasting of BinomialDevianceLoss
  // vtype f = exp(y % yhat);
  // vtype g = (pow(y, 2) % f)/pow(1 + f, 2);
  // *grad = -y/(g % (1 + f));

  // Second recasting of BinomialDevianceLoss
  // vtype f = exp(y % yhat);
  // vtype g = (pow(y, 2) % f);
  // *grad = -y/(g % (1 + f));

  // Proper decomposition of \Phi\left( y,\hat{y}\right) = \left( y-\hat{y}\right)^3
  vtype f = exp(-0.5 * (1 / pow(y - yhat, 2)));
  vtype g = exp(0.5 * (1 / pow(y, 2)));
  *grad = -y % f % g;

  // vtype f(y.n_cols, arma::fill::zeros);
  // *grad = -sign(y) % max(1 - sign(y) % yhat, f);

  // *grad = -sign(y) % max(sign(y) % (y - yhat), f);
  // *grad = -y % pow(y - yhat, 2);

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void SyntheticLossVar1<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  // Recasting of BinomialDevianceLoss
  // vtype f(y.n_cols, arma::fill::ones);
  // *hess = f;

  // Second recasting of BinomialDevianceLoss
  // vtype f = exp(y % yhat);
  // *hess = 1. / pow(1 + f, 2);

  // Proper decomposition of \Phi\left( y,\hat{y}\right) = \left( y-\hat{y}\right)^3
  vtype f = exp(-0.5 * (1 / pow(y - yhat, 2)));
  vtype g = exp(0.5 * (1 / pow(y, 2)));
  *hess = (y % f % g) / pow(y - yhat, 3);

  // UNUSED(yhat);

  // vtype f(y.n_cols, arma::fill::ones);
  // *hess = f;
}

template <typename DataType>
DataType SyntheticLossVar1<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType SyntheticLossVar1<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return sum(pow(y - yhat, 2));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real SyntheticLossVar1<DataType>::loss_reverse(
    const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

//////////////////////////////
// END SyntheticLossVariation1
//////////////////////////////

//////////////////////////////
// BEGIN SyntheticLossVariation2
//////////////////////////////

template <typename DataType>
DataType SyntheticLossVar2<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  // Proper decomposition of \Phi\left( y,\hat{y}\right) = y\left( y-\hat{y}\right)^2
  // vtype f = exp(-1.0 / (y % (y - yhat)));
  // vtype g = exp(1.0 / pow(y, 2));
  // *grad = -y % f % g;

  // Proper decomposition of \Phi\left( y,\hat{y}\right) = \left( y-\hat{y}\right)^{\frac{1}{3}}
  vtype p_yyhat = pow(abs(y - yhat), 1. / 3.);
  p_yyhat = pow(p_yyhat, 2);
  vtype p_y = pow(abs(y), 1. / 3.);
  p_y = pow(p_y, 2);

  vtype f = exp((3. / 2.) * p_yyhat);
  vtype g = exp(-(3. / 2.) * p_y);

  *grad = -y % f % g;

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void SyntheticLossVar2<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  // Naive implementation of \Phi \left( y,\hat{y}\right) = \left( y-\hat{y}\right)^3
  // vtype f(y.n_cols, arma::fill::ones);
  // *hess = f;

  // Proper decomposition of \Phi\left( y,\hat{y}\right) = y\left( y-\hat{y}\right)^2
  // vtype f = exp(-1.0 / (y % (y - yhat)));
  // vtype g = exp(1.0 / pow(y, 2));
  // *hess = (f % g) / pow(y - yhat, 2);

  // Proper decomposition of \Phi\left( y,\hat{y}\right) = \left( y-\hat{y}\right)^{\frac{1}{3}}
  vtype p_yyhat = pow(abs(y - yhat), 1. / 3.);
  p_yyhat = pow(p_yyhat, 2);
  vtype p_y = pow(abs(y), 1. / 3.);
  p_y = pow(p_y, 2);
  vtype den = pow(abs(y - yhat), 1. / 3.);

  vtype f = exp((3. / 2.) * p_yyhat);
  vtype g = exp(-(3. / 2.) * p_y);

  *hess = (1.0 / den) % f % g;
}

template <typename DataType>
DataType SyntheticLossVar2<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType SyntheticLossVar2<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return sum(pow(y - yhat, 2));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real SyntheticLossVar2<DataType>::loss_reverse(
    const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

//////////////////////////////
// END SyntheticLossVariation2
//////////////////////////////

/////////////////////////
// BEGIN CrossEntropyLoss
/////////////////////////

template <typename DataType>
DataType CrossEntropyLoss<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  *grad = sigmoid<DataType>(yhat) - y;

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void CrossEntropyLoss<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  UNUSED(y);
  vtype f = sigmoid<DataType>(yhat);
  *hess = f % (1 - f);
}

template <typename DataType>
DataType CrossEntropyLoss<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType CrossEntropyLoss<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return sum(pow(y - yhat, 2));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real CrossEntropyLoss<DataType>::loss_reverse(
    const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

/////////////////////////
// END CrossEntropyLoss
/////////////////////////

//////////////////////////////
// BEGIN SyntheticLossVariation3
//////////////////////////////

template <typename DataType>
DataType SyntheticLossVar3<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  // Proper decomposition of \Phi \left( y,\hat{y}\right) = y\left( 1-y\hat{y}\right)^3
  vtype f = exp(-1.0 / (pow(y, 2) % pow(1 - y % yhat, 2)));
  vtype g = exp(1.0 / pow(y, 2));
  *grad = -y % f % g;

#ifdef AUTODIFF
  ArrayXreal yhatr = LossUtils::static_cast_eigen<DataType>(yhat).eval();
  ArrayXreal yr = LossUtils::static_cast_eigen<DataType>(y).eval();

  return static_cast<DataType>(loss_reverse(yr, yhatr).val());
#else
  return static_cast<DataType>(loss_reverse_arma(y, yhat));
#endif
}

template <typename DataType>
void SyntheticLossVar3<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  vtype f = exp(-1.0 / (pow(y, 2) % pow(1 - y % yhat, 2)));
  vtype g = exp(1.0 / pow(y, 2));
  vtype den = pow(1 - y % yhat, 2);
  *hess = (2.0 / den) % f % g;
}

template <typename DataType>
DataType SyntheticLossVar3<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType SyntheticLossVar3<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return sum(pow(y - yhat, 2));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real SyntheticLossVar3<DataType>::loss_reverse(
    const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

//////////////////////////////
// END SyntheticLossVariation3
//////////////////////////////

////////////////////
// BEGIN PowerLoss
////////////////////

template <typename DataType>
DataType PowerLoss<DataType>::gradient_(const vtype& yhat, const vtype& y, vtype* grad) {
  // Proper decomposition of \Phi \left( y,\hat{y}\right) = y\left( 1-y\hat{y}\right)^p
  if (p_ != 1) {
    vtype f = exp(1.0 / ((-p_ + 1) * pow(y, 2))) % pow(1 - y % yhat, -p_ + 1);
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
void PowerLoss<DataType>::hessian_(const vtype& yhat, const vtype& y, vtype* hess) {
  if (p_ != 1) {
    vtype f = exp(1.0 / ((-p_ + 1) * pow(y, 2))) % pow(1 - y % yhat, -p_ + 1);
    vtype g = exp(-1.0 / ((-p_ + 1) * pow(y, 2)));
    *hess = pow(1 - y % yhat, -p_) % f % g;
  } else {
    vtype f = exp(log(1 - y % yhat) / pow(y, 2));
    *hess = (1.0 / (1 - y % yhat)) % f;
  }
}

template <typename DataType>
DataType PowerLoss<DataType>::loss_reverse_arma(const vtype& yhat, const vtype& y) {
  UNUSED(yhat);
  UNUSED(y);
  return 0.;
}

template <typename DataType>
DataType PowerLoss<DataType>::loss_arma(const vtype& yhat, const vtype& y) {
  return sum(pow(y - yhat, 2));
}

#ifdef AUTODIFF
template <typename DataType>
autodiff::real PowerLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

////////////////////
// END PowerLoss
////////////////////

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

////////////////
// END Loss
////////////////

#endif
