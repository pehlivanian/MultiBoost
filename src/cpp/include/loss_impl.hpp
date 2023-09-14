#ifndef __LOSS_IMPL_HPP__
#define __LOSS_IMPL_HPP__

using namespace LossMeasures;

const bool AUTODIFF_ON = false;

template<typename DataType>
DataType
LossFunction<DataType>::loss(const rowvec& yhat, const rowvec& y, rowvec* grad, rowvec* hess,
			     bool clamp_gradient, DataType upper_val, DataType lower_val) {
#ifdef AUTODIFFBAD
  {
    // Slooooooowwww...
    // Since we only need gradient, hessian of quadratic approximation to loss and not
    // some deep neural network composition, we can usually compute and 
    // hard-code these for a given loss function, the approach we take here.
    autodiff::real u;
    ArrayXreal yhatr = LossUtils::static_cast_eigen(yhat).eval();
    ArrayXreal yr = LossUtils::static_cast_eigen(y).eval();
    
    std::function<autodiff::real(const ArrayXreal&)> loss_ = 
      std::bind(&LossFunction<DataType>::loss_reverse, this, yr, _1);
    Eigen::VectorXd grad_tmp = gradient(loss_, wrt(yhatr), at(yhatr), u);
    *grad = LossUtils::static_cast_arma(gradtmp);
    return static_cast<DataType>(u.val());
  }
#else
  {
    DataType r = gradient_(yhat, y, grad);

    if (clamp_gradient) {
      // Clamp gradient, leave hessian as is
      // Only relevant for {-1,1}-classifier
      for (std::size_t i=0; i<grad->n_elem; ++i) {
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

////////////////
// BEGIN LogLoss 
////////////////
template<typename DataType>
DataType
LogLoss<DataType>::gradient_(const rowvec& yhat, const rowvec& y, rowvec* grad) {
  // *grad = (y - yhat) / ((yhat - 1) % yhat);
  *grad = -1*(y - yhat);

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
LogLoss<DataType>::hessian_(const rowvec& yhat, const rowvec& y, rowvec* hess) {
  // *hess = (-2*y % yhat + y + pow(yhat, 2))/pow(yhat % (yhat-1), 2);
  *hess = yhat * (1 - yhat);
}

template<typename DataType>
DataType
LogLoss<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  return 0.;

  rowvec log_odds = log(yhat / (1 - yhat));
  return -1 * sum(y % log_odds - log(1 + exp(log_odds)));

  /*
    rowvec a = exp(yhat) / (1 + exp(yhat));
    double b = -1 * sum(y % log(yhat) + (1 - y) % log(1 - yhat));
    
    rowvec log_odds = log(yhat / (1 - yhat));
    double c = -1 * sum(log_odds - log(1 + exp(log_odds)));
    
    return -1 * sum(log_odds - log(1 + exp(log_odds)));
  */
}

#ifdef AUTODIFF
template<typename DataType>
autodiff::real
LogLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow(1 - y*yhat, 2).sum();
}
#endif

///////////////////
// BEGIN SquareLoss
///////////////////
template<typename DataType>
DataType
SquareLoss<DataType>::gradient_(const rowvec& yhat, const rowvec& y, rowvec* grad) {
  *grad = 2 * y % (y % yhat - 1);
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
SquareLoss<DataType>::hessian_(const rowvec& yhat, const rowvec& y, rowvec* hess) {
  (void)yhat;
  *hess = 2 * pow(y, 2);
}

template<typename DataType>
DataType
SquareLoss<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  // return 0.;
  return sum(pow(1 -  y % yhat, 2));
}

#ifdef AUTODIFF
template<typename DataType>
autodiff::real
SquareLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow(1 - y*yhat, 2).sum();
}
#endif

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
  (void)yhat;
  rowvec f(y.n_cols, arma::fill::ones);
  *hess = 2 * f;
}

template<typename DataType>
DataType
MSELoss<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  // return 0.;
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
  rowvec f = exp(y % yhat);
  *grad = -y/(1 + f);
  // rowvec f = exp(-y % yhat);
  // *grad = (1 + f)/-y;
  
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
  rowvec f = exp(y % yhat);
  *hess = (pow(y, 2) % f)/pow(1 + f, 2);
  // rowvec f(y.n_cols, arma::fill::ones);
  // *hess = f;
  
}

template<typename DataType>
DataType
BinomialDevianceLoss<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  // return 0.;
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
// BEGIN ExpLoss
////////////////

template<typename DataType>
DataType
ExpLoss<DataType>::gradient_(const rowvec& yhat, const rowvec& y, rowvec* grad) {
  *grad = -y % exp(-y % yhat);
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
ExpLoss<DataType>::hessian_(const rowvec& yhat, const rowvec& y, rowvec *hess) {
  *hess = pow(y, 2) % exp(-y % yhat);
}

template<typename DataType>
DataType
ExpLoss<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  // return 0.;
  return sum(exp(-y % yhat));
}

#ifdef AUTODIFF
template<typename DataType>
autodiff::real
ExpLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return ((-y * yhat).exp()).sum();
}
#endif

////////////////
// END ExpLoss
////////////////

////////////////
// BEGIN SavageLoss
////////////////

template<typename DataType>
DataType
SavageLoss<DataType>::gradient_(const rowvec& yhat, const rowvec& y, rowvec* grad) {
  rowvec f = exp(2* y % yhat);
  *grad = (4 * y % f) / pow(1 + f, 3);

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
  rowvec f = exp(2 * y % yhat);
  *hess = (8 * pow(y, 2) % f % (2*f - 1)) / pow(1 + f, 4);
}

template<typename DataType>
DataType
SavageLoss<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  // return 0.;
  return sum(1 / pow(1 + exp(2* y % yhat), 2));
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

////////////////
// BEGIN ArctanLoss
////////////////

template<typename DataType>
DataType
ArctanLoss<DataType>::gradient_(const rowvec& yhat, const rowvec& y, rowvec* grad) {
  *grad = 2 * atan(y - yhat)/(1 + pow(y - yhat, 2));

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
ArctanLoss<DataType>::hessian_(const rowvec& yhat, const rowvec& y, rowvec* hess) {
  rowvec f = pow(y, 2);
  rowvec g = pow(yhat, 2);
  *hess = (2 - 4 * ((y - yhat) % atan(y - yhat)))/pow(f - (2 * y % yhat) + g + 1, 2);
}

template<typename DataType>
DataType
ArctanLoss<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  // return 0.;
  return sum(pow(atan(y - yhat), 2));
}

#ifdef AUTODIFF
template<typename DataType>
autodiff::real
ArctanLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow(atan(y - yhat, 2), 2).sum();
}
#endif

////////////////
// END ArctanLoss
////////////////

////////////////
// BEGIN SyntheticLoss
////////////////

template<typename DataType>
DataType
SyntheticLoss<DataType>::gradient_(const rowvec& yhat, const rowvec& y, rowvec* grad) {
  *grad = -y % pow(y - yhat, 2);
  
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
SyntheticLoss<DataType>::hessian_(const rowvec& yhat, const rowvec& y, rowvec* hess) {

  (void)yhat;

  // OLD
  rowvec f(y.n_cols, arma::fill::ones);
  *hess = f;
  
  // NEW
  // *hess = (sign(y) % exp(yhat / (pow(y, 2) % (y - yhat)))) / (y % pow(y - yhat, 2));
  
  
}

template<typename DataType>
DataType
SyntheticLoss<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  // return 0.;
  return sum(pow((y - yhat), 2));
}

#ifdef AUTODIFF
template<typename DataType>
autodiff::real
SyntheticLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

////////////////
// END SyntheticLoss
////////////////

//////////////////////////
// BEGIN SyntheticLossVar1
//////////////////////////

template<typename DataType>
DataType
SyntheticLossVar1<DataType>::gradient_(const rowvec& yhat, const rowvec& y, rowvec* grad) {

  rowvec f(y.n_cols, arma::fill::zeros);
  *grad = -sign(y) % max(1 - sign(y) % yhat, f);
  // *grad = -sign(y) % max(sign(y) % (y - yhat), f);
  // *grad = -y % pow(y - yhat, 2);


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
SyntheticLossVar1<DataType>::hessian_(const rowvec& yhat, const rowvec& y, rowvec* hess) {

  (void)yhat;

  rowvec f(y.n_cols, arma::fill::ones);
  *hess = f;
}

template<typename DataType>
DataType
SyntheticLossVar1<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  // return 0.;
  return sum(pow((y - yhat), 2));
}

#ifdef AUTODIFF
template<typename DataType>
autodiff::real
SyntheticLossVar1<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

//////////////////////////////
// END SyntheticLossVariation1
//////////////////////////////

/////////////////////////
// BEGIN SyntheticRegLoss
/////////////////////////

template<typename DataType>
DataType
SyntheticRegLoss<DataType>::gradient_(const rowvec& yhat, const rowvec& y, rowvec* grad) {

  // rowvec f = exp( -0.5 / pow(y - yhat, 2));
  // f.transform([](double val) { return (std::isnan(val) ? 0. : val); });
  // *grad = f;
  
  // rowvec f = -exp( -0.5 / pow(y - yhat, 2)) % pow(y - yhat, 3);
  // f.transform([](double val){ return (std::isnan(val) ? 0. : val); });
  // *grad = f;

  // rowvec f = -2 * pow(y - yhat, 3);
  // f.transform([](double val){ return (std::isnan(val) ? 0. : val); });
  // *grad = f;

  rowvec f = -2. * sign(y - yhat) % pow(abs(y - yhat), .3333);
  f.transform([](double val){ return (std::isnan(val) ? 0. : val); });
  *grad = f;

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
SyntheticRegLoss<DataType>::hessian_(const rowvec& yhat, const rowvec& y, rowvec* hess) {
  (void)yhat;

  // rowvec f = -exp( -0.5 / pow(y - yhat, 2));
  // rowvec g = f / pow(y - yhat, 3);
  // g.transform([](double val) { return (std::isnan(val) ? 1. : val); });
  // *hess = g;

  // rowvec f = exp( -0.5 / pow(y - yhat, 2));
  // f.transform([](double val) { return (std::isnan(val) ? 1. : val); });
  // *hess = f;

  rowvec f(y.n_cols, arma::fill::ones);
  *hess = 2 * f;

}

template<typename DataType>
DataType
SyntheticRegLoss<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  // return 0.;
  return sum(pow((y - yhat), 2));
}

#ifdef AUTODIFF
template<typename DataType>
autodiff::real
SyntheticRegLoss<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

///////////////////////
// END SyntheticRegLoss
///////////////////////

//////////////////////////////
// BEGIN SyntheticLossVariation2
//////////////////////////////

template<typename DataType>
DataType
SyntheticLossVar2<DataType>::gradient_(const rowvec& yhat, const rowvec& y, rowvec* grad) {

  // Tent function
  // *grad = -sign(y) % ( y - yhat);

  // Cubic cutoff
  // rowvec f(y.n_cols, arma::fill::ones);
  // *grad = -sign(y) % max(-sign(y) * pow(yhat - y, 3), sign(y) % f);

  // Quadratic cutoff
  rowvec f(y.n_cols, arma::fill::zeros);
  *grad = -sign(y) % max(-sign(y) % sign(yhat - y) % pow(yhat - y, 2), f);

  // Quartic cutoff
  // rowvec f(y.n_cols, arma::fill::zeros);
  // *grad = -sign(y) % max(-sign(y) % sign(yhat - y) % pow(yhat - y, 4), f);

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
SyntheticLossVar2<DataType>::hessian_(const rowvec& yhat, const rowvec& y, rowvec* hess) {

  (void)yhat;

  rowvec f(y.n_cols, arma::fill::ones);
  *hess = f;
}

template<typename DataType>
DataType
SyntheticLossVar2<DataType>::loss_reverse_arma(const rowvec& yhat, const rowvec& y) {
  // return 0.;
  return sum(pow((y - yhat), 2));
}

#ifdef AUTODIFF
template<typename DataType>
autodiff::real
SyntheticLossVar2<DataType>::loss_reverse(const ArrayXreal& yhat, const ArrayXreal& y) {
  return pow((y - yhat), 2).sum();
}
#endif

//////////////////////////////
// END SyntheticLossVariation2
//////////////////////////////



////////////////
// END Loss
////////////////



#endif
