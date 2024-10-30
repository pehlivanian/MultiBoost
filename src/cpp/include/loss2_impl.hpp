#ifndef __LOSS2_IMPL_HPP__
#define __LOSS2_IMPL_HPP__

using namespace LossMeasures;

const bool AUTODIFF_ON = false;

template<typename DataType>
DataType
LossFunction<DataType>::loss(const vtype& yhat, const vtype& y, vtype* grad, vtype* hess,
			     bool clamp_gradient, DataType upper_val, DataType lower_val) {
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


#endif
