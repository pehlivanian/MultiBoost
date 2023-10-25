#ifndef __CONSTANTREGRESSOR_HPP__
#define __CONSTANTREGRESSOR_HPP__

#include <mlpack/core.hpp>

using namespace arma;

class ConstantTreeRegressor {
public:
  ConstantTreeRegressor() = default;
  ConstantTreeRegressor(ConstantTreeRegressor&) = default;

  ConstantTreeRegressor(const Mat<double>& dataset,
			Row<double>& labels,
			double leafValue) : leafValue_{leafValue} {}

  ConstantTreeRegressor(Mat<double>&& dataset,
			Row<double>& labels,
			double leafValue) : leafValue_{leafValue} {}

  ConstantTreeRegressor(const Mat<float>& dataset,
			Row<float>& labels,
			double leafValue) : leafValue_{leafValue} {}

  ConstantTreeRegressor(Mat<float>&& dataset,
			Row<float>& labels,
			double leafValue) : leafValue_{leafValue} {}

  void Predict(const Mat<double>& dataset,
	       Row<double>& prediction) {
    prediction = ones<Row<double>>(dataset.n_cols);
    prediction.fill(leafValue_);
  }

  void Predict(const Mat<float>& dataset,
	       Row<float>& prediction) {
    prediction = ones<Row<float>>(dataset.n_cols);
    prediction.fill(leafValue_);
  }

private:
  double leafValue_;
};

#endif
