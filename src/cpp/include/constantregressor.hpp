#ifndef __CONSTANTREGRESSOR_HPP__
#define __CONSTANTREGRESSOR_HPP__

#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)
#include <mlpack/core.hpp>

using namespace arma;

class ConstantTreeRegressor {
public:
  ConstantTreeRegressor() = default;
  ConstantTreeRegressor(ConstantTreeRegressor&) = default;

  ConstantTreeRegressor(const Mat<double>& dataset, Row<double>& labels) {
    UNUSED(dataset);
    init_(labels);
  }

  ConstantTreeRegressor(const Mat<float>& dataset, Row<float>& labels) {
    UNUSED(dataset);
    init_(labels);
  }

  ConstantTreeRegressor(double leafValue) : leafValue_{leafValue} {}

  void Predict(const Mat<double>& dataset, Row<double>& prediction) {
    prediction = ones<Row<double>>(dataset.n_cols);
    prediction.fill(leafValue_);
  }

  void Predict(const Mat<float>& dataset, Row<float>& prediction) {
    prediction = ones<Row<float>>(dataset.n_cols);
    prediction.fill(leafValue_);
  }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(leafValue_);
  }

private:
  template <typename F>
  void init_(Row<F>& labels) {
    Row<F> uniqueVals = unique(labels);
    assert(uniqueVals.n_cols == 1);
    leafValue_ = static_cast<double>(uniqueVals[0]);
  }

  double leafValue_;
};

#endif
