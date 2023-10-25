#ifndef __CONSTANTCLASSIFIER_HPP__
#define __CONSTANTCLASSIFIER_HPP__

#include <mlpack/core.hpp>

using namespace arma;

class ConstantTree {
public:
  ConstantTree() = default;
  ConstantTree(ConstantTree&) = default;

  ConstantTree(const Mat<double>& dataset,
	       Row<std::size_t>& labels,
	       std::size_t leafValue) : leafValue_{leafValue} {}

  ConstantTree(const Mat<float>& dataset,
	       Row<std::size_t>& labels,
	       std::size_t leafValue) : leafValue_{leafValue} {}

  void Classify(const Mat<double>& dataset, Row<std::size_t>& prediction) {
    prediction = ones<Row<std::size_t>>(dataset.n_cols);
    prediction.fill(leafValue_);
  }

  void Classify(Mat<double>&& dataset, Row<std::size_t>& prediction) {
    prediction = ones<Row<std::size_t>>(dataset.n_cols);
    prediction.fill(leafValue_);
  }
  
  void Classify(const Mat<float>& dataset, Row<std::size_t>& prediction) {
    prediction = ones<Row<std::size_t>>(dataset.n_cols);
    prediction.fill(leafValue_);
  }

  void Classify(Mat<float>&& dataset, Row<std::size_t>& prediction) {
    prediction = ones<Row<std::size_t>>(dataset.n_cols);
    prediction.fill(leafValue_);
  }


private:
  std::size_t leafValue_;
};


#endif
