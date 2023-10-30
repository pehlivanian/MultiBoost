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
	       std::size_t passThrough,
	       std::size_t thresh) : passThrough_{passThrough}, thresh_{thresh}
  { init_(labels); }

  ConstantTree(const Mat<float>& dataset,
	       Row<std::size_t>& labels,
	       std::size_t passThrough,
	       std::size_t thresh) : passThrough_{passThrough}, thresh_{thresh}
  { init_(labels); }

  ConstantTree(const Mat<double>& dataset,
	       Row<std::size_t>& labels) : passThrough_{0}, thresh_{0}
  { init_(labels); }

  ConstantTree(const Mat<float>& dataset,
	       Row<std::size_t>& labels) : passThrough_{0}, thresh_{0}
  { init_(labels); }

  ConstantTree(Row<std::size_t>& labels) : passThrough_{0}, thresh_{0}
  { init_(labels); }

  ConstantTree(std::size_t leafValue) : leafValue_{leafValue} {}

  void Classify(const Mat<double>& dataset, Row<std::size_t>& prediction) {
    prediction = ones<Row<std::size_t>>(dataset.n_cols);
    prediction.fill(leafValue_);
  }

  void Classify(Mat<double>&& dataset, Row<std::size_t>& prediction) {
    prediction = zeros<Row<std::size_t>>(dataset.n_cols);
    prediction.fill(leafValue_);
  }
  
  void Classify(const Mat<float>& dataset, Row<std::size_t>& prediction) {
    prediction = zeros<Row<std::size_t>>(dataset.n_cols);
    prediction.fill(leafValue_);
  }

  void Classify(Mat<float>&& dataset, Row<std::size_t>& prediction) {
    prediction = zeros<Row<std::size_t>>(dataset.n_cols);
    prediction.fill(leafValue_);
  }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(leafValue_);
  }

private:

  void init_(Row<std::size_t>& labels) {
    Row<std::size_t> uniqueVals = unique(labels);
    assert(uniqueVals.n_cols == 1);
    leafValue_ = uniqueVals[0];
  }

  std::size_t thresh_;
  std::size_t passThrough_;
  std::size_t leafValue_;

};


#endif
