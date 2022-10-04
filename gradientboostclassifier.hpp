#ifndef __GRADIENTBOOSTCLASSIFIER_HPP__
#define __GRADIENTBOOSTCLASSIFIER_HPP__

#include <list>
#include <memory>
#include <random>
#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_map>

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>

#include "dataset.hpp"
#include "decision_tree.hpp"
#include "loss.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

using namespace LossMeasures;

// Helpers for gdb
template<class mat>
void print_matrix(mat matrix) {
  matrix.print(std::cout);
}

template<class Row>
void print_row(Row row) {
  row.print(std::cout);
}


template<typename DataType>
class ClassifierBase {
public:
  ClassifierBase() = default;
  ~ClassifierBase() = default;

  virtual void Classify_(const mat& dataset, Row<DataType>& labels) = 0;
};

template<typename DataType>
class DecisionTreeRegressorClassifier : public ClassifierBase<DataType> {
public:
  using Classifier = DecisionTreeRegressor<MSEGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  
  DecisionTreeRegressorClassifier(const mat& dataset, 
				  rowvec& labels,
				  unsigned long minLeafSize=10,
				  double minGainSplit=1.e-7,
				  unsigned long maxDepth=10) :
    classifier_{dataset, labels, minLeafSize, minGainSplit, maxDepth} {}
  
  void Classify_(const mat& dataset, rowvec& labels) {
    classifier_.Predict(dataset, labels);
  }

private:
  Classifier classifier_;
};

template<typename DataType>
class DecisionTreeClassifier : public ClassifierBase<DataType> {
public:
  // Doesn't have a predict method which rerturns floats
  using Classifier = DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  
  DecisionTreeClassifier(const mat& dataset, Row<DataType>& labels) :
    classifier_{dataset, labels, 7, 10, 3} {}
    
  void Classify_(const mat& dataset, Row<DataType>& labels) {
    classifier_.Classify(dataset, labels);
  }
private:
  Classifier classifier_;
};

template<typename DataType>
class LeafOnlyClassifier : public ClassifierBase<DataType> {
public:
  LeafOnlyClassifier(Row<DataType> leaves) :
    leaves_{leaves} {}

  void Classify_(const mat& dataset, Row<DataType>& labels) {
    labels = leaves_;
  }
private:
  Row<DataType> leaves_;
};

template<typename DataType>
class GradientBoostClassifier {
public:

  using Partition = std::vector<int>;
  using PartitionList = std::vector<Partition>;
  using RegularizationList = std::vector<DataType>;
  using ClassifierList = std::vector<std::unique_ptr<DecisionTreeRegressorClassifier<DataType>>>;
  using Leaves = Row<DataType>;
  using LeavesValues = std::vector<Leaves>;
  using LeavesMap = std::unordered_map<DataType, int>;
  using Prediction = Row<DataType>;
  using PredictionList = std::vector<Prediction>;
  
  GradientBoostClassifier<DataType>(const mat& dataset, 
				    const Row<DataType>& labels, 
				    lossFunction loss,
				    int steps,
				    bool symmetrizeLabels) : 
    steps_{steps},
    symmetrized_{symmetrizeLabels},
    dataset_{dataset},
    loss_{loss},
    labels_{labels},
    row_subsample_ratio_{1.},
    col_subsample_ratio_{.75}, // .75
    current_classifier_ind_{0} { init_(); }

  void Classify(const mat&, Row<DataType>&);
  void Predict(const mat& dataset, Row<DataType>& labels) { Classify(dataset, labels); }

private:
  void init_();
  LeavesMap relabel(const Row<DataType>&, Row<std::size_t>&);
  Row<DataType> _constantLeaf() const;
  uvec subsampleRows(size_t);
  uvec subsampleCols(size_t);
  void symmetrizeLabels();
  void fit();

  rowvec generate_coefficients(const mat&, const Row<DataType>&);

  void setNextClassifier(const ClassifierBase<DataType>&);
  int steps_;
  mat dataset_;
  Row<DataType> labels_;

  lossFunction loss_;
  LossFunction<DataType>* lossFn_;

  double row_subsample_ratio_;
  double col_subsample_ratio_;

  uvec rowMask_;
  uvec colMask_;

  int n_;
  int m_;

  int current_classifier_ind_;

  LeavesValues leaves_;
  ClassifierList classifiers_;
  PartitionList partitions_;
  PredictionList predictions_;
  RegularizationList regularizations_;

  std::mt19937 mersenne_engine_{std::random_device{}()};

  bool symmetrized_;
};

#include "gradientboostclassifier_impl.hpp"

#endif
