#ifndef __GRADIENTBOOSTCLASSIFIER_HPP__
#define __GRADIENTBOOSTCLASSIFIER_HPP__

#include <list>
#include <utility>
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
#include "score.hpp"
#include "LTSS.hpp"
#include "DP.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

using namespace Objectives;
using namespace LossMeasures;

namespace PartitionSize {
  enum class SizeMethod { 
    FIXED = 0,
      FIXED_PROPORTION = 1,
      DECREASING = 2,
      INCREASING = 3,
      RANDOM = 4,
  };
} // namespace Partition

namespace LearningRate {
  enum class RateMethod {
    FIXED = 0,
      INCREASING = 1,
      DECREASING = 2,
  };
} // namespace LarningRate

namespace ClassifierContext {
  struct Context {
    lossFunction loss;
    std::size_t partitionSize;
    double learningRate;
    int steps;
    bool symmetrizeLabels;
    double rowSubsampleRatio;
    double colSubsampleRatio;
    bool preExtrapolate;
    bool postExtrapolate;
    PartitionSize::SizeMethod partitionSizeMethod;
    LearningRate::RateMethod learningRateMethod;
  };
} // namespace ClassifierContext

// Helpers for gdb
template<class mat>
void print_matrix(mat matrix) {
  matrix.print(std::cout);
}

template<class Row>
void print_row(Row row) {
  row.print(std::cout);
}

class PartitionUtils {
public:
  static std::vector<int> _shuffle(int sz) {
    std::vector<int> ind(sz), r(sz);
    std::iota(ind.begin(), ind.end(), 0);
    
    std::vector<std::vector<int>::iterator> v(static_cast<int>(ind.size()));
    std::iota(v.begin(), v.end(), ind.begin());
    
    std::shuffle(v.begin(), v.end(), std::mt19937{std::random_device{}()});
    
    for (int i=0; i<v.size(); ++i) {
      r[i] = *(v[i]);
    }
}

};

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
  using Classifier = DecisionTreeRegressor<MADGain>;
  DecisionTreeRegressorClassifier(const mat& dataset,
				  rowvec& labels,
				  unsigned long minLeafSize=1,
				  double minGainSplit=0.0) :
    classifier_{dataset, labels, minLeafSize, minGainSplit} {}
  /*
    Doesn't work that well
    using Classifier = DecisionTreeRegressor<MSEGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
    
    DecisionTreeRegressorClassifier(const mat& dataset, 
    rowvec& labels,
    unsigned long minLeafSize=10,
    double minGainSplit=1.e-7,
    unsigned long maxDepth=10) :
    classifier_{dataset, labels, minLeafSize, minGainSplit, maxDepth} {}
  */
  
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

  using Partition = std::vector<std::vector<int>>;
  using PartitionList = std::vector<Partition>;
  using Classifier = DecisionTreeRegressorClassifier<DataType>;
  using ClassifierList = std::vector<std::unique_ptr<Classifier>>;
  using Leaves = Row<DataType>;
  using LeavesList = std::vector<Leaves>;
  using Prediction = Row<DataType>;
  using PredictionList = std::vector<Prediction>;
  using MaskList = std::vector<uvec>;
  
  GradientBoostClassifier<DataType>(const mat& dataset, 
				    const Row<DataType>& labels, 
				    lossFunction loss,
				    std::size_t partitionSize,
				    double learningRate,
				    int steps,
				    bool symmetrizeLabels) : 
    dataset_{dataset},
    labels_{labels},
    steps_{steps},
    symmetrized_{symmetrizeLabels},
    loss_{loss},
    partitionSize_{partitionSize},
    learningRate_{learningRate},
    row_subsample_ratio_{1.},
    col_subsample_ratio_{.05},
    preExtrapolate_{false},
    postExtrapolate_{true},
    current_classifier_ind_{0} 
  { init_(); }

  GradientBoostClassifier<DataType>(const mat& dataset, 
				    const Row<DataType>& labels,
				    ClassifierContext::Context context) :
    dataset_{dataset},
    labels_{labels},
    loss_{context.loss},
    partitionSize_{context.partitionSize},
    learningRate_{context.learningRate},
    steps_{context.steps},
    symmetrized_{context.symmetrizeLabels},
    row_subsample_ratio_{context.rowSubsampleRatio},
    col_subsample_ratio_{context.colSubsampleRatio},
    preExtrapolate_{context.preExtrapolate},
    postExtrapolate_{context.postExtrapolate},
    partitionSizeMethod_{context.partitionSizeMethod},
    learningRateMethod_{context.learningRateMethod},
    current_classifier_ind_{0} 
  { init_(); }

  void fit();

  void Classify(const mat&, Row<DataType>&);
  void Predict(Row<DataType>&);
  void Predict(Row<DataType>&, const uvec&);
  void Predict(const mat&, Row<DataType>&);

  mat getDataset() const { return dataset_; }
  Row<DataType> getLabels() const { return labels_; }

private:
  void init_();
  Row<DataType> _constantLeaf() const;
  uvec subsampleRows(size_t);
  uvec subsampleCols(size_t);
  void symmetrizeLabels();
  void deSymmetrize(Row<DataType>&);
  void fit_step(std::size_t);
  double computeLearningRate(std::size_t);
  std::size_t computePartitionSize(std::size_t);

  std::pair<rowvec,rowvec>  generate_coefficients(const mat&, const Row<DataType>&, const uvec&);
  Leaves computeOptimalSplit(rowvec&, rowvec&, mat, std::size_t, std::size_t);

  void setNextClassifier(const ClassifierBase<DataType>&);
  int steps_;
  mat dataset_;
  Row<DataType> labels_;
  std::size_t partitionSize_;

  lossFunction loss_;
  LossFunction<DataType>* lossFn_;
  
  double learningRate_;

  PartitionSize::SizeMethod partitionSizeMethod_;
  LearningRate::RateMethod learningRateMethod_;

  double row_subsample_ratio_;
  double col_subsample_ratio_;

  uvec rowMask_;
  uvec colMask_;

  int n_;
  int m_;

  int current_classifier_ind_;

  LeavesList leaves_;
  ClassifierList classifiers_;
  PartitionList partitions_;
  PredictionList predictions_;
  MaskList rowMasks_;
  MaskList colMasks_;

  std::mt19937 mersenne_engine_{std::random_device{}()};

  bool symmetrized_;

  bool preExtrapolate_;
  bool postExtrapolate_;
};

#include "gradientboostclassifier_impl.hpp"

#endif
