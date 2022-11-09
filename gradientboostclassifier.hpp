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
#include <unordered_map>
#include <cassert>

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

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
    double partitionRatio = .5;
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
template<class Matrix>
void print_matrix(Matrix matrix) {
  matrix.print(std::cout);
}

template<class Row>
void print_vector(Row row) {
  row.print(std::cout);
}
template void print_matrix<arma::mat>(arma::mat matrix);
template void print_vector<arma::rowvec>(arma::rowvec row);

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
  // Different flavors
  using Classifier = DecisionTreeRegressor<MADGain, BestBinaryNumericSplit>;
  // using Classifier = DecisionTreeRegressor<MADGain>;
  // using Classifier = DecisionTreeRegressor<>;
  // using Classifier = DecisionTreeRegressor<MSEGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  // using Classifier = DecisionTreeRegressor<InformationGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;

  const unsigned long minLeafSize = 1;
  const double minGainSplit = 0.0;
  const unsigned long maxDepth = 100;

  DecisionTreeRegressorClassifier(const mat& dataset,
				  rowvec& labels,
				  unsigned long minLeafSize=5,
				  double minGainSplit=0.,
				  unsigned long maxDepth=5) :
    classifier_{dataset, labels, minLeafSize, minGainSplit, maxDepth} {}
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
class RandomForestClassifier : public ClassifierBase<DataType> {
public:
  
  using LeavesMap = std::unordered_map<std::size_t, DataType>;
  using Classifier = RandomForest<>;

  RandomForestClassifier(const mat& dataset, Row<DataType>& labels, std::size_t numClasses) {    
    Row<std::size_t> labels_t(labels.n_cols);
    leavesMap_ = encode(labels, labels_t);
    classifier_ = Classifier{dataset, labels_t, numClasses, 5, 3}; 
  }
  
  void Classify_(const mat& dataset, Row<DataType>& labels) {
    Row<std::size_t> labels_t;
    classifier_.Classify(dataset, labels_t);
    labels = Row<DataType>(labels_t.n_cols);
    decode(labels_t, labels);

    // Check error
    /*
      Row<std::size_t> prediction;
      classifier_.Classify(dataset, prediction);
      const double trainError = arma::accu(prediction != labels_t) * 100. / labels_t.n_elem;
      cout << "Training error: " << trainError << "%." << endl;
    */
  }
  
private:
  LeavesMap encode(const Row<DataType>&, Row<std::size_t>&);
  void decode(const Row<std::size_t>&, Row<DataType>&);

  Classifier classifier_;
  LeavesMap leavesMap_;
};

template<typename DataType>
class DecisionTreeClassifier : public ClassifierBase<DataType> {
public:

  using LeavesMap = std::unordered_map<std::size_t, DataType>;
  // Doesn't have a predict method which rerturns floats
  // using Classifier = DecisionTree<GiniGain, 
  //                      BestBinaryNumericSplit, 
  //			  AllCategoricalSplit, 
  //			  AllDimensionSelect, 
  //			  true>;
  using Classifier = DecisionTree<GiniGain, BestBinaryNumericSplit>;
  
  DecisionTreeClassifier(const mat& dataset, Row<DataType>& labels, std::size_t numClasses) 
  {    
    Row<std::size_t> labels_t(labels.n_cols);
    leavesMap_ = encode(labels, labels_t);
    classifier_ = Classifier{dataset, labels_t, numClasses, 1, 0.}; 

    // Check error
    /*
      Row<std::size_t> prediction;
      classifier_.Classify(dataset, prediction);
      const double trainError = arma::accu(prediction != labels_t) * 100. / labels_t.n_elem;
      cout << "Training error: " << trainError << "%." << endl;
    */
    
  }
    
  void Classify_(const mat& dataset, Row<DataType>& labels) {
    Row<std::size_t> labels_t;
    classifier_.Classify(dataset, labels_t);
    labels = Row<DataType>(labels_t.n_cols);
    decode(labels_t, labels);
  }

private:
  LeavesMap encode(const Row<DataType>&, Row<std::size_t>&);
  void decode(const Row<std::size_t>&, Row<DataType>&);

  Classifier classifier_;
  LeavesMap leavesMap_;

};

template<typename DataType>
class LeafOnlyClassifier : public ClassifierBase<DataType> {
public:
  LeafOnlyClassifier(const mat& dataset, const Row<DataType>& leaves) :
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

  // using Classifier = DecisionTreeRegressorClassifier<DataType>;
  using Classifier = DecisionTreeClassifier<DataType>;

  using Partition = std::vector<std::vector<int>>;
  using PartitionList = std::vector<Partition>;
  using ClassifierList = std::vector<std::unique_ptr<ClassifierBase<DataType>>>;
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
    partitionRatio_{context.partitionRatio},
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
  { 
    init_(); 
  }

  GradientBoostClassifier<DataType>(const mat& dataset_is,
				    const Row<DataType>& labels_is,
				    const mat& dataset_oos,
				    const Row<DataType>& labels_oos,
				    ClassifierContext::Context context) : 
    GradientBoostClassifier<DataType>(dataset_is, labels_is, context)
  {
    hasOOSData_ = true;
    dataset_oos_ = dataset_oos;
    labels_oos_ = labels_oos;
  }

  void fit();

  void Classify(const mat&, Row<DataType>&);

  // 3 Predict methods
  // predict on member dataset; loop through and sum step prediction vectors
  void Predict(Row<DataType>&);
  // predict on subset of dataset defined by uvec; sum step prediction vectors
  void Predict(Row<DataType>&, const uvec&);
  // predict OOS, loop through and call Classify_ on individual classifiers, sum
  void Predict(const mat&, Row<DataType>&);

  mat getDataset() const { return dataset_; }
  Row<DataType> getLabels() const { return labels_; }

private:
  void init_();
  Row<DataType> _constantLeaf() const;
  Row<DataType> _randomLeaf(std::size_t numVals=20) const;
  uvec subsampleRows(size_t);
  uvec subsampleCols(size_t);
  void symmetrizeLabels();
  void symmetrize(Row<DataType>&);
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
  mat dataset_oos_;
  Row<DataType> labels_oos_;
  std::size_t partitionSize_;
  double partitionRatio_;

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

  double a_;
  double b_;

  int current_classifier_ind_;

  LeavesList leaves_;
  ClassifierList classifiers_;
  PartitionList partitions_;
  PredictionList predictions_;
  MaskList rowMasks_;
  MaskList colMasks_;

  std::mt19937 mersenne_engine_{std::random_device{}()};
  std::default_random_engine default_engine_;
  std::uniform_int_distribution<std::size_t> partitionDist_{1, 
      static_cast<std::size_t>(m_ * col_subsample_ratio_)};
  // call by partitionDist_(default_engine_)

  bool symmetrized_;

  bool preExtrapolate_;
  bool postExtrapolate_;

  bool hasOOSData_;
};

#include "gradientboostclassifier_impl.hpp"

#endif
