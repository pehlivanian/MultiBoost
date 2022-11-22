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
      MULTISCALE = 5
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
    Context(std::size_t minLeafSize=1,
	    double minimumGainSplit=0.0,
	    std::size_t maxDepth=100,
	    std::size_t numTrees=10) : 
      minLeafSize{minLeafSize},
      minimumGainSplit{minimumGainSplit},
      maxDepth{maxDepth},
      numTrees{numTrees} {}
      
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
    std::size_t minLeafSize;
    double minimumGainSplit;
    std::size_t maxDepth;
    std::size_t numTrees;
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

  static std::vector<std::vector<int>> _fullPartition(int sz) {
    std::vector<int> subset{sz};
    std::iota(subset.begin(), subset.end(), 0);
    std::vector<std::vector<int>> p{1, subset};
    return p;
  }
};

/**********************/
/* CLASSIFIER CLASSES */
/**********************/

namespace ClassifierTypes {
  using DecisionTreeRegressorType = DecisionTreeRegressor<MADGain, BestBinaryNumericSplit>;
  using RandomForestClassifierType = RandomForest<>;
  using DecisionTreeClassifierType = DecisionTree<>;

  // using DecisionTreeClassifierType = DecisionTree<GiniGain, BestBinaryNumericSplit>;
  // using DecisionTreeClassifierType = DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  // using DecisionTreeClassifierType = DecisionTreeRegressor<MADGain>;
  // using DecisionTreeClassifierType = DecisionTreeRegressor<>;
  // using DecisionTreeClassifierType = DecisionTreeRegressor<MSEGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  // using DecisionTreeClassifierType = DecisionTreeRegressor<InformationGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  
};

template<typename DataType, typename ClassifierType, typename... Args>
class ClassifierBase {
public:
  using data_type = DataType;

  ClassifierBase() = default;
  ~ClassifierBase() = default;

};

template<typename DataType, typename ClassifierType, typename... Args>
class DiscreteClassifierBase : public ClassifierBase<DataType, ClassifierType, Args...> {
public:
  using LeavesMap = std::unordered_map<std::size_t, DataType>;

  DiscreteClassifierBase(const mat& dataset, Row<DataType>& labels, Args&&... args) : 
    ClassifierBase<DataType, ClassifierType, Args...>()
  {
    dataset_ = dataset;
    labels_t_ = Row<std::size_t>(labels.n_cols);
    encode(labels, labels_t_);
    setClassifier(dataset_, labels_t_, std::forward<Args>(args)...);
    
    // Check error
    /*
      Row<std::size_t> prediction;
      classifier_->Classify(dataset, prediction);
      const double trainError = arma::accu(prediction != labels_t_) * 100. / labels_t_.n_elem;
      for (size_t i=0; i<25; ++i)
      std::cout << labels_t_[i] << " ::(1) " << prediction[i] << std::endl;
      std::cout << "dataset size:    " << dataset.n_rows << " x " << dataset.n_cols << std::endl;
      std::cout << "prediction size: " << prediction.n_rows << " x " << prediction.n_cols << std::endl;
      std::cout << "Training error (1): " << trainError << "%." << std::endl;
    */
  
  }

  ~DiscreteClassifierBase() = default;

  void setClassifier(const mat&, Row<std::size_t>&, Args&&...);
  void Classify_(const mat&, Row<DataType>&);

private:
  void encode(const Row<DataType>&, Row<std::size_t>&); 
  void decode(const Row<std::size_t>&, Row<DataType>&);

  mat dataset_;
  Row<std::size_t> labels_t_;
  LeavesMap leavesMap_;
  std::unique_ptr<ClassifierType> classifier_;

};

template<typename DataType, typename ClassifierType, typename... Args>
class ContinuousClassifierBase : public ClassifierBase<DataType, ClassifierType, Args...> {
public:  
  ContinuousClassifierBase(const mat& dataset, Row<DataType>& labels, Args&&... args) : 
    ClassifierBase<DataType, ClassifierType, Args...>() 
  {
    setClassifier(dataset, labels, std::forward<Args>(args)...);
  }
  ~ContinuousClassifierBase() = default;
  
  void setClassifier(const mat&, Row<DataType>&, Args&&...);
  void Classify_(const mat&, Row<DataType>&);

private:
  std::unique_ptr<ClassifierType> classifier_;
};

class RandomForestClassifier : public DiscreteClassifierBase<double, 
							     ClassifierTypes::RandomForestClassifierType,
							     std::size_t,
							     std::size_t,
							     std::size_t> {
public:
  
  RandomForestClassifier(const mat& dataset, 
			 Row<double>& labels, 
			 std::size_t numClasses,
			 std::size_t numTrees,
			 std::size_t minLeafSize) : 
    DiscreteClassifierBase<double, 
			   ClassifierTypes::RandomForestClassifierType, 
			   std::size_t,
			   std::size_t, 
			   std::size_t>(dataset, 
					labels, 
					std::move(numClasses),
					std::move(numTrees),
					std::move(minLeafSize))
  {}
  
};

class DecisionTreeClassifier : public DiscreteClassifierBase<double, 
							     ClassifierTypes::DecisionTreeClassifierType,
							     std::size_t,
							     std::size_t,
							     double,
							     std::size_t> {
public:
  DecisionTreeClassifier(const mat& dataset, 
			 Row<double>& labels,
			 std::size_t numClasses,
			 std::size_t minLeafSize,
			 double minimumGainSplit,
			 std::size_t maxDepth) : 
    DiscreteClassifierBase<double, 
			   ClassifierTypes::DecisionTreeClassifierType, 
			   std::size_t, 
			   std::size_t,
			   double,
			   std::size_t>(dataset, 
					labels, 
					std::move(numClasses), 
					std::move(minLeafSize),
					std::move(minimumGainSplit), 
					std::move(maxDepth))
    {}
    
  };

class DecisionTreeRegressorClassifier : public ContinuousClassifierBase<double, 
									ClassifierTypes::DecisionTreeRegressorType,
									unsigned long,
									double,
									unsigned long> {
public:

  const unsigned long minLeafSize = 1;
  const double minGainSplit = 0.0;
  const unsigned long maxDepth = 100;

  DecisionTreeRegressorClassifier(const mat& dataset,
				  rowvec& labels,
				  unsigned long minLeafSize=5,
				  double minGainSplit=0.,
				  unsigned long maxDepth=5) : 
    ContinuousClassifierBase<double, 
			     ClassifierTypes::DecisionTreeRegressorType,
			     unsigned long,
			     double,
			     unsigned long>(dataset, 
					    labels, 
					    std::move(minLeafSize), 
					    std::move(minGainSplit), 
					    std::move(maxDepth))
  {}
  
};


class LeafOnlyClassifier {
public:
  LeafOnlyClassifier(const mat& dataset, const Row<double>& leaves)
  {
    leaves_ = Row<double>{leaves}; 
  }
  
  void Classify_(const mat& dataset, Row<double>& labels) {
    labels = leaves_;
  }
private:
  Row<double> leaves_;
};

template<typename T>
struct classifier_traits {
  using datatype = double;
  using labeltype = std::size_t;
};


class GradientBoostClassifier {
public:

  using ClassifierType = DecisionTreeClassifier;
  // using ClassifierType = RandomForestClassifier;
  using DataType = classifier_traits<ClassifierType>::datatype;
  using LabelType = classifier_traits<ClassifierType>::labeltype;

  using Partition = std::vector<std::vector<int>>;
  using PartitionList = std::vector<Partition>;
  using ClassifierList = std::vector<std::unique_ptr<ClassifierType>>;
  using Leaves = Row<double>;
  using LeavesList = std::vector<Leaves>;
  using Prediction = Row<double>;
  using PredictionList = std::vector<Prediction>;
  using MaskList = std::vector<uvec>;
  
  GradientBoostClassifier(const mat& dataset, 
			  const Row<LabelType>& labels, 
			  lossFunction loss,
			  std::size_t partitionSize,
			  double learningRate,
			  int steps,
			  bool symmetrizeLabels) : 
    dataset_{dataset},
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
  { 
    labels_ = conv_to<Row<double>>::from(labels);    
    init_(); 
  }

  GradientBoostClassifier(const mat& dataset, 
			  const Row<LabelType>& labels,
			  ClassifierContext::Context context) :
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
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
    current_classifier_ind_{0},
    minLeafSize_{context.minLeafSize},
    minimumGainSplit_{context.minimumGainSplit},
    maxDepth_{context.maxDepth},
    numTrees_{context.numTrees}
  { 
    init_(); 
  }

  GradientBoostClassifier(const mat& dataset_is,
			  const Row<LabelType>& labels_is,
			  const mat& dataset_oos,
			  const Row<LabelType>& labels_oos,
			  ClassifierContext::Context context) : 
    GradientBoostClassifier(dataset_is, labels_is, context)
  {
    hasOOSData_ = true;
    dataset_oos_ = dataset_oos;
    labels_oos_ = conv_to<Row<double>>::from(labels_oos);
  }

  void fit();

  void Classify(const mat&, Row<double>&);

  // 3 Predict methods
  // predict on member dataset; loop through and sum step prediction vectors
  void Predict(Row<double>&);
  // predict on subset of dataset defined by uvec; sum step prediction vectors
  void Predict(Row<double>&, const uvec&);
  // predict OOS, loop through and call Classify_ on individual classifiers, sum
  void Predict(const mat&, Row<double>&);

  // overloaded versions of above
  void Predict(Row<LabelType>&);
  void Predict(Row<LabelType>&, const uvec&);
  void Predict(const mat&, Row<LabelType>&);

  mat getDataset() const { return dataset_; }
  Row<double> getLabels() const { return labels_; }

private:
  void init_();
  Row<double> _constantLeaf() const;
  Row<double> _randomLeaf(std::size_t numVals=20) const;
  uvec subsampleRows(size_t);
  uvec subsampleCols(size_t);
  void symmetrizeLabels();
  void symmetrize(Row<double>&);
  void deSymmetrize(Row<double>&);
  void fit_step(std::size_t);
  double computeLearningRate(std::size_t);
  std::size_t computePartitionSize(std::size_t, const uvec&);

  std::pair<rowvec,rowvec>  generate_coefficients(const Row<double>&, const uvec&);
  Leaves computeOptimalSplit(rowvec&, rowvec&, mat, std::size_t, std::size_t, const uvec&);

  void setNextClassifier(const ClassifierType&);
  int steps_;
  mat dataset_;
  Row<double> labels_;
  mat dataset_oos_;
  Row<double> labels_oos_;
  std::size_t partitionSize_;
  double partitionRatio_;

  lossFunction loss_;
  LossFunction<double>* lossFn_;
  
  double learningRate_;

  PartitionSize::SizeMethod partitionSizeMethod_;
  LearningRate::RateMethod learningRateMethod_;

  double row_subsample_ratio_;
  double col_subsample_ratio_;

  int n_;
  int m_;

  double a_;
  double b_;

  int current_classifier_ind_;

  std::size_t minLeafSize_;
  double minimumGainSplit_;
  std::size_t maxDepth_;
  std::size_t numTrees_;

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
