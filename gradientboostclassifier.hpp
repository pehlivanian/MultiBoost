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
#include <type_traits>
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
	    std::size_t numTrees=10,
	    bool recursiveFit=false) : 
      minLeafSize{minLeafSize},
      minimumGainSplit{minimumGainSplit},
      maxDepth{maxDepth},
      numTrees{numTrees},
      removeRedundantLabels{false},
      recursiveFit{recursiveFit},
      hasOOSData{false},
      reuseColMask{false}
    {}
      
    lossFunction loss;
    std::size_t partitionSize;
    double partitionRatio = .5;
    double learningRate;
    int steps;
    bool symmetrizeLabels;
    bool removeRedundantLabels;
    double rowSubsampleRatio;
    double colSubsampleRatio;
    bool recursiveFit;
    PartitionSize::SizeMethod partitionSizeMethod;
    LearningRate::RateMethod learningRateMethod;
    std::size_t minLeafSize;
    double minimumGainSplit;
    std::size_t maxDepth;
    std::size_t numTrees;
    bool hasOOSData;
    bool reuseColMask;
    mat dataset_oos;
    Row<double> labels_oos;
    uvec colMask;
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
    std::vector<int> subset(sz);
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

template<typename DataType, typename ClassifierType>
class ClassifierBase {
public:
  virtual void Classify_(const mat&, Row<DataType>&) = 0;
  virtual void purge() = 0;
};

template<typename ClassifierType>
class AggregatorBase {
  ;
};

template<typename DataType, typename ClassifierType, typename... Args>
class DiscreteClassifierBase : public ClassifierBase<DataType, ClassifierType> {
public:
  using LeavesMap = std::unordered_map<std::size_t, DataType>;

  DiscreteClassifierBase(const mat& dataset, Row<DataType>& labels, Args&&... args) : 
    ClassifierBase<DataType, ClassifierType>()
  {
    labels_t_ = Row<std::size_t>(labels.n_cols);
    encode(labels, labels_t_);
    setClassifier(dataset, labels_t_, std::forward<Args>(args)...);
    
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
  void Classify_(const mat&, Row<DataType>&) override;
  void purge() override;

private:
  void encode(const Row<DataType>&, Row<std::size_t>&); 
  void decode(const Row<std::size_t>&, Row<DataType>&);

  Row<std::size_t> labels_t_;
  LeavesMap leavesMap_;
  std::unique_ptr<ClassifierType> classifier_;

};

template<typename DataType, typename ClassifierType, typename... Args>
class ContinuousClassifierBase : public ClassifierBase<DataType, ClassifierType> {
public:  
  ContinuousClassifierBase(const mat& dataset, Row<DataType>& labels, Args&&... args) : 
    ClassifierBase<DataType, ClassifierType>() 
  {
    setClassifier(dataset, labels, std::forward<Args>(args)...);
  }
  ~ContinuousClassifierBase() = default;
  
  void setClassifier(const mat&, Row<DataType>&, Args&&...);
  void Classify_(const mat&, Row<DataType>&) override;
  void purge() override {};

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

template<typename T>
struct classifier_traits {
  using datatype = double;
  using integrallabeltype = std::size_t;
  using classifier = ClassifierTypes::DecisionTreeClassifierType;
};

template<>
struct classifier_traits<DecisionTreeClassifier> {
  using datatype = double;
  using integrallabeltype = std::size_t;
  using classifier = ClassifierTypes::DecisionTreeClassifierType;
};


template<typename ClassifierType>
class GradientBoostClassifier : public ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
						      typename classifier_traits<ClassifierType>::classifier> {
public:

  using DataType = typename classifier_traits<ClassifierType>::datatype;
  using IntegralLabelType = typename classifier_traits<ClassifierType>::integrallabeltype;
  using Classifier = typename classifier_traits<ClassifierType>::classifier;

  using Partition = std::vector<std::vector<int>>;
  using PartitionList = std::vector<Partition>;
  using ClassifierList = std::vector<std::unique_ptr<ClassifierBase<DataType, Classifier>>>;
  using Leaves = Row<double>;
  using LeavesList = std::vector<Leaves>;
  using Prediction = Row<double>;
  using PredictionList = std::vector<Prediction>;
  using MaskList = std::vector<uvec>;
  
  GradientBoostClassifier(const mat& dataset, 
			  const Row<std::size_t>& labels,
			  ClassifierContext::Context context) :
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    loss_{context.loss},
    partitionSize_{context.partitionSize},
    partitionRatio_{context.partitionRatio},
    learningRate_{context.learningRate},
    steps_{context.steps},
    symmetrized_{context.symmetrizeLabels},
    removeRedundantLabels_{context.removeRedundantLabels},
    row_subsample_ratio_{context.rowSubsampleRatio},
    col_subsample_ratio_{context.colSubsampleRatio},
    recursiveFit_{context.recursiveFit},
    partitionSizeMethod_{context.partitionSizeMethod},
    learningRateMethod_{context.learningRateMethod},
    minLeafSize_{context.minLeafSize},
    minimumGainSplit_{context.minimumGainSplit},
    maxDepth_{context.maxDepth},
    numTrees_{context.numTrees},
    reuseColMask_{context.reuseColMask}
  { 
    if (hasOOSData_ = context.hasOOSData) {
      dataset_oos_ = context.dataset_oos;
      labels_oos_ = conv_to<Row<double>>::from(context.labels_oos);      
    }
    if (reuseColMask_ = context.reuseColMask) {
      colMask_ = context.colMask;
    }
    init_(); 
  }

  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  ClassifierContext::Context context) :

    dataset_{dataset},
    labels_{labels},
    loss_{context.loss},
    partitionSize_{context.partitionSize},
    partitionRatio_{context.partitionRatio},
    learningRate_{context.learningRate},
    steps_{context.steps},
    symmetrized_{context.symmetrizeLabels},
    removeRedundantLabels_{context.removeRedundantLabels},
    row_subsample_ratio_{context.rowSubsampleRatio},
    col_subsample_ratio_{context.colSubsampleRatio},
    recursiveFit_{context.recursiveFit},
    partitionSizeMethod_{context.partitionSizeMethod},
    learningRateMethod_{context.learningRateMethod},
    minLeafSize_{context.minLeafSize},
    minimumGainSplit_{context.minimumGainSplit},
    maxDepth_{context.maxDepth},
    numTrees_{context.numTrees},
    reuseColMask_{context.reuseColMask}
  { 
    if (hasOOSData_ = context.hasOOSData) {
      dataset_oos_ = context.dataset_oos;
      labels_oos_ = labels;
    }
    if (reuseColMask_ = context.reuseColMask) {
      colMask_ = context.colMask;
    }
    init_(); 
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

  // overloaded versions of above
  void Predict(Row<IntegralLabelType>&);
  void Predict(Row<IntegralLabelType>&, const uvec&);
  void Predict(const mat&, Row<IntegralLabelType>&);

  virtual void Classify_(const mat& dataset, Row<DataType>& prediction) { 
    Predict(dataset, prediction); 
  }

  mat getDataset() const { return dataset_; }
  Row<double> getLabels() const { return labels_; }
  void printStats(int);
  void purge();
  

private:
  void init_();
  Row<double> _constantLeaf() const;
  Row<double> _randomLeaf(std::size_t numVals=20) const;
  uvec subsampleRows(size_t);
  uvec subsampleCols(size_t);
  void symmetrizeLabels();
  Row<DataType> uniqueCloseAndReplace(Row<DataType>&);
  void symmetrize(Row<DataType>&);
  void deSymmetrize(Row<DataType>&);
  void fit_step(std::size_t);
  double computeLearningRate(std::size_t);
  std::size_t computePartitionSize(std::size_t, const uvec&);
  void updateClassifiers(std::unique_ptr<ClassifierBase<DataType, Classifier>>&&, Row<DataType>&);

  std::pair<rowvec,rowvec> generate_coefficients(const Row<DataType>&, const uvec&);
  std::pair<rowvec,rowvec> generate_coefficients(const Row<DataType>&, const Row<DataType>&, const uvec&);
  Leaves computeOptimalSplit(rowvec&, rowvec&, mat, std::size_t, std::size_t, const uvec&);

  void setNextClassifier(const ClassifierType&);
  int steps_;
  mat dataset_;
  Row<double> labels_;
  mat dataset_oos_;
  Row<double> labels_oos_;
  std::size_t partitionSize_;
  double partitionRatio_;
  Row<DataType> latestPrediction_;

  lossFunction loss_;
  LossFunction<double>* lossFn_;
  
  double learningRate_;

  PartitionSize::SizeMethod partitionSizeMethod_;
  LearningRate::RateMethod learningRateMethod_;

  double row_subsample_ratio_;
  double col_subsample_ratio_;

  uvec colMask_;

  int n_;
  int m_;

  double a_;
  double b_;

  std::size_t minLeafSize_;
  double minimumGainSplit_;
  std::size_t maxDepth_;
  std::size_t numTrees_;

  ClassifierList classifiers_;
  PartitionList partitions_;
  PredictionList predictions_;

  std::mt19937 mersenne_engine_{std::random_device{}()};
  std::default_random_engine default_engine_;
  std::uniform_int_distribution<std::size_t> partitionDist_{1, 
      static_cast<std::size_t>(m_ * col_subsample_ratio_)};
  // call by partitionDist_(default_engine_)

  bool symmetrized_;
  bool removeRedundantLabels_;
  bool reuseColMask_;

  bool recursiveFit_;

  bool hasOOSData_;
};

#include "gradientboostclassifier_impl.hpp"

#endif
