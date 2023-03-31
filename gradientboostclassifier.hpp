#ifndef __GRADIENTBOOSTCLASSIFIER_HPP__
#define __GRADIENTBOOSTCLASSIFIER_HPP__

// #define DEBUG() __debug dd{__FILE__, __FUNCTION__, __LINE__};

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
#include <typeinfo>
#include <chrono>
#include <limits>
#include <exception>

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/access.hpp>

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

#include "loss.hpp"
#include "score2.hpp"
#include "LTSS.hpp"
#include "DP.hpp"
#include "utils.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

using namespace Objectives;
using namespace LossMeasures;
using namespace ClassifierContext;
using namespace PartitionSize;
using namespace LearningRate;

struct predictionAfterClearedClassifiersException : public std::exception {
  const char* what() const throw () {
    return "Attempting to predict on a classifier that has been serialized and cleared";
  };
};

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

class __debug {
public:
  __debug(const char* fl, const char* fn, int ln) :
    fl_{fl},
    fn_{fn},
    ln_{ln} 
  {
    std::cerr << "===> ENTER FILE: " << fl_
	      << " FUNCTION: " << fn_
	      <<" LINE: " << ln_ << std::endl;
  }
  ~__debug() {
    std::cerr << "===< EXIT FILE:  " << fl_
	      << " FUNCTION: " << fn_
	      <<" LINE: " << ln_ << std::endl;
  }
private:
  const char* fl_;
  const char* fn_;
  int ln_;
};

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
  ClassifierBase() = default;
  ClassifierBase(std::string id) : id_{id} {}
  virtual void Classify_(const mat&, Row<DataType>&) = 0;
  virtual void purge() = 0;

  std::string get_id() const { return id_; }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(id_);
  }
private:
  std::string id_;
};

template<typename DataType, typename ClassifierType, typename... Args>
class DiscreteClassifierBase : public ClassifierBase<DataType, ClassifierType> {
public:
  using LeavesMap = std::unordered_map<std::size_t, DataType>;

  DiscreteClassifierBase(const mat& dataset, Row<DataType>& labels, Args&&... args) : 
    ClassifierBase<DataType, ClassifierType>(typeid(*this).name())
  {

    labels_t_ = Row<std::size_t>(labels.n_cols);
    encode(labels, labels_t_);
    setClassifier(dataset, labels_t_, std::forward<Args>(args)...);
    args_ = std::tuple<Args...>(args...);
    
    // Check error
    /*
      Row<std::size_t> prediction;
      classifier_->Classify(dataset, prediction);
      const double trainError = err(prediction, labels_t_);
      for (size_t i=0; i<25; ++i)
      std::cout << labels_t_[i] << " ::(1) " << prediction[i] << std::endl;
      std::cout << "dataset size:    " << dataset.n_rows << " x " << dataset.n_cols << std::endl;
      std::cout << "prediction size: " << prediction.n_rows << " x " << prediction.n_cols << std::endl;
      std::cout << "Training error (1): " << trainError << "%." << std::endl;
    */
  
  }

  DiscreteClassifierBase(const LeavesMap& leavesMap, std::unique_ptr<ClassifierType> classifier) : 
    leavesMap_{leavesMap},
    classifier_{std::move(classifier)} {}

  DiscreteClassifierBase() = default;
  ~DiscreteClassifierBase() = default;

  void setClassifier(const mat&, Row<std::size_t>&, Args&&...);
  void Classify_(const mat&, Row<DataType>&) override;
  void Classify_(const mat&, Row<DataType>&, mat&);
  void purge() override;

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::base_class<ClassifierBase<DataType, ClassifierType>>(this), CEREAL_NVP(leavesMap_));
    ar(cereal::base_class<ClassifierBase<DataType, ClassifierType>>(this), CEREAL_NVP(classifier_));
  }

private:
  void encode(const Row<DataType>&, Row<std::size_t>&); 
  void decode(const Row<std::size_t>&, Row<DataType>&);

  Row<std::size_t> labels_t_;
  LeavesMap leavesMap_;
  std::unique_ptr<ClassifierType> classifier_;
  std::tuple<Args...> args_;

};

template<typename DataType, typename ClassifierType, typename... Args>
class ContinuousClassifierBase : public ClassifierBase<DataType, ClassifierType> {
public:  
  ContinuousClassifierBase(const mat& dataset, Row<DataType>& labels, Args&&... args) : 
    ClassifierBase<DataType, ClassifierType>(typeid(*this).name()) 
  {

    setClassifier(dataset, labels, std::forward<Args>(args)...);
    args_ = std::tuple<Args...>(args...);
  }

  ContinuousClassifierBase(std::unique_ptr<ClassifierType> classifier) : classifier_{std::move(classifier)} {}

  ContinuousClassifierBase() = default;
  ~ContinuousClassifierBase() = default;
  
  void setClassifier(const mat&, Row<DataType>&, Args&&...);
  void Classify_(const mat&, Row<DataType>&) override;
  void purge() override {};

  template<class Archive>
  void serialize(Archive &ar) {

    ar(cereal::base_class<ClassifierBase<DataType, ClassifierType>>(this), CEREAL_NVP(classifier_));
  }

private:
  std::unique_ptr<ClassifierType> classifier_;
  std::tuple<Args...> args_;
};

class RandomForestClassifier : public DiscreteClassifierBase<double, 
							     ClassifierTypes::RandomForestClassifierType,
							     std::size_t,
							     std::size_t,
							     std::size_t> {
public:
  RandomForestClassifier() = default;
  
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
  DecisionTreeClassifier() = default;

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
  DecisionTreeRegressorClassifier() = default;

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
  using ClassifierList = std::vector<std::unique_ptr<ClassifierBase<DataType, Classifier>>>;

  using Partition = std::vector<std::vector<int>>;
  using PartitionList = std::vector<Partition>;

  using Leaves = Row<double>;
  using LeavesList = std::vector<Leaves>;
  using Prediction = Row<double>;
  using PredictionList = std::vector<Prediction>;
  
  GradientBoostClassifier() = default;
  
  // 1
  GradientBoostClassifier(const mat& dataset, 
			  const Row<std::size_t>& labels,
			  ClassifierContext::Context context) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::classifier>(typeid(*this).name()),
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    hasOOSData_{false},
    hasInitialPrediction_{false},
    reuseColMask_{false}
  { 
    contextInit_(std::move(context));
    init_(); 
  }

  // 2
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  ClassifierContext::Context context) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::classifier>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    hasOOSData_{false},
    hasInitialPrediction_{false},
    reuseColMask_{false}
  { 
    contextInit_(std::move(context));
    init_(); 
  }

  // 3
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const mat& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  ClassifierContext::Context context) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::classifier>(typeid(*this).name()),
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    dataset_oos_{dataset_oos},
    labels_oos_{conv_to<Row<double>>::from(labels_oos)},
    hasOOSData_{true},
    hasInitialPrediction_{false},
    reuseColMask_{false}
  {
    contextInit_(std::move(context));
    init_();
  }

  // 4
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  ClassifierContext::Context context) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::classifier>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    dataset_oos_{dataset_oos},
    labels_oos_{conv_to<Row<double>>::from(labels_oos)},
    hasOOSData_{true},
    hasInitialPrediction_{false},
    reuseColMask_{false}
  {
    contextInit_(std::move(context));
    init_();
  }

  // 5
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  ClassifierContext::Context context) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::classifier>(typeid(*this).name()),
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    hasOOSData_{false},
    hasInitialPrediction_{true},
    reuseColMask_{true},
    latestPrediction_{latestPrediction},
    colMask_{colMask}
  {
    contextInit_(std::move(context));
    init_();
  }

  // 6
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const Row<double>& latestPrediction,
			  ClassifierContext::Context context) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::classifier>(typeid(*this).name()),
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    hasOOSData_{false},
    hasInitialPrediction_{true},
    reuseColMask_{false},
    latestPrediction_{latestPrediction}
  {
    contextInit_(std::move(context));
    init_();
  }
   
  // 7
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  ClassifierContext::Context context) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::classifier>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    hasOOSData_{false},
    hasInitialPrediction_{true},
    reuseColMask_{true},
    latestPrediction_{latestPrediction},
    colMask_{colMask}
  {
    contextInit_(std::move(context));
    init_();
  }

  // 8
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const Row<double>& latestPrediction,
			  ClassifierContext::Context context) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::classifier>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    hasOOSData_{false},
    hasInitialPrediction_{true},
    reuseColMask_{false},
    latestPrediction_{latestPrediction}
  {
    contextInit_(std::move(context));
    init_();
  }

  // 9
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const mat& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  ClassifierContext::Context context) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::classifier>(typeid(*this).name()),
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    dataset_oos_{dataset_oos},
    labels_oos_{conv_to<Row<double>>::from(labels_oos)},
    hasOOSData_{true},
    hasInitialPrediction_{true},
    reuseColMask_{true},
    latestPrediction_{latestPrediction},
    colMask_{colMask}
  {
    contextInit_(std::move(context));
    init_();
  }

  // 10
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const mat& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  const Row<double>& latestPrediction,
			  ClassifierContext::Context context) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::classifier>(typeid(*this).name()),
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    dataset_oos_{dataset_oos},
    labels_oos_{conv_to<Row<double>>::from(labels_oos)},
    hasOOSData_{true},
    hasInitialPrediction_{true},
    reuseColMask_{false},
    latestPrediction_{latestPrediction}
  {
    contextInit_(std::move(context));
    init_();
  }

  // 11
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  ClassifierContext::Context context) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::classifier>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    dataset_oos_{dataset_oos},
    labels_oos_{labels_oos},
    hasOOSData_{true},
    hasInitialPrediction_{true},
    reuseColMask_{true},
    latestPrediction_{latestPrediction},
    colMask_{colMask}
  {
    contextInit_(std::move(context));
    init_();
  }

  // 12
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  const Row<double>& latestPrediction,
			  ClassifierContext::Context context) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::classifier>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    dataset_oos_{dataset_oos},
    labels_oos_{labels_oos},
    hasOOSData_{true},
    hasInitialPrediction_{true},
    reuseColMask_{false},
    latestPrediction_{latestPrediction}
  {
    contextInit_(std::move(context));
    init_();
  }

  void fit();

  virtual void Classify(const mat&, Row<DataType>&);

  // 4 Predict methods
  // predict on member dataset; use latestPrediction_
  virtual void Predict(Row<DataType>&);
  // predict on subset of dataset defined by uvec; sum step prediction vectors
  virtual void Predict(Row<DataType>&, const uvec&);
  // predict OOS, loop through and call Classify_ on individual classifiers, sum
  virtual void Predict(const mat&, Row<DataType>&, bool=false);

  // 3 overloaded versions of above based based on label datatype
  virtual void Predict(Row<IntegralLabelType>&);
  virtual void Predict(Row<IntegralLabelType>&, const uvec&);
  virtual void Predict(const mat&, Row<IntegralLabelType>&);

  // 2 overloaded versions for archive classifier
  virtual void Predict(std::string, Row<DataType>&, bool=false);
  virtual void Predict(std::string, const mat&, Row<DataType>&, bool=false);

  void Classify_(const mat& dataset, Row<DataType>& prediction) { 
    Predict(dataset, prediction); 
  }

  mat getDataset() const { return dataset_; }
  Row<DataType> getLatestPrediction() const { return latestPrediction_; }
  int getNRows() const { return n_; }
  int getNCols() const { return m_; }
  Row<double> getLabels() const { return labels_; }
  ClassifierList getClassifiers() const { return classifiers_; }

  std::string getIndexName() const { return indexName_; }

  std::pair<double, double> getAB() const {return std::make_pair(a_, b_); }
  
  virtual void printStats(int);
  void purge();
  std::string write();  
  std::string writeDataset();
  std::string writeDatasetOOS();
  std::string writeLabels();
  std::string writeLabelsOOS();
  std::string writePrediction();
  std::string writeColMask();
  void read(GradientBoostClassifier&, std::string);
  void commit();
  void checkAccuracyOfArchive();

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this), CEREAL_NVP(classifiers_));
    ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this), symmetrized_);
    ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this), a_);
    ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this), b_);
    // ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this), latestPrediction_);
  }

private:
  void childContext(ClassifierContext::Context&, std::size_t, double, std::size_t);
  void contextInit_(ClassifierContext::Context&&);
  void init_();
  Row<double> _constantLeaf() const;
  Row<double> _randomLeaf() const;
  uvec subsampleRows(size_t);
  uvec subsampleCols(size_t);
  void symmetrizeLabels(Row<DataType>&);
  void symmetrizeLabels();
  Row<DataType> uniqueCloseAndReplace(Row<DataType>&);
  void symmetrize(Row<DataType>&);
  void deSymmetrize(Row<DataType>&);
  void fit_step(std::size_t);
  double computeLearningRate(std::size_t);
  std::size_t computePartitionSize(std::size_t, const uvec&);

  double computeSubLearningRate(std::size_t);
  std::size_t computeSubPartitionSize(std::size_t);
  std::size_t computeSubStepSize(std::size_t);

  void updateClassifiers(std::unique_ptr<ClassifierBase<DataType, Classifier>>&&, Row<DataType>&);

  std::pair<rowvec,rowvec> generate_coefficients(const Row<DataType>&, const uvec&);
  std::pair<rowvec,rowvec> generate_coefficients(const Row<DataType>&, const Row<DataType>&, const uvec&);
  Leaves computeOptimalSplit(rowvec&, rowvec&, std::size_t, std::size_t, double, const uvec&);

  void setNextClassifier(const ClassifierType&);
  int steps_;
  int baseSteps_;
  mat dataset_;
  Row<double> labels_;
  mat dataset_oos_;
  Row<double> labels_oos_;
  std::size_t partitionSize_;
  double partitionRatio_;
  Row<DataType> latestPrediction_;
  std::vector<std::string> fileNames_;

  lossFunction loss_;
  LossFunction<double>* lossFn_;
  
  double learningRate_;

  PartitionSize::PartitionSizeMethod partitionSizeMethod_;
  LearningRate::LearningRateMethod learningRateMethod_;
  StepSize::StepSizeMethod stepSizeMethod_;

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
  bool quietRun_;
  bool reuseColMask_;

  bool recursiveFit_;
  bool serialize_;
  bool serializePrediction_;
  bool serializeColMask_;
  bool serializeDataset_;
  bool serializeLabels_;

  bool hasOOSData_;
  bool hasInitialPrediction_;

  std::size_t serializationWindow_;
  std::string indexName_;
};

using DTC = ClassifierTypes::DecisionTreeClassifierType;
using CTC = ClassifierTypes::DecisionTreeRegressorType;
using RFC = ClassifierTypes::RandomForestClassifierType;
using DiscreteClassifierBaseDTC = DiscreteClassifierBase<double, 
						       DTC, 
						       std::size_t,
						       std::size_t,
						       double,
						       std::size_t>;
using DiscreteClassifierBaseRFC = DiscreteClassifierBase<double,
							 RFC,
							 std::size_t,
							 std::size_t,
							 std::size_t>;
using ContinuousClassifierBaseD = ContinuousClassifierBase<double, 
							   CTC,
							   unsigned long,
							   double,
							   unsigned long>;
using GradientBoostClassifierD = GradientBoostClassifier<DTC>;
using ClassifierBaseDD = ClassifierBase<double, DTC>;
using ClassifierBaseRD = ClassifierBase<double, RFC>;
using ClassifierBaseCD = ClassifierBase<double, CTC>;

// Register class with cereal
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTC);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFC);
CEREAL_REGISTER_TYPE(ContinuousClassifierBaseD);

CEREAL_REGISTER_TYPE(DecisionTreeClassifier);
CEREAL_REGISTER_TYPE(RandomForestClassifier);
CEREAL_REGISTER_TYPE(DecisionTreeRegressorClassifier);

CEREAL_REGISTER_TYPE(GradientBoostClassifier<DecisionTreeClassifier>);

// Register class hierarchy with cereal
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDD, GradientBoostClassifierD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDD, DecisionTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRD, RandomForestClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseCD, DecisionTreeRegressorClassifier);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDD, DiscreteClassifierBaseDTC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRD, DiscreteClassifierBaseRFC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseCD, ContinuousClassifierBaseD);

template<typename DataType, typename ClassifierType, typename... Args>
class ContinuousClassifierBase;

template<typename DataType, typename ClassifierType, typename... Args>
class DiscreteClassifierBase;

template<typename DataType>
using LeavesMap = std::unordered_map<std::size_t, DataType>;

namespace cereal {
  
  template<typename DataType>
  using LeavesMap = std::unordered_map<std::size_t, DataType>;

  template<typename DataType, typename ClassifierType, typename... Args> 
  struct LoadAndConstruct<ContinuousClassifierBase<DataType, ClassifierType, Args...>> {
    template<class Archive>
    static void load_and_construct(Archive &ar, cereal::construct<ContinuousClassifierBase<DataType, ClassifierType, Args...>> &construct) {
      std::unique_ptr<ClassifierType> classifier;
      ar(CEREAL_NVP(classifier));
      construct(std::move(classifier));
    }
  };


  template<typename DataType, typename ClassifierType, typename... Args>
  struct LoadAndConstruct<DiscreteClassifierBase<DataType, ClassifierType, Args...>> {
    template<class Archive>
    static void load_and_construct(Archive &ar, cereal::construct<DiscreteClassifierBase<DataType, ClassifierType, Args...>> &construct) {
      LeavesMap<DataType> leavesMap;
      std::unique_ptr<ClassifierType> classifier;
      ar(CEREAL_NVP(leavesMap));
      ar(CEREAL_NVP(classifier));
      construct(leavesMap, std::move(classifier));
    }
  };

} // namespace cereal
  

#include "gradientboostclassifier_impl.hpp"

#endif
