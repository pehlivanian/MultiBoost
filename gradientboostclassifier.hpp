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
#include "DP.hpp"
#include "utils.hpp"
#include "model.hpp"
#include "classifier_traits.hpp"

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

using namespace IB_utils;

template<typename ClassifierType>
class GradientBoostClassifier : public ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
						      typename classifier_traits<ClassifierType>::classifier> {
public:

  using DataType = typename classifier_traits<ClassifierType>::datatype;
  using IntegralLabelType = typename classifier_traits<ClassifierType>::integrallabeltype;
  using Classifier = typename classifier_traits<ClassifierType>::classifier;
  using ClassifierArgs = typename classifier_traits<ClassifierType>::classifierArgs;
  using ClassifierList = std::vector<std::unique_ptr<ClassifierBase<DataType, Classifier>>>;

  using Partition = std::vector<std::vector<int>>;
  using PartitionList = std::vector<Partition>;

  using Leaves = Row<double>;
  using LeavesList = std::vector<Leaves>;
  using Prediction = Row<double>;
  using PredictionList = std::vector<Prediction>;
  
  GradientBoostClassifier() = default;
  
  // 1
  // mat	: arma::Mat<double>
  // labels	: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // context	: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset, 
			  const Row<std::size_t>& labels,
			  Context<ClassifierType> context) :
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
  // mat	: arma::Mat<double>
  // labels	: arma::Row<double>
  // context	: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  Context<ClassifierType> context) :
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
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<std::size_t> <- CONVERTED TO Row<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const mat& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  Context<ClassifierType> context) :
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
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  Context<ClassifierType> context) :
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
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // colMask		: uvec
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context<ClassifierType> context) :
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
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const Row<double>& latestPrediction,
			  Context<ClassifierType> context) :
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
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context<ClassifierType> context) :
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
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const Row<double>& latestPrediction,
			  Context<ClassifierType> context) :
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
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<std::size_t> <- CONVERTED TO Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const mat& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context<ClassifierType> context) :
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
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<std::size_t> <- CONVERTED TO Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const mat& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  const Row<double>& latestPrediction,
			  Context<ClassifierType> context) :
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
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context<ClassifierType> context) :
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
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  const Row<double>& latestPrediction,
			  Context<ClassifierType> context) :
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
  void childContext(Context<ClassifierType>&, std::size_t, double, std::size_t);
  void contextInit_(Context<ClassifierType>&&);
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

  ClassifierArgs classifierArgs_;

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
using ClassifierBaseDD = ClassifierBase<double, DTC>;
using ClassifierBaseRD = ClassifierBase<double, RFC>;
using ClassifierBaseCD = ClassifierBase<double, CTC>;

using DecisionTreeRegressorClassifierBaseLDL = DecisionTreeRegressorClassifier<unsigned long, double, unsigned long>;
using DecisionTreeClassifierBaseLLDL = DecisionTreeClassifier<unsigned long, unsigned long, double, unsigned long>;
using RandomForestClassifierBaseLLL = RandomForestClassifier<unsigned long, unsigned long, unsigned long>;

using GradientBoostClassifierDTC = GradientBoostClassifier<DecisionTreeClassifier>);
using GradientBoostClassifierRFC = GradientBoostClassifier<RandomForestClassifier>);
using GradientBoostClassifierCTC = GradientBoostClassifier<DecisionTreeRegressorClassifier>);

// Register class with cereal
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTC);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFC);
CEREAL_REGISTER_TYPE(ContinuousClassifierBaseD);

CEREAL_REGISTER_TYPE(DecisionTreeClassifierBaseLLDL);
CEREAL_REGISTER_TYPE(RandomForestClassifierBaseLLL);
CEREAL_REGISTER_TYPE(DecisionTreeRegressorClassifierBaseLDL);

CEREAL_REGISTER_TYPE(GradientBoostClassifierDTC);
CEREAL_REGISTER_TYPE(GradientBoostClassifierRFC);
CEREAL_REGISTER_TYPE(GradientBoostClassifierCTC);


// Register class with cereal
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTC);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFC);
CEREAL_REGISTER_TYPE(ContinuousClassifierBaseD);

CEREAL_REGISTER_TYPE(DecisionTreeClassifierLLDL);
CEREAL_REGISTER_TYPE(RandomForestClassifierLLL);
CEREAL_REGISTER_TYPE(DecisionTreeRegressorClassifierLDL);

CEREAL_REGISTER_TYPE(GradientBoostClassifier<DecisionTreeClassifierLLDL>);
CEREAL_REGISTER_TYPE(GradientBoostClassifier<DecisionTreeRegressorClassifierLDL>);

// Register class hierarchy with cereal
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDD, DecisionTreeClassifierLLDL);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRD, RandomForestClassifierLLL);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseCD, DecisionTreeRegressorClassifierLDL);

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
    static void load_anod_construct(Archive &ar, cereal::construct<ContinuousClassifierBase<DataType, ClassifierType, Args...>> &construct) {
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
