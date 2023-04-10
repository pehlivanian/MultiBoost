#ifndef __CLASSIFIER_HPP__
#define __CLASSIFIER_HPP__

#include <memory>
#include <vector>

#include <mlpack/core.hpp>

#include "utils.hpp"
#include "score2.hpp"
#include "model.hpp"
#include "DP.hpp"
#include "model_traits.hpp"

using namespace arma;
using namespace Model_Traits;
using namespace ModelContext;

template<typename DataType>
class ClassifierBase : public Model<DataType> {
public:
  ClassifierBase() = default;
  ClassifierBase(std::string id) : Model<DataType>(id) {}

  virtual void Classify(const mat& data, Row<DataType>& pred) { Classify_(data, pred); }

private:
  virtual void purge() = 0;
  virtual void Classify_(const mat&, Row<DataType>&) = 0;

  void Project_(const mat& data, Row<DataType>& pred) override { Classify_(data, pred); }
  
};

template<typename DataType, typename ClassifierType, typename... Args>
class DiscreteClassifierBase : public ClassifierBase<DataType> {
public:
  using LeavesMap = std::unordered_map<std::size_t, DataType>;

  DiscreteClassifierBase(const mat& dataset, Row<DataType>& labels, Args&&... args) : 
    ClassifierBase<DataType>(typeid(*this).name())
  {

    labels_t_ = Row<std::size_t>(labels.n_cols);
    encode(labels, labels_t_);
    setClassifier(dataset, labels_t_, std::forward<Args>(args)...);
    args_ = std::tuple<Args...>(args...);
    
  }

  DiscreteClassifierBase(const LeavesMap& leavesMap, std::unique_ptr<ClassifierType> classifier) : 
    leavesMap_{leavesMap},
    classifier_{std::move(classifier)} {}

  DiscreteClassifierBase() = default;
  ~DiscreteClassifierBase() = default;

  void setClassifier(const mat&, Row<std::size_t>&, Args&&...);

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::base_class<ClassifierBase<DataType>>(this), CEREAL_NVP(leavesMap_));
    ar(cereal::base_class<ClassifierBase<DataType>>(this), CEREAL_NVP(classifier_));
  }

private:
  void encode(const Row<DataType>&, Row<std::size_t>&); 
  void decode(const Row<std::size_t>&, Row<DataType>&);

  Row<std::size_t> labels_t_;
  LeavesMap leavesMap_;
  std::unique_ptr<ClassifierType> classifier_;
  std::tuple<Args...> args_;

  void Classify_(const mat&, Row<DataType>&) override;
  void purge() override;

};


template<typename ClassifierType>
class CompositeClassifier : public ClassifierBase<typename model_traits<ClassifierType>::datatype> {
public:  
  using DataType		= typename model_traits<ClassifierType>::datatype;
  using IntegralLabelType	= typename model_traits<ClassifierType>::integrallabeltype;
  using Classifier		= typename model_traits<ClassifierType>::model;
  using ClassifierArgs		= typename model_traits<ClassifierType>::modelArgs;
  using ClassifierList		= std::vector<std::unique_ptr<ClassifierBase<DataType>>>;

  using Partition		= std::vector<std::vector<int>>;
  using PartitionList		= std::vector<Partition>;

  using Leaves			= Row<double>;
  using LeavesList		= std::vector<Leaves>;
  using Prediction		= Row<double>;
  using PredictionList		= std::vector<Prediction>;

  CompositeClassifier() = default;

  // 1
  // mat	: arma::Mat<double>
  // labels	: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // context	: ClassifierContext::Context
  CompositeClassifier(const mat& dataset, 
		      const Row<std::size_t>& labels,
		      Context context) :
    ClassifierBase<typename model_traits<ClassifierType>::datatype>(typeid(*this).name()),
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
  CompositeClassifier(const mat& dataset,
			  const Row<double>& labels,
			  Context context) :
    ClassifierBase<typename model_traits<ClassifierType>::datatype>(typeid(*this).name()),
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
  CompositeClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const mat& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  Context context) :
    ClassifierBase<typename model_traits<ClassifierType>::datatype>(typeid(*this).name()),
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
  CompositeClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  Context context) :
    ClassifierBase<typename model_traits<ClassifierType>::datatype>(typeid(*this).name()),
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
  CompositeClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context context) :
    ClassifierBase<typename model_traits<ClassifierType>::datatype>(typeid(*this).name()),
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
  CompositeClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const Row<double>& latestPrediction,
			  Context context) :
    ClassifierBase<typename model_traits<ClassifierType>::datatype>(typeid(*this).name()),
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
  CompositeClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context context) :
    ClassifierBase<typename model_traits<ClassifierType>::datatype>(typeid(*this).name()),
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
  CompositeClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const Row<double>& latestPrediction,
			  Context context) :
    ClassifierBase<typename model_traits<ClassifierType>::datatype>(typeid(*this).name()),
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
  CompositeClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const mat& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context context) :
    ClassifierBase<typename model_traits<ClassifierType>::datatype>(typeid(*this).name()),
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
  CompositeClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const mat& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  const Row<double>& latestPrediction,
			  Context context) :
    ClassifierBase<typename model_traits<ClassifierType>::datatype>(typeid(*this).name()),
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
  CompositeClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context context) :
    ClassifierBase<typename model_traits<ClassifierType>::datatype>(typeid(*this).name()),
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
  CompositeClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  const Row<double>& latestPrediction,
			  Context context) :
    ClassifierBase<typename model_traits<ClassifierType>::datatype>(typeid(*this).name()),
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

  virtual void fit();

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
  void read(CompositeClassifier&, std::string);
  void commit();
  void checkAccuracyOfArchive();

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::base_class<ClassifierBase<DataType>>(this), CEREAL_NVP(classifiers_));
    ar(cereal::base_class<ClassifierBase<DataType>>(this), symmetrized_);
    ar(cereal::base_class<ClassifierBase<DataType>>(this), a_);
    ar(cereal::base_class<ClassifierBase<DataType>>(this), b_);
    // ar(cereal::base_class<ClassifierBase<DataType>>(this), latestPrediction_);
  }

private:
  void childContext(Context&, std::size_t, double, std::size_t);
  void contextInit_(Context&&);
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

  void updateClassifiers(std::unique_ptr<ClassifierBase<DataType>>&&, Row<DataType>&);

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

#include "classifier_impl.hpp"

#endif
