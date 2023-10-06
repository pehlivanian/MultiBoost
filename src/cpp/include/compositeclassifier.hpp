#ifndef __COMPOSITE_CLASSIFIER_HPP__
#define __COMPOSITE_CLASSIFIER_HPP__

#include <tuple>
#include <memory>
#include <vector>
#include <unordered_map>

#include <boost/filesystem.hpp>

#include <mlpack/core.hpp>

#include "utils.hpp"
#include "DP.hpp"
#include "score2.hpp"
#include "classifier.hpp"
#include "model_traits.hpp"

using namespace arma;

using namespace ModelContext;
using namespace Model_Traits;

template<typename ClassifierType>
class CompositeClassifier : public ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
						  typename classifier_traits<ClassifierType>::model> {

public:  
  using DataType		= typename classifier_traits<ClassifierType>::datatype;
  using IntegralLabelType	= typename classifier_traits<ClassifierType>::integrallabeltype;
  using Classifier		= typename classifier_traits<ClassifierType>::model;
  using ClassifierList		= std::vector<std::unique_ptr<ClassifierBase<DataType, Classifier>>>;

  using Leaves			= Row<double>;
  using Prediction		= Row<double>;
  using PredictionList		= std::vector<Prediction>;

  CompositeClassifier() = default;

  // 1
  // mat	: arma::Mat<double>
  // labels	: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // context	: ModelContext::Context
  CompositeClassifier(const mat& dataset, 
		      const Row<std::size_t>& labels,
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    hasOOSData_{false},
    hasInitialPrediction_{false},
    reuseColMask_{false},
    folderName_{folderName}
  { 
    init_(std::move(context));
  }

  // 2
  // mat	: arma::Mat<double>
  // labels	: arma::Row<double>
  // context	: ModelContext::Context
  CompositeClassifier(const mat& dataset,
		      const Row<double>& labels,
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    hasOOSData_{false},
    hasInitialPrediction_{false},
    reuseColMask_{false},
    folderName_{folderName}
  { 
    init_(std::move(context));
  }

  // 2a
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // colMask		: uvec
  // context		: ModelContext::Context
  CompositeClassifier(const mat& dataset,
		      const Row<std::size_t>& labels,
		      const uvec& colMask,
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    hasOOSData_{false},
    hasInitialPrediction_{false},
    reuseColMask_{true},
    colMask_{colMask},
    folderName_{folderName}
  { 
    init_(std::move(context));
  }

  // 2b
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // colMask		: uvec
  // context		: ModelContext::Context
  CompositeClassifier(const mat& dataset,
		      const Row<double>& labels,
		      const uvec& colMask,
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    hasOOSData_{false},
    hasInitialPrediction_{false},
    reuseColMask_{true},
    colMask_{colMask},
    folderName_{folderName}
  { 
    init_(std::move(context));
  }



  // 3
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<std::size_t> <- CONVERTED TO Row<double>
  // context		: ModelContext::Context
  CompositeClassifier(const mat& dataset,
		      const Row<std::size_t>& labels,
		      const mat& dataset_oos,
		      const Row<std::size_t>& labels_oos,
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    dataset_oos_{dataset_oos},
    labels_oos_{conv_to<Row<double>>::from(labels_oos)},
    hasOOSData_{true},
    hasInitialPrediction_{false},
    reuseColMask_{false},
    folderName_{folderName}
  {
    init_(std::move(context));
  }

  // 4
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // context		: ModelContext::Context
  CompositeClassifier(const mat& dataset,
		      const Row<double>& labels,
		      const mat& dataset_oos,
		      const Row<double>& labels_oos,
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    dataset_oos_{dataset_oos},
    labels_oos_{labels_oos},
    hasOOSData_{true},
    hasInitialPrediction_{false},
    reuseColMask_{false},
    folderName_{folderName}
  {
    init_(std::move(context));
  }

  // 5
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // colMask		: uvec
  // context		: ModelContext::Context
  CompositeClassifier(const mat& dataset,
		      const Row<std::size_t>& labels,
		      const Row<double>& latestPrediction,
		      const uvec& colMask,
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    hasOOSData_{false},
    hasInitialPrediction_{true},
    reuseColMask_{true},
    latestPrediction_{latestPrediction},
    colMask_{colMask},
    folderName_{folderName}
  {
    init_(std::move(context));
  }

  // 6
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ModelContext::Context
  CompositeClassifier(const mat& dataset,
		      const Row<std::size_t>& labels,
		      const Row<double>& latestPrediction,
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    hasOOSData_{false},
    hasInitialPrediction_{true},
    reuseColMask_{false},
    latestPrediction_{latestPrediction},
    folderName_{folderName}
  {
    init_(std::move(context));
  }
   
  // 7
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ModelContext::Context
  CompositeClassifier(const mat& dataset,
		      const Row<double>& labels,
		      const Row<double>& latestPrediction,
		      const uvec& colMask,
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    hasOOSData_{false},
    hasInitialPrediction_{true},
    reuseColMask_{true},
    latestPrediction_{latestPrediction},
    colMask_{colMask},
    folderName_{folderName}
  {
    init_(std::move(context));
  }

  // 8
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ModelContext::Context
  CompositeClassifier(const mat& dataset,
		      const Row<double>& labels,
		      const Row<double>& latestPrediction,
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    hasOOSData_{false},
    hasInitialPrediction_{true},
    reuseColMask_{false},
    latestPrediction_{latestPrediction},
    folderName_{folderName}
  {
    init_(std::move(context));
  }

  // 9
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<std::size_t> <- CONVERTED TO Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ModelContext::Context
  CompositeClassifier(const mat& dataset,
		      const Row<std::size_t>& labels,
		      const mat& dataset_oos,
		      const Row<std::size_t>& labels_oos,
		      const Row<double>& latestPrediction,
		      const uvec& colMask,
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    dataset_oos_{dataset_oos},
    labels_oos_{conv_to<Row<double>>::from(labels_oos)},
    hasOOSData_{true},
    hasInitialPrediction_{true},
    reuseColMask_{true},
    latestPrediction_{latestPrediction},
    colMask_{colMask},
    folderName_{folderName}
  {
    init_(std::move(context));
  }

  // 10
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<std::size_t> <- CONVERTED TO Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ModelContext::Context
  CompositeClassifier(const mat& dataset,
		      const Row<std::size_t>& labels,
		      const mat& dataset_oos,
		      const Row<std::size_t>& labels_oos,
		      const Row<double>& latestPrediction,
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},
    dataset_oos_{dataset_oos},
    labels_oos_{conv_to<Row<double>>::from(labels_oos)},
    hasOOSData_{true},
    hasInitialPrediction_{true},
    reuseColMask_{false},
    latestPrediction_{latestPrediction},
    folderName_{folderName}
  {
    init_(std::move(context));
  }

  // 11
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ModelContext::Context
  CompositeClassifier(const mat& dataset,
		      const Row<double>& labels,
		      const mat& dataset_oos,
		      const Row<double>& labels_oos,
		      const Row<double>& latestPrediction,
		      const uvec& colMask,
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    dataset_oos_{dataset_oos},
    labels_oos_{labels_oos},
    hasOOSData_{true},
    hasInitialPrediction_{true},
    reuseColMask_{true},
    latestPrediction_{latestPrediction},
    colMask_{colMask},
    folderName_{folderName}
  {
    init_(std::move(context));
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
		      Context context,
		      const std::string& folderName=std::string{}) :
    ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
		   typename classifier_traits<ClassifierType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    dataset_oos_{dataset_oos},
    labels_oos_{labels_oos},
    hasOOSData_{true},
    hasInitialPrediction_{true},
    reuseColMask_{false},
    latestPrediction_{latestPrediction},
    folderName_{folderName}
  {
    init_(std::move(context));
  }

  virtual void fit();

  virtual void Classify(const mat&, Row<DataType>&) override;

 // 4 Predict methods
  // predict on member dataset; use latestPrediction_
  virtual void Predict(Row<DataType>&);
  // predict on subset of dataset defined by uvec; sum step prediction vectors
  virtual void Predict(Row<DataType>&, const uvec&);
  // predict OOS, loop through and call Classify_ on individual classifiers, sum
  virtual void Predict(const mat&, Row<DataType>&, bool=false);
  virtual void Predict(mat&&, Row<DataType>&, bool=false);

  // 3 overloaded versions of above based based on label datatype
  virtual void Predict(Row<IntegralLabelType>&);
  virtual void Predict(Row<IntegralLabelType>&, const uvec&);
  virtual void Predict(const mat&, Row<IntegralLabelType>&);
  virtual void Predict(mat&&, Row<IntegralLabelType>&);

  // 2 overloaded versions for archive classifier
  virtual void Predict(std::string, Row<DataType>&, bool=false);
  virtual void Predict(std::string, const mat&, Row<DataType>&, bool=false);
  virtual void Predict(std::string, mat&&, Row<DataType>&, bool=false);

  template<typename MatType>
  void _predict_in_loop(MatType&&, Row<DataType>&, bool=false);
  
  template<typename MatType>
  void _predict_in_loop_archive(std::vector<std::string>&, MatType&&, Row<DataType>&, bool=false);


  mat getDataset() const { return dataset_; }
  Row<DataType> getLatestPrediction() const { return latestPrediction_; }
  int getNRows() const { return n_; }
  int getNCols() const { return m_; }
  Row<double> getLabels() const { return labels_; }
  ClassifierList getClassifiers() const { return classifiers_; }

  std::string getIndexName() const { return indexName_; }
  boost::filesystem::path getFldr() const { return fldr_; }

  void symmetrizeLabels(Row<DataType>&);
  void symmetrizeLabels();

  std::pair<double, double> getAB() const {return std::make_pair(a_, b_); }
  
  virtual void printStats(int);
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
    ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this), CEREAL_NVP(classifiers_));
    ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this), symmetrized_);
    ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this), a_);
    ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this), b_);
    // Don't serialize latestPrediction_, we will generate from (classifier, dataste)
    // if necessary
    // ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this), latestPrediction_);
  }

private:
  void childContext(Context&);
  void contextInit_(Context&&);
  void init_(Context&&);
  Row<double> _constantLeaf() const;
  Row<double> _constantLeaf(double) const;
  Row<double> _randomLeaf() const;
  uvec subsampleRows(size_t);
  uvec subsampleCols(size_t);
  Row<DataType> uniqueCloseAndReplace(Row<DataType>&);
  void symmetrize(Row<DataType>&);
  void deSymmetrize(Row<DataType>&);

  void fit_step(std::size_t);

  void Classify_(const mat& dataset, Row<DataType>& prediction) override { 
    Predict(dataset, prediction); 
  }
  void Classify_(mat&& dataset, Row<DataType>& prediction) override {
    Predict(std::move(dataset), prediction);
  }

  void purge_() override;
  
  void createRootClassifier(std::unique_ptr<ClassifierType>&, 
			    uvec, 
			    uvec, const Row<double>&);
  template<typename... Ts>
  void setRootClassifier(std::unique_ptr<ClassifierType>&, 
			 const mat&,
			 Row<double>&,
			 std::tuple<Ts...> const&);

  std::tuple<std::size_t, std::size_t, double, double> computeChildPartitionInfo();
  std::tuple<std::size_t, std::size_t, double> computeChildModelInfo();

  void updateClassifiers(std::unique_ptr<ClassifierBase<DataType, Classifier>>&&, Row<DataType>&);

  std::pair<rowvec,rowvec> generate_coefficients(const Row<DataType>&, const uvec&);
  Leaves computeOptimalSplit(rowvec&, rowvec&, std::size_t, std::size_t, double, double, const uvec&);

  void setNextClassifier(const ClassifierType&);
  AllClassifierArgs allClassifierArgs(std::size_t);

  int steps_;
  int baseSteps_;
  mat dataset_;
  Row<double> labels_;
  mat dataset_oos_;
  Row<double> labels_oos_;

  bool hasOOSData_;
  bool hasInitialPrediction_;
  bool reuseColMask_;

  std::size_t partitionSize_;
  Row<DataType> latestPrediction_;
  std::vector<std::string> fileNames_;

  lossFunction loss_;
  LossFunction<double>* lossFn_;

  float lossPower_ = -1.;

  bool clamp_gradient_;
  double upper_val_, lower_val_;
  
  double learningRate_;
  double activePartitionRatio_;

  double row_subsample_ratio_;
  double col_subsample_ratio_;

  uvec rowMask_;
  uvec colMask_;
  std::string folderName_;

  int n_;
  int m_;

  double a_;
  double b_;

  std::size_t minLeafSize_;
  double minimumGainSplit_;
  std::size_t maxDepth_;
  std::size_t numTrees_;

  AllClassifierArgs classifierArgs_;

  ClassifierList classifiers_;
  PredictionList predictions_;

  std::mt19937 mersenne_engine_{std::random_device{}()};
  std::default_random_engine default_engine_;
  std::uniform_int_distribution<std::size_t> partitionDist_;
  // call by partitionDist_(default_engine_)

  bool symmetrized_;
  bool removeRedundantLabels_;
  bool quietRun_;

  bool recursiveFit_;
  bool serializeModel_;
  bool serializePrediction_;
  bool serializeColMask_;
  bool serializeDataset_;
  bool serializeLabels_;

  std::vector<std::size_t> childPartitionSize_;
  std::vector<std::size_t> childNumSteps_;
  std::vector<double> childLearningRate_;
  std::vector<double> childActivePartitionRatio_;

  std::vector<std::size_t> childMinLeafSize_;
  std::vector<std::size_t> childMaxDepth_;
  std::vector<double> childMinimumGainSplit_;

  std::size_t serializationWindow_;
  std::string indexName_;
  std::size_t depth_;

  boost::filesystem::path fldr_{};

};

#include "compositeclassifier_impl.hpp"

#endif
