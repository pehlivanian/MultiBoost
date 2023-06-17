#ifndef __COMPOSITEREGRESSOR_HPP__
#define __COMPOSITEREGRESSOR_HPP__

#include <tuple>
#include <memory>
#include <vector>
#include <unordered_map>
#include <iterator>
#include <algorithm>

#include <boost/filesystem.hpp>

#include <mlpack/core.hpp>

#include "utils.hpp"
#include "DP.hpp"
#include "score2.hpp"
#include "regressor.hpp"
#include "model_traits.hpp"

using namespace arma;

using namespace ModelContext;
using namespace Model_Traits;

template<typename RegressorType>
class CompositeRegressor : public RegressorBase<typename regressor_traits<RegressorType>::datatype,
						typename regressor_traits<RegressorType>::model> {
public:
  using DataType = typename regressor_traits<RegressorType>::datatype;
  using Regressor = typename regressor_traits<RegressorType>::model;
  using RegressorList = std::vector<std::unique_ptr<RegressorBase<DataType, Regressor>>>;
  
  using Partition = std::vector<std::vector<int>>;
  using PartitionList = std::vector<Partition>;

  using Leaves = Row<double>;
  using Prediction = Row<double>;
  using PredictionList = std::vector<Prediction>;

  CompositeRegressor() = default;

  // 1
  // mat	: arma::Mat<double>
  // labels	: arma::Row<double>
  // context	: ModelContext::Context
  CompositeRegressor(const mat& dataset,
		     const Row<double>& labels,
		     Context context,
		     const std::string& folderName=std::string{}) :
    RegressorBase<typename regressor_traits<RegressorType>::datatype,
		  typename regressor_traits<RegressorType>::model>(typeid(*this).name()),
    dataset_{dataset},
    labels_{labels},
    hasOOSData_{false},
    hasInitialPrediction_{false},
    reuseColMask_{false},
    folderName_{folderName}
  {
    init_(std::move(context));
  }

  // 2
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: arma::Row<std::double>
  // context		: ModelContext::Context
  CompositeRegressor(const mat& dataset,
		     const Row<double>& labels,
		     const mat& dataset_oos,
		     const Row<double> labels_oos,
		     Context context,
		     const std::string& folderName=std::string{}) :
    RegressorBase<typename regressor_traits<RegressorType>::datatype,
		  typename regressor_traits<RegressorType>::model>(typeid(*this).name()),
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

  // 3
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ModelContext::Context
  CompositeRegressor(const mat& dataset,
		     const Row<double>& labels,
		     const Row<double>& latestPrediction,
		     const uvec& colMask,
		     Context context,
		     const std::string& folderName=std::string{}) :
    RegressorBase<typename regressor_traits<RegressorType>::datatype,
		  typename regressor_traits<RegressorType>::model>(typeid(*this).name()),
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

  // 4
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ModelContext::Context
  CompositeRegressor(const mat& dataset,
		     const Row<double>& labels,
		     const Row<double>& latestPrediction,
		     Context context,
		     const std::string& folderName=std::string{}) :
    RegressorBase<typename regressor_traits<RegressorType>::datatype,
		  typename regressor_traits<RegressorType>::model>(typeid(*this).name()),
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

  
  // 5
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ModelContext::Context
  CompositeRegressor(const mat& dataset,
		     const Row<double>& labels,
		     const mat& dataset_oos,
		     const Row<double>& labels_oos,
		     const Row<double>& latestPrediction,
		     const uvec& colMask,
		     Context context,
		     const std::string& folderName=std::string{}) :
    RegressorBase<typename regressor_traits<RegressorType>::datatype,
		  typename regressor_traits<RegressorType>::model>(typeid(*this).name()),
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

  // 6
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ModelContext::Context
  CompositeRegressor(const mat& dataset,
		     const Row<double>& labels,
		     const mat& dataset_oos,
		     const Row<double>& labels_oos,
		     const Row<double>& latestPrediction,
		     Context context,
		     const std::string& folderName=std::string{}) :
    RegressorBase<typename regressor_traits<RegressorType>::datatype,
		  typename regressor_traits<RegressorType>::model>(typeid(*this).name()),
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

  // 4 Predict methods
  // predict on member dataset; use latestPrediction_
  virtual void Predict(Row<DataType>&);
  // predict on subset of dataset defined by uvec; sum step prediction vectors
  virtual void Predict(Row<DataType>&, const uvec&);
  // predict OOS, loop through and call Predict_ on individual regressors, sum
  virtual void Predict(const mat&, Row<DataType>&) override;
  virtual void Predict(mat&&, Row<DataType>&) override;
  
  // 2 overloaded versions for archive regressor
  virtual void Predict(std::string, Row<DataType>&);
  virtual void Predict(std::string, const mat&, Row<DataType>&);
  virtual void Predict(std::string, mat&&, Row<DataType>&);

  template<typename MatType>
  void _predict_in_loop(MatType&&, Row<DataType>&);
  
  template<typename MatType>
  void _predict_in_loop_archive(std::vector<std::string>&, MatType&&, Row<DataType>&);
  
  mat getDataset() const { return dataset_; }
  Row<DataType> getLatestPrediction() const { return latestPrediction_; }
  int getNRows() const { return n_; }
  int getNCols() const { return m_; }
  Row<double> getLabels() const { return labels_; }
  RegressorList getRegressors() const { return regressors_; }

  std::string getIndexName() const { return indexName_; }
  boost::filesystem::path getFldr() const { return fldr_; }

  lossFunction getLoss() const { return loss_; }

  double loss(const Row<DataType>& yhat);
  double loss(const Row<DataType>& y, const Row<DataType>& yhat);

  virtual void printStats(int);
  std::string write();  
  std::string writeDataset();
  std::string writeDatasetOOS();
  std::string writeLabels();
  std::string writeLabelsOOS();
  std::string writePrediction();
  std::string writeColMask();
  void read(CompositeRegressor&, std::string);
  void commit();
  void checkAccuracyOfArchive();

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::base_class<RegressorBase<DataType, Regressor>>(this), CEREAL_NVP(regressors_));
    // We choose not to seriazlize latestPrediction_, we will generate from (regressor, dataset) 
    // if necessary.
    // ar(cereal::base_class<RegressorBase<DataType, Regressor>>(this), latestPrediction_);
  }

private:
  void childContext(Context&, std::size_t, double, std::size_t);
  void contextInit_(Context&&);
  void init_(Context&&);
  Row<double> _constantLeaf() const;
  Row<double> _randomLeaf() const;
  uvec subsampleRows(size_t);
  uvec subsampleCols(size_t);
  void fit_step(std::size_t);
  double computeLearningRate(std::size_t);
  std::size_t computePartitionSize(std::size_t, const uvec&);
  void childInfoInit_();
  
  void Predict_(const mat& dataset, Row<DataType>& prediction) override { 
    Predict(dataset, prediction);
  }
  void Predict_(mat&& dataset, Row<DataType>& prediction) override {
    Predict(std::move(dataset), prediction);
  }

  void purge_() override;
  
  template<typename... Ts>
  void createRegressor(std::unique_ptr<RegressorType>&, 
		       const mat&,
		       rowvec&,
		       std::tuple<Ts...> const&);

  double computeSubLearningRate(std::size_t);
  std::size_t computeSubPartitionSize(std::size_t);
  std::size_t computeSubStepSize(std::size_t);
  std::tuple<std::size_t, std::size_t, double> computeChildPartitionInfo(std::size_t);

  void updateRegressors(std::unique_ptr<RegressorBase<DataType, Regressor>>&&, Row<DataType>&);

  std::pair<rowvec,rowvec> generate_coefficients(const Row<DataType>&, const uvec&);
  std::pair<rowvec,rowvec> generate_coefficients(const Row<DataType>&, const Row<DataType>&, const uvec&);
  Leaves computeOptimalSplit(rowvec&, rowvec&, std::size_t, std::size_t, double, const uvec&);

  void setNextRegressor(const RegressorType&);
  AllRegressorArgs allRegressorArgs();

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
  std::string folderName_;

  int n_;
  int m_;

  std::size_t minLeafSize_;
  double minimumGainSplit_;
  std::size_t maxDepth_;
  std::size_t numTrees_;

  AllRegressorArgs regressorArgs_;

  RegressorList regressors_;
  PartitionList partitions_;
  PredictionList predictions_;

  std::mt19937 mersenne_engine_{std::random_device{}()};
  std::default_random_engine default_engine_;
  std::uniform_int_distribution<std::size_t> partitionDist_;
  // call by partitionDist_(default_engine_)

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

  using childInfoType = std::tuple<std::size_t,
				   std::size_t,
				   double>;
  std::unordered_map<std::size_t, childInfoType> childInfo_;

  std::size_t serializationWindow_;
  std::string indexName_;

  boost::filesystem::path fldr_{};

		     
};

#include "compositeregressor_impl.hpp"

#endif
