#ifndef __COMPOSITEREGRESSOR_HPP__
#define __COMPOSITEREGRESSOR_HPP__

#include <tuple>
#include <memory>
#include <vector>
#include <unordered_map>
#include <iterator>
#include <algorithm>
#include <optional>

#include <boost/filesystem.hpp>

#include <mlpack/core.hpp>

#include "utils.hpp"
#include "DP.hpp"
#include "score2.hpp"
#include "constantregressor.hpp"
#include "contextmanager.hpp"
#include "regressor.hpp"
#include "recursivemodel.hpp"
#include "regressor_loss.hpp"
#include "model_traits.hpp"

using namespace arma;

using namespace ModelContext;
using namespace Model_Traits;

template<typename RegressorType>
class CompositeRegressor : 
  public RegressorBase<typename model_traits<RegressorType>::datatype,
		       typename model_traits<RegressorType>::model>,
  public RecursiveModel<typename model_traits<RegressorType>::datatype,
			CompositeRegressor<RegressorType>>
{
  
public:
  using DataType = typename model_traits<RegressorType>::datatype;
  using Regressor = typename model_traits<RegressorType>::model;
  using RegressorList = std::vector<std::unique_ptr<Model<DataType>>>;
  // using RegressorList = std::vector<std::unique_ptr<RegressorBase<DataType, Regressor>>>;
  
  using Leaves = Row<DataType>;
  using Prediction = Row<DataType>;
  using PredictionList = std::vector<Prediction>;
  using optLeavesInfo = std::tuple<Leaves,
				   std::optional<std::vector<std::vector<int>>>>;
  
  using BaseModel_t				= RecursiveModel<typename model_traits<RegressorType>::datatype,
								 CompositeRegressor<RegressorType>>;


  friend ContextManager;
  friend RecursiveModel<typename model_traits<RegressorType>::datatype,
			CompositeRegressor<RegressorType>>;

  CompositeRegressor() = default;

  // 1
  // mat	: arma::Mat<double>
  // labels	: arma::Row<double>
  // context	: ModelContext::Context
  CompositeRegressor(const Mat<DataType>& dataset,
		     const Row<DataType>& labels,
		     Context context,
		     const std::string& folderName=std::string{}) :
    RegressorBase<typename model_traits<RegressorType>::datatype,
		  typename model_traits<RegressorType>::model>(typeid(*this).name()),
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
  CompositeRegressor(const Mat<DataType>& dataset,
		     const Row<DataType>& labels,
		     const Mat<DataType>& dataset_oos,
		     const Row<DataType> labels_oos,
		     Context context,
		     const std::string& folderName=std::string{}) :
    RegressorBase<typename model_traits<RegressorType>::datatype,
		  typename model_traits<RegressorType>::model>(typeid(*this).name()),
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
  
  // 2a
  CompositeRegressor(const Mat<DataType>& dataset,
		     const Row<DataType>& labels,
		     const uvec& colMask,
		     Context context,
		     const std::string& folderName=std::string{}) :
    RegressorBase<typename model_traits<RegressorType>::datatype,
		  typename model_traits<RegressorType>::model>(typeid(*this).name()),    
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
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ModelContext::Context
  CompositeRegressor(const Mat<DataType>& dataset,
		     const Row<DataType>& labels,
		     const Row<DataType>& latestPrediction,
		     const uvec& colMask,
		     Context context,
		     const std::string& folderName=std::string{}) :
    RegressorBase<typename model_traits<RegressorType>::datatype,
		  typename model_traits<RegressorType>::model>(typeid(*this).name()),
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
  CompositeRegressor(const Mat<DataType>& dataset,
		     const Row<DataType>& labels,
		     const Row<DataType>& latestPrediction,
		     Context context,
		     const std::string& folderName=std::string{}) :
    RegressorBase<typename model_traits<RegressorType>::datatype,
		  typename model_traits<RegressorType>::model>(typeid(*this).name()),
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
  CompositeRegressor(const Mat<DataType>& dataset,
		     const Row<DataType>& labels,
		     const Mat<DataType>& dataset_oos,
		     const Row<DataType>& labels_oos,
		     const Row<DataType>& latestPrediction,
		     const uvec& colMask,
		     Context context,
		     const std::string& folderName=std::string{}) :
    RegressorBase<typename model_traits<RegressorType>::datatype,
		  typename model_traits<RegressorType>::model>(typeid(*this).name()),
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
  CompositeRegressor(const Mat<DataType>& dataset,
		     const Row<DataType>& labels,
		     const Mat<DataType>& dataset_oos,
		     const Row<DataType>& labels_oos,
		     const Row<DataType>& latestPrediction,
		     Context context,
		     const std::string& folderName=std::string{}) :
    RegressorBase<typename model_traits<RegressorType>::datatype,
		  typename model_traits<RegressorType>::model>(typeid(*this).name()),
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
  virtual void Predict(const Mat<DataType>&, Row<DataType>&) override;
  virtual void Predict(Mat<DataType>&&, Row<DataType>&) override;
  
  // 2 overloaded versions for archive regressor
  virtual void Predict(std::string, Row<DataType>&);
  virtual void Predict(std::string, const Mat<DataType>&, Row<DataType>&);
  virtual void Predict(std::string, Mat<DataType>&&, Row<DataType>&);

  template<typename MatType>
  void _predict_in_loop(MatType&&, Row<DataType>&);
  
  template<typename MatType>
  void _predict_in_loop_archive(std::vector<std::string>&, MatType&&, Row<DataType>&);
  
  Mat<DataType> getDataset() const { return dataset_; }
  Row<DataType> getLatestPrediction() const { return latestPrediction_; }
  int getNRows() const { return n_; }
  int getNCols() const { return m_; }
  Row<DataType> getLabels() const { return labels_; }
  RegressorList getRegressors() const { return regressors_; }

  std::string getIndexName() const { return indexName_; }
  boost::filesystem::path getFldr() const { return fldr_; }

  regressorLossFunction getLoss() const { return loss_; }

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
    // We choose not to serialize latestPrediction_, we will generate from (regressor, dataset) 
    // if necessary.
    // ar(cereal::base_class<RegressorBase<DataType, Regressor>>(this), latestPrediction_);
  }

private:
  void childContext(Context&);
  void contextInit_(Context&&);
  void init_(Context&&);
  uvec subsampleRows(size_t);
  uvec subsampleCols(size_t);
  void fit_step(std::size_t);
  
  void Predict_(const Mat<DataType>& dataset, Row<DataType>& prediction) override { 
    Predict(dataset, prediction);
  }
  void Predict_(Mat<DataType>&& dataset, Row<DataType>& prediction) override {
    Predict(std::move(dataset), prediction);
  }

  void purge_() override;
  
  template<typename... Ts>
  void setRootRegressor(std::unique_ptr<RegressorType>&,
			const Mat<DataType>&,
			Row<DataType>&,
			Row<DataType>&,
			std::tuple<Ts...> const&);

  template<typename... Ts>
  void setRootRegressor(std::unique_ptr<RegressorType>&,
			const Mat<DataType>&,
			Row<DataType>&,
			std::tuple<Ts...> const&);



  template<typename... Ts>
  void createRootRegressor(std::unique_ptr<RegressorType>&, 
			   uvec,
			   uvec,
			   const Row<DataType>&);

  void updateRegressors(std::unique_ptr<Model<DataType>>&&, Row<DataType>&);

  void calcWeights();
  void setWeights();
  auto generate_coefficients(const Row<DataType>&, const uvec&) -> std::pair<Row<DataType>,Row<DataType>>;
  auto computeOptimalSplit(Row<DataType>&, Row<DataType>&, std::size_t, std::size_t, double, double, const uvec&, bool=false) -> optLeavesInfo;

  void setNextRegressor(const RegressorType&);
  AllRegressorArgs allRegressorArgs();

  int steps_;
  int baseSteps_;
  Mat<DataType> dataset_;
  Row<DataType> labels_;
  Row<DataType> weights_;
  Mat<DataType> dataset_oos_;
  Row<DataType> labels_oos_;

  bool hasOOSData_;
  bool hasInitialPrediction_;
  bool reuseColMask_;

  std::size_t partitionSize_;
  Row<DataType> latestPrediction_;
  std::vector<std::string> fileNames_;

  regressorLossFunction loss_;
  RegressorLossFunction<DataType>* lossFn_;

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

  std::size_t minLeafSize_;
  double minimumGainSplit_;
  std::size_t maxDepth_;
  std::size_t numTrees_;

  AllRegressorArgs regressorArgs_;

  RegressorList regressors_;
  PredictionList predictions_;

  std::mt19937 mersenne_engine_{std::random_device{}()};
  std::default_random_engine default_engine_;
  std::uniform_int_distribution<std::size_t> partitionDist_;
  // call: partitionDist_(default_engine_)

  bool quietRun_;

  bool useWeights_;

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
  
  std::vector<std::size_t> childNumTrees_;
  std::vector<std::size_t> childMaxDepth_;
  std::vector<std::size_t> childMinLeafSize_;
  std::vector<double> childMinimumGainSplit_;

  std::size_t serializationWindow_;
  std::size_t depth_;
  std::string indexName_;

  boost::filesystem::path fldr_{};

		     
};

#include "compositeregressor_impl.hpp"

#endif
