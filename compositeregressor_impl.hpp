#ifndef __COMPOSITEREGRESSOR_IMPL_HPP__
#define __COMPOSITEREGRESSOR_IMPL_HPP__

using row_d = Row<double>;

using namespace PartitionSize;
using namespace LearningRate;
using namespace LossMeasures;
using namespace ModelContext;
using namespace Objectives;
using namespace IB_utils;

namespace FileScope {
  const bool POST_EXTRAPOLATE = false;
  const bool DIAGNOSTICS = false;
} // namespace FileScope

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::childContext(Context& context, 
						std::size_t subPartitionSize,
						double subLearningRate,
						std::size_t stepSize) {

  context.loss			= loss_;
  context.partitionRatio	= std::min(1., 2*partitionRatio_);
  context.learningRate		= subLearningRate;
  context.baseSteps		= baseSteps_;
  context.symmetrizeLabels	= false;
  context.removeRedundantLabels	= false;
  context.rowSubsampleRatio	= row_subsample_ratio_;
  context.colSubsampleRatio	= col_subsample_ratio_;
  context.recursiveFit		= true;
  context.stepSizeMethod	= stepSizeMethod_;
  context.partitionSizeMethod	= partitionSizeMethod_;
  context.learningRateMethod	= learningRateMethod_;    
  context.steps			= stepSize;

  // Part of model args
  context.numTrees		= numTrees_;
  context.partitionSize		= subPartitionSize;
  context.minLeafSize		= minLeafSize_;
  context.maxDepth		= maxDepth_;
  context.minimumGainSplit	= minimumGainSplit_;
}

template<typename RegressorType>
AllRegressorArgs
CompositeRegressor<RegressorType>::allRegressorArgs() {
  return std::make_tuple(minLeafSize_, minimumGainSplit_, maxDepth_);
}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::contextInit_(Context&& context) {

  loss_				= context.loss;
  partitionSize_		= context.partitionSize;
  partitionRatio_		= context.partitionRatio;
  learningRate_			= context.learningRate;
  steps_			= context.steps;
  baseSteps_			= context.baseSteps;
  quietRun_			= context.quietRun;
  row_subsample_ratio_		= context.rowSubsampleRatio;
  col_subsample_ratio_		= context.colSubsampleRatio;
  recursiveFit_			= context.recursiveFit;
  partitionSizeMethod_		= context.partitionSizeMethod;
  learningRateMethod_		= context.learningRateMethod;
  stepSizeMethod_		= context.stepSizeMethod;
  minLeafSize_			= context.minLeafSize;
  minimumGainSplit_		= context.minimumGainSplit;
  maxDepth_			= context.maxDepth;
  numTrees_			= context.numTrees;
  serialize_			= context.serialize;
  serializePrediction_		= context.serializePrediction;
  serializeColMask_		= context.serializeColMask;
  serializeDataset_		= context.serializeDataset;
  serializeLabels_		= context.serializeLabels;
  serializationWindow_		= context.serializationWindow;

}

template<typename RegressorType>
row_d
CompositeRegressor<RegressorType>::_constantLeaf() const {

  row_d r;
  r.zeros(dataset_.n_cols);
  return r;
}

template<typename RegressorType>
row_d
CompositeRegressor<RegressorType>::_randomLeaf() const {

  row_d r(dataset_.n_cols, arma::fill::none);
  std::mt19937 rng;
  std::uniform_real_distribution<DataType> dist{-learningRate_, learningRate_};
  r.imbue([&](){ return dist(rng);});
  return r;

}

template<typename RegressorType>
template<typename... Ts>
void
CompositeRegressor<RegressorType>::createRegressor(std::unique_ptr<RegressorType>& regressor,
						   const mat& dataset,
						   rowvec& labels,
						   std::tuple<Ts...> const& args) {
  // mimic:
  // regressor.reset(new RegressorType(dataset_,
  //  				      constantLabels,
  // 				      std::forward<typename RegressorType::Args>(regressorArgs)));

  // Instantiation of RegressorType should include fit stage; look at profile results
  auto _c = [&regressor, &dataset, &labels](Ts const&... classArgs) { 
    regressor.reset(new RegressorType(dataset,
				      labels,
				      classArgs...)
		    );
  };
  std::apply(_c, args);
}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::updateRegressors(std::unique_ptr<RegressorBase<DataType, Regressor>>&& regressor,
						    Row<DataType>& prediction) {
  latestPrediction_ += prediction;
  regressor->purge();
  regressors_.push_back(std::move(regressor));
}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::init_() {

  
  // Serialize dataset, labels first
  if (serializeDataset_) {
    std::string path;
    path = writeDataset();
    fileNames_.push_back(path);
    path = writeDatasetOOS();
    fileNames_.push_back(path);
  }

  if (serializeLabels_) {
    std::string path;
    path = writeLabels();
    fileNames_.push_back(path);
    path = writeLabelsOOS();
    fileNames_.push_back(path);
  }

  // Note these are flipped
  n_ = dataset_.n_rows; 
  m_ = dataset_.n_cols;

  // Initialize rng  
  std::size_t a=1, b=std::max(1, static_cast<int>(m_ * col_subsample_ratio_));
  partitionDist_ = std::uniform_int_distribution<std::size_t>(a, b);							      

  // partitions
  Partition partition = PartitionUtils::_fullPartition(m_);
  partitions_.push_back(partition);

  // regressors
  // don't overfit on first regressor
  row_d constantLabels = _constantLeaf();
  // row_d constantLabels = _randomLeaf();
 
  // numClasses is always the first parameter for the regressor
  // form parameter pack based on RegressorType
  std::unique_ptr<RegressorType> regressor;
  const typename RegressorType::Args& regressorArgs = RegressorType::_args(allRegressorArgs());
  createRegressor(regressor, dataset_, constantLabels, regressorArgs);

  // regressor.reset(new RegressorType(dataset_,
  //  				      constantLabels,
  //				      std::forward<typename RegressorType::Args>(regressorArgs)));

  // first prediction
  if (!hasInitialPrediction_){
    latestPrediction_ = zeros<Row<DataType>>(dataset_.n_cols);
  }

  Row<DataType> prediction;
  regressor->Predict(dataset_, prediction);

  // update regressor, predictions
  updateRegressors(std::move(regressor), prediction);

  // set loss function
  lossFn_ = lossMap<DataType>[loss_];

  // ensure this is a leaf regressor for lowest-level call
  if (partitionSize_ == 1) {
    recursiveFit_ = false;
  }

}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::Predict(Row<DataType>& prediction) {

  prediction = latestPrediction_;
}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::Predict(Row<DataType>& prediction, const uvec& colMask) {

  Predict(prediction);
  prediction = prediction.submat(zeros<uvec>(1), colMask);

}

template<typename RegressorType>
template<typename MatType>
void
CompositeRegressor<RegressorType>::_predict_in_loop(MatType&& dataset, Row<DataType>& prediction) {
  prediction = zeros<Row<DataType>>(dataset.n_cols);

  for (const auto& regressor : regressors_) {
    Row<DataType> predictionStep;
    regressor->Predict(std::forward<MatType>(dataset), predictionStep);
    prediction += predictionStep;    
  }  

}
template<typename RegressorType>
void
CompositeRegressor<RegressorType>::Predict(const mat& dataset, Row<DataType>& prediction) {

  if (serialize_ && indexName_.size()) {
    throw predictionAfterClearedClassifiersException();
    return;
  }

  _predict_in_loop(dataset, prediction);
}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::Predict(mat&& dataset, Row<DataType>& prediction) {
  
  if (serialize_ && indexName_.size()) {
    throw predictionAfterClearedClassifiersException();
    return;
  }

  _predict_in_loop(std::move(dataset), prediction);
}

template<typename RegressorType>
uvec
CompositeRegressor<RegressorType>::subsampleRows(size_t numRows) {

  // XXX
  // Necessary?
  // uvec r = sort(randperm(n_, numRows));
  // uvec r = randperm(n_, numRows);
  uvec r = PartitionUtils::sortedSubsample2(n_, numRows);
  return r;
}

template<typename RegressorType>
uvec
CompositeRegressor<RegressorType>::subsampleCols(size_t numCols) {

  // XXX
  // Necessary?
  // uvec r = sort(randperm(m_, numCols));
  // uvec r = randperm(m_, numCols);
  uvec r = PartitionUtils::sortedSubsample2(n_, numCols);
  return r;
}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::fit_step(std::size_t stepNum) {

  if (!reuseColMask_) {
    int colRatio = static_cast<size_t>(m_ * col_subsample_ratio_);
    // Equivalent, second a little faster, see benchmarks
    // colMask_ = subsampleCols(colRatio);
    colMask_ = PartitionUtils::sortedSubsample2(m_, colRatio);
  }

  row_d labels_slice = labels_.submat(zeros<uvec>(1), colMask_);

  Leaves allLeaves = zeros<row_d>(m_), best_leaves;

  Row<DataType> prediction, prediction_slice;
  
  std::unique_ptr<RegressorType> regressor;

  //////////////////////////
  // BEGIN RECURSIVE STEP //
  //////////////////////////
  if (recursiveFit_ && partitionSize_ > 2) {
    // Compute new partition size
    std::size_t subPartitionSize = computeSubPartitionSize(stepNum);

    // Compute new learning rate
    double subLearningRate = computeSubLearningRate(stepNum);

    // Compute new steps
    std::size_t subStepSize = computeSubStepSize(stepNum);

    // Generate coefficients g, h
    std::pair<rowvec, rowvec> coeffs = generate_coefficients(labels_slice, colMask_);

    best_leaves = computeOptimalSplit(coeffs.first, 
				      coeffs.second, 
				      stepNum, 
				      subPartitionSize, 
				      subLearningRate, 
				      colMask_);

    allLeaves(colMask_) = best_leaves;

    Context context{};      
    childContext(context, subPartitionSize, subLearningRate, subStepSize);

    // allLeaves may not strictly fit the definition of labels here - 
    // aside from the fact that it is of double type, it may have more 
    // than one class. So we don't want to symmetrize, but we want 
    // to remap the redundant values.

    std::unique_ptr<CompositeRegressor<RegressorType>> regressor;
    regressor.reset(new CompositeRegressor<RegressorType>(dataset_, 
							  labels_, 
							  latestPrediction_, 
							  colMask_, 
							  context));

    regressor->fit();

    regressor->Predict(dataset_, prediction);

    updateRegressors(std::move(regressor), prediction);

  } 
  ////////////////////////
  // END RECURSIVE STEP //
  ////////////////////////
  
  // If we are in recursive mode and partitionSize <= 2, fall through
  // to this case for the leaf regressor

  if (FileScope::DIAGNOSTICS)
    std::cout << "FITTING REGRESSOR FOR (PARTITIONSIZE, STEPNUM, NUMSTEPS): ("
	      << partitionSize_ << ", "
	      << stepNum << ", "
	      << steps_ << ")"
	      << std::endl;
  
  // Generate coefficients g, h
  std::pair<rowvec, rowvec> coeffs = generate_coefficients(labels_slice, colMask_);

  // Compute partition size
  std::size_t partitionSize = computePartitionSize(stepNum, colMask_);
  
  // Compute learning rate
  double learningRate = computeLearningRate(stepNum);

  // Compute optimal leaf choice on unrestricted dataset
  best_leaves = computeOptimalSplit(coeffs.first, 
				    coeffs.second, 
				    stepNum, 
				    partitionSize, 
				    learningRate,
				    colMask_);
  
  if (FileScope::POST_EXTRAPOLATE) {
    // Fit regressor on {dataset_slice, best_leaves}, both subsets of the original data
    // There will be no post-padding of zeros as that is not defined for OOS prediction, we
    // just use the regressor below to predict on the larger dataset for this step's
    // prediction
    uvec rowMask = linspace<uvec>(0, -1+n_, n_);
    auto dataset_slice = dataset_.submat(rowMask, colMask_);
    
    const typename RegressorType::Args& rootRegressorArgs = RegressorType::_args(allRegressorArgs());
    createRegressor(regressor, dataset_slice, best_leaves, rootRegressorArgs);

    // regressor.reset(new RegressorType(dataset_slice,
    //				best_leaves,
    //				std::forward(rootRegressorArgs)));		     
  } else {
    // Fit regressor on {dataset, padded best_leaves}
    // Zero pad labels first
    allLeaves(colMask_) = best_leaves;

    const typename RegressorType::Args& rootRegressorArgs = RegressorType::_args(allRegressorArgs());
    createRegressor(regressor, dataset_, allLeaves, rootRegressorArgs);
    
    // regressor.reset(new RegressorType(dataset_,
    // 				allLeaves,
    // 				std::forward(rootRegressorArgs)));
  }

  regressor->Predict(dataset_, prediction);

  updateRegressors(std::move(regressor), prediction);

}

template<typename RegressorType>
typename CompositeRegressor<RegressorType>::Leaves
CompositeRegressor<RegressorType>::computeOptimalSplit(rowvec& g,
						       rowvec& h,
						       std::size_t stepNum, 
						       std::size_t partitionSize,
						       double learningRate,
						       const uvec& colMask) {


  (void)stepNum;

  // We should implement several methods here
  // XXX
  std::vector<double> gv = arma::conv_to<std::vector<double>>::from(g);
  std::vector<double> hv = arma::conv_to<std::vector<double>>::from(h);

  int n = colMask.n_rows, T = partitionSize;
  bool risk_partitioning_objective = true;
  bool use_rational_optimization = true;
  bool sweep_down = false;
  double gamma = 0.;
  double reg_power=1.;
  bool find_optimal_t = false;

  // std::cout << "PARTITION SIZE: " << T << std::endl;

  DPSolver<double> dp;

  dp = DPSolver(n, T, gv, hv,
		objective_fn::RationalScore,
		risk_partitioning_objective,
		use_rational_optimization,
		gamma,
		reg_power,
		sweep_down,
		find_optimal_t
		);
  
  auto subsets = dp.get_optimal_subsets_extern();
  
  rowvec leaf_values = arma::zeros<rowvec>(n);
  
  for (auto &subset : subsets) {
    uvec ind = arma::conv_to<uvec>::from(subset);
    double val = -1. * learningRate * sum(g(ind))/sum(h(ind));
    for (auto i: ind) {
      leaf_values(i) = val;
    }
  }

  partitions_.emplace_back(subsets);

  return leaf_values;
    
}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::purge_() {

  dataset_ = ones<mat>(0,0);
  labels_ = ones<Row<double>>(0);
  dataset_oos_ = ones<mat>(0,0);
  labels_oos_ = ones<Row<double>>(0);
  std::vector<Partition>().swap(partitions_);

  // dataset_.clear();
  // labels_.clear();
  // dataset_oos_.clear();
  // labels_oos_.clear();
  // partitions_.clear();
}

template<typename RegressorType>
std::string
CompositeRegressor<RegressorType>::write() {

  using CerealT = CompositeRegressor<RegressorType>;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  std::string fileName = dumps<CerealT, CerealIArch, CerealOArch>(*this, SerializedType::REGRESSOR);
  return fileName;
}

template<typename RegressorType>
std::string
CompositeRegressor<RegressorType>::writeColMask() {

  return IB_utils::writeColMask(colMask_);
}

template<typename RegressorType>
std::string
CompositeRegressor<RegressorType>::writePrediction() {

  return IB_utils::writePrediction(latestPrediction_);
}

template<typename RegressorType>
std::string
CompositeRegressor<RegressorType>::writeDataset() {
  return IB_utils::writeDatasetIS(dataset_);
}

template<typename RegressorType>
std::string
CompositeRegressor<RegressorType>::writeDatasetOOS() {
  return IB_utils::writeDatasetOOS(dataset_oos_);
}

template<typename RegressorType>
std::string
CompositeRegressor<RegressorType>::writeLabels() {
  return IB_utils::writeLabelsIS(labels_);
}

template<typename RegressorType>
std::string
CompositeRegressor<RegressorType>::writeLabelsOOS() {
  return IB_utils::writeLabelsOOS(labels_oos_);
}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::read(CompositeRegressor<RegressorType>& rhs,
					std::string fileName) {

  using CerealT = CompositeRegressor<RegressorType>;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;  

  loads<CerealT, CerealIArch, CerealOArch>(rhs, fileName);
}

template<typename RegressorType>
template<typename MatType>
void
CompositeRegressor<RegressorType>::_predict_in_loop_archive(std::vector<std::string>& fileNames, 
							    MatType&& dataset, 
							    Row<DataType>& prediction) {

  using C = CompositeRegressor<RegressorType>;
  std::unique_ptr<C> regressorNew = std::make_unique<C>();
  prediction = zeros<Row<DataType>>(dataset.n_cols);
  Row<DataType> predictionStep;

  for (auto & fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "CLS") {
      fileName = strJoin(tokens, '_', 1);
      read(*regressorNew, fileName);
      regressorNew->Predict(std::forward<MatType>(dataset), predictionStep);
      prediction += predictionStep;
    }
  }
  
}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::Predict(std::string index, const mat& dataset, Row<DataType>& prediction) {

  std::vector<std::string> fileNames;
  readIndex(index, fileNames);

  _predict_in_loop_archive(fileNames, dataset, prediction);

}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::Predict(std::string index, mat&& dataset, Row<DataType>& prediction) {

  std::vector<std::string> fileNames;
  readIndex(index, fileNames);

  _predict_in_loop_archive(fileNames, std::move(dataset), prediction);
}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::Predict(std::string index, Row<DataType>& prediction) {

  Predict(index, dataset_, prediction);
}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::commit() {

  std::string path, predictionPath, colMaskPath;
  path = write();
  fileNames_.push_back(path);

  if (serializePrediction_) {
    predictionPath = writePrediction();
    fileNames_.push_back(predictionPath);
  }
  if (serializeColMask_) {
    colMaskPath = writeColMask();
    fileNames_.push_back(colMaskPath);
  }
  // std::copy(fileNames_.begin(), fileNames_.end(),std::ostream_iterator<std::string>(std::cout, "\n"));
  indexName_ = writeIndex(fileNames_);  
  RegressorList{}.swap(regressors_);
}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::checkAccuracyOfArchive() {

  Row<DataType> yhat;
  Predict(yhat); 

  Row<DataType> prediction;
  Predict(indexName_, prediction);
  
  float eps = std::numeric_limits<float>::epsilon();
  for (int i=0; i<prediction.n_elem; ++i) {
    if (fabs(prediction[i] - yhat[i]) > eps) {
      std::cerr << "VIOLATION: (i, yhat[i], prediction[i]): " 
		<< "( " << yhat[i] 
		<< ", " << prediction[i]
		<< ") : "
		<< "(diff, eps) = " << "(" << fabs(prediction[i]-yhat[i])
		<< ", " << eps << ")" << std::endl;
    }
  }   
  std::cerr << "ACCURACY CHECKED" << std::endl;
}

template<typename RegressorType>
double
CompositeRegressor<RegressorType>::loss(const Row<DataType>& yhat, const Row<DataType>& y) {
  return lossFn_->loss(yhat, y);
}

template<typename RegressorType>
double
CompositeRegressor<RegressorType>::loss(const Row<DataType>& yhat) {
  return lossFn_->loss(yhat, labels_);
}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::printStats(int stepNum) {

  Row<DataType> yhat;
  double r;

  if (serialize_) {
    // Prediction from current archive
    Predict(indexName_, yhat);
    r = lossFn_->loss(yhat, labels_);
    // checkAccuracyOfArchive();
  } else {
    // Prediction from nonarchived regressor
    Predict(yhat); 
    r = lossFn_->loss(yhat, labels_);
  }

  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  // auto UTC = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  
  std::stringstream datetime;
  datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
  auto suff = datetime.str();

  // Only print stats for top level of recursive call
  if (hasOOSData_) {
    double error_is = err(yhat, labels_);
    std::cout << suff << ": " 
	      << "(PARTITION SIZE = " << partitionSize_
	      << ", STEPS = " << steps_ << ") "
	      << "STEP: " << stepNum 
	      << " IS LOSS: " << r
	      << " IS ERROR: " << error_is << "%" << std::endl;
  }
  
  if (hasOOSData_) {
    Row<DataType> yhat_oos;
    if (serialize_) {
      Predict(indexName_, dataset_oos_, yhat_oos);
    } else {
      Predict(dataset_oos_, yhat_oos);
    }
    
    const double loss_oos = lossFn_->loss(yhat_oos, labels_oos_);
    
    std::cout << suff<< ": "
	      << "(PARTITION SIZE = " << partitionSize_
	      << ", STEPS = " << steps_ << ") "
	      << "STEP: " << stepNum
	      << " OOS ERROR: " << loss_oos << std::endl;
  }

}

template<typename RegressorType>
void
CompositeRegressor<RegressorType>::fit() {

  for (int stepNum=1; stepNum<=steps_; ++stepNum) {
    fit_step(stepNum);
          
    if (serialize_) {
      commit();
    }
    if (!quietRun_) {
      printStats(stepNum);
    }
    
  }
  
  // Serialize residual
  if (serialize_) {
    commit();
  }
  
  // print final stats
  if (!quietRun_) {
    printStats(steps_);
  }
  
}

template<typename RegressorType>
double
CompositeRegressor<RegressorType>::computeLearningRate(std::size_t stepNum) {

  double learningRate = learningRate_;

  if (learningRateMethod_ == LearningRateMethod::FIXED) {

    learningRate = learningRate_;
  } else if (learningRateMethod_ == LearningRateMethod::DECREASING) {

    double A = learningRate_, B = -log(.5) / static_cast<double>(steps_);
    learningRate = A * exp(-B * (-1 + stepNum));
  } else if (learningRateMethod_ == LearningRateMethod::INCREASING) {

    double A = learningRate_, B = log(2.) / static_cast<double>(steps_);

    learningRate = A * exp(B * (-1 + stepNum));

  }

  // if ((stepNum%100)==0)
  //   std::cout << "stepNum: " << stepNum << " LEARNING RATE: " << learningRate << std::endl;

  return learningRate;
}

template<typename RegressorType>
std::size_t
CompositeRegressor<RegressorType>::computeSubPartitionSize(std::size_t stepNum) {

  (void)stepNum;

  return static_cast<std::size_t>(partitionSize_/2);
}

template<typename RegressorType>
double
CompositeRegressor<RegressorType>::computeSubLearningRate(std::size_t stepNum) {

  (void)stepNum;

  return learningRate_;
}

template<typename RegressorType>
std::size_t
CompositeRegressor<RegressorType>::computeSubStepSize(std::size_t stepNum) {

  (void)stepNum;

  // XXX
  double mult = 1.;
  return std::max(1, static_cast<int>(mult * std::log(steps_)));    
}

template<typename RegressorType>
std::size_t
CompositeRegressor<RegressorType>::computePartitionSize(std::size_t stepNum, const uvec& colMask) {

  // stepNum is in range [1,...,context.steps]

  std::size_t partitionSize = partitionSize_;
  double lowRatio = .05;
  double highRatio = .95;
  int attach = 1000;

  if (partitionSizeMethod_ == PartitionSizeMethod::FIXED) {

    partitionSize = partitionSize_;

  } else if (partitionSizeMethod_ == PartitionSizeMethod::FIXED_PROPORTION) {

    partitionSize = static_cast<std::size_t>(partitionRatio_ * row_subsample_ratio_ * colMask.n_rows);
  } else if (partitionSizeMethod_ == PartitionSizeMethod::DECREASING) {

    double A = colMask.n_rows, B = log(colMask.n_rows)/steps_;
    partitionSize = std::max(1, static_cast<int>(A * exp(-B * (-1 + stepNum))));
  } else if (partitionSizeMethod_ == PartitionSizeMethod::INCREASING) {

    double A = 2., B = log(colMask.n_rows)/static_cast<double>(steps_);
    partitionSize = std::max(1, static_cast<int>(A * exp(B * (-1 + stepNum))));
  } else if (partitionSizeMethod_ == PartitionSizeMethod::RANDOM) {

    partitionSize = partitionDist_(default_engine_);

  } else if (partitionSizeMethod_ == PartitionSizeMethod::MULTISCALE) {

    if ((stepNum%attach) < static_cast<std::size_t>(attach/2)) {

      partitionSize = static_cast<std::size_t>(lowRatio * col_subsample_ratio_ * colMask.n_rows);
      partitionSize = partitionSize >= 1 ? partitionSize : 1;
    } else {

      partitionSize = static_cast<std::size_t>(highRatio * col_subsample_ratio_ * colMask.n_rows);
      partitionSize = partitionSize >= 1 ? partitionSize : 1;
    }
  }
  
  // if ((stepNum%100)==0)
  //   std::cout << "stepNum: " << stepNum << " PARTITIONSIZE: " << partitionSize << std::endl;

  return partitionSize;
}

template<typename RegressorType>
std::pair<rowvec, rowvec>
CompositeRegressor<RegressorType>::generate_coefficients(const Row<DataType>& labels, const uvec& colMask) {

  rowvec yhat;
  Predict(yhat, colMask);

  rowvec g, h;
  lossFn_->loss(yhat, labels, &g, &h);

  return std::make_pair(g, h);

}

template<typename RegressorType>
std::pair<rowvec, rowvec>
CompositeRegressor<RegressorType>::generate_coefficients(const Row<DataType>& yhat,
							 const Row<DataType>& y,
							 const uvec& colMask) {
  
  (void)colMask;

  rowvec g, h;
  lossFn_->loss(yhat, y, &g, &h);
  
  return std::make_pair(g, h);
}

#endif
