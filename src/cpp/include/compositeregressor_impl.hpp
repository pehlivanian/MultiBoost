#ifndef __COMPOSITEREGRESSOR_IMPL_HPP__
#define __COMPOSITEREGRESSOR_IMPL_HPP__

#define DEBUG() __debug dd{__FILE__, __FUNCTION__, __LINE__};

#include "path_utils.hpp"

using namespace PartitionSize;
using namespace LearningRate;
using namespace LossMeasures;
using namespace ModelContext;
using namespace Objectives;
using namespace IB_utils;

namespace RegressorFileScope {
constexpr bool POST_EXTRAPOLATE = false;
constexpr bool W_CYCLE_PREFIT = true;
constexpr bool DIAGNOSTICS_0_ = false;
constexpr bool DIAGNOSTICS_1_ = false;
constexpr bool TIMER = true;
const std::string DIGEST_PATH = IB_utils::resolve_path("digest/regress");
}  // namespace RegressorFileScope

template <typename RegressorType>
inline AllRegressorArgs CompositeRegressor<RegressorType>::allRegressorArgs() {
  return std::make_tuple(minLeafSize_, minimumGainSplit_, maxDepth_);
}

template <typename RegressorType>
inline void CompositeRegressor<RegressorType>::updateRegressors(
    std::unique_ptr<Model<DataType>>&& regressor, Row<DataType>& prediction) {
  latestPrediction_ += prediction;
  regressor->purge();
  regressors_.push_back(std::move(regressor));
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::init_(Context&& context) {
  ContextManager::contextInit(*this, context);

  if (serializeModel_ || serializePrediction_ || serializeColMask_ || serializeDataset_ ||
      serializeLabels_) {
    if (folderName_.size()) {
      fldr_ = boost::filesystem::path{folderName_};
    } else {
      fldr_ =
          IB_utils::FilterDigestLocation(boost::filesystem::path{RegressorFileScope::DIGEST_PATH});
      boost::filesystem::create_directory(fldr_);
    }

    // Will keep overwriting context
    std::string contextFilename = "_Context_0.cxt";
    writeBinary<Context>(contextFilename, context, fldr_);
  }

  // Set weights
  weights_ = ones<Row<DataType>>(labels_.n_cols);

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

  // Initialize rowMask
  rowMask_ = linspace<uvec>(0, -1 + n_, n_);

  // Initialize rng
  std::size_t a = 1, b = std::max(1, static_cast<int>(m_ * col_subsample_ratio_));
  partitionDist_ = std::uniform_int_distribution<std::size_t>(a, b);

  // Set latestPrediction to 0 if not passed
  if (!hasInitialPrediction_) {
    latestPrediction_ = BaseModel_t::_constantLeaf(0.0);
  }

  // set loss function
  lossFn_ = createLoss<DataType>(loss_, lossPower_);
  if (RegressorFileScope::DIAGNOSTICS_0_) {
    std::cerr << "Loss Function: [" 
	      << static_cast<int>(loss_) 
	      << "] : Loss Power: ["
	      << lossPower_ 
	      << "]" << std::endl;
  }

  // ensure this is a leaf regressor for lowest-level call
  if (childPartitionSize_.size() <= 1) {
    recursiveFit_ = false;
  }
}

template <typename RegressorType>
template <typename... Ts>
void CompositeRegressor<RegressorType>::setRootRegressor(
    std::unique_ptr<RegressorType>& regressor,
    const Mat<DataType>& dataset,
    Row<DataType>& labels,
    Row<DataType>& weights,
    std::tuple<Ts...> const& args) {
  // The calling convention for mlpack regressors with weight specification:
  // cls{dataset, responses, weights, args...)
  std::unique_ptr<RegressorType> reg;

  auto _c = [&reg, &dataset, &labels, &weights](Ts const&... classArgs) {
    reg = std::make_unique<RegressorType>(dataset, labels, weights, classArgs...);
  };
  std::apply(_c, args);

  // Feedback
  // ...

  regressor = std::move(reg);
}

template <typename RegressorType>
template <typename... Ts>
void CompositeRegressor<RegressorType>::setRootRegressor(
    std::unique_ptr<RegressorType>& regressor,
    const Mat<DataType>& dataset,
    Row<DataType>& labels,
    std::tuple<Ts...> const& args) {
  std::unique_ptr<RegressorType> reg;

  auto _c = [&reg, &dataset, &labels](Ts const&... classArgs) {
    reg = std::make_unique<RegressorType>(dataset, labels, classArgs...);
  };
  std::apply(_c, args);

  regressor = std::move(reg);
}

template <typename RegressorType>
template <typename... Ts>
void CompositeRegressor<RegressorType>::createRootRegressor(
    std::unique_ptr<RegressorType>& regressor,
    uvec rowMask,
    uvec colMask,
    const Row<DataType>& best_leaves) {
  const typename RegressorType::Args& rootRegressorArgs = RegressorType::_args(allRegressorArgs());

  if (RegressorFileScope::POST_EXTRAPOLATE) {
    // Fit regressor on {dataset_slice, best_leaves}, both subsets of the original data
    // There will be no post-padding of zeros as that is not defined for OOS prediction, we
    // just use the regressor below to predict on the larger dataset for this step's
    // prediction

    auto dataset_slice = dataset_.submat(rowMask, colMask);
    Leaves allLeaves = best_leaves;

    if (useWeights_ && true) {
      calcWeights();
      setRootRegressor(regressor, dataset_slice, weights_, rootRegressorArgs);
    } else {
      setRootRegressor(regressor, dataset_slice, allLeaves, rootRegressorArgs);
    }

  } else {
    // Fit regressor on {dataset, padded best_leaves}

    // Zero pad labels first
    Leaves allLeaves = zeros<Row<DataType>>(m_);
    allLeaves(colMask) = best_leaves;

    if (useWeights_ && true) {
      calcWeights();
      setRootRegressor(regressor, dataset_, allLeaves, weights_, rootRegressorArgs);
    } else {
      setRootRegressor(regressor, dataset_, allLeaves, rootRegressorArgs);
    }
  }
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::Predict(Row<DataType>& prediction) {
  prediction = latestPrediction_;
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::Predict(Row<DataType>& prediction, const uvec& colMask) {
  Predict(prediction);
  prediction = prediction.submat(zeros<uvec>(1), colMask);
}

template <typename RegressorType>
template <typename MatType>
void CompositeRegressor<RegressorType>::_predict_in_loop(
    MatType&& dataset, Row<DataType>& prediction) {
  prediction = zeros<Row<DataType>>(dataset.n_cols);

  for (const auto& regressor : regressors_) {
    Row<DataType> predictionStep;
    regressor->Project(std::forward<MatType>(dataset), predictionStep);
    prediction += predictionStep;
  }
}
template <typename RegressorType>
void CompositeRegressor<RegressorType>::Predict(
    const Mat<DataType>& dataset, Row<DataType>& prediction) {
  if (serializeModel_ && indexName_.size()) {
    throw predictionAfterClearedModelException();
    return;
  }

  _predict_in_loop(dataset, prediction);
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::Predict(
    Mat<DataType>&& dataset, Row<DataType>& prediction) {
  if (serializeModel_ && indexName_.size()) {
    throw predictionAfterClearedModelException();
    return;
  }

  _predict_in_loop(std::move(dataset), prediction);
}

template <typename RegressorType>
uvec CompositeRegressor<RegressorType>::subsampleRows(size_t numRows) {
  // XXX
  // Necessary?
  // uvec r = sort(randperm(n_, numRows));
  // uvec r = randperm(n_, numRows);
  uvec r = PartitionUtils::sortedSubsample2(n_, numRows);
  return r;
}

template <typename RegressorType>
uvec CompositeRegressor<RegressorType>::subsampleCols(size_t numCols) {
  // XXX
  // Necessary?
  // uvec r = sort(randperm(m_, numCols));
  // uvec r = randperm(m_, numCols);
  uvec r = PartitionUtils::sortedSubsample2(n_, numCols);
  return r;
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::fit_step(std::size_t stepNum) {
  // Implementation of W-cycle

  if (!reuseColMask_) {
    int colRatio = static_cast<size_t>(m_ * col_subsample_ratio_);
    colMask_ = PartitionUtils::sortedSubsample2(m_, colRatio);
  }

  Row<DataType> labels_slice = labels_.submat(zeros<uvec>(1), colMask_);
  std::pair<Row<DataType>, Row<DataType>> coeffs;

  Row<DataType> prediction;
  std::unique_ptr<RegressorType> regressor;

  if (!hasInitialPrediction_) {
    latestPrediction_ = BaseModel_t::_constantLeaf(0.0);

    std::unique_ptr<ConstantTreeRegressorRegressor> reg_;
    Row<DataType> constantLeaf = ones<Row<DataType>>(labels_.n_elem);
    constantLeaf.fill(mean(labels_slice));

    reg_ = std::make_unique<ConstantTreeRegressorRegressor>(dataset_, constantLeaf);

    updateRegressors(std::move(reg_), constantLeaf);
  }

  if (RegressorFileScope::W_CYCLE_PREFIT) {
    if (RegressorFileScope::DIAGNOSTICS_0_ || RegressorFileScope::DIAGNOSTICS_1_) {
      std::cerr << fit_prefix(depth_);
      std::cerr << "[*]PRE-FITTING COMPOSITE REGRESSOR FOR (PARTITIONSIZE, STEPNUM): ("
                << partitionSize_ << ", " << stepNum << " of " << steps_ << ")" << std::endl;
    }

    coeffs = generate_coefficients(labels_slice, colMask_);

    auto [best_leaves, subset_info] = computeOptimalSplit(
        coeffs.first,
        coeffs.second,
        stepNum,
        partitionSize_,
        learningRate_,
        activePartitionRatio_,
        colMask_,
        false);

    createRootRegressor(regressor, rowMask_, colMask_, best_leaves);

    regressor->Predict(dataset_, prediction);

    updateRegressors(std::move(regressor), prediction);

    hasInitialPrediction_ = true;

    if (RegressorFileScope::DIAGNOSTICS_1_) {
      Row<DataType> latestPrediction_slice = latestPrediction_.submat(zeros<uvec>(1), colMask_);
      Row<DataType> prediction_slice = prediction.submat(zeros<uvec>(1), colMask_);
      float eps = std::numeric_limits<float>::epsilon();

      std::cerr << "[PRE-FIT ";
      for (std::size_t i = 0; i < best_leaves.size(); ++i) {
        std::string status = "";
        if (fabs(best_leaves[i] - prediction_slice[i]) > eps) status = "MISPREDICTEDED";
        std::cerr << colMask_[i] << " : " << labels_slice[i] << " : " << latestPrediction_slice[i]
                  << " :: " << best_leaves[i] << " : " << prediction_slice[i]
                  << " :: " << coeffs.first[i] << " : " << coeffs.second[i] << " : " << status
                  << std::endl;
      }
      std::cerr << "]" << std::endl;
    }
  }

  //////////////////////////
  // BEGIN RECURSIVE STEP //
  //////////////////////////
  if (recursiveFit_ && (childPartitionSize_.size() > 1)) {
    if (RegressorFileScope::DIAGNOSTICS_1_ || RegressorFileScope::DIAGNOSTICS_0_) {
      std::cerr << fit_prefix(depth_);
      std::cerr << "[-]FITTING COMPOSITE REGRESSOR FOR (PARTITIONSIZE, STEPNUM): ("
                << partitionSize_ << ", " << stepNum << " of " << steps_ << ")" << std::endl;
    }

    Context context{};
    ContextManager::childContext(context, *this);

    // allLeaves may not strictly fit the definition of labels here -
    // aside from the fact that it is of double type, it may have more
    // than one class. So we don't want to symmetrize, but we want
    // to remap the redundant values.
    std::unique_ptr<CompositeRegressor<RegressorType>> regressor;
    if (hasInitialPrediction_) {
      regressor.reset(new CompositeRegressor<RegressorType>(
          dataset_, labels_, latestPrediction_, colMask_, context));
    } else {
      regressor.reset(new CompositeRegressor<RegressorType>(dataset_, labels_, colMask_, context));
    }

    if (RegressorFileScope::DIAGNOSTICS_1_) {
      std::cerr << "PREFIT: (PARTITIONSIZE, STEPNUM, NUMSTEPS): (" << partitionSize_ << ", "
                << stepNum << ", " << steps_ << ")" << std::endl;
    }

    regressor->fit();

    if (RegressorFileScope::DIAGNOSTICS_1_) {
      std::cerr << "POSTFIT: (PARTITIONSIZE, STEPNUM, NUMSTEPS): (" << partitionSize_ << ", "
                << stepNum << ", " << steps_ << ")" << std::endl;
    }

    regressor->Predict(dataset_, prediction);

    updateRegressors(std::move(regressor), prediction);

    hasInitialPrediction_ = true;
  }
  ////////////////////////
  // END RECURSIVE STEP //
  ////////////////////////

  // If we are in recursive mode and partitionSize <= 2, fall through
  // to this case for the leaf regressor

  if (RegressorFileScope::DIAGNOSTICS_0_ || RegressorFileScope::DIAGNOSTICS_1_) {
    std::cerr << fit_prefix(depth_);
    std::cerr << "[*]POST-FITTING LEAF REGRESSOR FOR (PARTITIONSIZE, STEPNUM): (" << partitionSize_
              << ", " << stepNum << " of " << steps_ << ")" << std::endl;
  }

  if (!hasInitialPrediction_) {
    latestPrediction_ = BaseModel_t::_constantLeaf(0.0);
  }

  // Generate coefficients g, h
  coeffs = generate_coefficients(labels_slice, colMask_);

  // Compute optimal leaf choice on unrestricted dataset
  auto [best_leaves, subset_info] = computeOptimalSplit(
      coeffs.first,
      coeffs.second,
      stepNum,
      partitionSize_,
      learningRate_,
      activePartitionRatio_,
      colMask_,
      false);

  createRootRegressor(regressor, rowMask_, colMask_, best_leaves);

  regressor->Predict(dataset_, prediction);

  updateRegressors(std::move(regressor), prediction);

  hasInitialPrediction_ = true;

  if (RegressorFileScope::DIAGNOSTICS_1_) {
    Row<DataType> latestPrediction_slice = latestPrediction_.submat(zeros<uvec>(1), colMask_);
    Row<DataType> prediction_slice = prediction.submat(zeros<uvec>(1), colMask_);
    float eps = std::numeric_limits<float>::epsilon();

    std::cerr << "[POST-FIT ";
    for (std::size_t i = 0; i < best_leaves.size(); ++i) {
      std::string status = "";
      if (fabs(best_leaves[i] - prediction_slice[i]) > eps) status = "MISPREDICTEDED";
      std::cerr << colMask_[i] << " : " << labels_slice[i] << " : " << latestPrediction_slice[i]
                << " :: " << best_leaves[i] << " : " << prediction_slice[i]
                << " :: " << coeffs.first[i] << " : " << coeffs.second[i] << " : " << status
                << std::endl;
    }
    std::cerr << "]" << std::endl;
  }
}

template <typename RegressorType>
auto CompositeRegressor<RegressorType>::computeOptimalSplit(
    Row<DataType>& g,
    Row<DataType>& h,
    std::size_t stepNum,
    std::size_t partitionSize,
    double learningRate,
    double activePartitionRatio,
    const uvec& colMask,
    bool includeSubsets) -> optLeavesInfo {
  (void)stepNum;

  int n = colMask.n_rows, T = partitionSize;
  objective_fn obj_fn = objective_fn::RationalScore;
  bool risk_partitioning_objective = false;
  bool use_rational_optimization = true;
  bool sweep_down = false;
  double gamma = 0.;
  double reg_power = 1.;
  bool find_optimal_t = false;
  bool reorder_by_weighted_priority = true;

  std::vector<DataType> gv = arma::conv_to<std::vector<DataType>>::from(g);
  std::vector<DataType> hv = arma::conv_to<std::vector<DataType>>::from(h);

  DPSolver<DataType> dp;

  {
    // This is the expensive call; DPSolver scales as ~ n^2*T
    // auto timer_ = __timer{"DPSolver instantiation"};

    dp = DPSolver(
        n,
        T,
        gv,
        hv,
        obj_fn,
        risk_partitioning_objective,
        use_rational_optimization,
        gamma,
        reg_power,
        sweep_down,
        find_optimal_t,
        reorder_by_weighted_priority);
  }

  auto subsets = dp.get_optimal_subsets_extern();
  // double end_ratio = 0.10; // Currently unused

  // printSubsets<DataType>(subsets, gv, hv, colMask);

  Row<DataType> leaf_values = arma::zeros<Row<DataType>>(n);

  {
    // auto timer_ = __timer{"DPSolver set leaves"};

    if (T > 1 || risk_partitioning_objective) {
      std::size_t start_ind =
          risk_partitioning_objective ? 0 : static_cast<std::size_t>(T * activePartitionRatio);
      // std::size_t end_ind = risk_partitioning_objective ? subsets.size() :
      //   static_cast<std::size_t>((1. - end_ratio) * static_cast<double>(T));
      // Currently unused - kept for potential future use

      for (std::size_t i = start_ind; i < subsets.size(); ++i) {
        // for (std::size_t i=start_ind; i<end_ind; ++i) {
        uvec ind = arma::conv_to<uvec>::from(subsets[i]);
        double val = -1. * learningRate * sum(g(ind)) / sum(h(ind));
        for (auto j : ind) {
          leaf_values(j) = val;
        }
      }
    }
  }

  if (includeSubsets) {
    return std::make_tuple(leaf_values, subsets);
  } else {
    return std::make_tuple(leaf_values, std::nullopt);
  }
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::purge_() {
  dataset_ = ones<Mat<DataType>>(0, 0);
  labels_ = ones<Row<DataType>>(0);
  dataset_oos_ = ones<Mat<DataType>>(0, 0);
  labels_oos_ = ones<Row<DataType>>(0);
}

template <typename RegressorType>
std::string CompositeRegressor<RegressorType>::write() {
  using CerealT = CompositeRegressor<RegressorType>;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  std::string fileName =
      dumps<CerealT, CerealIArch, CerealOArch>(*this, SerializedType::REGRESSOR, fldr_);
  return fileName;
}

template <typename RegressorType>
std::string CompositeRegressor<RegressorType>::writeColMask() {
  return IB_utils::writeColMask(colMask_, fldr_);
}

template <typename RegressorType>
std::string CompositeRegressor<RegressorType>::writePrediction() {
  return IB_utils::writePrediction(latestPrediction_, fldr_);
}

template <typename RegressorType>
std::string CompositeRegressor<RegressorType>::writeDataset() {
  return IB_utils::writeDatasetIS(dataset_, fldr_);
}

template <typename RegressorType>
std::string CompositeRegressor<RegressorType>::writeDatasetOOS() {
  return IB_utils::writeDatasetOOS(dataset_oos_, fldr_);
}

template <typename RegressorType>
std::string CompositeRegressor<RegressorType>::writeLabels() {
  return IB_utils::writeLabelsIS(labels_, fldr_);
}

template <typename RegressorType>
std::string CompositeRegressor<RegressorType>::writeLabelsOOS() {
  return IB_utils::writeLabelsOOS(labels_oos_, fldr_);
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::read(
    CompositeRegressor<RegressorType>& rhs, std::string fileName) {
  using CerealT = CompositeRegressor<RegressorType>;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  loads<CerealT, CerealIArch, CerealOArch>(rhs, fileName, fldr_);
}

template <typename RegressorType>
template <typename MatType>
void CompositeRegressor<RegressorType>::_predict_in_loop_archive(
    std::vector<std::string>& fileNames, MatType&& dataset, Row<DataType>& prediction) {
  using C = CompositeRegressor<RegressorType>;
  std::unique_ptr<C> regressorNew = std::make_unique<C>();
  prediction = zeros<Row<DataType>>(dataset.n_cols);
  Row<DataType> predictionStep;

  for (auto& fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "CLS") {
      fileName = strJoin(tokens, '_', 1);
      read(*regressorNew, fileName);
      regressorNew->Predict(std::forward<MatType>(dataset), predictionStep);
      prediction += predictionStep;
    }
  }
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::Predict(
    std::string index, const Mat<DataType>& dataset, Row<DataType>& prediction) {
  std::vector<std::string> fileNames;
  readIndex(index, fileNames, fldr_);

  _predict_in_loop_archive(fileNames, dataset, prediction);
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::Predict(
    std::string index, Mat<DataType>&& dataset, Row<DataType>& prediction) {
  std::vector<std::string> fileNames;
  readIndex(index, fileNames, fldr_);

  _predict_in_loop_archive(fileNames, std::move(dataset), prediction);
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::Predict(std::string index, Row<DataType>& prediction) {
  Predict(index, dataset_, prediction);
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::commit() {
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
  // std::copy(fileNames_.begin(), fileNames_.end(),std::ostream_iterator<std::string>(std::cout,
  // "\n"));
  indexName_ = writeIndex(fileNames_, fldr_);
  RegressorList{}.swap(regressors_);
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::checkAccuracyOfArchive() {
  Row<DataType> yhat;
  Predict(yhat);

  Row<DataType> prediction;
  Predict(indexName_, prediction);

  float eps = std::numeric_limits<float>::epsilon();
  for (int i = 0; i < prediction.n_elem; ++i) {
    if (fabs(prediction[i] - yhat[i]) > eps) {
      std::cerr << "VIOLATION: (i, yhat[i], prediction[i]): "
                << "( " << yhat[i] << ", " << prediction[i] << ") : "
                << "(diff, eps) = "
                << "(" << fabs(prediction[i] - yhat[i]) << ", " << eps << ")" << std::endl;
    }
  }
  std::cerr << "ACCURACY CHECKED" << std::endl;
}

template <typename RegressorType>
double CompositeRegressor<RegressorType>::loss(const Row<DataType>& yhat, const Row<DataType>& y) {
  return lossFn_->loss(yhat, y);
}

template <typename RegressorType>
double CompositeRegressor<RegressorType>::loss(const Row<DataType>& yhat) {
  return lossFn_->loss(yhat, labels_);
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::printStats(int stepNum) {
  Row<DataType> yhat;
  double r_squared_IS, r_squared_OOS;

  if (serializeModel_) {
    // Prediction from current archive
    Predict(indexName_, yhat);
    // checkAccuracyOfArchive();
  } else {
    // Prediction from nonarchived regressor
    Predict(yhat);
  }

  auto mn = mean(labels_);
  auto den = sum(pow((labels_ - mn), 2));
  auto num = sum(pow((labels_ - yhat), 2));
  r_squared_IS = 1. - (num / den);

  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  // auto UTC =
  // std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  std::stringstream datetime;
  datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
  auto suff = datetime.str();

  // Only print stats for top level of recursive call
  if (hasOOSData_) {
    std::cout << suff << ": "
              << "(PARTITION SIZE = " << partitionSize_ << ", STEPS = " << steps_ << ") "
              << "STEP: " << stepNum << " IS R^2: " << r_squared_IS << std::endl;
  }

  if (hasOOSData_) {
    Row<DataType> yhat_oos;
    if (serializeModel_) {
      Predict(indexName_, dataset_oos_, yhat_oos);
    } else {
      Predict(dataset_oos_, yhat_oos);
    }

    mn = mean(labels_oos_);
    den = sum(pow((labels_oos_ - mn), 2));
    num = sum(pow((labels_oos_ - yhat_oos), 2));
    r_squared_OOS = 1. - (num / den);

    std::cout << suff << ": "
              << "(PARTITION SIZE = " << partitionSize_ << ", STEPS = " << steps_ << ") "
              << "STEP: " << stepNum << " OOS R^2: " << r_squared_OOS << std::endl;
  }
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::fit() {
  for (int stepNum = 1; stepNum <= steps_; ++stepNum) {
    fit_step(stepNum);

    if (serializeModel_) {
      commit();
    }
    if (!quietRun_) {
      printStats(stepNum);
    }
  }

  // Serialize residual
  if (serializeModel_) {
    commit();
  }

  // print final stats
  if (!quietRun_) {
    printStats(steps_);
  }
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::calcWeights() {
  Row<DataType> yhat;
  Predict(yhat, colMask_);
  Row<DataType> labels_slice = labels_.submat(zeros<uvec>(1), colMask_);

  weights_ = abs(labels_slice - yhat);
  weights_ = weights_ * (static_cast<DataType>(weights_.n_cols) / sum(weights_));
}

template <typename RegressorType>
void CompositeRegressor<RegressorType>::setWeights() {
  ;
}

template <typename RegressorType>
auto CompositeRegressor<RegressorType>::generate_coefficients(
    const Row<DataType>& labels, const uvec& colMask) -> std::pair<Row<DataType>, Row<DataType>> {
  Row<DataType> yhat;
  Predict(yhat, colMask);

  Row<DataType> g, h;
  lossFn_->loss(yhat, labels, &g, &h);

  return std::make_pair(g, h);
}

#endif
