#ifndef __COMPOSITECLASSIFIER_IMPL_HPP__
#define __COMPOSITECLASSIFIER_IMPL_HPP__

#include "path_utils.hpp"
#include "utils.hpp"

using namespace PartitionSize;
using namespace LearningRate;
using namespace LossMeasures;
using namespace ModelContext;
using namespace Objectives;
using namespace IB_utils;

namespace ClassifierFileScope {
constexpr bool POST_EXTRAPOLATE = false;
constexpr bool W_CYCLE_PREFIT = true;
constexpr bool NEW_COLMASK_FOR_CHILD = false;
constexpr bool DIAGNOSTICS_0_ = false;
constexpr bool DIAGNOSTICS_1_ = false;
constexpr bool SUBSET_DIAGNOSTICS = false;
const std::string DIGEST_PATH = IB_utils::resolve_path("digest/classify");
}  // namespace ClassifierFileScope

template <typename ClassifierType>
inline AllClassifierArgs CompositeClassifier<ClassifierType>::allClassifierArgs(
    std::size_t numClasses) {
  return std::make_tuple(numClasses, minLeafSize_, minimumGainSplit_, numTrees_, maxDepth_);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::init_(Context&& context) {
  ContextManager::contextInit(*this, context);

  if (serializeModel_ || serializePrediction_ || serializeColMask_ || serializeDataset_ ||
      serializeLabels_) {
    if (folderName_.size()) {
      fldr_ = boost::filesystem::path{folderName_};
    } else {
      fldr_ =
          IB_utils::FilterDigestLocation(boost::filesystem::path{ClassifierFileScope::DIGEST_PATH});
      boost::filesystem::create_directory(fldr_);
    }
  }

  // Will keep overwriting context
  std::string contextFilename = "_Context_0.cxt";
  writeBinary<Context>(contextFilename, context, fldr_);

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

  // Make labels members of {-1,1}
  // Note that we pass labels_oos to this classifier for OOS testing
  // at regular intervals, but the external labels (hence labels_oos_)
  // may be in {0,1} and we leave things like that.
  assert(!(symmetrized_ && removeRedundantLabels_));

  if (symmetrized_) {
    symmetrizeLabels();
  } else if (removeRedundantLabels_) {
    auto uniqueVals = uniqueCloseAndReplace(labels_);
  }

  // Set latestPrediction to 0 if not passed
  if (!this->hasInitialPrediction_) {
    this->latestPrediction_ = this->_constantLeaf(0.0);
  }

  // set loss function
  lossFn_ = createLoss<DataType>(loss_, lossPower_);

  // ensure this is a leaf classifier for lowest-level call
  if (childPartitionSize_.size() <= 1) {
    recursiveFit_ = false;
  }
}

template <typename ClassifierType>
inline void CompositeClassifier<ClassifierType>::Predict(Row<DataType>& prediction) {
  prediction = this->latestPrediction_;
}

template <typename ClassifierType>
inline void CompositeClassifier<ClassifierType>::Predict(
    Row<DataType>& prediction, const uvec& colMask) {
  Predict(prediction);
  prediction = prediction.submat(zeros<uvec>(1), colMask);
}

template <typename ClassifierType>
template <typename MatType>
void CompositeClassifier<ClassifierType>::_predict_in_loop(
    MatType&& dataset, Row<DataType>& prediction, bool ignoreSymmetrization) {
  prediction = zeros<Row<DataType>>(dataset.n_cols);

  // Reserve memory for prediction step to avoid reallocations
  Row<DataType> predictionStep;
  predictionStep.set_size(dataset.n_cols);

  for (const auto& classifier : this->models_) {
    classifier->Project(dataset, predictionStep);
    prediction += predictionStep;
  }

  if (symmetrized_ && !ignoreSymmetrization) {
    deSymmetrize(prediction);
  }
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::Predict(
    const Mat<DataType>& dataset, Row<DataType>& prediction, bool ignoreSymmetrization) {
  if (serializeModel_ && indexName_.size()) {
    throw predictionAfterClearedModelException();
    return;
  }

  _predict_in_loop(dataset, prediction, ignoreSymmetrization);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::Predict(
    Mat<DataType>&& dataset, Row<DataType>& prediction, bool ignoreSymmetrization) {
  if (serializeModel_ && indexName_.size()) {
    throw predictionAfterClearedModelException();
    return;
  }

  _predict_in_loop(std::move(dataset), prediction, ignoreSymmetrization);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::Predict(
    Row<typename CompositeClassifier<ClassifierType>::IntegralLabelType>& prediction) {
  Row<DataType> prediction_d = conv_to<Row<DataType>>::from(prediction);
  Predict(prediction_d);
  prediction = conv_to<Row<std::size_t>>::from(prediction_d);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::Predict(
    Row<typename CompositeClassifier<ClassifierType>::IntegralLabelType>& prediction,
    const uvec& colMask) {
  Row<DataType> prediction_d = conv_to<Row<DataType>>::from(prediction);
  Predict(prediction_d, colMask);
  prediction = conv_to<Row<std::size_t>>::from(prediction_d);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::Predict(
    const Mat<DataType>& dataset,
    Row<typename CompositeClassifier<ClassifierType>::IntegralLabelType>& prediction) {
  Row<DataType> prediction_d;
  Predict(dataset, prediction_d);

  if (symmetrized_) {
    deSymmetrize(prediction_d);
  }

  prediction = conv_to<Row<std::size_t>>::from(prediction_d);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::Predict(
    Mat<DataType>&& dataset,
    Row<typename CompositeClassifier<ClassifierType>::IntegralLabelType>& prediction) {
  Row<DataType> prediction_d;
  Predict(std::move(dataset), prediction_d);

  if (symmetrized_) {
    deSymmetrize(prediction_d);
  }

  prediction = conv_to<Row<std::size_t>>::from(prediction_d);
}

template <typename ClassifierType>
inline uvec CompositeClassifier<ClassifierType>::subsampleRows(size_t numRows) {
  // Use the more efficient sortedSubsample2 method
  return PartitionUtils::sortedSubsample(n_, numRows);
}

template <typename ClassifierType>
inline uvec CompositeClassifier<ClassifierType>::subsampleCols(size_t numCols) {
  return PartitionUtils::sortedSubsample(n_, numCols);
}

template <typename ClassifierType>
auto CompositeClassifier<ClassifierType>::uniqueCloseAndReplace(Row<DataType>& labels)
    -> Row<DataType> {
  const Row<DataType> uniqueVals = unique(labels);
  constexpr double eps = static_cast<double>(std::numeric_limits<float>::epsilon());

  std::vector<std::pair<DataType, DataType>> uniqueByEps;
  std::vector<DataType> uniqueVals_;

  // Reserve memory to avoid reallocations
  uniqueByEps.reserve(uniqueVals.n_cols);
  uniqueVals_.reserve(uniqueVals.n_cols);

  uniqueVals_.push_back(uniqueVals[0]);

  for (std::size_t i = 1; i < uniqueVals.n_cols; ++i) {
    bool found = false;
    for (const auto& el : uniqueVals_) {
      if (std::abs(uniqueVals[i] - el) <= eps) {
        found = true;
        uniqueByEps.emplace_back(uniqueVals[i], el);
        break;  // Exit early once found
      }
    }
    if (!found) {
      uniqueVals_.push_back(uniqueVals[i]);
    }
  }

  // Replace redundant values in labels_
  for (const auto& [first, second] : uniqueByEps) {
    const uvec ind = find(labels_ == first);
    labels.elem(ind).fill(second);
  }

  return Row<DataType>(uniqueVals_);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::symmetrizeLabels(Row<DataType>& labels) {
  const Row<DataType> uniqueVals = uniqueCloseAndReplace(labels);
  const std::size_t num_unique = uniqueVals.n_cols;

  if (num_unique == 1) {
    a_ = 1.;
    b_ = 1.;
    labels.ones();
  } else if (num_unique == 2) {
    const auto [min_it, max_it] = std::minmax_element(uniqueVals.cbegin(), uniqueVals.cend());
    const double m = *min_it;
    const double M = *max_it;
    const double range = M - m;

    // Always use the standard symmetrization (the false branch)
    a_ = 2. / range;
    b_ = (m + M) / (m - M);
    labels = sign(a_ * labels + b_);
  } else if (num_unique == 3) {
    // Handle multiclass case with values in {0, 1, 2}
    const Row<DataType> sortedVals = sort(uniqueVals);
    constexpr double eps = static_cast<double>(std::numeric_limits<float>::epsilon());

    if ((std::abs(sortedVals[0]) <= eps) && (std::abs(sortedVals[1] - 0.5) <= eps) &&
        (std::abs(sortedVals[2] - 1.0) <= eps)) {
      a_ = 2.;
      b_ = -1.;
      labels = sign(a_ * labels - 1);
    }
  } else {
    assert(num_unique == 2);
  }
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::symmetrizeLabels() {
  symmetrizeLabels(labels_);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::symmetrize(Row<DataType>& prediction) {
  prediction = sign(a_ * prediction + b_);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::deSymmetrize(Row<DataType>& prediction) {
  // if (false && (loss_ == classifierossFunction::LogLoss)) {
  if (false) {
    // Normalized values were in $\left\lbrace 0,1\right\rightbrace$
    prediction = ((0.5 * sign(prediction) + 0.5) - b_) / a_;
  } else {
    prediction = (sign(prediction) - b_) / a_;
  }
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::setWeights() {
  ;
}

// SFINAE helper to detect if ClassifierType has setRootClassifier method
template<typename T, typename DataType, typename... Args>
class has_setRootClassifier {
private:
    template<typename U>
    static auto test(int) -> decltype(
        std::declval<U>().setRootClassifier(
            std::declval<std::unique_ptr<DecisionTreeClassifier>&>(),
            std::declval<const Mat<DataType>&>(),
            std::declval<Row<DataType>&>(),
            std::declval<std::tuple<Args...> const&>()
        ),
        std::true_type{});
    
    template<typename>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename ClassifierType>
template <typename... Ts>
void CompositeClassifier<ClassifierType>::setRootClassifier(
    std::unique_ptr<ClassifierType>& classifier,
    const Mat<DataType>& dataset,
    Row<CompositeClassifier<ClassifierType>::DataType>& labels,
    Row<CompositeClassifier<ClassifierType>::DataType>& weights,
    std::tuple<Ts...> const& args) {
  
  // Check if ClassifierType has setRootClassifier method (decorator pattern)
  constexpr bool isDecorator = has_setRootClassifier<ClassifierType, DataType, Ts...>::value;
  
  if constexpr (isDecorator) {
    std::cout << "DEBUG: Using decorator pattern - calling classifier's setRootClassifier" << std::endl;
    // Create decorator instance and call its setRootClassifier method
    classifier = std::make_unique<ClassifierType>();
    
    // Get the decorated type and create a classifier of that type
    std::unique_ptr<DecisionTreeClassifier> innerClassifier;
    classifier->setRootClassifier(innerClassifier, dataset, labels, weights, args);
  } else {
    std::cout << "DEBUG: Using direct construction for non-decorator classifier" << std::endl;
    // The calling convention for mlpack classifiers with weight specification:
    // cls{dataset, labels, numClasses, weights, args...)
    // We must remake the tuple in this case
    std::unique_ptr<ClassifierType> cls;

    auto _c = [&cls, &dataset, &labels, &weights](Ts const&... classArgs) {
      cls = std::make_unique<ClassifierType>(dataset, labels, weights, classArgs...);
    };
    std::apply(_c, args);

    // Feedback
    // ...

    classifier = std::move(cls);
  }
}

template <typename ClassifierType>
template <typename... Ts>
void CompositeClassifier<ClassifierType>::setRootClassifier(
    std::unique_ptr<ClassifierType>& classifier,
    const Mat<DataType>& dataset,
    Row<CompositeClassifier<ClassifierType>::DataType>& labels,
    std::tuple<Ts...> const& args) {
  
  // Check if ClassifierType has setRootClassifier method (decorator pattern)
  constexpr bool isDecorator = has_setRootClassifier<ClassifierType, DataType, Ts...>::value;
  
  if constexpr (isDecorator) {
    std::cout << "DEBUG: Using decorator pattern - calling classifier's setRootClassifier (no weights)" << std::endl;
    // Create decorator instance and call its setRootClassifier method
    classifier = std::make_unique<ClassifierType>();
    
    // Get the decorated type and create a classifier of that type
    std::unique_ptr<DecisionTreeClassifier> innerClassifier;
    classifier->setRootClassifier(innerClassifier, dataset, labels, args);
  } else {
    std::cout << "DEBUG: Using direct construction for non-decorator classifier (no weights)" << std::endl;
    std::unique_ptr<ClassifierType> cls;

    auto _c = [&cls, &dataset, &labels](Ts const&... classArgs) {
      cls = std::make_unique<ClassifierType>(dataset, labels, classArgs...);
    };
    std::apply(_c, args);

    Row<DataType> labels_it = labels, prediction;
    float beta = 0.025;
    const std::size_t FEEDBACK_ITERATIONS = 0;

    if (FEEDBACK_ITERATIONS > 0) {
      cls->Classify(dataset, prediction);

      auto _c0 = [&cls, &dataset](Row<DataType>& labels, Ts const&... classArgs) {
        cls = std::make_unique<ClassifierType>(dataset, labels, classArgs...);
      };

      for (std::size_t i = 0; i < FEEDBACK_ITERATIONS; ++i) {
        for (std::size_t j = 0; j < 10; ++j) {
          std::cerr << j << " : " << labels(j) << " : " << prediction(j) << std::endl;
        }
        labels_it = labels_it - beta * prediction;
        auto _cn = [&labels_it, &_c0](Ts const&... classArgs) { _c0(labels_it, classArgs...); };
        std::apply(_cn, args);
        cls->Classify(dataset, prediction);
      }
    }

    classifier = std::move(cls);
  }
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::createRootClassifier(
    std::unique_ptr<ClassifierType>& classifier,
    const Row<CompositeClassifier<ClassifierType>::DataType>& best_leaves) {
  const typename ClassifierType::Args& rootClassifierArgs =
      ClassifierType::_args(allClassifierArgs(partitionSize_ + 1));

  if (ClassifierFileScope::POST_EXTRAPOLATE) {
    // Fit classifier on {dataset_slice, best_leaves}, both subsets of the original data
    // There will be no post-padding of zeros as that is not well-defined for OOS prediction,
    // we just use the classifier below to predict on the larger dataset for this step's
    // prediction

    auto dataset_slice = dataset_.submat(rowMask_, colMask_);
    Leaves allLeaves = best_leaves;

    if (useWeights_ && true) {
      calcWeights();

      setRootClassifier(classifier, dataset_slice, allLeaves, weights_, rootClassifierArgs);
    } else {
      setRootClassifier(classifier, dataset_slice, allLeaves, rootClassifierArgs);
    }

  } else {
    // Fit classifier on {dataset, padded best_leaves}, where padded best_leaves is the
    // label slice padded with zeros to match original dataset size

    // Zero pad labels first
    Leaves allLeaves = zeros<Row<DataType>>(m_);
    allLeaves(colMask_) = best_leaves;

    if (useWeights_ && true) {
      // We opt to express weighted values via the coefficient generation
      // and partition selection, not during classification fitting

      calcWeights();
      setRootClassifier(classifier, dataset_, allLeaves, weights_, rootClassifierArgs);
    } else {
      setRootClassifier(classifier, dataset_, allLeaves, rootClassifierArgs);
    }
  }
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::fit() {
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
  if (serializeModel_) commit();

  // print final stats (only if more than 1 step to avoid duplication)
  if (!quietRun_ && steps_ > 1) {
    printStats(steps_);
  }
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::fit_step(std::size_t stepNum) {
  // Implementation of W-cycle

  if (!this->reuseColMask_) {
    const auto colRatio = static_cast<size_t>(m_ * col_subsample_ratio_);
    colMask_ = PartitionUtils::sortedSubsample2(m_, colRatio);
  }

  const Row<DataType> labels_slice = labels_.submat(zeros<uvec>(1), colMask_);
  std::pair<Row<DataType>, Row<DataType>> coeffs;

  Row<DataType> prediction;
  std::unique_ptr<ClassifierType> classifier;

  if (!this->hasInitialPrediction_) {
    this->latestPrediction_ = this->_constantLeaf(0.0);

    std::unique_ptr<ConstantTreeClassifier> cls_;
    Row<DataType> constantLeaf = ones<Row<DataType>>(labels_.n_elem);
    constantLeaf.fill(mean(labels_slice));

    cls_ = std::make_unique<ConstantTreeClassifier>(dataset_, constantLeaf);

    this->updateModels(std::move(cls_), constantLeaf);
  }

  if constexpr (ClassifierFileScope::W_CYCLE_PREFIT) {
    if constexpr (ClassifierFileScope::DIAGNOSTICS_0_ || ClassifierFileScope::DIAGNOSTICS_1_) {
      std::cerr << fit_prefix(depth_);
      std::cerr << "[*]PRE-FITTING LEAF CLASSIFIER FOR (PARTITIONSIZE, STEPNUM): ("
                << partitionSize_ << ", " << stepNum << " of " << steps_ << ")" << std::endl;
    }

    coeffs = generate_coefficients(labels_slice, colMask_);

    auto [subset_info, best_leaves] = computeOptimalSplit(
        coeffs.first,
        coeffs.second,
        stepNum,
        partitionSize_,
        learningRate_,
        activePartitionRatio_,
        ClassifierFileScope::SUBSET_DIAGNOSTICS);

    createRootClassifier(classifier, best_leaves);

    classifier->Classify(dataset_, prediction);

    this->updateModels(std::move(classifier), prediction);

    this->hasInitialPrediction_ = true;

    if constexpr (ClassifierFileScope::SUBSET_DIAGNOSTICS) {
      std::vector<DataType> gv0 = arma::conv_to<std::vector<DataType>>::from(coeffs.first);
      std::vector<DataType> hv0 = arma::conv_to<std::vector<DataType>>::from(coeffs.second);
      std::vector<DataType> preds = arma::conv_to<std::vector<DataType>>::from(prediction);
      std::vector<DataType> yv0 = arma::conv_to<std::vector<DataType>>::from(labels_);
      std::vector<DataType> yhatv0 =
          arma::conv_to<std::vector<DataType>>::from(this->latestPrediction_);
      std::vector<DataType> best_leavesv0 = arma::conv_to<std::vector<DataType>>::from(best_leaves);
      printSubsets<DataType>(
          subset_info.value(), best_leavesv0, preds, gv0, hv0, yv0, yhatv0, colMask_);
    }

    if constexpr (ClassifierFileScope::DIAGNOSTICS_1_) {
      const Row<DataType> latestPrediction_slice =
          this->latestPrediction_.submat(zeros<uvec>(1), colMask_);
      const Row<DataType> prediction_slice = prediction.submat(zeros<uvec>(1), colMask_);
      constexpr float eps = std::numeric_limits<float>::epsilon();

      std::cerr << "[PRE-FIT ";
      const std::size_t leaves_size = best_leaves.size();
      for (std::size_t i = 0; i < leaves_size; ++i) {
        const bool misclassified = std::abs(best_leaves[i] - prediction_slice[i]) > eps;
        const std::string_view status = misclassified ? "MISCLASSIFIED" : "";
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
    if (ClassifierFileScope::DIAGNOSTICS_1_ || ClassifierFileScope::DIAGNOSTICS_0_) {
      std::cerr << fit_prefix(depth_);
      std::cerr << "[-]FITTING COMPOSITE CLASSIFIER FOR (PARTITIONSIZE, STEPNUM): ("
                << partitionSize_ << ", " << stepNum << " of " << steps_ << ")" << std::endl;
    }

    uvec colMask;
    if (ClassifierFileScope::NEW_COLMASK_FOR_CHILD) {
      int colRatio = static_cast<size_t>(m_ * col_subsample_ratio_);
      colMask = PartitionUtils::sortedSubsample2(m_, colRatio);
    } else {
      colMask = colMask_;
    }

    Context context{};
    ContextManager::childContext(context, *this);

    // allLeaves may not strictly fit the definition of labels here -
    // aside from the fact that it is of double type, it may have more
    // than one class. So we don't want to symmetrize, but we want
    // to remap the redundant values.
    std::unique_ptr<CompositeClassifier<ClassifierType>> classifier;
    if (this->hasInitialPrediction_) {
      classifier.reset(new CompositeClassifier<ClassifierType>(
          dataset_, labels_, this->latestPrediction_, colMask, context));
    } else {
      classifier.reset(
          new CompositeClassifier<ClassifierType>(dataset_, labels_, colMask, context));
    }

    if (ClassifierFileScope::DIAGNOSTICS_1_) {
      std::cerr << "PREFIT: (PARTITIONSIZE, STEPNUM, NUMSTEPS): (" << partitionSize_ << ", "
                << stepNum << ", " << steps_ << ")" << std::endl;
    }

    classifier->fit();

    if (ClassifierFileScope::DIAGNOSTICS_1_) {
      std::cerr << "POSTFIT: (PARTITIONSIZE, STEPNUM, NUMSTEPS): (" << partitionSize_ << ", "
                << stepNum << ", " << steps_ << ")" << std::endl;
    }

    classifier->Predict(dataset_, prediction);

    this->updateModels(std::move(classifier), prediction);

    this->hasInitialPrediction_ = true;
  }
  ////////////////////////
  // END RECURSIVE STEP //
  ////////////////////////

  // If we are in recursive mode and partitionSize <= 2, fall through
  // to this case for the leaf classifier

  if (ClassifierFileScope::DIAGNOSTICS_0_ || ClassifierFileScope::DIAGNOSTICS_1_) {
    std::cerr << fit_prefix(depth_);
    std::cerr << "[*]POST-FITTING LEAF CLASSIFIER FOR (PARTITIONSIZE, STEPNUM): (" << partitionSize_
              << ", " << stepNum << " of " << steps_ << ")" << std::endl;
  }

  if (!this->hasInitialPrediction_) {
    // if (false&& (loss_ == lossFunction::LogLoss)) {
    if (false) {
      this->latestPrediction_ = this->_constantLeaf(mean(labels_slice));
    } else {
      this->latestPrediction_ = this->_constantLeaf(0.0);
    }
  }

  // Generate coefficients g, h
  coeffs = generate_coefficients(labels_slice, colMask_);

  // Compute optimal leaf choice on unrestricted dataset
  auto [subset_info, best_leaves] = computeOptimalSplit(
      coeffs.first,
      coeffs.second,
      stepNum,
      partitionSize_,
      learningRate_,
      activePartitionRatio_,
      ClassifierFileScope::SUBSET_DIAGNOSTICS);

  createRootClassifier(classifier, best_leaves);

  classifier->Classify(dataset_, prediction);

  this->updateModels(std::move(classifier), prediction);

  this->hasInitialPrediction_ = true;

  if (ClassifierFileScope::SUBSET_DIAGNOSTICS) {
    std::vector<DataType> gv0 = arma::conv_to<std::vector<DataType>>::from(coeffs.first);
    std::vector<DataType> hv0 = arma::conv_to<std::vector<DataType>>::from(coeffs.second);
    std::vector<DataType> preds = arma::conv_to<std::vector<DataType>>::from(prediction);
    std::vector<DataType> yv0 = arma::conv_to<std::vector<DataType>>::from(labels_);
    std::vector<DataType> yhatv0 =
        arma::conv_to<std::vector<DataType>>::from(this->latestPrediction_);
    std::vector<DataType> best_leavesv0 = arma::conv_to<std::vector<DataType>>::from(best_leaves);
    printSubsets<DataType>(
        subset_info.value(), best_leavesv0, preds, gv0, hv0, yv0, yhatv0, colMask_);
  }

  if (ClassifierFileScope::DIAGNOSTICS_1_) {
    Row<DataType> latestPrediction_slice = this->latestPrediction_.submat(zeros<uvec>(1), colMask_);
    Row<DataType> prediction_slice = prediction.submat(zeros<uvec>(1), colMask_);
    float eps = std::numeric_limits<float>::epsilon();

    std::cerr << "[POST-FIT ";
    for (std::size_t i = 0; i < best_leaves.size(); ++i) {
      std::string status = "";
      if (fabs(best_leaves[i] - prediction_slice[i]) > eps) status = "MISCLASSIFIED";
      std::cerr << colMask_[i] << " : " << labels_slice[i] << " : " << latestPrediction_slice[i]
                << " :: " << best_leaves[i] << " : " << prediction_slice[i]
                << " :: " << coeffs.first[i] << " : " << coeffs.second[i] << " : " << status
                << std::endl;
    }
    std::cerr << "]" << std::endl;
  }
}

template <typename ClassifierType>
auto CompositeClassifier<ClassifierType>::computeOptimalSplit(
    Row<CompositeClassifier<ClassifierType>::DataType>& g,
    Row<CompositeClassifier<ClassifierType>::DataType>& h,
    std::size_t stepNum,
    std::size_t partitionSize,
    double learningRate,
    double activePartitionRatio,
    bool includeSubsets) -> optLeavesInfo {
  (void)stepNum;

  // Compile-time constants for better optimization
  const int n = static_cast<int>(g.n_cols);
  const int T = static_cast<int>(partitionSize);
  constexpr objective_fn obj_fn = objective_fn::RationalScore;
  constexpr bool risk_partitioning_objective = false;
  constexpr bool use_rational_optimization = true;
  constexpr bool sweep_down = false;
  constexpr double gamma = 0.;
  constexpr double reg_power = 1.;
  constexpr bool find_optimal_t = false;
  constexpr bool reorder_by_weighted_priority = true;

  // More efficient conversion using reserve
  std::vector<DataType> gv0, hv0;
  gv0.reserve(n);
  hv0.reserve(n);

  std::copy(g.begin(), g.end(), std::back_inserter(gv0));
  std::copy(h.begin(), h.end(), std::back_inserter(hv0));

  auto dp0 = DPSolver(
      n,
      T,
      gv0,
      hv0,
      obj_fn,
      risk_partitioning_objective,
      use_rational_optimization,
      gamma,
      reg_power,
      sweep_down,
      find_optimal_t,
      reorder_by_weighted_priority);

  const auto subsets0 = dp0.get_optimal_subsets_extern();
  // constexpr double end_ratio = 0.10; // Currently unused

  Row<DataType> leaf_values0 = arma::zeros<Row<DataType>>(n);

  if (T > 1 || risk_partitioning_objective) {
    // XXX
    const std::size_t start_ind = static_cast<std::size_t>(T * activePartitionRatio);
    const std::size_t end_ind =
        static_cast<std::size_t>((1. - activePartitionRatio) * static_cast<double>(T));

    // const std::size_t start_ind =
    //     risk_partitioning_objective ? 0 : static_cast<std::size_t>(T * activePartitionRatio);
    const std::size_t subsets_size = subsets0.size();

    // Precompute negative learning rate for efficiency
    const double neg_lr = -learningRate;

    for (std::size_t i = start_ind; i < subsets_size; ++i) {
      // for (std::size_t i=0; i<end_ind; ++i) {
      const uvec ind = arma::conv_to<uvec>::from(subsets0[i]);
      const double g_sum = sum(g(ind));
      const double h_sum = sum(h(ind));

      // Avoid division by zero and compute leaf value efficiently
      if (h_sum != 0.0) {
        const double val = neg_lr * g_sum / h_sum;
        // Use vectorized assignment when possible
        leaf_values0.elem(ind).fill(val);
      }
    }
  }

  if (includeSubsets) {
    return std::make_tuple(subsets0, leaf_values0);
  } else {
    return std::make_tuple(std::nullopt, leaf_values0);
  }
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::purge_() {
  // Efficiently deallocate memory by assigning empty containers
  dataset_ = Mat<DataType>();
  labels_ = Row<DataType>();
  dataset_oos_ = Mat<DataType>();
  labels_oos_ = Row<DataType>();
}

template <typename ClassifierType>
std::string CompositeClassifier<ClassifierType>::write() {
  using CerealT = CompositeClassifier<ClassifierType>;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  std::string fileName =
      dumps<CerealT, CerealIArch, CerealOArch>(*this, SerializedType::CLASSIFIER, fldr_);
  return fileName;
}

template <typename ClassifierType>
std::string CompositeClassifier<ClassifierType>::writeColMask() {
  return IB_utils::writeColMask(colMask_, fldr_);
}

template <typename ClassifierType>
std::string CompositeClassifier<ClassifierType>::writePrediction() {
  return IB_utils::writePrediction(this->latestPrediction_, fldr_);
}

template <typename ClassifierType>
std::string CompositeClassifier<ClassifierType>::writeDataset() {
  return IB_utils::writeDatasetIS(dataset_, fldr_);
}

template <typename ClassifierType>
std::string CompositeClassifier<ClassifierType>::writeDatasetOOS() {
  return IB_utils::writeDatasetOOS(dataset_oos_, fldr_);
}

template <typename ClassifierType>
std::string CompositeClassifier<ClassifierType>::writeLabels() {
  return IB_utils::writeLabelsIS(labels_, fldr_);
}

template <typename ClassifierType>
std::string CompositeClassifier<ClassifierType>::writeWeights() {
  return IB_utils::writeWeightsIS(weights_, fldr_);
}

template <typename ClassifierType>
std::string CompositeClassifier<ClassifierType>::writeLabelsOOS() {
  return IB_utils::writeLabelsOOS(labels_oos_, fldr_);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::read(
    CompositeClassifier<ClassifierType>& rhs, std::string fileName) {
  using CerealT = CompositeClassifier<ClassifierType>;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  loads<CerealT, CerealIArch, CerealOArch>(rhs, fileName, fldr_);
}

template <typename ClassifierType>
template <typename MatType>
void CompositeClassifier<ClassifierType>::_predict_in_loop_archive(
    std::vector<std::string>& fileNames,
    MatType&& dataset,
    Row<DataType>& prediction,
    bool postSymmetrize) {
  using C = CompositeClassifier<ClassifierType>;
  std::unique_ptr<C> classifierNew = std::make_unique<C>();
  prediction = zeros<Row<DataType>>(dataset.n_cols);
  Row<DataType> predictionStep;

  bool ignoreSymmetrization = true;
  for (auto& fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "CLS") {
      fileName = strJoin(tokens, '_', 1);
      read(*classifierNew, fileName);
      classifierNew->Predict(std::forward<MatType>(dataset), predictionStep, ignoreSymmetrization);
      prediction += predictionStep;
    }
  }

  if (postSymmetrize) {
    deSymmetrize(prediction);
  }
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::Predict(
    std::string index,
    const Mat<DataType>& dataset,
    Row<DataType>& prediction,
    bool postSymmetrize) {
  std::vector<std::string> fileNames;
  readIndex(index, fileNames, fldr_);

  _predict_in_loop_archive(fileNames, dataset, prediction, postSymmetrize);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::Predict(
    std::string index, Mat<DataType>&& dataset, Row<DataType>& prediction, bool postSymmetrize) {
  std::vector<std::string> fileNames;
  readIndex(index, fileNames, fldr_);

  _predict_in_loop_archive(fileNames, dataset, prediction, postSymmetrize);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::Predict(
    std::string index, Row<DataType>& prediction, bool postSymmetrize) {
  Predict(index, dataset_, prediction, postSymmetrize);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::commit() {
  std::string path, predictionPath, colMaskPath, weightsPath;
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

  /* No - weights are attached to the classifier serialization
     if (useWeights_) {
     weightsPath = writeWeights();
     fileNames_.push_back(weightsPath);
     }
  */

  // std::copy(fileNames_.begin(), fileNames_.end(),std::ostream_iterator<std::string>(std::cout,
  // "\n"));
  indexName_ = writeIndex(fileNames_, fldr_);
  ClassifierList{}.swap(this->models_);
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::checkAccuracyOfArchive() {
  Row<DataType> yhat;
  Predict(yhat);

  Row<DataType> prediction;
  Predict(indexName_, prediction, false);

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

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::printStats(int stepNum) {
  Row<DataType> yhat;

  if (serializeModel_) {
    // Prediction from current archive
    Predict(indexName_, yhat, false);
    if (symmetrized_) {
      deSymmetrize(yhat);
      symmetrize(yhat);
    }
    // checkAccuracyOfArchive();
  } else {
    // Prediction from nonarchived classifier
    Predict(yhat);
    if (symmetrized_) {
      deSymmetrize(yhat);
      symmetrize(yhat);
    }
  }

  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  // auto UTC =
  // std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  std::stringstream datetime;
  datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
  auto suff = datetime.str();

  // Only print stats for top level of recursive call
  if (hasOOSData_) {
    // Convert continuous predictions to binary class predictions for classification error
    Row<DataType> predicted_classes = sign(this->latestPrediction_);

    Row<int> labels_i = conv_to<Row<int>>::from(labels_);
    Row<int> predicted_classes_i = conv_to<Row<int>>::from(predicted_classes);

    double error_is = err(predicted_classes, labels_);
    auto [prec, recall, F1] = precision(labels_i, predicted_classes_i);
    double imb = imbalance(labels_);

    std::cout << suff << " IS (error, precision, recall, F1, imbalance) : (" << error_is << ", "
              << prec << ", " << recall << ", " << F1 << ", " << imb << ")" << std::endl;

    // std::cout << suff << ": "
    //           << "(PARTITION SIZE = " << partitionSize_ << ", STEPS = " << steps_ << ")"
    //           << " STEP: " << stepNum << " IS ERROR: " << error_is << "%" << std::endl;
  }

  if (false and hasOOSData_) {
    Row<DataType> yhat_oos;
    if (serializeModel_) {
      Predict(indexName_, dataset_oos_, yhat_oos, true);
    } else {
      Predict(dataset_oos_, yhat_oos);
    }
    // Convert OOS predictions to binary class predictions for classification error
    Row<DataType> predicted_classes_oos = sign(yhat_oos);
    double error_oos = err(predicted_classes_oos, labels_oos_);
    std::cout << suff << ": "
              << "(PARTITION SIZE = " << partitionSize_ << ", STEPS = " << steps_ << ")"
              << " STEP: " << stepNum << " OOS ERROR: " << error_oos << "%" << std::endl;
  }
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::calcWeights() {
  Row<DataType> res = labels_;
  Row<DataType> uniqueVals = unique(res);

  for (const auto& el : uniqueVals) {
    uvec ind = find(res == el);
    DataType weight = static_cast<DataType>(m_) / static_cast<DataType>(ind.n_elem);
    weights_.elem(ind).fill(weight);
  }

  /*
    Row<DataType> yhat;
    Predict(yhat, colMask_);
    Row<DataType> labels_slice = labels_.submat(zeros<uvec>(1), colMask_);

    weights_ = abs(labels_slice - yhat);
    weights_ = weights_ * (static_cast<DataType>(weights_.n_cols)/sum(weights_));
  */
}

template <typename ClassifierType>
void CompositeClassifier<ClassifierType>::Classify(
    const Mat<DataType>& dataset, Row<DataType>& labels) {
  Predict(dataset, labels);
}

template <typename ClassifierType>
auto CompositeClassifier<ClassifierType>::generate_coefficients(
    const Row<DataType>& labels, const uvec& colMask) -> std::pair<Row<DataType>, Row<DataType>> {
  Row<DataType> yhat;
  Predict(yhat, colMask);

  Row<DataType> g, h;
  lossFn_->loss(yhat, labels, &g, &h, clamp_gradient_, upper_val_, lower_val_);

  if (useWeights_ && false) {
    // normalize weights
    calcWeights();
    g = g % weights_;
  }

  return std::make_pair(g, h);
}

#endif
