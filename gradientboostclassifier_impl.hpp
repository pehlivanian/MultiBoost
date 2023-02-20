#ifndef __GRADIENTCLASSIFIER_IMPL_HPP__
#define __GRADIENTCLASSIFIER_IMPL_HPP__

// #define DEBUG() __debug dd{__FILE__, __FUNCTION__, __LINE__};
#define DEBUG() ;


using row_d = Row<double>;
using row_t = Row<std::size_t>;

using namespace PartitionSize;
using namespace LearningRate;
using namespace LossMeasures;
using namespace IB_utils;

namespace {
  const bool DIAGNOSTICS = false;
}

template<typename DataType, typename ClassifierType, typename... Args>
void 
DiscreteClassifierBase<DataType, ClassifierType, Args...>::encode(const Row<DataType>& labels_d, Row<std::size_t>& labels_t) {

  DEBUG()

  Row<DataType> uniqueVals = sort(unique(labels_d));    
  for (auto it=uniqueVals.begin(); it!=uniqueVals.end(); ++it) {
    uvec ind = find(labels_d == *it);
    std::size_t equiv = std::distance(it, uniqueVals.end()) - 1;
    labels_t.elem(ind).fill(equiv);
    leavesMap_.insert(std::make_pair(static_cast<std::size_t>(equiv), (*it)));
  }
}

template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::purge() {
  DEBUG()

  labels_t_ = ones<Row<std::size_t>>(0);
  // labels_t_.clear();
}

template<typename DataType, typename ClassifierType, typename... Args>
void 
DiscreteClassifierBase<DataType, ClassifierType, Args...>::decode(const Row<std::size_t>& labels_t, Row<DataType>& labels_d) {

  DEBUG()

  labels_d = Row<DataType>(labels_t.n_elem);
  Row<std::size_t> uniqueVals = unique(labels_t);
  for (auto it=uniqueVals.begin(); it!=uniqueVals.end(); ++it) {
    uvec ind = find(labels_t == *it);
    double equiv = leavesMap_[*it];
    labels_d.elem(ind).fill(equiv);
  }    
}

template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::Classify_(const mat& dataset, Row<DataType>& labels) {

  DEBUG()

  Row<std::size_t> labels_t;
  classifier_->Classify(dataset, labels_t);
  decode(labels_t, labels);
  
  // Check error
  /*
    Row<std::size_t> prediction;
    classifier_->Classify(dataset_, prediction);
    const double trainError = err(prediction, labels_t_);
    for (size_t i=0; i<25; ++i)
    std::cout << labels_t_[i] << " ::(2) " << prediction[i] << std::endl;
    std::cout << "dataset size:    " << dataset.n_rows << " x " << dataset.n_cols << std::endl;
    std::cout << "prediction size: " << prediction.n_rows << " x " << prediction.n_cols << std::endl;
    std::cout << "Training error (2): " << trainError << "%." << std::endl;
  */
  
}

template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::Classify_(const mat& dataset, Row<DataType>& labels, mat& predictions) {
  Row<std::size_t> labels_t;
  labels = Row<DataType>(dataset.n_cols);
  classifier_->Classify(dataset, labels_t, predictions);
  decode(labels_t, labels);
}

template<typename DataType, typename ClassifierType, typename... Args>
void
ContinuousClassifierBase<DataType, ClassifierType, Args...>::Classify_(const mat& dataset, Row<DataType>& labels) {

  DEBUG() 
  classifier_->Predict(dataset, labels);
}

template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::setClassifier(const mat& dataset, Row<std::size_t>& labels, Args&&... args) {

  DEBUG()

  classifier_.reset(new ClassifierType(dataset, labels, std::forward<Args>(args)...));
}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::contextInit_(ClassifierContext::Context&& context) {

  loss_ = context.loss;
  partitionSize_ = context.partitionSize;
  partitionRatio_ = context.partitionRatio;
  learningRate_ = context.learningRate;
  steps_ = context.steps;
  symmetrized_ = context.symmetrizeLabels;
  removeRedundantLabels_ = context.removeRedundantLabels;
  row_subsample_ratio_ = context.rowSubsampleRatio;
  col_subsample_ratio_ = context.colSubsampleRatio;
  recursiveFit_ = context.recursiveFit;
  partitionSizeMethod_ = context.partitionSizeMethod;
  learningRateMethod_ = context.learningRateMethod;
  minLeafSize_ = context.minLeafSize;
  minimumGainSplit_ = context.minimumGainSplit;
  maxDepth_ = context.maxDepth;
  numTrees_ = context.numTrees;
  serialize_ = context.serialize;
  serializePrediction_ = context.serializePrediction;
  serializeColMask_ = context.serializeColMask;
  serializationWindow_ = context.serializationWindow;

}

template<typename ClassifierType>
row_d
GradientBoostClassifier<ClassifierType>::_constantLeaf() const {

  DEBUG()

  row_d r;
  r.zeros(dataset_.n_cols);
  return r;
}

template<typename ClassifierType>
row_d
GradientBoostClassifier<ClassifierType>::_randomLeaf(std::size_t numVals) const {

  DEBUG()

  row_d range = linspace<row_d>(-1, 1, numVals+2);
  row_d r(dataset_.n_cols, arma::fill::none);
  std::mt19937 rng;
  std::uniform_int_distribution<std::size_t> dist{1, numVals};
  r.imbue([&](){ return dist(rng);});
  return r;

}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::updateClassifiers(std::unique_ptr<ClassifierBase<DataType, Classifier>>&& classifier,
							   Row<DataType>& prediction) {

  DEBUG()

  latestPrediction_ += prediction;
  classifier->purge();
  classifiers_.push_back(std::move(classifier));
  // predictions_.emplace_back(prediction);
}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::init_() {

  DEBUG()

  // Note these are flipped
  n_ = dataset_.n_rows; 
  m_ = dataset_.n_cols;

  // Initialize rng  
  std::size_t a=1, b=std::max(1, static_cast<int>(m_ * col_subsample_ratio_));
  partitionDist_ = std::uniform_int_distribution<std::size_t>(a, b);							      

  // Make labels members of {-1,1}
  // Note that we pass labels_oos to this classifier for OOS testing
  // at regular intervals, but the external labels (hence labels_oos_)
  // may be in {0, 1} and we leave things like that.
  assert(!(symmetrized_ && removeRedundantLabels_));
  if (symmetrized_) {
    symmetrizeLabels();
  } else if (removeRedundantLabels_) {
    auto uniqueVals = uniqueCloseAndReplace(labels_);
  }

  // partitions
  Partition partition = PartitionUtils::_fullPartition(m_);
  partitions_.push_back(partition);

  // classifiers
  // don't overfit on first classifier
  row_d constantLabels = _constantLeaf();
  std::unique_ptr<ClassifierType> classifier;
  classifier.reset(new ClassifierType(dataset_, 
				      constantLabels,
				      partitionSize_,
				      minLeafSize_,
				      minimumGainSplit_,
				      maxDepth_));

  // first prediction
  if (!hasInitialPrediction_){
    latestPrediction_ = zeros<Row<DataType>>(dataset_.n_cols);
  }

  Row<DataType> prediction;
  classifier->Classify_(dataset_, prediction);

  // update classifier, predictions
  updateClassifiers(std::move(classifier), prediction);

  uvec colMask = linspace<uvec>(0, -1+m_, m_);
  
  if (loss_ == lossFunction::BinomialDeviance) {
    lossFn_ = new BinomialDevianceLoss<double>();
  }
  else if (loss_ == lossFunction::MSE) {
    lossFn_ = new MSELoss<double>();
  }
  else if (loss_ == lossFunction::Savage) {
    lossFn_ = new SavageLoss<double>();
  }
  else if (loss_ == lossFunction::Exp) {
    lossFn_ = new ExpLoss<double>();
  }
  else if (loss_ == lossFunction::Arctan) {
    lossFn_ = new ArctanLoss<double>();
  }
  else if (loss_ == lossFunction::Synthetic) {
    lossFn_ = new SyntheticLoss<double>();
  }

  if (partitionSize_ == 1) {
    recursiveFit_ = false;
    // steps_ = 0;
  }

}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::Predict(Row<DataType>& prediction) {

  DEBUG()

  prediction = latestPrediction_;
}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::Predict(Row<DataType>& prediction, const uvec& colMask) {

  DEBUG()

  Predict(prediction);
  prediction = prediction.submat(zeros<uvec>(1), colMask);

}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::Predict(const mat& dataset, Row<DataType>& prediction, bool ignoreSymmetrization) {

  DEBUG()

  if (serialize_ && indexName_.size()) {
    throw predictionAfterClearedClassifiersException();
  }

  prediction = zeros<Row<DataType>>(dataset.n_cols);

  for (const auto& classifier : classifiers_) {
    Row<DataType> predictionStep;
    classifier->Classify_(dataset, predictionStep);
    prediction += predictionStep;    
  }  

  if (symmetrized_ and not ignoreSymmetrization) {
    deSymmetrize(prediction);
  }

}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::Predict(Row<typename GradientBoostClassifier<ClassifierType>::IntegralLabelType>& prediction) {

  DEBUG()

  row_d prediction_d = conv_to<row_d>::from(prediction);
  Predict(prediction_d);
  prediction = conv_to<row_t>::from(prediction_d);
}


template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::Predict(Row<typename GradientBoostClassifier<ClassifierType>::IntegralLabelType>& prediction, const uvec& colMask) {

  DEBUG()

  row_d prediction_d = conv_to<row_d>::from(prediction);
  Predict(prediction_d, colMask);
  prediction = conv_to<row_t>::from(prediction_d);
}


template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::Predict(const mat& dataset, Row<typename GradientBoostClassifier<ClassifierType>::IntegralLabelType>& prediction) {

  DEBUG()

  row_d prediction_d;
  Predict(dataset, prediction_d);

  if (symmetrized_) {
    deSymmetrize(prediction_d);
  }

  prediction = conv_to<row_t>::from(prediction_d);

}

template<typename ClassifierType>
uvec
GradientBoostClassifier<ClassifierType>::subsampleRows(size_t numRows) {

  DEBUG()

  uvec r = sort(randperm(n_, numRows));
  // uvec r = randperm(n_, numRows);
  return r;
}

template<typename ClassifierType>
uvec
GradientBoostClassifier<ClassifierType>::subsampleCols(size_t numCols) {

  DEBUG()

  uvec r = sort(randperm(m_, numCols));
  // uvec r = randperm(m_, numCols);
  return r;
}

template<typename ClassifierType>
Row<typename GradientBoostClassifier<ClassifierType>::DataType>
GradientBoostClassifier<ClassifierType>::uniqueCloseAndReplace(Row<DataType>& labels) {

  DEBUG()

  Row<DataType> uniqueVals = unique(labels);
  double eps = static_cast<double>(std::numeric_limits<float>::epsilon());
  
  std::vector<std::pair<DataType, DataType>> uniqueByEps;
  std::vector<DataType> uniqueVals_;
  
  uniqueVals_.push_back(uniqueVals[0]);
  
  for (int i=1; i<uniqueVals.n_cols; ++i) {
    bool found = false;
    for (const auto& el : uniqueVals_) {
      if (fabs(uniqueVals[i] - el) <= eps) {
	found = true;
	uniqueByEps.push_back(std::make_pair(uniqueVals[i], el));
      }
    }
    if (!found) {
      uniqueVals_.push_back(uniqueVals[i]);
    }      
  }
  
  // Replace redundant values in labels_
  for(const auto& el : uniqueByEps) {
    uvec ind = find(labels_ == el.first);
    labels.elem(ind).fill(el.second);
  }

  // Now uniqueVals_ matches labels_ characteristics
  return uniqueVals_;
}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::symmetrizeLabels(Row<DataType>& labels) {

  DEBUG()

  Row<DataType> uniqueVals = uniqueCloseAndReplace(labels);

  if (uniqueVals.n_cols == 1) {
    // a_ = fabs(1./uniqueVals(0)); b_ = 0.;
    // labels = sign(labels);
    a_ = 1.; b_ = 1.;
    labels = ones<Row<double>>(labels.n_elem);
  } else if (uniqueVals.size() == 2) {
    double m = *std::min_element(uniqueVals.cbegin(), uniqueVals.cend());
    double M = *std::max_element(uniqueVals.cbegin(), uniqueVals.cend());
    a_ = 2./static_cast<double>(M-m);
    b_ = static_cast<double>(m+M)/static_cast<double>(m-M);
    labels = sign(a_*labels + b_);
    // labels = sign(2 * labels - 1);      
  } else if (uniqueVals.size() == 3) { // for the multiclass case, we may have values in {0, 1, 2}
    uniqueVals = sort(uniqueVals);
    double eps = static_cast<double>(std::numeric_limits<float>::epsilon());
    if ((fabs(uniqueVals[0]) <= eps) &&
	(fabs(uniqueVals[1]-.5) <= eps) &&
	(fabs(uniqueVals[2]-1.) <= eps)) {
      a_ = 2.; b_ = -1.;
      labels = sign(a_*labels - 1);
    }
  }
  else {
    assert(uniqueVals.size() == 2);
  }
  
}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::symmetrizeLabels() {

  DEBUG()

  symmetrizeLabels(labels_);
}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::symmetrize(Row<DataType>& prediction) {

  DEBUG()

  prediction = sign(a_*prediction + b_);
}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::deSymmetrize(Row<DataType>& prediction) {

  DEBUG()

  prediction = (sign(prediction) - b_)/ a_;
}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::fit_step(std::size_t stepNum) {

  DEBUG()

  if (!reuseColMask_) {
    int colRatio = static_cast<size_t>(m_ * col_subsample_ratio_);
    colMask_ = subsampleCols(colRatio);
  }

  row_d labels_slice = labels_.submat(zeros<uvec>(1), colMask_);

  // Compute partition size
  std::size_t partitionSize = computePartitionSize(stepNum, colMask_);
  
  if (DIAGNOSTICS)
    std::cout << "PARTITION SIZE: " << partitionSize 
	      << " (stepNum, steps): " << "(" << stepNum
	      << ", " << steps_ << ")" << std::endl;

  // Compute learning rate
  double learningRate = computeLearningRate(stepNum);

  // Find classifier fit for leaves choice
  Leaves allLeaves = zeros<row_d>(m_);

  Row<DataType> prediction;
  std::unique_ptr<ClassifierType> classifier;

  Row<DataType> prediction_slice;


  if (recursiveFit_ && partitionSize_ > 2) {
    // Reduce partition size
    std::size_t subPartitionSize = static_cast<std::size_t>(partitionSize/2);

    // When considering subproblems, colMask is full
    // uvec subColMask = linspace<uvec>(0, -1+m_, m_);


    // Generate coefficients g, h
    std::pair<rowvec, rowvec> coeffs = generate_coefficients(labels_slice, colMask_);

    // Regenerate coefficients with full colMask
    // coeffs = generate_coefficients(labels_, subColMask);    
    // Compute optimal leaf choice on unrestricted dataset
    // Leaves best_leaves = computeOptimalSplit(coeffs.first, coeffs.second, dataset_, stepNum, subPartitionSize, subColMask);
    // allLeaves = best_leaves;

    Leaves best_leaves = computeOptimalSplit(coeffs.first, coeffs.second, dataset_, stepNum, subPartitionSize, colMask_);

    allLeaves(colMask_) = best_leaves;

    ClassifierContext::Context context{}; 
    
    context.loss = loss_;
    context.partitionSize = subPartitionSize + 1;
    context.partitionRatio = std::min(1., 2*partitionRatio_);
    context.learningRate = learningRate_;
    // XXX
    context.steps = std::max(static_cast<int>(std::log(steps_)), 1);
    // context.steps = std::max(1, static_cast<int>(.25 * std::log(steps_)));
    context.symmetrizeLabels = false;
    context.removeRedundantLabels = true;
    context.rowSubsampleRatio = row_subsample_ratio_;
    context.colSubsampleRatio = col_subsample_ratio_;
    context.recursiveFit = true;
    context.partitionSizeMethod = partitionSizeMethod_;
    context.learningRateMethod = learningRateMethod_;    
    context.minLeafSize = minLeafSize_;
    // context.minLeafSize = static_cast<std::size_t>(.2 * m_ / partitionSize_);
    context.maxDepth = maxDepth_;
    context.minimumGainSplit = minimumGainSplit_;

    if (DIAGNOSTICS)
      std::cout << "SUBPARTITION SIZE: " << subPartitionSize 
		<< "(stepNum, steps): " << "(" << stepNum
		<< ", " << context.steps << ")" << std::endl;
    
    // allLeaves may not strictly fit the definition of labels here - 
    // aside from the fact that it is of double type, it may have more 
    // than one class. So we don't want to symmetrize, but we want 
    // to remap the redundant values.
    std::unique_ptr<GradientBoostClassifier<ClassifierType>> classifier;
    classifier.reset(new GradientBoostClassifier<ClassifierType>(dataset_, 
								 labels_, 
								 latestPrediction_, 
								 colMask_, 
								 context));

    classifier->fit();
    // classifier->Classify_(dataset_, prediction);
    classifier->Predict(dataset_, prediction);

    updateClassifiers(std::move(classifier), prediction);

  } 
  
  // If we are in recursive mode and partitionSize <= 2, fall through
  // to this case for the leaf classifier
  
  // Generate coefficients g, h
  std::pair<rowvec, rowvec> coeffs = generate_coefficients(labels_slice, colMask_);
  
  // Compute optimal leaf choice on unrestricted dataset
  Leaves best_leaves = computeOptimalSplit(coeffs.first, coeffs.second, dataset_, stepNum, partitionSize, colMask_);
  
  // Fit classifier on {dataset, padded best_leaves}
  // Zero pad labels first
  allLeaves(colMask_) = best_leaves;
  
  if (DIAGNOSTICS) {
    std::cerr << "FITTING LEAF CLASSIFIER...";
  }
  classifier.reset(new ClassifierType(dataset_, 
				      allLeaves, 
				      std::move(partitionSize+1), // Since 0 is an additional class value
				      std::move(minLeafSize_),
				      std::move(minimumGainSplit_),
				      std::move(maxDepth_)));
  if (DIAGNOSTICS) {
    std::cerr << "FINISHED." << std::endl;
  }  
  
  mat probabilities;
  classifier->Classify_(dataset_, prediction, probabilities);

  updateClassifiers(std::move(classifier), prediction);

}

template<typename ClassifierType>
typename GradientBoostClassifier<ClassifierType>::Leaves
GradientBoostClassifier<ClassifierType>::computeOptimalSplit(rowvec& g,
							     rowvec& h,
							     mat dataset,
							     std::size_t stepNum, 
							     std::size_t partitionSize,
							     const uvec& colMask) {


  DEBUG()

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

  auto dp = DPSolver(n, T, gv, hv,
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
  for (const auto& subset : subsets) {
    uvec ind = arma::conv_to<uvec>::from(subset);
    double val = -1. * learningRate_ * sum(g(ind))/sum(h(ind));
    for (auto i: ind) {
      leaf_values(i) = val;
    }
  }

  partitions_.emplace_back(subsets);

  return leaf_values;
    
}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::purge() {

  DEBUG()

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

template<typename ClassifierType>
std::string
GradientBoostClassifier<ClassifierType>::write() {

  DEBUG()

  using CerealT = GradientBoostClassifier<ClassifierType>;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  std::string fileName = dumps<CerealT, CerealIArch, CerealOArch>(*this, SerializedType::CLASSIFIER);
  return fileName;
}

template<typename ClassifierType>
std::string
GradientBoostClassifier<ClassifierType>::writeColMask() {

  using CerealT = ColMaskArchive;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  ColMaskArchive cma{colMask_};
  std::string fileName = dumps<CerealT, CerealIArch, CerealOArch>(cma, SerializedType::COLMASK);
  return fileName;

}

template<typename ClassifierType>
std::string
GradientBoostClassifier<ClassifierType>::writePrediction() {
  
  using CerealT = PredictionArchive<DataType>;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  PredictionArchive pa{latestPrediction_};
  std::string fileName = dumps<CerealT, CerealIArch, CerealOArch>(pa, SerializedType::PREDICTION);
  return fileName;
  
}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::read(GradientBoostClassifier<ClassifierType>& rhs,
					      std::string fileName) {

  DEBUG()

  using CerealT = GradientBoostClassifier<ClassifierType>;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  

  loads<CerealT, CerealIArch, CerealOArch>(rhs, fileName);
}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::Predict(std::string index, const mat& dataset, Row<DataType>& prediction, bool postSymmetrize) {

  DEBUG()

  std::vector<std::string> fileNames;
  readIndex(index, fileNames);

  using C = GradientBoostClassifier<ClassifierType>;
  std::unique_ptr<C> classifierNew = std::make_unique<C>();
  prediction = zeros<Row<DataType>>(dataset.n_cols);
  Row<DataType> predictionStep;

  bool ignoreSymmetrization = true;
  for (auto & fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "CLS") {
      fileName = strJoin(tokens, '_', 1);
      read(*classifierNew, fileName);
      classifierNew->Predict(dataset, predictionStep, ignoreSymmetrization);
      prediction += predictionStep;
    }
  }

  if (postSymmetrize) {
    deSymmetrize(prediction);
  }

}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::Predict(std::string index, Row<DataType>& prediction, bool postSymmetrize) {

  DEBUG()

  Predict(index, dataset_, prediction, postSymmetrize);
}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::commit() {

  DEBUG()

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
  ClassifierList{}.swap(classifiers_);
}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::checkAccuracyOfArchive() {

  DEBUG()

  Row<DataType> yhat;
  Predict(yhat); 

  Row<DataType> prediction;
  Predict(indexName_, prediction, false);
  
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

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::printStats(int stepNum) {

  DEBUG()

  Row<DataType> yhat;
  double r;

  if (serialize_) {
    // Prediction from current archive
    Predict(indexName_, yhat, false);
    r = lossFn_->loss(yhat, labels_);
    if (symmetrized_) {
      deSymmetrize(yhat);
      symmetrize(yhat);
    }
    // checkAccuracyOfArchive();
  } else {
    // Prediction from nonarchived classifier
    Predict(yhat); 
    r = lossFn_->loss(yhat, labels_);
    if (symmetrized_) {
      deSymmetrize(yhat); 
      symmetrize(yhat);
    }
  }

  auto now = std::chrono::system_clock::now();
  auto UTC = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  
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
      Predict(indexName_, dataset_oos_, yhat_oos, true);
    } else {
      Predict(dataset_oos_, yhat_oos);
    }
    double error_oos = err(yhat_oos, labels_oos_);
    std::cout << suff<< ": "
	      << "(PARTITION SIZE = " << partitionSize_
	      << ", STEPS = " << steps_ << ") "
	      << "STEP: " << stepNum
	      << " OOS ERROR: " << error_oos << "%" << std::endl;
  }

}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::fit() {

  DEBUG()

  for (std::size_t stepNum=1; stepNum<=steps_; ++stepNum) {
    fit_step(stepNum);
    
    if (DIAGNOSTICS && (stepNum > 5) && ((stepNum%100) == 1)) {
      std::cout << "STEP: " << stepNum << std::endl;
    }

    if ((stepNum > 5) && ((stepNum%serializationWindow_) == 1)) {
      if (serialize_) {
	commit();
      }
      printStats(stepNum);
    }
    
  }

  // Serialize residual
  if (serialize_)
    commit();

  // print final stats
  printStats(steps_);

}

template<typename ClassifierType>
void
GradientBoostClassifier<ClassifierType>::Classify(const mat& dataset, Row<DataType>& labels) {

  DEBUG()

  Predict(dataset, labels);
}

template<typename ClassifierType>
double
GradientBoostClassifier<ClassifierType>::computeLearningRate(std::size_t stepNum) {

  DEBUG()

  double learningRate;

  if (learningRateMethod_ == RateMethod::FIXED) {
    learningRate = learningRate_;
  } else if (learningRateMethod_ == RateMethod::DECREASING) {
    double A = learningRate_, B = -log(.5) / static_cast<double>(steps_);
    learningRate = A * exp(-B * (-1 + stepNum));
  } else if (learningRateMethod_ == RateMethod::INCREASING) {
    double A = learningRate_, B = log(2.) / static_cast<double>(steps_);
    learningRate = A * exp(B * (-1 + stepNum));
  }

  // if ((stepNum%100)==0)
  //   std::cout << "stepNum: " << stepNum << " LEARNING RATE: " << learningRate << std::endl;

  return learningRate;
}

template<typename ClassifierType>
std::size_t
GradientBoostClassifier<ClassifierType>::computePartitionSize(std::size_t stepNum, const uvec& colMask) {

  DEBUG()

  // stepNum is in range [1,...,context.steps]

  std::size_t partitionSize;
  double lowRatio = .05;
  double highRatio = .95;
  int attach = 1000;

  if (partitionSizeMethod_ == SizeMethod::FIXED) {
    return partitionSize_;
  } else if (partitionSizeMethod_ == SizeMethod::FIXED_PROPORTION) {
    partitionSize = static_cast<std::size_t>(partitionRatio_ * row_subsample_ratio_ * colMask.n_rows);
  } else if (partitionSizeMethod_ == SizeMethod::DECREASING) {
    double A = colMask.n_rows, B = log(colMask.n_rows)/steps_;
    partitionSize = std::max(1, static_cast<int>(A * exp(-B * (-1 + stepNum))));
  } else if (partitionSizeMethod_ == SizeMethod::INCREASING) {
    double A = 2., B = log(colMask.n_rows)/static_cast<double>(steps_);
    partitionSize = std::max(1, static_cast<int>(A * exp(B * (-1 + stepNum))));
  } else if (partitionSizeMethod_ == SizeMethod::RANDOM) {
    partitionSize = partitionDist_(default_engine_);
    ;
  } else if (partitionSizeMethod_ == SizeMethod::MULTISCALE) {
    if ((stepNum%attach) < (attach/2)) {
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

template<typename ClassifierType>
std::pair<rowvec, rowvec>
GradientBoostClassifier<ClassifierType>::generate_coefficients(const Row<DataType>& labels, const uvec& colMask) {

  DEBUG()

  rowvec yhat;
  Predict(yhat, colMask);

  rowvec g, h;
  lossFn_->loss(yhat, labels, &g, &h);

  /*
    std::cout << "GENERATE COEFFICIENTS\n";
    std::cout << "g size: " << g.n_rows << " x " << g.n_cols << std::endl;
    // g.print(std::cout);
    std::cout << "h size: " << h.n_rows << " x " << h.n_cols << std::endl;
    // h.print(std::cout);
    for (size_t i=0; i<5; ++i) {
    std::cout << labels[i] << " : " << yhat[i] << std::endl;
    }
  */

  return std::make_pair(g, h);

}

template<typename ClassifierType>
std::pair<rowvec, rowvec>
GradientBoostClassifier<ClassifierType>::generate_coefficients(const Row<DataType>& yhat,
							       const Row<DataType>& y,
							       const uvec& colMask) {

  DEBUG()

  rowvec g, h;
  lossFn_->loss(yhat, y, &g, &h);

  return std::make_pair(g, h);
}
/*
  double
  GradientBoostClassifier<ClassifierType>::imbalance() {
  ;
  }
  // 2.0*((sum(y_train==0)/len(y_train) - .5)**2 + (sum(y_train==1)/len(y_train) - .5)**2)
  */


#endif
