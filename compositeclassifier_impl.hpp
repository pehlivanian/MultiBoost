#ifndef __COMPOSITECLASSIFIER_IMPL_HPP__
#define __COMPOSITECLASSIFIER_IMPL_HPP__

using row_d = Row<double>;
using row_t = Row<std::size_t>;

using namespace PartitionSize;
using namespace LearningRate;
using namespace LossMeasures;
using namespace ModelContext;
using namespace Objectives;
using namespace IB_utils;

namespace {
  const bool POST_EXTRAPOLATE = false;
  const bool DIAGNOSTICS = false;
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::childContext(Context& context, 
						  std::size_t subPartitionSize,
						  double subLearningRate,
						  std::size_t stepSize) {

  context.loss			= loss_;
  context.partitionRatio	= std::min(1., 2*partitionRatio_);
  context.learningRate		= subLearningRate;
  context.baseSteps		= baseSteps_;
  context.symmetrizeLabels	= false;
  context.removeRedundantLabels	= true;
  context.rowSubsampleRatio	= row_subsample_ratio_;
  context.colSubsampleRatio	= col_subsample_ratio_;
  context.recursiveFit		= true;
  context.stepSizeMethod	= stepSizeMethod_;
  context.partitionSizeMethod	= partitionSizeMethod_;
  context.learningRateMethod	= learningRateMethod_;   
  context.steps			= stepSize;

  // Part of model args
  context.numTrees		= numTrees_;
  context.partitionSize		= subPartitionSize + 1;
  context.minLeafSize		= minLeafSize_;
  context.maxDepth		= maxDepth_;
  context.minimumGainSplit	= minimumGainSplit_;
}

template<typename ClassifierType>
AllClassifierArgs
CompositeClassifier<ClassifierType>::allClassifierArgs(std::size_t numClasses) {
  return std::make_tuple(numClasses, minLeafSize_, minimumGainSplit_, numTrees_, maxDepth_);
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::contextInit_(Context&& context) {

  loss_				= context.loss;
  partitionSize_		= context.partitionSize;
  partitionRatio_		= context.partitionRatio;
  learningRate_			= context.learningRate;
  steps_			= context.steps;
  baseSteps_			= context.baseSteps;
  symmetrized_			= context.symmetrizeLabels;
  removeRedundantLabels_	= context.removeRedundantLabels;
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

template<typename ClassifierType>
row_d
CompositeClassifier<ClassifierType>::_constantLeaf() const {

  row_d r;
  r.zeros(dataset_.n_cols);
  return r;
}

template<typename ClassifierType>
row_d
CompositeClassifier<ClassifierType>::_randomLeaf() const {

  row_d r(dataset_.n_cols, arma::fill::none);
  std::mt19937 rng;
  // std::uniform_int_distribution<std::size_t> dist{1, numVals};
  std::uniform_real_distribution<DataType> dist{-learningRate_, learningRate_};
  r.imbue([&](){ return dist(rng);});
  return r;

}

// XXX
template<typename ClassifierType>
template<typename... Ts>
void
CompositeClassifier<ClassifierType>::createClassifier(std::unique_ptr<ClassifierType>& classifier,
						      const mat& dataset,
						      rowvec& labels,
						      std::tuple<Ts...> const& args) {
  // mimic:
  // classifier.reset(new ClassifierType(dataset_,
  //  				      constantLabels,
  // 				      std::forward<typename ClassifierType::Args>(classifierArgs)));

  // Instantiation of ClassifierType should include fit stage; look at profile results
  auto _c = [&classifier, &dataset, &labels](Ts const&... classArgs) { 
    classifier.reset(new ClassifierType(dataset,
					labels,
					classArgs...)
					);
  };
  std::apply(_c, args);
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::updateClassifiers(std::unique_ptr<ClassifierBase<DataType, Classifier>>&& classifier,
						       Row<DataType>& prediction) {
  latestPrediction_ += prediction;
  classifier->purge();
  classifiers_.push_back(std::move(classifier));
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::init_(Context&& context) {

  contextInit_(std::move(context));

  if (serialize_ || serializePrediction_ ||
      serializeColMask_ || serializeDataset_ ||
      serializeLabels_) {

    if (folderName_.size()) {
      fldr_ = boost::filesystem::path{folderName_};
    } else {
      fldr_ = IB_utils::FilterDigestLocation();
      boost::filesystem::create_directory(fldr_);
    }

    // Will keep overwriting context
    std::string contextFilename = "_Context_0.cxt";
    writeBinary<Context>(contextFilename, context, fldr_);
    
  }

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

  // partitions
  Partition partition = PartitionUtils::_fullPartition(m_);
  partitions_.push_back(partition);

  // classifiers
  // don't overfit on first classifier
  row_d constantLabels = _constantLeaf();
  // row_d constantLabels = _randomLeaf();
 
  // numClasses is always the first parameter for the classifier
  // form parameter pack based on ClassifierType
  std::unique_ptr<ClassifierType> classifier;
  const typename ClassifierType::Args& classifierArgs = ClassifierType::_args(allClassifierArgs(partitionSize_));
  createClassifier(classifier, dataset_, constantLabels, classifierArgs);

  // classifier.reset(new ClassifierType(dataset_,
  //  				      constantLabels,
  //				      std::forward<typename ClassifierType::Args>(classifierArgs)));

  // first prediction
  if (!hasInitialPrediction_){
    latestPrediction_ = zeros<Row<DataType>>(dataset_.n_cols);
  }

  Row<DataType> prediction;
  classifier->Classify(dataset_, prediction);

  // update classifier, predictions
  updateClassifiers(std::move(classifier), prediction);

  // set loss function
  lossFn_ = lossMap<DataType>[loss_];

  // ensure this is a leaf classifier for lowest-level call
  if (partitionSize_ == 1) {
    recursiveFit_ = false;
  }

}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::Predict(Row<DataType>& prediction) {

  prediction = latestPrediction_;
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::Predict(Row<DataType>& prediction, const uvec& colMask) {

  Predict(prediction);
  prediction = prediction.submat(zeros<uvec>(1), colMask);

}

template<typename ClassifierType>
template<typename MatType>
void
CompositeClassifier<ClassifierType>::_predict_in_loop(MatType&& dataset, Row<DataType>& prediction, bool ignoreSymmetrization) {

  prediction = zeros<Row<DataType>>(dataset.n_cols);

  for (const auto& classifier : classifiers_) {
    Row<DataType> predictionStep;
    classifier->Classify(dataset, predictionStep);
    prediction += predictionStep;    
  }  

  if (symmetrized_ and not ignoreSymmetrization) {
    deSymmetrize(prediction);
  }
  
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::Predict(const mat& dataset, Row<DataType>& prediction, bool ignoreSymmetrization) {

  if (serialize_ && indexName_.size()) {
    throw predictionAfterClearedClassifiersException();
    return;
  }
  
  _predict_in_loop(dataset, prediction, ignoreSymmetrization);

}

template<typename ClassifierType>
void 
CompositeClassifier<ClassifierType>::Predict(mat&& dataset, Row<DataType>& prediction, bool ignoreSymmetrization) {

  if (serialize_ && indexName_.size()) {
    throw predictionAfterClearedClassifiersException();
    return;
  }
  
  _predict_in_loop(std::move(dataset), prediction, ignoreSymmetrization);
  
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::Predict(Row<typename CompositeClassifier<ClassifierType>::IntegralLabelType>& prediction) {

  row_d prediction_d = conv_to<row_d>::from(prediction);
  Predict(prediction_d);
  prediction = conv_to<row_t>::from(prediction_d);
}


template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::Predict(Row<typename CompositeClassifier<ClassifierType>::IntegralLabelType>& prediction, const uvec& colMask) {

  row_d prediction_d = conv_to<row_d>::from(prediction);
  Predict(prediction_d, colMask);
  prediction = conv_to<row_t>::from(prediction_d);
}


template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::Predict(const mat& dataset, Row<typename CompositeClassifier<ClassifierType>::IntegralLabelType>& prediction) {

  row_d prediction_d;
  Predict(dataset, prediction_d);

  if (symmetrized_) {
    deSymmetrize(prediction_d);
  }

  prediction = conv_to<row_t>::from(prediction_d);

}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::Predict(mat&& dataset, Row<typename CompositeClassifier<ClassifierType>::IntegralLabelType>& prediction) {

  row_d prediction_d;
  Predict(std::move(dataset), prediction_d);

  if (symmetrized_) {
    deSymmetrize(prediction_d);
  }

  prediction = conv_to<row_t>::from(prediction_d);
}

template<typename ClassifierType>
uvec
CompositeClassifier<ClassifierType>::subsampleRows(size_t numRows) {

  // XXX
  // Necessary?
  // uvec r = sort(randperm(n_, numRows));
  // uvec r = randperm(n_, numRows);
  uvec r = PartitionUtils::sortedSubsample2(n_, numRows);
  return r;
}

template<typename ClassifierType>
uvec
CompositeClassifier<ClassifierType>::subsampleCols(size_t numCols) {

  // XXX
  // Necessary?
  // uvec r = sort(randperm(m_, numCols));
  // uvec r = randperm(m_, numCols);
  uvec r = PartitionUtils::sortedSubsample2(n_, numCols);
  return r;
}

template<typename ClassifierType>
Row<typename CompositeClassifier<ClassifierType>::DataType>
CompositeClassifier<ClassifierType>::uniqueCloseAndReplace(Row<DataType>& labels) {

  Row<DataType> uniqueVals = unique(labels);
  double eps = static_cast<double>(std::numeric_limits<float>::epsilon());
  
  std::vector<std::pair<DataType, DataType>> uniqueByEps;
  std::vector<DataType> uniqueVals_;
  
  uniqueVals_.push_back(uniqueVals[0]);
  
  for (std::size_t i=1; i<uniqueVals.n_cols; ++i) {
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
CompositeClassifier<ClassifierType>::symmetrizeLabels(Row<DataType>& labels) {

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
CompositeClassifier<ClassifierType>::symmetrizeLabels() {

  symmetrizeLabels(labels_);
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::symmetrize(Row<DataType>& prediction) {

  prediction = sign(a_*prediction + b_);
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::deSymmetrize(Row<DataType>& prediction) {

  prediction = (sign(prediction) - b_)/ a_;
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::fit_step(std::size_t stepNum) {

  if (!reuseColMask_) {
    int colRatio = static_cast<size_t>(m_ * col_subsample_ratio_);
    // Equivalent, second a little faster, see benchmarks
    // colMask_ = subsampleCols(colRatio);
    colMask_ = PartitionUtils::sortedSubsample2(m_, colRatio);
  }

  row_d labels_slice = labels_.submat(zeros<uvec>(1), colMask_);

  Leaves allLeaves = zeros<row_d>(m_), best_leaves;

  Row<DataType> prediction, prediction_slice;
  std::unique_ptr<ClassifierType> classifier;

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
    std::unique_ptr<CompositeClassifier<ClassifierType>> classifier;
    classifier.reset(new CompositeClassifier<ClassifierType>(dataset_, 
							     labels_, 
							     latestPrediction_, 
							     colMask_, 
							     context));

    classifier->fit();

    classifier->Predict(dataset_, prediction);

    updateClassifiers(std::move(classifier), prediction);

  } 
  ////////////////////////
  // END RECURSIVE STEP //
  ////////////////////////
  
  // If we are in recursive mode and partitionSize <= 2, fall through
  // to this case for the leaf classifier

  if (DIAGNOSTICS)
    std::cout << "FITTING CLASSIFIER FOR (PARTITIONSIZE, STEPNUM, NUMSTEPS): ("
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
  
  if (POST_EXTRAPOLATE) {
    // Fit classifier on {dataset_slice, best_leaves}, both subsets of the original data
    // There will be no post-padding of zeros as that is not defined for OOS prediction, we
    // just use the classifier below to predict on the larger dataset for this step's
    // prediction
    uvec rowMask = linspace<uvec>(0, -1+n_, n_);
    auto dataset_slice = dataset_.submat(rowMask, colMask_);
    
    const typename ClassifierType::Args& rootClassifierArgs = ClassifierType::_args(allClassifierArgs(partitionSize+1));
    createClassifier(classifier, dataset_slice, best_leaves, rootClassifierArgs);

    // classifier.reset(new ClassifierType(dataset_slice,
    //				best_leaves,
    //				std::forward(rootClassifierArgs)));		     
  } else {
    // Fit classifier on {dataset, padded best_leaves}
    // Zero pad labels first
    allLeaves(colMask_) = best_leaves;

    const typename ClassifierType::Args& rootClassifierArgs = ClassifierType::_args(allClassifierArgs(partitionSize+1));
    createClassifier(classifier, dataset_, allLeaves, rootClassifierArgs);
    
    // classifier.reset(new ClassifierType(dataset_,
    // 				allLeaves,
    // 				std::forward(rootClassifierArgs)));
  }

  classifier->Classify(dataset_, prediction);

  updateClassifiers(std::move(classifier), prediction);

}

template<typename ClassifierType>
typename CompositeClassifier<ClassifierType>::Leaves
CompositeClassifier<ClassifierType>::computeOptimalSplit(rowvec& g,
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

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::purge_() {

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
CompositeClassifier<ClassifierType>::write() {

  using CerealT = CompositeClassifier<ClassifierType>;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  std::string fileName = dumps<CerealT, CerealIArch, CerealOArch>(*this, SerializedType::CLASSIFIER, fldr_);
  return fileName;
}

template<typename ClassifierType>
std::string
CompositeClassifier<ClassifierType>::writeColMask() {

  return IB_utils::writeColMask(colMask_, fldr_);
}

template<typename ClassifierType>
std::string
CompositeClassifier<ClassifierType>::writePrediction() {

  return IB_utils::writePrediction(latestPrediction_, fldr_);
}

template<typename ClassifierType>
std::string
CompositeClassifier<ClassifierType>::writeDataset() {
  return IB_utils::writeDatasetIS(dataset_, fldr_);
}

template<typename ClassifierType>
std::string
CompositeClassifier<ClassifierType>::writeDatasetOOS() {
  return IB_utils::writeDatasetOOS(dataset_oos_, fldr_);
}

template<typename ClassifierType>
std::string
CompositeClassifier<ClassifierType>::writeLabels() {
  return IB_utils::writeLabelsIS(labels_, fldr_);
}

template<typename ClassifierType>
std::string
CompositeClassifier<ClassifierType>::writeLabelsOOS() {
  return IB_utils::writeLabelsOOS(labels_oos_, fldr_);
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::read(CompositeClassifier<ClassifierType>& rhs,
					  std::string fileName) {

  using CerealT = CompositeClassifier<ClassifierType>;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;  

  loads<CerealT, CerealIArch, CerealOArch>(rhs, fileName, fldr_);
}

template<typename ClassifierType>
template<typename MatType>
void
CompositeClassifier<ClassifierType>::_predict_in_loop_archive(std::vector<std::string>& fileNames, 
							      MatType&& dataset, 
							      Row<DataType>& prediction, 
							      bool postSymmetrize) {

  using C = CompositeClassifier<ClassifierType>;
  std::unique_ptr<C> classifierNew = std::make_unique<C>();
  prediction = zeros<Row<DataType>>(dataset.n_cols);
  Row<DataType> predictionStep;

  bool ignoreSymmetrization = true;
  for (auto & fileName : fileNames) {
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

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::Predict(std::string index, const mat& dataset, Row<DataType>& prediction, bool postSymmetrize) {

  std::vector<std::string> fileNames;
  readIndex(index, fileNames, fldr_);

  _predict_in_loop_archive(fileNames, dataset, prediction, postSymmetrize);

}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::Predict(std::string index, mat&& dataset, Row<DataType>& prediction, bool postSymmetrize) {
  
  std::vector<std::string> fileNames;
  readIndex(index, fileNames, fldr_);

  _predict_in_loop_archive(fileNames, dataset, prediction, postSymmetrize);
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::Predict(std::string index, Row<DataType>& prediction, bool postSymmetrize) {

  Predict(index, dataset_, prediction, postSymmetrize);
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::commit() {

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
  indexName_ = writeIndex(fileNames_, fldr_);  
  ClassifierList{}.swap(classifiers_);
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::checkAccuracyOfArchive() {

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
CompositeClassifier<ClassifierType>::printStats(int stepNum) {

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
CompositeClassifier<ClassifierType>::fit() {

  for (int stepNum=1; stepNum<=steps_; ++stepNum) {
    fit_step(stepNum);
    
    if ((stepNum > 5) && ((stepNum%serializationWindow_) == 1)) {
      if (serialize_) {
	commit();
      }
      if (!quietRun_) {
	printStats(stepNum);
      }
    }
    
  }

  // Serialize residual
  if (serialize_)
    commit();

  // print final stats
  if (!quietRun_) {
    printStats(steps_);
  }

}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::Classify(const mat& dataset, Row<DataType>& labels) {

  Predict(dataset, labels);
}

template<typename ClassifierType>
double
CompositeClassifier<ClassifierType>::computeLearningRate(std::size_t stepNum) {

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

template<typename ClassifierType>
std::size_t
CompositeClassifier<ClassifierType>::computeSubPartitionSize(std::size_t stepNum) {
  
  (void)stepNum;

  return static_cast<std::size_t>(partitionSize_/2);
}

template<typename ClassifierType>
double
CompositeClassifier<ClassifierType>::computeSubLearningRate(std::size_t stepNum) {

  (void)stepNum;

  return learningRate_;
}

template<typename ClassifierType>
std::size_t
CompositeClassifier<ClassifierType>::computeSubStepSize(std::size_t stepNum) {

  (void)stepNum;

  double mult = 1.;
  return std::max(1, static_cast<int>(mult * std::log(steps_)));    
}

template<typename ClassifierType>
std::size_t
CompositeClassifier<ClassifierType>::computePartitionSize(std::size_t stepNum, const uvec& colMask) {

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

template<typename ClassifierType>
std::pair<rowvec, rowvec>
CompositeClassifier<ClassifierType>::generate_coefficients(const Row<DataType>& labels, const uvec& colMask) {

  rowvec yhat;
  Predict(yhat, colMask);

  rowvec g, h;
  lossFn_->loss(yhat, labels, &g, &h);

  return std::make_pair(g, h);

}

template<typename ClassifierType>
std::pair<rowvec, rowvec>
CompositeClassifier<ClassifierType>::generate_coefficients(const Row<DataType>& yhat,
							   const Row<DataType>& y,
							   const uvec& colMask) {
  
  (void)colMask;

  rowvec g, h;
  lossFn_->loss(yhat, y, &g, &h);
  
  return std::make_pair(g, h);
}
/*
  double
  CompositeClassifier<ClassifierType>::imbalance() {
  ;
  }
  // 2.0*((sum(y_train==0)/len(y_train) - .5)**2 + (sum(y_train==1)/len(y_train) - .5)**2)
  */

#endif
