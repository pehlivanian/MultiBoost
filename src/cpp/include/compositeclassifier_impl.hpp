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

namespace ClassifierFileScope{
  const bool POST_EXTRAPOLATE = false;
  const bool W_CYCLE_PREFIT = true;
  const bool DIAGNOSTICS_0_ = true;
  const bool DIAGNOSTICS_1_ = false;
  const std::string DIGEST_PATH = 
    "/home/charles/src/C++/sandbox/Inductive-Boost/digest/classify";
} // namespace ClassifierFileScope

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::childContext(Context& context) {

  auto [partitionSize, 
	stepSize, 
	learningRate, 
	activePartitionRatio] = computeChildPartitionInfo();
  auto [maxDepth, 
	minLeafSize, 
	minimumGainSplit] = computeChildModelInfo();

  context.loss			= loss_;
  context.clamp_gradient	= clamp_gradient_;
  context.upper_val		= upper_val_;
  context.lower_val		= lower_val_;
  context.baseSteps		= baseSteps_;
  context.symmetrizeLabels	= false;
  context.removeRedundantLabels	= true;
  context.rowSubsampleRatio	= row_subsample_ratio_;
  context.colSubsampleRatio	= col_subsample_ratio_;
  context.recursiveFit		= true;
  context.numTrees		= numTrees_;

  context.partitionSize		= partitionSize+1;
  context.steps			= stepSize;
  context.learningRate		= learningRate;
  context.activePartitionRatio	= activePartitionRatio;

  context.maxDepth		= maxDepth;
  context.minLeafSize		= minLeafSize;
  context.minimumGainSplit	= minimumGainSplit;

  auto it = std::find(childPartitionSize_.cbegin(), childPartitionSize_.cend(), partitionSize);
  auto ind = std::distance(childPartitionSize_.cbegin(), it);
  // Must ensure that ind > 0; may happen if partition size is the same through 2 steps
  ind = ind > 0 ? ind : 1;

  context.childPartitionSize	= std::vector<std::size_t>(childPartitionSize_.cbegin()+ind,
							   childPartitionSize_.cend());
  context.childNumSteps		= std::vector<std::size_t>(childNumSteps_.cbegin()+ind,
							   childNumSteps_.cend());
  context.childLearningRate	= std::vector<double>(childLearningRate_.cbegin()+ind,
						      childLearningRate_.cend());
  context.childActivePartitionRatio = std::vector<double>(childActivePartitionRatio_.cbegin()+ind,
							  childActivePartitionRatio_.cend());
  // Model args
  context.childMinLeafSize	= std::vector<std::size_t>(childMinLeafSize_.cbegin()+ind,
							   childMinLeafSize_.cend());
  context.childMaxDepth		= std::vector<std::size_t>(childMaxDepth_.cbegin()+ind,
							   childMaxDepth_.cend());
  context.childMinimumGainSplit	= std::vector<double>(childMinimumGainSplit_.cbegin()+ind,
						      childMinimumGainSplit_.cend());

  context.depth			= depth_ + 1;

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
  clamp_gradient_		= context.clamp_gradient;
  upper_val_			= context.upper_val;
  lower_val_			= context.lower_val;

  partitionSize_		= context.childPartitionSize[0];
  steps_			= context.childNumSteps[0];
  learningRate_			= context.childLearningRate[0];
  activePartitionRatio_		= context.childActivePartitionRatio[0];

  minLeafSize_			= context.childMinLeafSize[0];
  minimumGainSplit_		= context.childMinimumGainSplit[0];
  maxDepth_			= context.childMaxDepth[0];
  
  baseSteps_			= context.baseSteps;
  symmetrized_			= context.symmetrizeLabels;
  removeRedundantLabels_	= context.removeRedundantLabels;
  quietRun_			= context.quietRun;
  row_subsample_ratio_		= context.rowSubsampleRatio;
  col_subsample_ratio_		= context.colSubsampleRatio;
  recursiveFit_			= context.recursiveFit;

  childPartitionSize_		= context.childPartitionSize;
  childNumSteps_		= context.childNumSteps;
  childLearningRate_		= context.childLearningRate;
  childActivePartitionRatio_	= context.childActivePartitionRatio;

  childMinLeafSize_		= context.childMinLeafSize;
  childMaxDepth_		= context.childMaxDepth;
  childMinimumGainSplit_	= context.childMinimumGainSplit;

  numTrees_			= context.numTrees;
  serializeModel_		= context.serializeModel;
  serializePrediction_		= context.serializePrediction;
  serializeColMask_		= context.serializeColMask;
  serializeDataset_		= context.serializeDataset;
  serializeLabels_		= context.serializeLabels;
  serializationWindow_		= context.serializationWindow;

  depth_			= context.depth;

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
CompositeClassifier<ClassifierType>::_constantLeaf(double val) const {
  
  row_d r;
  r.ones(dataset_.n_cols);
  r *= val;
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

  if (serializeModel_ || serializePrediction_ ||
      serializeColMask_ || serializeDataset_ ||
      serializeLabels_) {

    if (folderName_.size()) {
      fldr_ = boost::filesystem::path{folderName_};
    } else {
      fldr_ = IB_utils::FilterDigestLocation(boost::filesystem::path{ClassifierFileScope::DIGEST_PATH});
      boost::filesystem::create_directory(fldr_);
    }
    
  }

  // Will keep overwriting context
  std::string contextFilename = "_Context_0.cxt";
  writeBinary<Context>(contextFilename, context, fldr_);

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
  rowMask_ = linspace<uvec>(0, -1+n_, n_);

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

  // Set latestPrediction to 0 if not passed
  if (!hasInitialPrediction_) {
    latestPrediction_ = _constantLeaf(0.0);
  }

  // set loss function
  lossFn_ = lossMap<DataType>[loss_];

  // ensure this is a leaf classifier for lowest-level call
  if (childPartitionSize_.size() <= 1) {
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

  if (serializeModel_ && indexName_.size()) {
    throw predictionAfterClearedClassifiersException();
    return;
  }
  
  _predict_in_loop(dataset, prediction, ignoreSymmetrization);

}

template<typename ClassifierType>
void 
CompositeClassifier<ClassifierType>::Predict(mat&& dataset, Row<DataType>& prediction, bool ignoreSymmetrization) {

  if (serializeModel_ && indexName_.size()) {
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
  // Necessary? unittest fail without sort
  // uvec r = sort(randperm(n_, numRows));
  // uvec r = randperm(n_, numRows);
  uvec r = PartitionUtils::sortedSubsample2(n_, numRows);
  return r;
}

template<typename ClassifierType>
uvec
CompositeClassifier<ClassifierType>::subsampleCols(size_t numCols) {

  // XXX
  // Necessary? unittest fail without sort
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
    a_ = 1.; b_ = 1.;
    labels = ones<Row<double>>(labels.n_elem);
  } else if (uniqueVals.size() == 2) {
    double m = *std::min_element(uniqueVals.cbegin(), uniqueVals.cend());
    double M = *std::max_element(uniqueVals.cbegin(), uniqueVals.cend());
    if (false && (loss_ == lossFunction::LogLoss)) {
      // Normalize so that $y \in \left\lbrace 0,1\right\rbrace$
      a_ = 1./(M-m);
      b_ = -1*static_cast<double>(m)/static_cast<double>(M-m);
      labels = a_*labels + b_;
    } else {
      a_ = 2./static_cast<double>(M-m);
      b_ = static_cast<double>(m+M)/static_cast<double>(m-M);
      labels = sign(a_*labels + b_);
    }
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

  if (false && (loss_ == lossFunction::LogLoss)) {
      // Normalized values were in $\left\lbrace 0,1\right\rightbrace$
      prediction = ((0.5*sign(prediction)+0.5) - b_)/ a_;
    }
    else {
      prediction = (sign(prediction) - b_)/ a_;
    }
}

template<typename ClassifierType>
template<typename... Ts>
void
CompositeClassifier<ClassifierType>::setRootClassifier(std::unique_ptr<ClassifierType>& classifier,
						       const mat& dataset,
						       rowvec& labels,
						       std::tuple<Ts...> const& args) {
  // mimic:
  // classifier.reset(new ClassifierType(dataset_,
  //  				      constantLabels,
  // 				      std::forward<typename ClassifierType::Args>(classifierArgs)));

  // Instantiation of ClassifierType should include fit stage; no need to call explicity
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
CompositeClassifier<ClassifierType>::createRootClassifier(std::unique_ptr<ClassifierType>& classifier,
							  uvec rowMask, 
							  uvec colMask, 
							  const row_d& best_leaves) {

  const typename ClassifierType::Args& rootClassifierArgs = ClassifierType::_args(allClassifierArgs(partitionSize_+1));
  
  if (ClassifierFileScope::POST_EXTRAPOLATE) {
    // Fit classifier on {dataset_slice, best_leaves}, both subsets of the original data
    // There will be no post-padding of zeros as that is not well-defined for OOS prediction,
    // we just use the classifier below to predict on the larger dataset for this step's
    // prediction

    auto dataset_slice = dataset_.submat(rowMask, colMask);
    Leaves allLeaves = best_leaves;
    
    setRootClassifier(classifier, dataset_slice, allLeaves, rootClassifierArgs);

  } else {
    // Fit classifier on {dataset, padded best_leaves}, where padded best_leaves is the
    // label slice padded with zeros to match original dataset size

    // Zero pad labels first
    Leaves allLeaves = zeros<row_d>(m_);
    allLeaves(colMask) = best_leaves;

    setRootClassifier(classifier, dataset_, allLeaves, rootClassifierArgs);
    
  }
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::fit_step(std::size_t stepNum) {
  // Implementation of W-cycle
  
  if (!reuseColMask_) {
    int colRatio = static_cast<size_t>(m_ * col_subsample_ratio_);
    colMask_ = PartitionUtils::sortedSubsample2(m_, colRatio);
  }

  row_d labels_slice = labels_.submat(zeros<uvec>(1), colMask_);
  row_d best_leaves;
  std::pair<rowvec, rowvec> coeffs;
  
  Row<DataType> prediction;
  std::unique_ptr<ClassifierType> classifier;

  if (!hasInitialPrediction_) {
    if (false && (loss_ == lossFunction::LogLoss)) {
      latestPrediction_ = _constantLeaf(mean(labels_slice));
    } else {
      latestPrediction_ = _constantLeaf(0.0);
    }
  }

  if (ClassifierFileScope::W_CYCLE_PREFIT) {

    if (ClassifierFileScope::DIAGNOSTICS_0_ || ClassifierFileScope::DIAGNOSTICS_1_) {
      std::cerr << fit_prefix(depth_);
      std::cerr << "[*]PRE-FITTING LEAF CLASSIFIER FOR (PARTITIONSIZE, STEPNUM): ("
		<< partitionSize_ << ", "
		<< stepNum << " of "
		<< steps_ << ")"
		<< std::endl;
    }
    
    coeffs = generate_coefficients(labels_slice, colMask_);
    
    best_leaves = computeOptimalSplit(coeffs.first,
				      coeffs.second,
				      stepNum,
				      partitionSize_,
				      learningRate_,
				      activePartitionRatio_,
				      colMask_);
    
    createRootClassifier(classifier, rowMask_, colMask_, best_leaves);
    
    classifier->Classify(dataset_, prediction);
    
    if (ClassifierFileScope::DIAGNOSTICS_1_) {    
      rowvec yhat_debug;
      Predict(yhat_debug, colMask_);
      
      for (std::size_t i=0; i<best_leaves.size(); ++i) {
	std::cerr << labels_slice[i] << " : "
		  << yhat_debug[i] << " : "
		  << best_leaves[i] << " : "
		  << prediction[i] << " : "
		  << coeffs.first[i] << " : " 
		  << coeffs.second[i] << std::endl;
      }
    }
    
    updateClassifiers(std::move(classifier), prediction);
    
    hasInitialPrediction_ = true;
  }

  //////////////////////////
  // BEGIN RECURSIVE STEP //
  //////////////////////////
  if (recursiveFit_ && (childPartitionSize_.size() > 1)) {

    if (ClassifierFileScope::DIAGNOSTICS_1_ || ClassifierFileScope::DIAGNOSTICS_0_) {
      std::cerr << fit_prefix(depth_);
      std::cerr << "[-]FITTING COMPOSITE CLASSIFIER FOR (PARTITIONSIZE, STEPNUM): ("
		<< partitionSize_ << ", "
		<< stepNum << " of "
		<< steps_ << ")"
		<< std::endl;      
    }

    Context context{};      
    childContext(context);

    // allLeaves may not strictly fit the definition of labels here - 
    // aside from the fact that it is of double type, it may have more 
    // than one class. So we don't want to symmetrize, but we want 
    // to remap the redundant values.
    std::unique_ptr<CompositeClassifier<ClassifierType>> classifier;
    if (hasInitialPrediction_) {
      classifier.reset(new CompositeClassifier<ClassifierType>(dataset_, 
							       labels_, 
							       latestPrediction_, 
							       colMask_, 
							       context));
    } else {
      classifier.reset(new CompositeClassifier<ClassifierType>(dataset_,
							       labels_,
							       colMask_,
							       context));
    }

    if (ClassifierFileScope::DIAGNOSTICS_1_) {

      std::cerr << "PREFIT: (PARTITIONSIZE, STEPNUM, NUMSTEPS): ("
		<< partitionSize_ << ", "
		<< stepNum << ", "
		<< steps_ << ")"
		<< std::endl;

    }

    classifier->fit();

    if (ClassifierFileScope::DIAGNOSTICS_1_) {

      std::cerr << "POSTFIT: (PARTITIONSIZE, STEPNUM, NUMSTEPS): ("
		<< partitionSize_ << ", "
		<< stepNum << ", "
		<< steps_ << ")"
		<< std::endl;

    }


    classifier->Predict(dataset_, prediction);

    updateClassifiers(std::move(classifier), prediction);

    hasInitialPrediction_ = true;

  } 
  ////////////////////////
  // END RECURSIVE STEP //
  ////////////////////////
  
  // If we are in recursive mode and partitionSize <= 2, fall through
  // to this case for the leaf classifier

  if (ClassifierFileScope::DIAGNOSTICS_0_ || ClassifierFileScope::DIAGNOSTICS_1_) {
    std::cerr << fit_prefix(depth_);
    std::cerr << "[*]POST-FITTING LEAF CLASSIFIER FOR (PARTITIONSIZE, STEPNUM): ("
	      << partitionSize_ << ", "
	      << stepNum << " of "
	      << steps_ << ")"
	      << std::endl;
  }

  if (!hasInitialPrediction_){
    if (false&& (loss_ == lossFunction::LogLoss)) {
      latestPrediction_ = _constantLeaf(mean(labels_slice));
    } else {
      latestPrediction_ = _constantLeaf(0.0);
    }
  }
  
  // Generate coefficients g, h
  coeffs = generate_coefficients(labels_slice, colMask_);
  
  // Compute optimal leaf choice on unrestricted dataset
  best_leaves = computeOptimalSplit(coeffs.first, 
				    coeffs.second, 
				    stepNum, 
				    partitionSize_, 
				    learningRate_,
				    activePartitionRatio_,
				    colMask_);

  createRootClassifier(classifier, rowMask_, colMask_, best_leaves);

  classifier->Classify(dataset_, prediction);

  if (ClassifierFileScope::DIAGNOSTICS_1_) {
    
    rowvec yhat_debug;
    Predict(yhat_debug, colMask_);
    
    for (std::size_t i=0; i<best_leaves.size(); ++i) {
      std::cerr << labels_slice[i] << " : "
		<< yhat_debug[i] << " : "
		<< best_leaves[i] << " : "
		<< prediction[i] << " : "
		<< coeffs.first[i] << " : " 
		<< coeffs.second[i] << std::endl;
    }
  }

  updateClassifiers(std::move(classifier), prediction);
  
  hasInitialPrediction_ = true;

}

template<typename ClassifierType>
typename CompositeClassifier<ClassifierType>::Leaves
CompositeClassifier<ClassifierType>::computeOptimalSplit(rowvec& g,
							 rowvec& h,
							 std::size_t stepNum, 
							 std::size_t partitionSize,
							 double learningRate,
							 double activePartitionRatio,
							 const uvec& colMask) {

  (void)stepNum;

  // We should implement several methods here
  int n = colMask.n_rows, T = partitionSize;
  objective_fn obj_fn					= objective_fn::RationalScore;
  bool risk_partitioning_objective			= false;
  bool use_rational_optimization			= true;
  bool sweep_down					= false;
  double gamma						= 0.;
  double reg_power					= 1.;
  bool find_optimal_t					= false;

  std::vector<double> gv0 = arma::conv_to<std::vector<double>>::from(g);
  std::vector<double> hv0 = arma::conv_to<std::vector<double>>::from(h);
  /*
  std::vector<double> gv1(n), hv1(n);
  std::vector<double> gv2(n), hv2(n);
  for (std::size_t i=0; i<n; ++i) {
    gv1[i] = g[i]/h[i];
    hv1[i] = 1.;
  }
  for (std::size_t i=0; i<n; ++i) {
    gv2[i] = 1.;
    hv2[i] = h[i]/g[i];
  }
  */

  // First solver
  auto dp0 = DPSolver(n, T, std::move(gv0), std::move(hv0),
		     obj_fn,
		     risk_partitioning_objective,
		     use_rational_optimization,
		     gamma,
		     reg_power,
		     sweep_down,
		     find_optimal_t
		     );
  
  auto subsets0 = dp0.get_optimal_subsets_extern();

  rowvec leaf_values0 = arma::zeros<rowvec>(n);

  if (T > 1 || risk_partitioning_objective) {
    std::size_t start_ind = risk_partitioning_objective ? 0 : static_cast<std::size_t>(T*activePartitionRatio);

    for (std::size_t i=start_ind; i<subsets0.size(); ++i) {      
      uvec ind = arma::conv_to<uvec>::from(subsets0[i]);
      double val = -1. * learningRate * sum(g(ind))/sum(h(ind));
      for (auto j: ind) {
	leaf_values0(j) = val;
      }
    }
  }

  /*
  // Second solver
  auto dp1 = DPSolver(n, T, std::move(gv1), std::move(hv1),
		      obj_fn,
		      risk_partitioning_objective,
		      use_rational_optimization,
		      gamma,
		      reg_power,
		      sweep_down,
		      find_optimal_t
		      );

  auto subsets1 = dp1.get_optimal_subsets_extern();
  
  rowvec leaf_values1 = arma::zeros<rowvec>(n);

  if (T > 1 || risk_partitioning_objective) {
    std::size_t start_ind = risk_partitioning_objective ? 0 : static_cast<std::size_t>(T*activePartitionRatio);

    for (std::size_t i=start_ind; i<subsets1.size(); ++i) {      
      uvec ind = arma::conv_to<uvec>::from(subsets1[i]);
      double val = -1. * learningRate * sum(g(ind))/sum(h(ind));
      for (auto j: ind) {
	leaf_values1(j) = val;
      }
    }
  }

  // Third solver
  auto dp2 = DPSolver(n, T, std::move(gv2), std::move(hv2),
		      obj_fn,
		      risk_partitioning_objective,
		      use_rational_optimization,
		      gamma,
		      reg_power,
		      sweep_down,
		      find_optimal_t
		      );

  auto subsets2 = dp2.get_optimal_subsets_extern();
  
  rowvec leaf_values2 = arma::zeros<rowvec>(n);

  if (T > 1 || risk_partitioning_objective) {
    std::size_t start_ind = risk_partitioning_objective ? 0 : static_cast<std::size_t>(T*activePartitionRatio);

    for (std::size_t i=start_ind; i<subsets2.size(); ++i) {      
      uvec ind = arma::conv_to<uvec>::from(subsets2[i]);
      double val = -1. * learningRate * sum(g(ind))/sum(h(ind));
      for (auto j: ind) {
	leaf_values2(j) = val;
      }
    }
  }
  */

  // return (1./3.) * (leaf_values0 + leaf_values1 + leaf_values2);
  // return 0.5 * (leaf_values0 + leaf_values1);
  return leaf_values0;
    
}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::purge_() {

  dataset_ = ones<mat>(0,0);
  labels_ = ones<Row<double>>(0);
  dataset_oos_ = ones<mat>(0,0);
  labels_oos_ = ones<Row<double>>(0);

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
std::tuple<std::size_t, std::size_t, double, double>
CompositeClassifier<ClassifierType>::computeChildPartitionInfo() {

  return std::make_tuple(childPartitionSize_[1],
			 childNumSteps_[1],
			 childLearningRate_[1],
			 childActivePartitionRatio_[1]);

}

template<typename ClassifierType>
std::tuple<std::size_t, std::size_t, double>
CompositeClassifier<ClassifierType>::computeChildModelInfo() {
  return std::make_tuple(childMaxDepth_[1],
			 childMinLeafSize_[1],
			 childMinimumGainSplit_[1]);
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

  if (serializeModel_) {
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
	      << ", STEPS = " << steps_ << ")"
	      << " STEP: " << stepNum 
	      << " IS LOSS: " << r
	      << " IS ERROR: " << error_is << "%" << std::endl;
  }
  
  if (hasOOSData_) {
    Row<DataType> yhat_oos;
    if (serializeModel_) {
      Predict(indexName_, dataset_oos_, yhat_oos, true);
    } else {
      Predict(dataset_oos_, yhat_oos);
    }
    double error_oos = err(yhat_oos, labels_oos_);
    std::cout << suff<< ": "
	      << "(PARTITION SIZE = " << partitionSize_
	      << ", STEPS = " << steps_ << ")"
	      << " STEP: " << stepNum
	      << " OOS ERROR: " << error_oos << "%" << std::endl;
  }

}

template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::fit() {

  for (int stepNum=1; stepNum<=steps_; ++stepNum) {
    fit_step(stepNum);
    
    if (serializeModel_) {
      commit();
    }
    if (!quietRun_) {
      printStats(stepNum);
    }

  }

  // Serialize residual
  if (serializeModel_)
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
std::pair<rowvec, rowvec>
CompositeClassifier<ClassifierType>::generate_coefficients(const Row<DataType>& labels, const uvec& colMask) {

  rowvec yhat;
  Predict(yhat, colMask);

  rowvec g, h;
  lossFn_->loss(yhat, labels, &g, &h, clamp_gradient_, upper_val_, lower_val_);

  return std::make_pair(g, h);

}

#endif
