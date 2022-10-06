#ifndef __GRADIENTBOOSTCLASSIFIER_IMPL_HPP__
#define __GRADIENTBOOSTCLASSIFIER_IMPL_HPP__

using namespace PartitionSize;
using namespace LearningRate;

template<typename DataType>
Row<DataType> 
GradientBoostClassifier<DataType>::_constantLeaf() const {
  Row<DataType> r;
  r.zeros(dataset_.n_cols);
  return r;
}

template<typename DataType>
void
GradientBoostClassifier<DataType>::init_() {
  
  // Note these are flipped
  n_ = dataset_.n_rows; 
  m_ = dataset_.n_cols;

  if (symmetrized_) {
    // Make labels members of {-1,1}
    symmetrizeLabels();
  }
  // regularisations
  regularizations_ = RegularizationList(steps_, DataType{0});

  // leaf values
  leaves_ = LeavesValues(steps_);
  for(Leaves &l : leaves_) 
    l = std::vector<DataType>(m_, 0.);

  // partitions
  std::vector<int> part{m_};
  std::iota(part.begin(), part.end(), 0);
  partitions_.push_back(part);

  // classifiers
  // XXX
  // Out of the box currently
  Row<DataType> constantLabels = _constantLeaf();
  classifiers_.push_back(std::make_unique<DecisionTreeRegressorClassifier<DataType>>(dataset_, constantLabels));

  // first prediction
  current_classifier_ind_ = 0;
  Row<DataType> prediction;
  classifiers_[current_classifier_ind_]->Classify_(dataset_, prediction);
  predictions_.push_back(prediction);

  // XXX
  rowMask_ = linspace<uvec>(0, -1+m_, m_);
  colMask_ = linspace<uvec>(0, -1+n_, n_);

  if (loss_ == lossFunction::BinomialDeviance) {
    lossFn_ = new BinomialDevianceLoss<DataType>();
  }
  else if (loss_ == lossFunction::MSE) {
    lossFn_ = new MSELoss<DataType>();
  }

  // fit
  fit();
}

template<typename DataType>
typename GradientBoostClassifier<DataType>::LeavesMap
GradientBoostClassifier<DataType>::relabel(const Row<DataType>& label,
					   Row<std::size_t>& newLabel) {
  std::unordered_map<DataType, int> leavesMap;
  Row<DataType> uniqueVals = unique(label);
  for(auto it=uniqueVals.begin(); it!=uniqueVals.end(); ++it) {
    uvec ind = find(label == *(it));
    std::size_t equiv = std::distance(it, uniqueVals.end());
    newLabel.elem(ind).fill(equiv);
    leavesMap.insert(std::make_pair(*(it), static_cast<int>(equiv)));
  }
  
  return leavesMap;
}
	
template<typename DataType>
uvec
GradientBoostClassifier<DataType>::subsampleRows(size_t numRows) {
  uvec r = randperm(numRows);
  return r;
}

template<typename DataType>
uvec
GradientBoostClassifier<DataType>::subsampleCols(size_t numCols) {
  uvec r = randperm(numCols);
  return r;
}

template<typename DataType>
void
GradientBoostClassifier<DataType>::symmetrizeLabels() {
  for (auto &el : labels_)
    el = 2*el - 1;
}

template<typename DataType>
void
GradientBoostClassifier<DataType>::fit_step(std::size_t stepNum) {
  int rowRatio = static_cast<size_t>(n_ * row_subsample_ratio_);
  int colRatio = static_cast<size_t>(m_ * col_subsample_ratio_);
  rowMask_ = subsampleRows(rowRatio);
  colMask_ = subsampleCols(colRatio);
  auto dataset_slice = dataset_.submat(rowMask_, colMask_);
  
  // Doesn't seem like there's a better way to do this
  size_t slice_size = colMask_.n_rows;
  Row<DataType> labels_slice(slice_size);
  for (size_t i=0; i<slice_size; ++i) {
    labels_slice.col(i) = labels_.col(colMask_(i));
  }
  

  std::pair<rowvec, rowvec> coeffs = generate_coefficients(dataset_slice, labels_slice);

  std::size_t partitionSize = computePartitionSize(stepNum);
  double learningRate = computeLearningRate(stepNum);

  Leaves best_leaves = computeOptimalSplit(coeffs.first, coeffs.second, dataset_slice, stepNum, partitionSize);

  

}

template<typename DataType>
typename GradientBoostClassifier<DataType>::Leaves
GradientBoostClassifier<DataType>::computeOptimalSplit(rowvec& g,
						       rowvec& h,
						       mat dataset,
						       std::size_t stepNum, 
						       std::size_t partitionSize) {

  // We should implement several methods here
  // XXX
  std::cout << "dataset size:  " << colMask_.n_rows << std::endl;
  std::cout << "partitionSize: " << partitionSize << std::endl;

  std::vector<float> gv = arma::conv_to<std::vector<float>>::from(g);
  std::vector<float> hv = arma::conv_to<std::vector<float>>::from(h);

  int n = colMask_.n_rows, T = partitionSize;
  bool risk_partitioning_objective = true;
  bool use_rational_optimization = true;
  bool sweep_down = false;
  float gamma = 0.;
  float reg_power=1;
  bool find_optimal_t = false;

  /*
    auto dp = DPSolver(n, T, gv, hv,
    objective_fn::Gaussian,
    use_rational_optimization,
    gamma,
    reg_power,
    sweep_down,
    find_optimal_t
    );
    
    auto dp_opt = dp.get_optimal_subsets_extern();
  */

  rowvec r;
  return r;
}

template<typename DataType>
void
GradientBoostClassifier<DataType>::fit() {

  for (std::size_t stepNum=1; stepNum<=steps_; ++stepNum)
    fit_step(stepNum);

}

template<typename DataType>
void
GradientBoostClassifier<DataType>::Classify(const mat& dataset, Row<DataType>& labels) {
  rowvec prediction = zeros<rowvec>(dataset.n_cols);
  for( auto const& classifier : classifiers_) {
    rowvec predictions;
    classifier->Classify_(dataset, predictions);
    prediction += predictions;
  }
  labels = prediction;

  /*
    std::cout << "labels_\n";
    labels_.submat(rowMask_, colMask_).print(std::cout);
    std::cout << "prediction\n";
    prediction.print(std::cout);sub
  */
  
}

template<typename DataType>
double
GradientBoostClassifier<DataType>::computeLearningRate(std::size_t stepNum) {
  if (learningRateMethod_ == RateMethod::FIXED) {
    return learningRate_;
  } else if (learningRateMethod_ == RateMethod::DECREASING) {
    double A = learningRate_, B = -log(.5) / static_cast<double>(steps_);
    return A * exp(-B * (-1 + stepNum));
  } else if (learningRateMethod_ == RateMethod::INCREASING) {
    double A = learningRate_, B = log(2.) / static_cast<double>(steps_);
    return A * exp(B * (-1 + stepNum));
  }
}
template<typename DataType>
std::size_t
GradientBoostClassifier<DataType>::computePartitionSize(std::size_t stepNum) {
  if (partitionSizeMethod_ == SizeMethod::FIXED) {
    return partitionSize_;
  } else if (partitionSizeMethod_ == SizeMethod::FIXED_PROPORTION) {
    return static_cast<double>(partitionSize_) * row_subsample_ratio_ * colMask_.n_rows;
  } else if (partitionSizeMethod_ == SizeMethod::DECREASING) {
    double A = m_, B = log(colMask_.n_rows)/steps_;
    return std::max(1, static_cast<int>(A * exp(-B * (-1 + stepNum))));
  } else if (partitionSizeMethod_ == SizeMethod::INCREASING) {
    double A = 2., B = log(colMask_.n_rows)/static_cast<double>(steps_);
    return std::max(1, static_cast<int>(A * exp(B * (-1 + stepNum))));
  } else if (partitionSizeMethod_ == SizeMethod::RANDOM) {
    // to be implemented
    // XXX
    ;
  }
}

template<typename DataType>
std::pair<rowvec, rowvec>
GradientBoostClassifier<DataType>::generate_coefficients(const mat& dataset, const Row<DataType>& labels) {

  rowvec yhat;
  Predict(dataset, yhat);

  rowvec g, h;
  lossFn_->loss(yhat, labels, &g, &h);
  
  /*
  std::cout << "labels size: " << labels.n_rows << " x " << labels.n_cols << std::endl;
  labels.print(std::cout);
  std::cout << "yhat size: " << yhat.n_rows << " x " << yhat.n_cols << std::endl;  
  yhat.print(std::cout);
  std::cout << "g size: " << g.n_rows << " x " << g.n_cols << std::endl;
  g.print(std::cout);
  std::cout << "h size: " << h.n_rows << " x " << h.n_cols << std::endl;
  h.print(std::cout);
  */

  return std::make_pair(g, h);
}

#endif
