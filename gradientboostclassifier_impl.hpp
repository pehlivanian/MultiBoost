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

  // leaf values
  leaves_ = LeavesValues(steps_);
  for(Leaves &l : leaves_) 
    l = std::vector<DataType>(m_, 0.);

  // partitions
  std::vector<int> subset{m_};
  std::iota(subset.begin(), subset.end(), 0);
  std::vector<std::vector<int>> partition{1, subset};
  partitions_.push_back(partition);

  // classifiers
  // XXX
  // Out of the box currently
  Row<DataType> constantLabels = _constantLeaf();
  classifiers_.push_back(std::make_unique<DecisionTreeRegressorClassifier<DataType>>(dataset_, constantLabels));
  leaves_.push_back(constantLabels);

  // first prediction
  Row<DataType> prediction;
  classifiers_[current_classifier_ind_]->Classify_(dataset_, prediction);
  predictions_.push_back(prediction);

  rowMask_ = linspace<uvec>(0, -1+m_, m_);
  colMask_ = linspace<uvec>(0, -1+n_, n_);
  
  rowMasks_.push_back(rowMask_);
  colMasks_.push_back(colMask_);

  if (loss_ == lossFunction::BinomialDeviance) {
    lossFn_ = new BinomialDevianceLoss<DataType>();
  }
  else if (loss_ == lossFunction::MSE) {
    lossFn_ = new MSELoss<DataType>();
  }

  current_classifier_ind_ = 0;

  // fit
  fit();
}

template<typename DataType>
void
GradientBoostClassifier<DataType>::Predict(const mat& dataset, Row<DataType>& prediction) {

  prediction = zeros<Row<DataType>>(m_);
  for (const auto& step : predictions_) {
    prediction %= step;
  }

  /*
    prediction = zeros<Row<DataType>>(m_);
    
    uvec rm, cm;
    for (size_t i=0; i<classifiers_.size(); ++i) {
    std::tie(rm, cm) = std::make_tuple(rowMasks_[i], colMasks_[i]);
    auto dataset_slice = dataset_.submat(rowMask_, colMask_);
    Row<DataType> p;
    classifiers_[i]->Classify_(dataset_slice, p);
    
    for (size_t i=0; i<colMask_.n_rows; ++i) {
    prediction.col(colMask_(i)) += p.col(i);
    }    
    }
  */

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
  uvec r = sort(randperm(n_, numRows));
  return r;
}

template<typename DataType>
uvec
GradientBoostClassifier<DataType>::subsampleCols(size_t numCols) {
  uvec r = sort(randperm(m_, numCols));
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
  
  // Apply row reduction with colMask_
  size_t slice_size = colMask_.n_rows;
  Row<DataType> labels_slice(slice_size);
  for (size_t i=0; i<slice_size; ++i) {
    labels_slice.col(i) = labels_.col(colMask_(i));
  }
  rowMasks_.push_back(rowMask_);
  colMasks_.push_back(colMask_);

  // Generate coefficients g, h
  std::pair<rowvec, rowvec> coeffs = generate_coefficients(dataset_slice, labels_slice, colMask_);

  // Compute partition size
  std::size_t partitionSize = computePartitionSize(stepNum);

  // Compute learning rate
  double learningRate = computeLearningRate(stepNum);

  // Compute leaves
  Leaves best_leaves = computeOptimalSplit(coeffs.first, coeffs.second, dataset_slice, stepNum, partitionSize);
  Leaves allLeaves = zeros<Row<DataType>>(m_);

  // Find classifier fit for leaves choice;
  // POST_EXTRAPOLATE
  // PRE_EXTRAPOLATE
  using dtr = DecisionTreeRegressorClassifier<DataType>;
  Row<DataType> prediction;
  std::unique_ptr<dtr> classifier;

  // XXX
  // For debugging; put back in the condition scope
  Row<DataType> prediction_slice;

  if (POST_EXTRAPOLATE_) {

    // Fit on restricted {dataset_slice, best_leaves}
    prediction = zeros<Row<DataType>>(m_);
    classifier.reset(new dtr(dataset_slice, best_leaves));
    // std::unique_ptr<dtr> classifier(new dtr(dataset_slice, best_leaves));
    classifier->Classify_(dataset_slice, prediction_slice);

    // Zero-pad to extend to all of dataset
    prediction(colMask_) = prediction_slice;

  } 
  
  else if (PRE_EXTRAPOLATE_) {

    // Zero pad labels first
    allLeaves(colMask_) = best_leaves;

    // Fit classifier on {dataset, padded best_leaves}
    classifier.reset(new dtr(dataset_, allLeaves));
    // std::unique_ptr<dtr> classifier(new dtr(dataset_, allLeaves));
    classifier->Classify_(dataset_, prediction);
  }

  classifiers_.push_back(std::move(classifier));
  predictions_.push_back(prediction);
  current_classifier_ind_++;

  std::cout << "PREDICTIONS\n";
  colMask_.print(std::cout);
  dataset_slice.print(std::cout);
  best_leaves.print(std::cout);
  prediction_slice.print(std::cout);

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

  std::vector<double> gv = arma::conv_to<std::vector<double>>::from(g);
  std::vector<double> hv = arma::conv_to<std::vector<double>>::from(h);

  int n = colMask_.n_rows, T = partitionSize;
  bool risk_partitioning_objective = true;
  bool use_rational_optimization = true;
  bool sweep_down = false;
  double gamma = 0.;
  double reg_power=1.;
  bool find_optimal_t = false;

  std::cout << "Finding optimal partition...\n";
  auto dp = DPSolver(n, T, gv, hv,
		     objective_fn::Gaussian,
		     true,
		     true,
		     0.,
		     1.,
		     false,
		     false
		     );
  
  auto subsets = dp.get_optimal_subsets_extern();
  std::cout << "...optimal partition found\n";


  rowvec leaf_values = arma::zeros<rowvec>(n);
  for (const auto& subset : subsets) {
    uvec ind = arma::conv_to<uvec>::from(subset);
    DataType val = -1. * learningRate_ * sum(g(ind))/sum(h(ind));
    for (auto i: ind) {
      leaf_values(i) = val;
    }
  }

  leaves_.push_back(leaf_values);
  partitions_.push_back(subsets);

  return leaf_values;
    
    /*
      g.print(std::cout);
      h.print(std::cout);
      ind.print(std::cout);
      leaf_values.print(std::cout);
      std::cout << learningRate_ << std::endl;
      std::cout << val << std::endl;
      }
      
      partitions_.push_back(subsets);
      classifiers_.push_back(std::make_unique<DecisionTreeRegressorClassifier<DataType>>(dataset, leaf_values, 1));  
      predictions_.push_back(prediction);
    */

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
  Predict(dataset, labels);
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
GradientBoostClassifier<DataType>::generate_coefficients(const mat& dataset, const Row<DataType>& labels, const uvec& colMask) {

  rowvec yhat;
  Predict(dataset, yhat);

  Row<DataType> yhat_slice = yhat.submat(zeros<uvec>(1), colMask);

  rowvec g, h;
  lossFn_->loss(yhat_slice, labels, &g, &h);
  
  std::cout << "GENERATE COEFFICIENTS\n";
  std::cout << "labels size: " << labels.n_rows << " x " << labels.n_cols << std::endl;
  labels.print(std::cout);
  std::cout << "yhat_slice size: " << yhat_slice.n_rows << " x " << yhat_slice.n_cols << std::endl;  
  yhat_slice.print(std::cout);
  std::cout << "g size: " << g.n_rows << " x " << g.n_cols << std::endl;
  g.print(std::cout);
  std::cout << "h size: " << h.n_rows << " x " << h.n_cols << std::endl;
  h.print(std::cout);

  return std::make_pair(g, h);
}

#endif
