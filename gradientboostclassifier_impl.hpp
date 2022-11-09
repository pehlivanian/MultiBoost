#ifndef __GRADIENTBOOSTCLASSIFIER_IMPL_HPP__
#define __GRADIENTBOOSTCLASSIFIER_IMPL_HPP__

using namespace PartitionSize;
using namespace LearningRate;


template<typename DataType>
typename RandomForestClassifier<DataType>::LeavesMap
RandomForestClassifier<DataType>::encode(const Row<DataType>& labels_d, Row<std::size_t>& labels_t) {
  LeavesMap leavesMap;
  Row<DataType> uniqueVals = unique(labels_d);
  for (auto it=uniqueVals.begin(); it!=uniqueVals.end(); ++it) {
    uvec ind = find(labels_d == *it);
    std::size_t equiv = std::distance(it, uniqueVals.end()) - 1;
    labels_t.elem(ind).fill(equiv);
    leavesMap.insert(std::make_pair(static_cast<std::size_t>(equiv), (*it)));
  }
  return leavesMap;
}

template<typename DataType>
void
RandomForestClassifier<DataType>::decode(const Row<std::size_t>& labels_t,
					  Row<DataType>& labels_d) {
  Row<std::size_t> uniqueVals = unique(labels_t);
  for (auto it=uniqueVals.begin(); it!=uniqueVals.end(); ++it) {
    uvec ind = find(labels_t == *it);
    DataType equiv = leavesMap_[*it];
    labels_d.elem(ind).fill(equiv);
  }
}

template<typename DataType>
typename DecisionTreeClassifier<DataType>::LeavesMap
DecisionTreeClassifier<DataType>::encode(const Row<DataType>& labels_d, Row<std::size_t>& labels_t) {
  LeavesMap leavesMap;
  Row<DataType> uniqueVals = unique(labels_d);
  for (auto it=uniqueVals.begin(); it!=uniqueVals.end(); ++it) {
    uvec ind = find(labels_d == *it);
    std::size_t equiv = std::distance(it, uniqueVals.end()) - 1;
    labels_t.elem(ind).fill(equiv);
    leavesMap.insert(std::make_pair(static_cast<std::size_t>(equiv), (*it)));
  }
  return leavesMap;
}

template<typename DataType>
void
DecisionTreeClassifier<DataType>::decode(const Row<std::size_t>& labels_t,
					  Row<DataType>& labels_d) {
  Row<std::size_t> uniqueVals = unique(labels_t);
  for (auto it=uniqueVals.begin(); it!=uniqueVals.end(); ++it) {
    uvec ind = find(labels_t == *it);
    DataType equiv = leavesMap_[*it];
    labels_d.elem(ind).fill(equiv);
  }
}


template<typename DataType>
Row<DataType> 
GradientBoostClassifier<DataType>::_constantLeaf() const {
  Row<DataType> r;
  r.zeros(dataset_.n_cols);
  return r;
}

template<typename DataType>
Row<DataType>
GradientBoostClassifier<DataType>::_randomLeaf(std::size_t numVals) const {
  
  // Look how clumsy this is
  Row<DataType> range = linspace<Row<DataType>>(0, 1, numVals+2);
  std::default_random_engine eng;
  std::uniform_int_distribution<std::size_t> dist{1, numVals};
  Row<DataType> r(dataset_.n_cols, arma::fill::none);
  for (size_t i=0; i<dataset_.n_cols; ++i) {
    auto j = dist(eng);    
    r[i] = range[j];
  }  

  return r;
}

template<typename DataType>
void
GradientBoostClassifier<DataType>::init_() {
  
  // Note these are flipped
  n_ = dataset_.n_rows; 
  m_ = dataset_.n_cols;

  // Initialize rng  
  std::size_t a=1, b=std::max(1, static_cast<int>(m_ * col_subsample_ratio_));
  partitionDist_ = std::uniform_int_distribution<std::size_t>(a, b);
							      

  if (symmetrized_) {
    // Make labels members of {-1,1}
    symmetrizeLabels();
  }

  // leaf values
  leaves_.push_back(std::vector<DataType>(m_, 0));

  // partitions
  std::vector<int> subset{m_};
  std::iota(subset.begin(), subset.end(), 0);
  std::vector<std::vector<int>> partition{1, subset};
  partitions_.push_back(partition);

  // classifiers
  // Row<DataType> constantLabels = _constantLeaf();
  Row<DataType> randomLabels = _randomLeaf(partitionSize_);
  

  // first prediction
  // DecisionTreeRegressorClassifier
  // classifiers_.push_back(std::make_unique<DecisionTreeRegressorClassifier<DataType>>(dataset_, randomLabels));
  
  // DecisionTreeClassifier
  classifiers_.push_back(std::make_unique<DecisionTreeClassifier<DataType>>(dataset_, randomLabels, partitionSize_));

  // RandomForestClassifier
  // classifiers_.push_back(std::make_unique<RandomForestClassifier<DataType>>(dataset_, randomLabels, partitionSize_));

  // LeafOnlyClassifier - unclear how to get OOS predictions
  // classifiers_.push_back(std::make_unique<LeafOnlyClassifier<DataType>>(dataset_, randomLabels));

  leaves_.push_back(randomLabels);

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

}

template<typename DataType>
void
GradientBoostClassifier<DataType>::Predict(Row<DataType>& prediction) {
  prediction = zeros<Row<DataType>>(m_);
  for (const auto& step : predictions_) {
    prediction += step;
  }
}
template<typename DataType>
void
GradientBoostClassifier<DataType>::Predict(Row<DataType>& prediction, const uvec& colMask) {

  Predict(prediction);
  prediction = prediction.submat(zeros<uvec>(1), colMask);

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
void
GradientBoostClassifier<DataType>::Predict(const mat& dataset, Row<DataType>& prediction) {

  prediction = zeros<Row<DataType>>(dataset.n_cols);

  if (true) {
    for (const auto& classifier : classifiers_) {
      Row<DataType> predictionStep;
      classifier->Classify_(dataset, predictionStep);
      prediction += predictionStep;    
    }  
  }
  else if (false) {
    uvec rm, cm;
    for (size_t i=0; i<classifiers_.size(); ++i) {
      std::tie(rm, cm) = std::make_tuple(rowMasks_[i], colMasks_[i]);
      auto dataset_slice = dataset.submat(rowMask_, colMask_);
      Row<DataType> p;
      classifiers_[i]->Classify_(dataset_slice, p);
      
      for (std::size_t i=0; i<colMask_.n_rows; ++i) {
	prediction.col(colMask_(i)) += p.col(i);
      }
    }
    
  }
  if (symmetrized_) {
    deSymmetrize(prediction);
  }
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
  Row<DataType> uniqueVals = unique(labels_);

  assert (uniqueVals.n_cols == 2);

  double m = min(uniqueVals), M=max(uniqueVals);
  a_ = 2./static_cast<double>(M-m);
  b_ = static_cast<double>(m+M)/static_cast<double>(m-M);
  labels_ = sign(a_*labels_ + b_);
  // labels_ = sign(2 * labels_ - 1);
}

template<typename DataType>
void
GradientBoostClassifier<DataType>::symmetrize(Row<DataType>& prediction) {
  prediction = sign(a_*prediction + b_);
  // prediction = sign(2 * prediction - 1);
}

template<typename DataType>
void
GradientBoostClassifier<DataType>::deSymmetrize(Row<DataType>& prediction) {
  prediction = (sign(prediction) - b_)/ a_;
  // prediction = (1 + sign(prediction)) / 2.;
}

template<typename DataType>
void
GradientBoostClassifier<DataType>::fit_step(std::size_t stepNum) {
  int rowRatio = static_cast<size_t>(n_ * row_subsample_ratio_);
  int colRatio = static_cast<size_t>(m_ * col_subsample_ratio_);
  rowMask_ = subsampleRows(rowRatio);
  colMask_ = subsampleCols(colRatio);

  mat dataset_slice = dataset_.submat(rowMask_, colMask_);  
  Row<DataType> labels_slice = labels_.submat(zeros<uvec>(1), colMask_);

  rowMasks_.push_back(rowMask_);
  colMasks_.push_back(colMask_);

  // Generate coefficients g, h
  std::pair<rowvec, rowvec> coeffs = generate_coefficients(dataset_slice, labels_slice, colMask_);

  // Compute partition size
  std::size_t partitionSize = computePartitionSize(stepNum);

  // Compute learning rate
  double learningRate = computeLearningRate(stepNum);

  // Compute leaves
  // Leaves best_leaves = computeOptimalSplit(coeffs.first, coeffs.second, dataset_slice, stepNum, partitionSize);
  Leaves allLeaves = zeros<Row<DataType>>(m_);

  // Find classifier fit for leaves choice
  // using dtr = DecisionTreeRegressorClassifier<DataType>;
  using dtr = DecisionTreeClassifier<DataType>;
  // using dtr = RandomForestClassifier<DataType>;
  Row<DataType> prediction;
  std::unique_ptr<dtr> classifier;

  // XXX
  // For debugging; put back in the condition scope
  Row<DataType> prediction_slice;

  if (postExtrapolate_) {

    // Compute optimal leaf choice on restricted dataset
    Leaves best_leaves = computeOptimalSplit(coeffs.first, coeffs.second, dataset_slice, stepNum, partitionSize);
    
    // Fit classifier on restricted {dataset_slice, best_leaves}
    prediction = zeros<Row<DataType>>(m_);

    classifier.reset(new dtr(dataset_slice, best_leaves, partitionSize + 1));
    // std::unique_ptr<dtr> classifier(new dtr(dataset_slice, best_leaves));
    classifier->Classify_(dataset_slice, prediction_slice);

    // Zero-pad to extend to all of dataset
    prediction(colMask_) = prediction_slice;

  } 
  
  else if (preExtrapolate_) {
    
    // Compute optimal leaf choice on unrestricted dataset
    Leaves best_leaves = computeOptimalSplit(coeffs.first, coeffs.second, dataset_, stepNum, partitionSize);
    
    // Fit classifier on {dataset, padded best_leaves}
    // Zero pad labels first
    allLeaves(colMask_) = best_leaves;

    classifier.reset(new dtr(dataset_, allLeaves, partitionSize + 1));
    // std::unique_ptr<dtr> classifier(new dtr(dataset_, allLeaves));
    classifier->Classify_(dataset_, prediction);

  }

  classifiers_.push_back(std::move(classifier));
  predictions_.push_back(prediction);
  current_classifier_ind_++;

  // colMask_.print(std::cout);
  // dataset_slice.print(std::cout);
  
  /*
    std::cout << "PREDICTIONS\n";
    for (size_t i=500; i<600; ++i) {
    std::cout << best_leaves(i) << " : " << prediction_slice(i) << std::endl;
    }
  */

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
  std::vector<double> gv = arma::conv_to<std::vector<double>>::from(g);
  std::vector<double> hv = arma::conv_to<std::vector<double>>::from(h);

  int n = colMask_.n_rows, T = partitionSize;
  bool risk_partitioning_objective = true;
  bool use_rational_optimization = true;
  bool sweep_down = false;
  double gamma = 0.;
  double reg_power=1.;
  bool find_optimal_t = false;

  // std::cout << "PARTITION SIZE: " << T << std::endl;

  auto dp = DPSolver(n, T, gv, hv,
		     objective_fn::RationalScore,
		     true,
		     false,
		     0.,
		     1.,
		     false,
		     false
		     );
  
  auto subsets = dp.get_optimal_subsets_extern();
  
  // Use subset of subsets
  // =====================
  bool USE_FIRST_SUBSET = false;
  size_t NUM_SUBSETS = 2;
  std::vector<std::vector<int>> v;
  if (USE_FIRST_SUBSET) {
    if (g.n_elem > 1) {
      // for (size_t i=partitionSize-1; i>(partitionSize-1-NUM_SUBSETS); --i) 
	for (size_t i=0; i<NUM_SUBSETS; ++i)
	v.push_back(subsets[i]);
      subsets = v;
    }
  }
  // =====================

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
    
}

template<typename DataType>
void
GradientBoostClassifier<DataType>::fit() {

  for (std::size_t stepNum=1; stepNum<=steps_; ++stepNum) {
    fit_step(stepNum);
    

    if ((stepNum%100) == 0) {
      Row<DataType> yhat, yhat_sym;
      Predict(yhat);
      DataType r = lossFn_->loss(yhat, labels_);
      deSymmetrize(yhat); symmetrize(yhat);

      DataType error_is = accu(yhat != labels_) * 100. / labels_.n_elem;
      
      std::cout << "STEP: " << stepNum 
		<< " IS LOSS: " << r
		<< " IS ERROR: " << error_is << "%" << std::endl;
      
      if (hasOOSData_) {
	Row<DataType> yhat_oos;
	Predict(dataset_oos_, yhat_oos);
	deSymmetrize(yhat_oos); symmetrize(yhat_oos);
	DataType error_oos = accu(yhat_oos != labels_oos_) * 100. / labels_oos_.n_elem;
	std::cout << "OOS ERROR: " << error_oos << "%" << std::endl;
      }
    }
  }

}

template<typename DataType>
void
GradientBoostClassifier<DataType>::Classify(const mat& dataset, Row<DataType>& labels) {
  Predict(dataset, labels);
}

template<typename DataType>
double
GradientBoostClassifier<DataType>::computeLearningRate(std::size_t stepNum) {

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

  // std::cout << "LEARNING RATE: " << learningRate << std::endl;

  return learningRate;
}
template<typename DataType>
std::size_t
GradientBoostClassifier<DataType>::computePartitionSize(std::size_t stepNum) {

  // stepNum is in range [1,...,context.steps]

  std::size_t partitionSize;

  if (partitionSizeMethod_ == SizeMethod::FIXED) {
    return partitionSize_;
  } else if (partitionSizeMethod_ == SizeMethod::FIXED_PROPORTION) {
    partitionSize = static_cast<std::size_t>(partitionRatio_ * row_subsample_ratio_ * colMask_.n_rows);
  } else if (partitionSizeMethod_ == SizeMethod::DECREASING) {
    double A = colMask_.n_rows, B = log(colMask_.n_rows)/steps_;
    partitionSize = std::max(1, static_cast<int>(A * exp(-B * (-1 + stepNum))));
  } else if (partitionSizeMethod_ == SizeMethod::INCREASING) {
    double A = 2., B = log(colMask_.n_rows)/static_cast<double>(steps_);
    partitionSize = std::max(1, static_cast<int>(A * exp(B * (-1 + stepNum))));
  } else if (partitionSizeMethod_ == SizeMethod::RANDOM) {
    partitionSize = partitionDist_(default_engine_);
    ;
  }

  return partitionSize;
}

template<typename DataType>
std::pair<rowvec, rowvec>
GradientBoostClassifier<DataType>::generate_coefficients(const mat& dataset, const Row<DataType>& labels, const uvec& colMask) {

  rowvec yhat;
  Predict(yhat, colMask);

  rowvec g, h;
  lossFn_->loss(yhat, labels, &g, &h);
  
  /* 
  std::cout << "GENERATE COEFFICIENTS\n";
  std::cout << "labels size: " << labels.n_rows << " x " << labels.n_cols << std::endl;
  labels.print(std::cout);
  std::cout << "yhat_slice size: " << yhat_slice.n_rows << " x " << yhat_slice.n_cols << std::endl;  
  yhat_slice.print(std::cout);
  std::cout << "g size: " << g.n_rows << " x " << g.n_cols << std::endl;
  g.print(std::cout);
  std::cout << "h size: " << h.n_rows << " x " << h.n_cols << std::endl;
  h.print(std::cout);
  */

  return std::make_pair(g, h);
}

/*
  template<typename DataType>
  DataType
  GradientBoostClassifier<DataType>::imbalance() {
  ;
  }
  // 2.0*((sum(y_train==0)/len(y_train) - .5)**2 + (sum(y_train==1)/len(y_train) - .5)**2)
  */


#endif

