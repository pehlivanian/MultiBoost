#ifndef __GRADIENTBOOSTCLASSIFIER_IMPL_HPP__
#define __GRADIENTBOOSTCLASSIFIER_IMPL_HPP__

std::vector<int> _shuffle(int sz) {
  std::vector<int> ind(sz), r(sz);
  std::iota(ind.begin(), ind.end(), 0);
  
  std::vector<std::vector<int>::iterator> v(static_cast<int>(ind.size()));
  std::iota(v.begin(), v.end(), ind.begin());

  std::shuffle(v.begin(), v.end(), std::mt19937{std::random_device{}()});
  
  for (int i=0; i<v.size(); ++i) {
    r[i] = *(v[i]);
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
    lossFn_ = std::make_unique<BinomialDevianceLoss<DataType>>();
  }
  else if (loss_ == lossFunction::MSE) {
    lossFn_ = std::make_unique<MSELoss<DataType>>();
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
GradientBoostClassifier<DataType>::fit() {

  int rowRatio = static_cast<size_t>(n_ * row_subsample_ratio_);
  int colRatio = static_cast<size_t>(m_ * col_subsample_ratio_);
  rowMask_ = subsampleRows(rowRatio);
  colMask_ = subsampleCols(colRatio);
  cout << "ratios: " << rowRatio << " : "  << colRatio << endl;
  auto dataset_slice = dataset_.submat(rowMask_, colMask_);

  rowvec coeffs = generate_coefficients(dataset_slice);
}

template<typename DataType>
rowvec
GradientBoostClassifier<DataType>::generate_coefficients(const mat& dataset) {

  

  rowvec r = linspace<rowvec>(0, -1+n_, n_);
  return r;
}

#endif
