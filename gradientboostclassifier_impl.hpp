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

Row<std::size_t> _constantLeaf(int m) {
  return zeros<Row<std::size_t>>(m);
}

template<typename DataType>
void
GradientBoostClassifier<DataType>::init_() {
  
  // Note these are flipped
  int n_ = dataset_.n_rows; 
  int m_ = dataset_.n_cols;

  // regularisations
  regularizations_ = RegularizationList(steps_, DataType{0});

  // leaf values
  leaves_ = LeavesValues{}; leaves_.reserve(steps_);
  for(Leaves &l : leaves_) 
    l = std::vector<DataType>(m_, 0.);

  // classifiers
  // XXX
  // Out of the box currently
  Row<std::size_t> constantLabels = _constantLeaf(m_);
  // std::unique_ptr<DecisionTreeClassifier<DataType>> ptr = std::make_unique<DecisionTreeClassifier<DataType>>(dataset_, constantLabels);
  // std::unique_ptr<ClassifierBase<DataType>> ptrBase(std::move(ptr));
  classifiers_.push_back(std::make_unique<DecisionTreeClassifier<DataType>>(dataset_, constantLabels));

  // partitions
  std::vector<int> part{m_};
  std::iota(part.begin(), part.end(), 0);
  partitions_.push_back(part);
  
  // first prediction
  current_classifier_ind_ = 0;
  Row<std::size_t> prediction;
  classifiers_[current_classifier_ind_]->Classify_(dataset_, prediction);
  predictions_.push_back(prediction);

  std::vector<int> rowMaskVec{m_};
  std::iota(rowMaskVec.begin(), rowMaskVec.end(), 0);  
  rowMask_ = Row<int>{rowMaskVec};
  
}

template<typename DataType>
typename GradientBoostClassifier<DataType>::LeavesMap
GradientBoostClassifier<DataType>::relabel(const Row<double>& label,
					   Row<std::size_t>& newLabel) {
  std::unordered_map<double, int> leavesMap;
  Row<double> uniqueVals = unique(label);
  for(auto it=uniqueVals.begin(); it!=uniqueVals.end(); ++it) {
    uvec ind = find(label == *(it));
    std::size_t equiv = std::distance(it, uniqueVals.end());
    newLabel.elem(ind).fill(equiv);
    leavesMap.insert(std::make_pair(*(it), static_cast<int>(equiv)));
  }
  
  return leavesMap;
}
	
template<typename DataType>
void
GradientBoostClassifier<DataType>::fit() {

  ;
}

template<typename DataType>
std::pair<std::unique_ptr<ClassifierBase<DataType>>, Row<std::size_t>>
GradientBoostClassifier<DataType>::predict(const Row<double>& labels) {
  Row<std::size_t> relabels;
  relabel(labels, relabels);
  ClassifierBase<DataType> c{dataset_, relabels};
  
  Row<std::size_t> predictions;
  c.Classify(dataset_, predictions);
  
  return std::make_pair(c, predictions);
}

#endif
