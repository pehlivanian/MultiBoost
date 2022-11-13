#ifndef __GRADIENTCLASSIFIER_IMPL_HPP__
#define __GRADIENTCLASSIFIER_IMPL_HPP__

template<typename DataType, typename ClassifierType, typename... Args>
void 
DiscreteClassifierBase<DataType, ClassifierType, Args...>::encode(const Row<DataType>& labels_d, Row<std::size_t>& labels_t) {
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
DiscreteClassifierBase<DataType, ClassifierType, Args...>::decode(const Row<std::size_t>& labels_t, Row<DataType>& labels_d) {
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
  Row<std::size_t> labels_t;
  labels = Row<DataType>(dataset.n_cols);
  classifier_->Classify(dataset, labels_t);
  decode(labels_t, labels);
  
  // Check error
  Row<std::size_t> prediction;
  classifier_->Classify(dataset_, prediction);
  const double trainError = arma::accu(prediction != labels_t_) * 100. / labels_t_.n_elem;
  for (size_t i=0; i<25; ++i)
    std::cout << labels_t_[i] << " ::(2) " << prediction[i] << std::endl;
  std::cout << "dataset size:    " << dataset.n_rows << " x " << dataset.n_cols << std::endl;
  std::cout << "prediction size: " << prediction.n_rows << " x " << prediction.n_cols << std::endl;
  std::cout << "Training error (2): " << trainError << "%." << std::endl;
  
}

template<typename DataType, typename ClassifierType, typename... Args>
void 
DiscreteClassifierBase<DataType, ClassifierType, Args...>::setClassifier(const mat& dataset, Row<std::size_t>& labels, Args&&... args) {
  classifier_.reset(new ClassifierType(dataset, labels, std::forward<Args>(args)...));
  
  
  // Check predictions
  /*
    Row<std::size_t> prediction;
    classifier_->Classify(dataset, prediction);
    const double trainError = arma::accu(prediction != labels) * 100. / labels.n_elem;
    cout << "Training error (1): " << trainError << "%." << endl;
  */
}

template<typename DataType, typename ClassifierType, typename... Args>
void 
ContinuousClassifierBase<DataType, ClassifierType, Args...>::setClassifier(const mat& dataset, Row<DataType>& labels, Args&&... args) {
  classifier_.reset(new ClassifierType(dataset, labels, std::forward<Args>(args)...));
}

template<typename DataType, typename ClassifierType, typename... Args>
void 
ContinuousClassifierBase<DataType, ClassifierType, Args...>::Classify_(const mat& dataset, Row<DataType>& labels) {
      classifier_->Predict(dataset, labels);
}
#endif
