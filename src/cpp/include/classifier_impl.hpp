#ifndef __CLASSIFIER_IMPL_HPP__
#define __CLASSIFIER_IMPL_HPP__

template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::init_(const Mat<DataType>& dataset, Row<DataType>& labels, bool useWeights, Args&&... args) {
  labels_t_ = Row<std::size_t>(labels.n_cols);
  encode(labels, labels_t_, useWeights);
  setClassifier(dataset, labels_t_, std::forward<Args>(args)...);
  args_ = std::tuple<Args...>(args...);
}

template<typename DataType, typename ClassifierType, typename... Args>
void 
DiscreteClassifierBase<DataType, ClassifierType, Args...>::encode(const Row<DataType>& labels_d, Row<std::size_t>& labels_t, bool useWeights) {

  Row<DataType> uniqueVals = sort(unique(labels_d));    

  for (auto it=uniqueVals.begin(); it!=uniqueVals.end(); ++it) {

    uvec ind = find(labels_d == *it);
    std::size_t equiv = std::distance(it, uniqueVals.end()) - 1;

    labels_t.elem(ind).fill(equiv);
    if (useWeights) {
      DataType weightEquiv = weights_[ind[0]];
      weights_.elem(ind).fill(weightEquiv);
    }
    leavesMap_.insert(std::make_pair(static_cast<std::size_t>(equiv), (*it)));

  }
}

template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::purge_() {

  labels_t_ = ones<Row<std::size_t>>(0);

}

template<typename DataType, typename ClassifierType, typename... Args>
void 
DiscreteClassifierBase<DataType, ClassifierType, Args...>::decode(const Row<std::size_t>& labels_t, Row<DataType>& labels_d) {

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
DiscreteClassifierBase<DataType, ClassifierType, Args...>::setClassifier(const Mat<DataType>& dataset, Row<std::size_t>& labels, Args&&... args) {

  // Implicit 
  // void fit(const Mat<DataType>&, Row<std::size_t>&) 
  // called on ClasifierType

  classifier_ = std::make_unique<ClassifierType>(dataset, labels, std::forward<Args>(args)...);

}

template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::Classify_(const Mat<DataType>& dataset, Row<DataType>& labels) {

  Row<std::size_t> labels_t;

  // Implicit 
  // void Classify(const Mat<DataType>&, Row<std::size_t>&) called 
  // called on ClassifierType

  classifier_->Classify(dataset, labels_t);
  decode(labels_t, labels);
  
}

template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::Classify_(Mat<DataType>&& dataset, Row<DataType>& labels) {

  Row<std::size_t> labels_t;
  
  classifier_->Classify(std::move(dataset), labels_t);
  decode(labels_t, labels);
}


#endif
