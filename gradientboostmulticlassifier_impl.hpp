const unsigned int NUM_WORKERS = 6;

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::init_() {

  using C = GradientBoostClassClassifier<ClassifierType>;
  using ClassPair = std::pair<std::size_t, std::size_t>;

  Row<DataType> uniqueVals = sort(unique(labels_));
  numClasses_ = uniqueVals.size();
  
  if (allVOne_) {
    for (auto it=uniqueVals.begin(); it!=uniqueVals.end(); ++it) {
      uvec ind = find(labels_ == *it);
      Row<double> oneHot = zeros<Row<double>>(labels_.n_elem);
      oneHot.elem(ind).fill(1.);
      
      std::unique_ptr<C> classClassifier;
      classClassifier.reset(new C(dataset_, oneHot, context_.context, *it));
      classClassifiers_.push_back(std::move(classClassifier));    
    }
  } else {
    for (auto it1=uniqueVals.begin(); it1!=uniqueVals.end(); ++it1) {
      for (auto it2=it1+1; it2!=uniqueVals.end(); ++it2) {
	uvec ind1 = find(labels_ == *it1), ind2 = find(labels_ == *it2);
	Row<double> aVb = zeros<Row<double>>(labels_.n_elem);
	aVb.elem(ind1).fill(.5);
	aVb.elem(ind2).fill(1.);

	std::unique_ptr<C> classClassifier;
	classClassifier.reset(new C(dataset_, aVb, context_.context, ClassPair(*it1, *it2)));
	classClassifiers_.push_back(std::move(classClassifier));
	
      }
    }
  }

}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::fit() {

  using C = GradientBoostClassClassifier<ClassifierType>;

  ThreadsafeQueue<int> results_queue;
  std::vector<ThreadPool::TaskFuture<void>> futures;

  auto task = [&results_queue](C& classifier)
    { 
      classifier.fit(); 
      results_queue.push(0); 
    };

  for (auto &classClassifier : classClassifiers_) {
    futures.push_back(DefaultThreadPool::submitJob(task, std::ref(*classClassifier)));
  }
  
  for (auto& item : futures)
    item.get();

  int result;
  while (!results_queue.empty()) {
    bool valid = results_queue.waitPop(result);
  }
  
}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::purge() {
 
  dataset_ = ones<mat>(0,0);
  labels_ = ones<Row<double>>(0);
  dataset_oos_ = ones<mat>(0,0);
  labels_oos_ = ones<Row<double>>(0);

}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Predict(const mat& dataset, Row<DataType>& prediction) {
  ;
}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Classify_(const mat& dataset, Row<DataType>& prediction) {
  ;
}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Predict(Row<DataType>& prediction) {
  ;
}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Predict(Row<DataType>& prediction, const uvec& colMask) {
  ;
}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Predict(const mat& dataset, Row<DataType>& prediction, bool ignoreSymmetrization) {
  ;
}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Predict(std::string indexName, Row<DataType>& prediction, bool postSymmetrize) {
  ;
}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Predict(std::string indexName, const mat& dataset, Row<DataType>& prediction, bool postSymmetrize) {
  ;
}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Predict(Row<IntegralLabelType>& prediction) {
  ;
}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Predict(Row<IntegralLabelType>& prediction, const uvec& colMask) {
  ;
}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Predict(const mat& dataset, Row<IntegralLabelType>& prediction) {
  ;
}
