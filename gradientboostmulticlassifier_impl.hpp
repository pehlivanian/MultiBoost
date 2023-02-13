const unsigned int NUM_WORKERS = 6;

template<typename ClassifierType>
void
GradientBoostClassClassifier<ClassifierType>::Classify_(const mat& dataset, Row<DataType>& prediction) {
  GradientBoostClassifier<ClassifierType>::Classify_(dataset, prediction);
}

template<typename ClassifierType>
void
GradientBoostClassClassifier<ClassifierType>::info(const mat& dataset) {
  if (allVOne_) {
    std::cout << "AllVOne: (" << classValue_ << ")" << std::endl;
  } else {
    std::cout << "OneVOne: (" << classValues_.first << ", " << classValues_.second << ")" << std::endl;
    std::cout << "Counts:  (" << num1_ << ", " << num2_ << ")" << std::endl;
  }

  Row<DataType> prediction;
  Classify_(dataset, prediction);

  std::cout << "Prediction" << std::endl;
  for(std::size_t i=0; i<10; ++i)
    std::cout << "[i] = " << prediction[i] << std::endl;

}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::init_() {

  using C = GradientBoostClassClassifier<ClassifierType>;
  using ClassPair = std::pair<std::size_t, std::size_t>;

  Row<DataType> uniqueVals = sort(unique(labels_));
  uniqueVals_ = conv_to<Row<std::size_t>>::from(uniqueVals);
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
	uvec ind0 = find((labels_ == *it1) || (labels_ == *it2));
	uvec ind1 = find(labels_ == *it1);
	uvec ind2 = find(labels_ == *it2);
	
	Row<double> labels_aVb = labels_.submat(zeros<uvec>(1), ind0);
	mat dataset_aVb = dataset_.cols(ind0);

	// If context contains OOS data we will have to create new
	// copies to slice for OOS aVb samples anyway; just copy now
	ClassifierContext::Context context_oos = context_.context;

	if (context_.context.hasOOSData) {
	  mat dataset_oos = context_.context.dataset_oos;
	  Row<DataType> labels_oos = context_.context.labels_oos;
	  uvec ind = find((labels_oos == *it1) || (labels_oos == *it2));
	  
	  Row<double> labels_aVb_oos = labels_oos.submat(zeros<uvec>(1), ind);
	  mat dataset_aVb_oos = dataset_oos.cols(ind);

	  context_oos.hasOOSData = true;
	  context_oos.dataset_oos = dataset_aVb_oos;
	  context_oos.labels_oos = labels_aVb_oos;

	  std::cout << "ALL V ALL (" << *it1 << ", " << *it2 << ")" << std::endl;

	  std::cout << "IS DATASET: (" << dataset_aVb.n_cols << " x " 
		    << dataset_aVb.n_rows << ")" << std::endl;
	  std::cout << "IS LABELS: (" << labels_aVb.n_cols << " x " 
		    << labels_aVb.n_rows << ")" << std::endl;

	  std::cout << "OOS DATASET: (" << context_oos.dataset_oos.n_cols << " x " 
		    << context_oos.dataset_oos.n_rows << ")" << std::endl;
	  std::cout << "OOS LABELS: (" << context_oos.labels_oos.n_cols << " x " 
		    << context_oos.labels_oos.n_rows << ")" << std::endl;

	}

	std::unique_ptr<C> classClassifier;
	classClassifier.reset(new C(dataset_aVb, 
				    labels_aVb, 
				    context_oos, 
				    ClassPair(*it1, *it2), 
				    ind1.n_elem, 
				    ind2.n_elem));
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

  // All class classifiers are not fit
  // printStats();

  std::cout << "HERE" << std::endl;
  Row<double> prediction;
  Predict(dataset_, prediction);
  
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
GradientBoostMultiClassifier<ClassifierType>::deSymmetrize(Row<DataType>& prediction) {
  ;
}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Predict(const mat& dataset, Row<DataType>& prediction, bool ignoreSymmetrization) {
  if (serialize_ && indexName_.size()) {
    throw predictionAfterClearedClassifiersException();
  }
  
  prediction = Row<DataType>(dataset.n_cols);
  std::vector<Row<DataType>> classPredictions;

  for (const auto& classClassifier : classClassifiers_) {
    classClassifier->info(dataset);
    Row<DataType> predictionStep;
    classClassifier->Classify_(dataset, predictionStep);
    classPredictions.push_back(predictionStep);
  }

  for (std::size_t i=0; i<dataset.n_cols; ++i) {

    // Set up unordered map
    std::unordered_map<std::size_t, std::size_t> votes;
    for(auto it=uniqueVals_.begin(); it!=uniqueVals_.end(); ++it) {
      votes[*it] = 0;
    }

    for (std::size_t j=0; j<classPredictions.size(); ++j) {
      Row<DataType> p = classPredictions[j];      
      votes[p.at(0,i)] += 1;
    }

    // Find majority vote
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    for (auto &i : votes) {
      pairs.push_back(i);
    }
    std::sort(pairs.begin(), pairs.end(), comp);

    if (pairs[0].second > pairs[1].second) {
      prediction(i) = pairs[0].first;
    } else {
      std::cout << "HERE" << std::endl;
    }
    
    std::cout << "HERE" << std::endl;
  
  }

  if (symmetrized_ and not ignoreSymmetrization) {
    deSymmetrize(prediction);
  }

}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Classify_(const mat& dataset, Row<DataType>& prediction) {
  Predict(dataset, prediction, false);
}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Predict(Row<DataType>& prediction) {
  return GradientBoostClassifier<ClassifierType>::latestPrediction_;
}

template<typename ClassifierType>
void
GradientBoostMultiClassifier<ClassifierType>::Predict(Row<DataType>& prediction, const uvec& colMask) {
  ;
}

///////////////////
// Archive versions
///////////////////
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
