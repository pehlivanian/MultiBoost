// #define DEBUG() __debug dd{__FILE__, __FUNCTION__, __LINE__};
#define DEBUG() ;

const unsigned int NUM_WORKERS = 6;

template <typename ClassifierType>
void GradientBoostClassClassifier<ClassifierType>::Classify_(
    const mat& dataset, Row<DataType>& prediction) {
  GradientBoostClassifier<ClassifierType>::Classify_(dataset, prediction);
}

template <typename ClassifierType>
void GradientBoostClassClassifier<ClassifierType>::info(const mat& dataset) {
  DEBUG()

  if (allVOne_) {
    std::cout << "AllVOne: (" << classValue_ << ")" << std::endl;
  } else {
    std::cout << "OneVOne: (" << classValues_.first << ", " << classValues_.second << ")"
              << std::endl;
    std::cout << "Counts:  (" << num1_ << ", " << num2_ << ")" << std::endl;
  }

  Row<DataType> prediction;
  Classify_(dataset, prediction);
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::contextInit_(
    MultiClassifierContext::CombinedContext<ClassifierType>&& context) {
  // XXX
  DEBUG();
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::init_() {
  DEBUG()

  using C = GradientBoostClassClassifier<ClassifierType>;
  using ClassPair = std::pair<std::size_t, std::size_t>;

  Row<DataType> uniqueVals = sort(unique(labels_));
  uniqueVals_ = conv_to<Row<std::size_t>>::from(uniqueVals);
  numClasses_ = uniqueVals.size();

  std::unique_ptr<C> classClassifier;
  Context<ClassifierType> context = context_.context;

  if (allVOne_) {
    for (auto it = uniqueVals.begin(); it != uniqueVals.end(); ++it) {
      uvec ind0 = find(labels_ == *it);
      Row<double> labels_aVo = zeros<Row<double>>(labels_.n_elem);
      labels_aVo.elem(ind0).fill(1.);

      if (hasOOSData_) {
        uvec ind = find(labels_oos_ == *it);
        Row<double> labels_aVo_oos = zeros<Row<double>>(labels_oos_.n_elem);
        labels_aVo_oos.elem(ind).fill(1.);

        classClassifier.reset(
            new C(dataset_, labels_aVo, dataset_oos_, labels_aVo_oos, context, *it));

      } else {
        classClassifier.reset(new C(dataset_, labels_aVo, context, *it));
      }

      classClassifiers_.push_back(std::move(classClassifier));
    }
  } else {
    for (auto it1 = uniqueVals.begin(); it1 != uniqueVals.end(); ++it1) {
      for (auto it2 = it1 + 1; it2 != uniqueVals.end(); ++it2) {
        uvec ind0 = find((labels_ == *it1) || (labels_ == *it2));
        uvec ind1 = find(labels_ == *it1);
        uvec ind2 = find(labels_ == *it2);

        Row<double> labels_aVb = labels_.submat(zeros<uvec>(1), ind0);
        mat dataset_aVb = dataset_.cols(ind0);

        if (hasOOSData_) {
          uvec ind = find((labels_oos_ == *it1) || (labels_oos_ == *it2));

          Row<double> labels_aVb_oos = labels_oos_.submat(zeros<uvec>(1), ind);
          mat dataset_aVb_oos = dataset_oos_.cols(ind);

          std::cout << "ALL V ALL (" << *it1 << ", " << *it2 << ")" << std::endl;

          std::cout << "IS DATASET: (" << dataset_aVb.n_cols << " x " << dataset_aVb.n_rows << ")"
                    << std::endl;
          std::cout << "IS LABELS: (" << labels_aVb.n_cols << " x " << labels_aVb.n_rows << ")"
                    << std::endl;

          std::cout << "OOS DATASET: (" << dataset_aVb_oos.n_cols << " x " << dataset_aVb_oos.n_rows
                    << ")" << std::endl;
          std::cout << "OOS LABELS: (" << labels_aVb_oos.n_cols << " x " << labels_aVb_oos.n_rows
                    << ")" << std::endl;

          classClassifier.reset(new C(
              dataset_aVb,
              labels_aVb,
              dataset_aVb_oos,
              labels_aVb_oos,
              context,
              ClassPair(*it1, *it2),
              ind1.n_elem,
              ind2.n_elem));
        } else {
          classClassifier.reset(new C(
              dataset_aVb, labels_aVb, context, ClassPair(*it1, *it2), ind1.n_elem, ind2.n_elem));
        }
        classClassifiers_.push_back(std::move(classClassifier));
      }
    }
  }
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::printStats(int stepNum) {
  DEBUG()

  auto now = std::chrono::system_clock::now();
  auto UTC = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream datetime;
  datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
  auto suff = datetime.str();

  Row<double> yhat;
  Predict(yhat);

  double error_is = err(yhat, labels_);
  std::cout << suff << ": "
            << "(STEPS = " << steps_ << ") "
            << "STEP: " << stepNum << " IS ERROR: " << error_is << "%" << std::endl;

  Row<double> yhat_oos;
  Predict(dataset_oos_, yhat_oos);

  double error_oos = err(yhat_oos, labels_oos_);
  std::cout << suff << ": "
            << "(STEPS = " << steps_ << ") "
            << "STEP: " << stepNum << " OOS ERROR: " << error_oos << "%" << std::endl;
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::fit() {
  DEBUG()

  for (std::size_t stepNum = 1; stepNum <= steps_; ++stepNum) {
    fit_step(stepNum);

    if (serialize_) {
      commit();
    }
  }

  // Serialize residual
  if (serialize_) commit();

  // print final stats
  printStats(steps_);
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::commit() {
  DEBUG()

  std::cerr << "CURRENTLY NOT SUPPORTED" << std::endl;
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::fit_step(int stepNum) {
  DEBUG()

  using ClassClassifier = GradientBoostClassClassifier<ClassifierType>;

  ThreadsafeQueue<int> results_queue;
  std::vector<ThreadPool::TaskFuture<int>> futures;

  auto task = [&results_queue](ClassClassifier& classifier) {
    classifier.fit();
    results_queue.push(0);
    return 0;
  };

  for (auto& classClassifier : classClassifiers_) {
    // futures.push_back(DefaultThreadPool::submitJob_n<3>(task, std::ref(*classClassifier)));
    futures.push_back(DefaultThreadPool::submitJob(task, std::ref(*classClassifier)));
  }

  for (auto& item : futures) int fresult = item.get();

  int result;
  while (!results_queue.empty()) {
    bool valid = results_queue.waitPop(result);
  }
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::purge() {
  DEBUG()

  dataset_ = ones<mat>(0, 0);
  labels_ = ones<Row<double>>(0);
  dataset_oos_ = ones<mat>(0, 0);
  labels_oos_ = ones<Row<double>>(0);
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::deSymmetrize(Row<DataType>& prediction) {
  DEBUG();
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::Predict(
    const mat& dataset, Row<DataType>& prediction, bool ignoreSymmetrization) {
  DEBUG()

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

  for (std::size_t i = 0; i < dataset.n_cols; ++i) {
    // Set up unordered map
    std::unordered_map<std::size_t, std::size_t> votes;
    for (auto it = uniqueVals_.begin(); it != uniqueVals_.end(); ++it) {
      votes[*it] = 0;
    }

    for (std::size_t j = 0; j < classPredictions.size(); ++j) {
      Row<DataType> p = classPredictions[j];
      votes[p.at(0, i)] += 1;
    }

    // Find majority vote
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    for (auto& i : votes) {
      pairs.push_back(i);
    }
    std::sort(pairs.begin(), pairs.end(), comp);

    if (pairs[0].second > pairs[1].second) {
      prediction(i) = pairs[0].first;
    } else {
      std::cerr << "MAJORITY RULE VIOLATION" << std::endl;
    }
  }

  if (symmetrized_ and not ignoreSymmetrization) {
    deSymmetrize(prediction);
  }
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::Classify_(
    const mat& dataset, Row<DataType>& prediction) {
  DEBUG()

  Predict(dataset, prediction, false);
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::Predict(Row<DataType>& prediction) {
  DEBUG()

  Predict(dataset_, prediction);
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::Predict(
    Row<DataType>& prediction, const uvec& colMask) {
  DEBUG()

      ;
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::Predict(
    Row<typename GradientBoostMultiClassifier<ClassifierType>::IntegralLabelType>& prediction) {
  DEBUG()

  Row<DataType> prediction_d = conv_to<Row<DataType>>::from(prediction);
  Predict(prediction_d);
  prediction =
      conv_to<Row<typename GradientBoostMultiClassifier<ClassifierType>::IntegralLabelType>>::from(
          prediction_d);
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::Predict(
    Row<typename GradientBoostMultiClassifier<ClassifierType>::IntegralLabelType>& prediction,
    const uvec& colMask) {
  DEBUG()

  Row<DataType> prediction_d = conv_to<Row<DataType>>::from(prediction);
  Predict(prediction_d, colMask);
  prediction =
      conv_to<Row<typename GradientBoostMultiClassifier<ClassifierType>::IntegralLabelType>>::from(
          prediction_d);
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::Predict(
    const mat& dataset,
    Row<typename GradientBoostMultiClassifier<ClassifierType>::IntegralLabelType>& prediction) {
  DEBUG()

  Row<DataType> prediction_d = conv_to<Row<DataType>>::from(prediction);
  Predict(dataset, prediction_d);
  prediction =
      conv_to<Row<typename GradientBoostMultiClassifier<ClassifierType>::IntegralLabelType>>::from(
          prediction_d);
}

///////////////////
// Archive versions
///////////////////
template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::Predict(
    std::string indexName, Row<DataType>& prediction, bool postSymmetrize) {
  DEBUG()

      ;
}

template <typename ClassifierType>
void GradientBoostMultiClassifier<ClassifierType>::Predict(
    std::string indexName, const mat& dataset, Row<DataType>& prediction, bool postSymmetrize) {
  DEBUG()

      ;
}
