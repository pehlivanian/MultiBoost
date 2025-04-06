#ifndef __RECURSIVEMODEL_IMPL_HPP__
#define __RECURSIVEMODEL_IMPL_HPP__

template <typename DataType, typename ModelType>
auto RecursiveModel<DataType, ModelType>::_constantLeaf() -> Row<DataType> const {
  Row<DataType> r;
  r.zeros(static_cast<ModelType*>(this)->dataset_.n_cols);
  return r;
}
template <typename DataType, typename ModelType>
auto RecursiveModel<DataType, ModelType>::_constantLeaf(double val) -> Row<DataType> const {
  Row<DataType> r;
  r.ones(static_cast<ModelType*>(this)->dataset_.n_cols);
  r *= val;
  return r;
}

template <typename DataType, typename ModelType>
auto RecursiveModel<DataType, ModelType>::_randomLeaf() -> Row<DataType> const {
  Row<DataType> r(static_cast<ModelType*>(this)->dataset_.n_cols);
  std::mt19937 rng;
  std::uniform_real_distribution<DataType> dist{-1., 1.};
  r.imbue([&]() { return dist(rng); });
  return r;
}

template <typename DataType, typename ModelType>
void RecursiveModel<DataType, ModelType>::updateModels(
    std::unique_ptr<Model<DataType>>&& classifier, Row<DataType>& prediction) {
  latestPrediction_ += prediction;
  classifier->purge();
  models_.push_back(std::move(classifier));
}

/*
template<typename ClassifierType>
void
CompositeClassifier<ClassifierType>::fit_step(std::size_t stepNum) {
  // Implementation of W-cycle

  if (!reuseColMask_) {
    int colRatio = static_cast<size_t>(m_ * col_subsample_ratio_);
    colMask_ = PartitionUtils::sortedSubsample2(m_, colRatio);
  }

  Row<DataType> labels_slice = labels_.submat(zeros<uvec>(1), colMask_);
  std::pair<Row<DataType>, Row<DataType>> coeffs;

  Row<DataType> prediction;
  std::unique_ptr<ClassifierType> classifier;

  if (!hasInitialPrediction_) {

    this->latestPrediction_ = this->_constantLeaf(0.0);

    std::unique_ptr<ConstantTreeClassifier> cls_;
    Row<DataType> constantLeaf = ones<Row<DataType>>(labels_.n_elem);
    constantLeaf.fill(mean(labels_slice));

    cls_ = std::make_unique<ConstantTreeClassifier>(dataset_, constantLeaf);

    this->updateModels(std::move(cls_), constantLeaf);

  }

  if (ClassifierFileScope::W_CYCLE_PREFIT) {

    if (ClassifierFileScope::DIAGNOSTICS_0_ || ClassifierFileScope::DIAGNOSTICS_1_) {
      std::cerr << fit_prefix(depth_);
      std::cerr << "[*]PRE-FITTING LEAF CLASSIFIER FOR (PARTITIONSIZE, STEPNUM): ("
                << partitionSize_ << ", "
                << stepNum << " of "
                << steps_ << ")"
                << std::endl;
    }

    coeffs = generate_coefficients(labels_slice, colMask_);

    auto [subset_info, best_leaves] = computeOptimalSplit(coeffs.first,
                                                          coeffs.second,
                                                          stepNum,
                                                          partitionSize_,
                                                          learningRate_,
                                                          activePartitionRatio_,
                                                          ClassifierFileScope::SUBSET_DIAGNOSTICS);

    createRootClassifier(classifier, best_leaves);

    classifier->Classify(dataset_, prediction);

    this->updateModels(std::move(classifier), prediction);

    hasInitialPrediction_ = true;


    if (ClassifierFileScope::SUBSET_DIAGNOSTICS) {
      std::vector<DataType> gv0 = arma::conv_to<std::vector<DataType>>::from(coeffs.first);
      std::vector<DataType> hv0 = arma::conv_to<std::vector<DataType>>::from(coeffs.second);
      std::vector<DataType> preds = arma::conv_to<std::vector<DataType>>::from(prediction);
      std::vector<DataType> yv0 = arma::conv_to<std::vector<DataType>>::from(labels_);
      std::vector<DataType> yhatv0 =
arma::conv_to<std::vector<DataType>>::from(this->latestPrediction_); std::vector<DataType>
best_leavesv0 = arma::conv_to<std::vector<DataType>>::from(best_leaves);
      printSubsets<DataType>(subset_info.value(), best_leavesv0, preds, gv0, hv0, yv0, yhatv0,
colMask_);
    }

    if (ClassifierFileScope::DIAGNOSTICS_1_) {
      Row<DataType> latestPrediction_slice = this->latestPrediction_.submat(zeros<uvec>(1),
colMask_); Row<DataType> prediction_slice = prediction.submat(zeros<uvec>(1), colMask_); float eps =
std::numeric_limits<float>::epsilon();

      std::cerr << "[PRE-FIT ";
      for (std::size_t i=0; i<best_leaves.size(); ++i) {
        std::string status = "";
        if (fabs(best_leaves[i]-prediction_slice[i]) > eps)
          status = "MISCLASSIFIED";
        std::cerr << colMask_[i] << " : "
                  << labels_slice[i] << " : "
                  << latestPrediction_slice[i] << " :: "
                  << best_leaves[i] << " : "
                  << prediction_slice[i] << " :: "
                  << coeffs.first[i] << " : "
                  << coeffs.second[i] << " : "
                  << status << std::endl;
      }
      std::cerr << "]" << std::endl;
    }

  }

  //////////////////////////
  // BEGIN RECURSIVE STEP //
  //////////////////////////
  if (recursiveFit_ && (childPartitionSize_.size() > 1)) {

    if (ClassifierFileScope::DIAGNOSTICS_1_ || ClassifierFileScope::DIAGNOSTICS_0_) {
      std::cerr << fit_prefix(depth_);
      std::cerr << "[-]FITTING COMPOSITE CLASSIFIER FOR (PARTITIONSIZE, STEPNUM): ("
                << partitionSize_ << ", "
                << stepNum << " of "
                << steps_ << ")"
                << std::endl;
    }

    uvec colMask;
    if (ClassifierFileScope::NEW_COLMASK_FOR_CHILD) {
      int colRatio = static_cast<size_t>(m_ * col_subsample_ratio_);
      colMask = PartitionUtils::sortedSubsample2(m_, colRatio);
    } else {
      colMask = colMask_;
    }

    Context context{};
    ContextManager::childContext(context, *this);

    // allLeaves may not strictly fit the definition of labels here -
    // aside from the fact that it is of double type, it may have more
    // than one class. So we don't want to symmetrize, but we want
    // to remap the redundant values.
    std::unique_ptr<CompositeClassifier<ClassifierType>> classifier;
    if (hasInitialPrediction_) {
      classifier.reset(new CompositeClassifier<ClassifierType>(dataset_,
                                                               labels_,
                                                               this->latestPrediction_,
                                                               colMask,
                                                               context));
    } else {
      classifier.reset(new CompositeClassifier<ClassifierType>(dataset_,
                                                               labels_,
                                                               colMask,
                                                               context));
    }

    if (ClassifierFileScope::DIAGNOSTICS_1_) {

      std::cerr << "PREFIT: (PARTITIONSIZE, STEPNUM, NUMSTEPS): ("
                << partitionSize_ << ", "
                << stepNum << ", "
                << steps_ << ")"
                << std::endl;

    }

    classifier->fit();

    if (ClassifierFileScope::DIAGNOSTICS_1_) {

      std::cerr << "POSTFIT: (PARTITIONSIZE, STEPNUM, NUMSTEPS): ("
                << partitionSize_ << ", "
                << stepNum << ", "
                << steps_ << ")"
                << std::endl;

    }


    classifier->Predict(dataset_, prediction);

    this->updateModels(std::move(classifier), prediction);

    hasInitialPrediction_ = true;

  }
  ////////////////////////
  // END RECURSIVE STEP //
  ////////////////////////

  // If we are in recursive mode and partitionSize <= 2, fall through
  // to this case for the leaf classifier

  if (ClassifierFileScope::DIAGNOSTICS_0_ || ClassifierFileScope::DIAGNOSTICS_1_) {
    std::cerr << fit_prefix(depth_);
    std::cerr << "[*]POST-FITTING LEAF CLASSIFIER FOR (PARTITIONSIZE, STEPNUM): ("
              << partitionSize_ << ", "
              << stepNum << " of "
              << steps_ << ")"
              << std::endl;
  }

  if (!hasInitialPrediction_){
    // if (false&& (loss_ == lossFunction::LogLoss)) {
    if (false) {
      this->latestPrediction_ = this->_constantLeaf(mean(labels_slice));
    } else {
      this->latestPrediction_ = this->_constantLeaf(0.0);
    }
  }

  // Generate coefficients g, h
  coeffs = generate_coefficients(labels_slice, colMask_);

  // Compute optimal leaf choice on unrestricted dataset
  auto [subset_info, best_leaves] = computeOptimalSplit(coeffs.first,
                                                        coeffs.second,
                                                        stepNum,
                                                        partitionSize_,
                                                        learningRate_,
                                                        activePartitionRatio_,
                                                        ClassifierFileScope::SUBSET_DIAGNOSTICS);

  createRootClassifier(classifier, best_leaves);

  classifier->Classify(dataset_, prediction);

  this->updateModels(std::move(classifier), prediction);

  hasInitialPrediction_ = true;

  if (ClassifierFileScope::SUBSET_DIAGNOSTICS) {
    std::vector<DataType> gv0 = arma::conv_to<std::vector<DataType>>::from(coeffs.first);
    std::vector<DataType> hv0 = arma::conv_to<std::vector<DataType>>::from(coeffs.second);
    std::vector<DataType> preds = arma::conv_to<std::vector<DataType>>::from(prediction);
    std::vector<DataType> yv0 = arma::conv_to<std::vector<DataType>>::from(labels_);
    std::vector<DataType> yhatv0 =
arma::conv_to<std::vector<DataType>>::from(this->latestPrediction_); std::vector<DataType>
best_leavesv0 = arma::conv_to<std::vector<DataType>>::from(best_leaves);
    printSubsets<DataType>(subset_info.value(), best_leavesv0, preds, gv0, hv0, yv0, yhatv0,
colMask_);
  }


  if (ClassifierFileScope::DIAGNOSTICS_1_) {
    Row<DataType> latestPrediction_slice = this->latestPrediction_.submat(zeros<uvec>(1), colMask_);
    Row<DataType> prediction_slice = prediction.submat(zeros<uvec>(1), colMask_);
    float eps = std::numeric_limits<float>::epsilon();

    std::cerr << "[POST-FIT ";
    for (std::size_t i=0; i<best_leaves.size(); ++i) {
      std::string status = "";
      if (fabs(best_leaves[i]-prediction_slice[i]) > eps)
        status = "MISCLASSIFIED";
      std::cerr << colMask_[i] << " : "
                << labels_slice[i] << " : "
                << latestPrediction_slice[i] << " :: "
                << best_leaves[i] << " : "
                << prediction_slice[i] << " :: "
                << coeffs.first[i] << " : "
                << coeffs.second[i] << " : "
                << status << std::endl;

    }
    std::cerr << "]" << std::endl;
  }

}
*/

#endif
