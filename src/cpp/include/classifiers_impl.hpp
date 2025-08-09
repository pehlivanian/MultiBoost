#ifndef __CLASSIFIERS_IMPL_HPP__
#define __CLASSIFIERS_IMPL_HPP__

template <typename ClassifierType, typename... Args>
template <typename... Ts>
void NegativeFeedback<ClassifierType, Args...>::setRootClassifier(
    std::unique_ptr<ClassifierType>& classifier,
    const Mat<typename Model_Traits::model_traits<ClassifierType>::datatype>& dataset,
    Row<typename Model_Traits::model_traits<ClassifierType>::datatype>& labels,
    std::tuple<Ts...> const& args) {
  using DataType = typename Model_Traits::model_traits<ClassifierType>::datatype;
  // std::cout << "DEBUG: NegativeFeedback::setRootClassifier called with beta=" << beta_ << ", iterations=" << iterations_ << std::endl;
  std::unique_ptr<ClassifierType> cls;

  auto _c = [&cls, &dataset, &labels](Ts const&... classArgs) {
    cls = std::make_unique<ClassifierType>(dataset, labels, classArgs...);
  };
  std::apply(_c, args);

  Row<DataType> labels_it = labels, prediction;

  if (iterations_ > 0) {
    cls->Classify(dataset, prediction);
    
    for (std::size_t i = 0; i < iterations_; ++i) {
      labels_it = labels_it - beta_ * prediction;
      
      // Convert continuous labels back to discrete labels for classifier creation
      // The negative feedback algorithm modifies labels to be continuous, but DecisionTreeClassifier needs discrete labels
      Row<DataType> labels_discrete(labels_it.n_elem);
      
      // Simple discretization: convert continuous values back to nearest discrete classes
      // For binary classification, map negative values to 0, positive to 1
      for (std::size_t j = 0; j < labels_it.n_elem; ++j) {
        if (labels_it[j] < 0.5) {
          labels_discrete[j] = 0.0;
        } else {
          labels_discrete[j] = 1.0;
        }
      }
      
      auto _c2 = [&cls, &dataset, &labels_discrete](Ts const&... classArgs) {
        cls = std::make_unique<ClassifierType>(dataset, labels_discrete, classArgs...);
      };
      std::apply(_c2, args);
      cls->Classify(dataset, prediction);
    }
  }

  // IMPORTANT: Set the classifier for return AND for this decorator's base class
  classifier = std::move(cls);
  
  // The decorator itself needs to be initialized with the ORIGINAL labels, not the processed ones
  // The negative feedback algorithm modifies labels_it, but the base class should use original labels
  // Now that we have protected access, we can call init_ directly with original labels
  auto _init = [this, &dataset, &labels](Ts const&... classArgs) {
    this->init_(dataset, labels, false, const_cast<Ts&&>(classArgs)...);
  };
  std::apply(_init, args);
  
  // std::cout << "DEBUG: NegativeFeedback::setRootClassifier completed - classifier_ should now be initialized" << std::endl;
}

template <typename ClassifierType, typename... Args>
template <typename... Ts>
void NegativeFeedback<ClassifierType, Args...>::setRootClassifier(
    std::unique_ptr<ClassifierType>& classifier,
    const Mat<typename Model_Traits::model_traits<ClassifierType>::datatype>& dataset,
    Row<typename Model_Traits::model_traits<ClassifierType>::datatype>& labels,
    Row<typename Model_Traits::model_traits<ClassifierType>::datatype>& weights,
    std::tuple<Ts...> const& args) {
  using DataType = typename Model_Traits::model_traits<ClassifierType>::datatype;
  // std::cout << "DEBUG: NegativeFeedback::setRootClassifier (with weights) called with beta=" << beta_ << ", iterations=" << iterations_ << std::endl;
  std::unique_ptr<ClassifierType> cls;

  auto _c = [&cls, &dataset, &labels, &weights](Ts const&... classArgs) {
    cls = std::make_unique<ClassifierType>(dataset, labels, weights, classArgs...);
  };
  std::apply(_c, args);

  Row<DataType> labels_it = labels, prediction;

  if (iterations_ > 0) {
    cls->Classify(dataset, prediction);

    for (std::size_t i = 0; i < iterations_; ++i) {
      labels_it = labels_it - beta_ * prediction;
      
      // Convert continuous labels back to discrete labels for classifier creation
      // The negative feedback algorithm modifies labels to be continuous, but DecisionTreeClassifier needs discrete labels
      Row<DataType> labels_discrete(labels_it.n_elem);
      
      // Simple discretization: convert continuous values back to nearest discrete classes
      // For binary classification, map negative values to 0, positive to 1
      for (std::size_t j = 0; j < labels_it.n_elem; ++j) {
        if (labels_it[j] < 0.5) {
          labels_discrete[j] = 0.0;
        } else {
          labels_discrete[j] = 1.0;
        }
      }
      
      auto _c2 = [&cls, &dataset, &labels_discrete, &weights](Ts const&... classArgs) {
        cls = std::make_unique<ClassifierType>(dataset, labels_discrete, weights, classArgs...);
      };
      std::apply(_c2, args);
      cls->Classify(dataset, prediction);
    }
  }

  // IMPORTANT: Set the classifier for return AND for this decorator's base class
  classifier = std::move(cls);
  
  // The decorator itself needs to be initialized with the ORIGINAL labels and weights, not processed ones
  // The negative feedback algorithm modifies labels_it, but the base class should use original labels
  // Now that we have protected access, we can call init_ directly with original labels and weights
  this->weights_ = weights;
  auto _init = [this, &dataset, &labels](Ts const&... classArgs) {
    this->init_(dataset, labels, true, const_cast<Ts&&>(classArgs)...);
  };
  std::apply(_init, args);
  
  // std::cout << "DEBUG: NegativeFeedback::setRootClassifier (with weights) completed - classifier_ should now be initialized" << std::endl;
}

#endif
