#ifndef __CLASSIFIERS_IMPL_HPP__
#define __CLASSIFIERS_IMPL_HPP__

// Helper function to create classifier from tuple of arguments
template <typename DecoratedType, typename DataType, typename... Args>
std::unique_ptr<DecoratedType> createClassifier(
    const Mat<DataType>& dataset,
    Row<DataType>& labels,
    const std::tuple<Args...>& args) {
  return std::apply([&](const Args&... classArgs) {
    return std::make_unique<DecoratedType>(dataset, labels, classArgs...);
  }, args);
}

template <typename DecoratedType, typename DataType, typename... Args>
std::unique_ptr<DecoratedType> createClassifierWithWeights(
    const Mat<DataType>& dataset,
    Row<DataType>& labels,
    Row<DataType>& weights,
    const std::tuple<Args...>& args) {
  return std::apply([&](const Args&... classArgs) {
    return std::make_unique<DecoratedType>(dataset, labels, weights, classArgs...);
  }, args);
}

template <typename DecoratedType, typename... DecoratedArgs>
template <typename... Ts>
void NegativeFeedback<DecoratedType, DecoratedArgs...>::setRootClassifier(
    std::unique_ptr<DecoratedType>& classifier,
    const Mat<typename Model_Traits::model_traits<DecoratedType>::datatype>& dataset,
    Row<typename Model_Traits::model_traits<DecoratedType>::datatype>& labels,
    std::tuple<Ts...> const& args) {
  
  // Create the base classifier
  classifier = createClassifier<DecoratedType>(dataset, labels, args);
  
  // Initialize the decorator base class - just pass the arguments directly without modification
  // since the base class constructors should handle const casting internally
}

template <typename DecoratedType, typename... DecoratedArgs>
template <typename... Ts>
void NegativeFeedback<DecoratedType, DecoratedArgs...>::setRootClassifier(
    std::unique_ptr<DecoratedType>& classifier,
    const Mat<typename Model_Traits::model_traits<DecoratedType>::datatype>& dataset,
    Row<typename Model_Traits::model_traits<DecoratedType>::datatype>& labels,
    Row<typename Model_Traits::model_traits<DecoratedType>::datatype>& weights,
    std::tuple<Ts...> const& args) {
  
  // Create the base classifier with weights
  classifier = createClassifierWithWeights<DecoratedType>(dataset, labels, weights, args);
  
  // Initialize the decorator base class with weights
  this->weights_ = weights;
}

#endif
