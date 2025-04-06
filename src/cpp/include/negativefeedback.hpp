#ifndef __NEGATIVEFEEDBACK_HPP__
#define __NEGATIVEFEEDBACK_HPP__

#include <memory>
#include <mlpack/core.hpp>
#include <tuple>

#include "classifier.hpp"
#include "classifiers.hpp"

using namespace arma;

template <typename ClassifierType, typename... Args>
class NegativeFeedback : DecoratorClassifier<
                             ClassifierType,
                             typename Model_Traits::model_traits<ClassifierType>::modelArgs> {
public:
  using DataType = typename Model_Traits::model_traits<ClassifierType>::datatype;
  using modelArgs = typename Model_Traits::model_traits<ClassifierType>::modelArgs;

  NegativeFeedback(){};

  NegativeFeedback(const Mat<DataType>& dataset, Row<DataType>& labels, Args&&... args)
      : DecoratorClassifier<ClassifierType, modelArgs>(
            dataset, labels, std::forward<Args>(args)...) {}

  NegativeFeedback(
      const Mat<DataType>& dataset, Row<DataType>& labels, Row<DataType>& weights, Args&&... args)
      : DecoratorClassifier<ClassifierType, modelArgs>(
            dataset, labels, weights, std::forward<Args>(args)...) {}

  template <typename... Ts>
  void setRootClassifier(
      std::unique_ptr<ClassifierType>&,
      const Mat<DataType>&,
      Row<DataType>&,
      std::tuple<Ts...> const&,
      float,
      std::size_t);
};
#endif
