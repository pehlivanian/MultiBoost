#ifndef __NEGATIVEFEEDBACK_HPP__
#define __NEGATIVEFEEDBACK_HPP__

#include <tuple>
#include <memory>

#include <mlpack/core.hpp>

#include "classifier.hpp"

using namespace arma;

template<typename DataType, typename ClassifierType, typename... Args>
class NegativeFeedback {
public:
  NegativeFeedback(const DiscreteClassifierBase<DataType, ClassifierType, Args...>& c) : c_{new ClassifierType{}}
  {}

  template<typename... Ts>
  void setRootClassifier(std::unique_ptr<ClassifierType>& ,
		    const Mat<DataType>&,
		    Row<DataType>&,
		    std::tuple<Ts...> const&,
		    float,
		    std::size_t);

  // From DiscreteClassifierBase<DataType, ClassifierType, Args...>
  //
  // virtual void Classify_(const Mat<DataType>&, Row<DataType>&) = 0;
  // virtual void Classify_(Mat<DataType>&&, Row<DataType>&) = 0;
  // 
  // So we dispatch
  //
  void Classify_(const Mat<DataType>& dataset, Row<DataType>& pred) { c_->Classify_(dataset, pred); }
  void Classify_(Mat<DataType>&& dataset, Row<DataType>& pred) { c_->Classify_(dataset, pred); }

  
		    
private:
  std::unique_ptr<ClassifierType> c_;
};
#endif
