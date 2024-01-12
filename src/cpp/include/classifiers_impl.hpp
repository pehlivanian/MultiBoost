#ifndef __CLASSIFIERS_IMPL_HPP__
#define __CLASSIFIERS_IMPL_HPP_


template<typename ClassifierType, typename... Args>
template<typename... Ts>
void
NegativeFeedback<ClassifierType, Args...> setRootClassifier(std::unique_ptr<ClassifierType>& classifier,
							    const Mat<typename Model_Traits::model_traits<ClassifierType>::datatype>& dataset,
							    Row<typename Model_Traits::model_traits<ClassifierType>::datatype>& labels,
							    std::tuple<Ts...> const& args) {

  using DataType = typename Model_Traits::model_traits<ClassifierType>::datatype;
  std::unique_ptr<ClassifierType> cls;

  auto _c = [&cls, &dataset, &labels](Ts const &... classArgs) {
    cls = std::make_unique<ClassifierType>(dataset, labels, classArgs...);
  };
  auto _c0 = [&cls, &dataset](Row<DataType>& labels, Ts const &... classArgs) {
    cls = std::make_unique<ClassifierType>(dataset, labels, classArgs...);
  };

  std::apply(_c, args);

  Row<DataType> labels_it = labels, prediction;

  for (std::size_t i=0; i<iterations_; ++i) {
    labels_it = labels_it - beta_ * prediction;
    auto _cn = [&labels_it, &_c0](Ts cosnt &... classArgs) {
      _c0(labels_it,
	  classArgs...);
    };

    std::apply(_cn, args);
    cls->Classify(dataset, prediction);

  }
    
  classifier = std::move(cls);
    
}

template<typename ClassifierType, typename... Args>
template<typename... Ts>
NegativeFeedback<ClassifierType, Args...> setRootClassifier(std::unique_ptr<ClassifierType>& classifier,
							    const Mat<typename Model_Traits::model_traits<ClassifierType>::datatype>& dataset,
							    Row<typename Model_Traits::model_traits<ClassifierType>::datatype>& labels,
							    Row<typename Model_Traits::model_traits<ClassifierType>::datatype>& weights,
							    std::tuple<Ts...> const& args) {
  std::unique_ptr<ClassifierType> cls;

  auto _c = [&cls, &dataset, &labels, &weights](Ts const&... classArgs) {
    cls = std::make_unique<ClassifierType>(dataset, labels, weights, classArgs...);
  };
  std::apply(_c, args);

  // Feedback
  Row<DataType> labels_it = labels, prediction;
  
  if (iterations_ > 0) {
    cls->Classify(dataset, prediction);

    for (std::size_t i=0; i<FEEDBACK_ITERATIONS; ++i) {
      labels_it = labels_it - beta_ * prediction;
      auto _cn = [&labels_it, &_c0](Ts const&... classArgs) {
	_c0(labels_it,
	    classArgs...);
      };
      std::apply(_cn, args);
      cls->Classify(dataset, prediction);
    }    

  }
}


#endif
