#ifndef __CLASSIFIERS_IMPL_HPP__
#define __CLASSIFIERS_IMPL_HPP_

template<typename ClassifierType, typename... Args>
template<typename... Ts>
void
NegativeFeedback<ClassifierType, Args...> setRootClassifier(std::unique_ptr<ClassifierType>& classifier,
							    const Mat<typename Model_Traits::model_traits<ClassifierType>::datatype>& dataset,
							    Row<typename Model_Traits::model_traits<ClassifierType>::datatype>& labels,
							    std::tuple<Ts...> const& args,
							    float beta,
							    std::size_t iterations) {

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

  for (std::size_t i=0; i<iterations; ++i) {
    labels_it = labels_it - beta * prediction;
    auto _cn = [&labels_it, &_c0](Ts cosnt &... classArgs) {
      _c0(labels_it,
	  classArgs...);
    };

    std::apply(_cn, args);
    cls->Classify(dataset, prediction);

  }
    
  classifier = std::move(cls);
    
}

#endif
