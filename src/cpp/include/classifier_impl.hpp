#ifndef __CLASSIFIER_IMPL_HPP__
#define __CLASSIFIER_IMPL_HPP__


// Some utilities for manipulating tuples
namespace {

  template<typename... Ts>
  decltype(auto) to_tuple(Ts &&... ts) {
    auto tup = std::make_tuple(std::forward<Ts>(ts)...);
    return tup;
  }

  template<typename T, std::size_t... I>
  void print_tuple(const T& tup, std::index_sequence<I...>) {
    std::cout << "(";
    (..., (std::cout << (I == 0? "" : ", ") << std::get<I>(tup)));
  }
  
  template<typename... T>
  void print_tuple(const std::tuple<T...>& tup) {
    print_tuple(tup, std::make_index_sequence<sizeof...(T)>());
  }

  template<std::size_t Ofst, class Tuple, std::size_t... I>
  constexpr auto slice_impl(Tuple&& t, std::index_sequence<I...>) {
    return std::forward_as_tuple(std::get<I + Ofst>(std::forward<Tuple>(t))...);
  }
  
  template<std::size_t I1, std::size_t I2, class Cont>
  constexpr auto tuple_slice(Cont&& t) {
    static_assert(I2 >= I1, "invalid slice");
    static_assert(std::tuple_size<std::decay_t<Cont>>::value >= I2,
		  "slice index out of bounds");
    return slice_impl<I1>(std::forward<Cont>(t),
			  std::make_index_sequence<I2 - I1>{});
  }
  
  template<std::size_t N, typename... Types>
  auto remove_element_from_tuple(const std::tuple<Types...>& t) {
    return std::tuple_cat(
			  tuple_slice<0, N>(t),
			  tuple_slice<N + 1, sizeof...(Types)>(t)
			  );
  }

  template<std::size_t N, typename... T, std::size_t... I>
  std::tuple<std::tuple_element_t<N+I, std::tuple<T...>>...>
  sub(std::index_sequence<I...>);
  
  template<std::size_t N, typename... T>
  using subpack = decltype(sub<N, T...>(std::make_index_sequence<sizeof...(T) - N>{}));

  template<typename DataType, typename... Args>
  struct dispatcher {
    dispatcher(const Mat<DataType>& dataset, Row<DataType>& labels, std::tuple<Args...>& params) :
      dataset_{dataset},
      labels_{labels},
      params_{params} {}

    template<std::size_t... I>
    void call_func(std::index_sequence<I...>) {
      set_classifier(dataset_, labels_, std::get<I>(params_)...);
    }

    void dispatch() {
      call_func(std::index_sequence_for<Args...>{});
    }

    // Try to avoid copying
    std::tuple<Args...> params_;
    Mat<DataType> dataset_;
    Row<DataType>& labels_;
  
  };

} // namespace 

template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::init_(const Mat<DataType>& dataset, Row<DataType>& labels, bool useWeights, Args&&... args) {
  labels_t_ = Row<std::size_t>(labels.n_cols);
  encode(labels, labels_t_, useWeights);
  if (useWeights) {
    constexpr auto N = sizeof...(args);
    if constexpr (N > 1) {
      auto args_tup = to_tuple(std::forward<Args>(args)...);
      auto numClasses = std::get<0>(args_tup);
      auto args_short = remove_element_from_tuple<0>(args_tup);
      auto I = std::index_sequence_for<decltype(args_short)>{};
      // setClassifier(dataset, labels_t_, numClasses, weights_, std::get<I>(args_short)...);
      setClassifier<Args...>(dataset, labels_t_, numClasses, weights_, std::forward<Args>(args)...);
    } else {
      setClassifier<Args...>(dataset, labels_t_, std::forward<Args>(args)...);
    }
  } else {
    setClassifier<Args...>(dataset, labels_t_, std::forward<Args>(args)...);
  }

  args_ = std::tuple<Args...>(args...);
}

/*
template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::init_(const Mat<DataType>& dataset,
								 Row<DataType>& labels,
								 bool useWeights,
								 Args... args) {
  init_(dataset, labels, useWeights, std::move(args)...);
}
*/


template<typename DataType, typename ClassifierType, typename... Args>
template<typename... ClassArgs>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::setClassifier(const Mat<DataType>& dataset, Row<std::size_t>& labels, std::size_t numClasses, const Row<DataType>& weights, ClassArgs &&... args) {
    classifier_ = std::make_unique<ClassifierType>(dataset, labels, std::forward<ClassArgs>(args)...);
    // classifier_ = std::make_unique<ClassifierType>(dataset, labels, numClasses, weights_, std::forward<ClassArgs>(args)...);
}

template<typename DataType, typename ClassifierType, typename... Args>
template<typename... ClassArgs>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::setClassifier(const Mat<DataType>& dataset, Row<std::size_t>& labels, ClassArgs &&... args) {
    classifier_ = std::make_unique<ClassifierType>(dataset, labels, std::forward<ClassArgs>(args)...);
}

/*
template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::setClassifier(const Mat<DataType>& dataset, Row<std::size_t>& labels, bool useWeights, Args&&... args) {

  // Implicit 
  // void fit(const Mat<DataType>&, Row<std::size_t>&) 
  // called on ClasifierType

  if (useWeights) {
    // Unfortunately the numClasses parameter comes before the 
    // weights vector, we have to unroll the parameter pack
    constexpr auto N = sizeof...(args);
    auto args_tup = to_tuple(std::forward<Args>(args)...);
    if constexpr (N > 1) {
      auto args_extended = std::tuple_cat(tuple_slice<0,1>(args_tup),
					  std::make_tuple(weights_),
					  tuple_slice<1,N>(args_tup));

      classifier_ = std::make_unique<ClassifierType>(dataset, labels, std::forward<Args>(args)...);
    } else {
      classifier_ = std::make_unique<ClassifierType>(dataset, labels, std::forward<Args>(args)...);
    }
  } else {

    auto c_ = [&dataset, &labels](Args...){};

    classifier_ = std::make_unique<ClassifierType>(dataset, labels, std::forward<Args>(args)...);
  }

}
*/

template<typename DataType, typename ClassifierType, typename... Args>
void 
DiscreteClassifierBase<DataType, ClassifierType, Args...>::encode(const Row<DataType>& labels_d, Row<std::size_t>& labels_t, bool useWeights) {

  Row<DataType> uniqueVals = sort(unique(labels_d));    

  for (auto it=uniqueVals.cbegin(); it!=uniqueVals.cend(); ++it) {

    uvec ind = find(labels_d == *it);
    std::size_t equiv = std::distance(it, uniqueVals.cend()) - 1;

    labels_t.elem(ind).fill(equiv);
    leavesMap_.insert(std::make_pair(static_cast<std::size_t>(equiv), (*it)));

  }

}

template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::purge_() {

  labels_t_ = ones<Row<std::size_t>>(0);

}

template<typename DataType, typename ClassifierType, typename... Args>
void 
DiscreteClassifierBase<DataType, ClassifierType, Args...>::decode(const Row<std::size_t>& labels_t, Row<DataType>& labels_d) {

  labels_d = Row<DataType>(labels_t.n_elem);
  Row<std::size_t> uniqueVals = unique(labels_t);

  for (auto it=uniqueVals.begin(); it!=uniqueVals.end(); ++it) {

    uvec ind = find(labels_t == *it);
    double equiv = leavesMap_[*it];
    labels_d.elem(ind).fill(equiv);

  }    
}

template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::Classify_(const Mat<DataType>& dataset, Row<DataType>& labels) {

  Row<std::size_t> labels_t;

  // Implicit 
  // void Classify(const Mat<DataType>&, Row<std::size_t>&) called 
  // called on ClassifierType

  classifier_->Classify(dataset, labels_t);
  decode(labels_t, labels);
  
}

template<typename DataType, typename ClassifierType, typename... Args>
void
DiscreteClassifierBase<DataType, ClassifierType, Args...>::Classify_(Mat<DataType>&& dataset, Row<DataType>& labels) {

  Row<std::size_t> labels_t;
  
  classifier_->Classify(std::move(dataset), labels_t);
  decode(labels_t, labels);
}


#endif
