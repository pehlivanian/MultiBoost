#ifndef __CLASSIFIER_TRAITS_HPP__
#define __CLASSIFIER_TRAITS_HPP__

#include <any>
#include <utility>

#include "classifiers.hpp"

template<typename T>
struct classifier_traits {
  using datatype = double;
  using integrallabeltype = std::size_t;
  using classifier = ClassifierTypes::DecisionTreeClassifierType;
  using classifierArgs = std::any;
};

template<>
struct classifier_traits<DecisionTreeClassifier> {
  using datatype = double;
  using integrallabeltype = std::size_t;
  using classifier = ClassifierTypes::DecisionTreeClassifierType;
  using classifierArgs = std::tuple<std::size_t, std::size_t, double, std::size_t>;
};

template<>
struct classifier_traits<DecisionTreeRegressorClassifier> {
  using datatype = double;
  using integrallabeltype = std::size_t;
  using classifier = ClassifierTypes::DecisionTreeRegressorType;
  using classifierArgs = std::tuple<std::size_t, double, std::size_t>;
};

template<>
struct classifier_traits<RandomForestClassifier> {
  using datatype = double;
  using integrallabeltype = std::size_t;
  using classifier = ClassifierTypes::RandomForestClassifierType;
  using classifierArgs = std::tuple<std::size_t, std::size_t, std::size_t>;
};


#endif
