#ifndef __GRADIENTBOOSTCLASSIFIER_HPP__
#define __GRADIENTBOOSTCLASSIFIER_HPP__

// #define DEBUG() __debug dd{__FILE__, __FUNCTION__, __LINE__};

#include <list>
#include <utility>
#include <memory>
#include <random>
#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <type_traits>
#include <cassert>
#include <typeinfo>
#include <chrono>
#include <limits>
#include <exception>

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/access.hpp>

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

#include "loss.hpp"
#include "score2.hpp"
#include "DP.hpp"
#include "utils.hpp"
#include "model.hpp"
#include "classifier.hpp"
#include "compositeclassifier.hpp"
#include "model_traits.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

using namespace Objectives;
using namespace LossMeasures;
using namespace ModelContext;
using namespace PartitionSize;
using namespace LearningRate;

using namespace IB_utils;

template<typename ClassifierType>
class GradientBoostClassifier : public CompositeClassifier<ClassifierType> {
public:

  // 1
  // mat	: arma::Mat<double>
  // labels	: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // context	: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset, 
			  const Row<std::size_t>& labels,
			  Context context) :
    CompositeClassifier<ClassifierType>(dataset, labels, context) {}

  // 2
  // mat	: arma::Mat<double>
  // labels	: arma::Row<double>
  // context	: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  Context context) :
    CompositeClassifier<ClassifierType>(dataset, labels, context) {}
  
  // 3
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<std::size_t> <- CONVERTED TO Row<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const mat& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  Context context) :
    CompositeClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, context) {}

  // 4
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  Context context) :
    CompositeClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, context) {}

  // 5
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // colMask		: uvec
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context context) :
    CompositeClassifier<ClassifierType>(dataset, labels, latestPrediction, colMask, context) {}

  // 6
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const Row<double>& latestPrediction,
			  Context context) :
    CompositeClassifier<ClassifierType>(dataset, labels, latestPrediction, context) {}

  // 7
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context context) :
    CompositeClassifier<ClassifierType>(dataset, labels, latestPrediction, colMask, context) {}

  // 8
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const Row<double>& latestPrediction,
			  Context context) :
    CompositeClassifier<ClassifierType>(dataset, labels, latestPrediction, context) {}

  // 9
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<std::size_t> <- CONVERTED TO Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const mat& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context context) :
    CompositeClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, latestPrediction, colMask) {}

  // 10
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<std::size_t> <- CONVERTED TO Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<std::size_t>& labels,
			  const mat& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  const Row<double>& latestPrediction,
			  Context context) :
    CompositeClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, latestPrediction, context) {}

  // 11
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context context) :
    CompositeClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, latestPrediction, colMask) {}

  // 12
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  const Row<double>& latestPrediction,
			  Context context) :
    CompositeClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, latestPrediction, context) {}

};

/*
  using DTC = ClassifierTypes::DecisionTreeClassifierType;
  using CTC = ClassifierTypes::DecisionTreeRegressorType;
  using RFC = ClassifierTypes::RandomForestClassifierType;

  using DiscreteClassifierBaseDTC = DiscreteClassifierBase<double, 
  DTC, 
  std::size_t,
  std::size_t,
  double,
  std::size_t>;
  using DiscreteClassifierBaseRFC = DiscreteClassifierBase<double,
  RFC,
  std::size_t,
  std::size_t,
  std::size_t>;
  using ContinuousClassifierBaseD = ContinuousClassifierBase<double, 
  CTC,
  unsigned long,
  double,
  unsigned long>;
  using ClassifierBaseDD = ClassifierBase<double, DTC>;
  using ClassifierBaseRD = ClassifierBase<double, RFC>;
  using ClassifierBaseCD = ClassifierBase<double, CTC>;

  using DecisionTreeRegressorClassifierBaseLDL = DecisionTreeRegressorClassifier<unsigned long, double, unsigned long>;
  using DecisionTreeClassifierBaseLLDL = DecisionTreeClassifier<unsigned long, unsigned long, double, unsigned long>;
  using RandomForestClassifierBaseLLL = RandomForestClassifier<unsigned long, unsigned long, unsigned long>;

  using GradientBoostClassifierDTC = GradientBoostClassifier<DecisionTreeClassifier>);
  using GradientBoostClassifierRFC = GradientBoostClassifier<RandomForestClassifier>);
  using GradientBoostClassifierCTC = GradientBoostClassifier<DecisionTreeRegressorClassifier>);

  // Register class with cereal
  CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTC);
  CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFC);
  CEREAL_REGISTER_TYPE(ContinuousClassifierBaseD);

  CEREAL_REGISTER_TYPE(DecisionTreeClassifierBaseLLDL);
  CEREAL_REGISTER_TYPE(RandomForestClassifierBaseLLL);
  CEREAL_REGISTER_TYPE(DecisionTreeRegressorClassifierBaseLDL);

  CEREAL_REGISTER_TYPE(GradientBoostClassifierDTC);
  CEREAL_REGISTER_TYPE(GradientBoostClassifierRFC);
  CEREAL_REGISTER_TYPE(GradientBoostClassifierCTC);


  // Register class with cereal
  CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTC);
  CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFC);
  CEREAL_REGISTER_TYPE(ContinuousClassifierBaseD);

  CEREAL_REGISTER_TYPE(DecisionTreeClassifierLLDL);
  CEREAL_REGISTER_TYPE(RandomForestClassifierLLL);
  CEREAL_REGISTER_TYPE(DecisionTreeRegressorClassifierLDL);

  CEREAL_REGISTER_TYPE(GradientBoostClassifier<DecisionTreeClassifierLLDL>);
  CEREAL_REGISTER_TYPE(GradientBoostClassifier<DecisionTreeRegressorClassifierLDL>);

  // Register class hierarchy with cereal
  CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDD, DecisionTreeClassifierLLDL);
  CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRD, RandomForestClassifierLLL);
  CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseCD, DecisionTreeRegressorClassifierLDL);

  CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDD, DiscreteClassifierBaseDTC);
  CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRD, DiscreteClassifierBaseRFC);
  CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseCD, ContinuousClassifierBaseD);

  template<typename DataType, typename ClassifierType, typename... Args>
  class ContinuousClassifierBase;

  template<typename DataType, typename ClassifierType, typename... Args>
  class DiscreteClassifierBase;

  template<typename DataType>
  using LeavesMap = std::unordered_map<std::size_t, DataType>;

  namespace cereal {
  
  template<typename DataType>
  using LeavesMap = std::unordered_map<std::size_t, DataType>;

  template<typename DataType, typename ClassifierType, typename... Args> 
  struct LoadAndConstruct<ContinuousClassifierBase<DataType, ClassifierType, Args...>> {
  template<class Archive>
  static void load_anod_construct(Archive &ar, cereal::construct<ContinuousClassifierBase<DataType, ClassifierType, Args...>> &construct) {
  std::unique_ptr<ClassifierType> classifier;
  ar(CEREAL_NVP(classifier));
  construct(std::move(classifier));
  }
  };


  template<typename DataType, typename ClassifierType, typename... Args>
  struct LoadAndConstruct<DiscreteClassifierBase<DataType, ClassifierType, Args...>> {
  template<class Archive>
  static void load_and_construct(Archive &ar, cereal::construct<DiscreteClassifierBase<DataType, ClassifierType, Args...>> &construct) {
  LeavesMap<DataType> leavesMap;
  std::unique_ptr<ClassifierType> classifier;
  ar(CEREAL_NVP(leavesMap));
  ar(CEREAL_NVP(classifier));
  construct(leavesMap, std::move(classifier));
  }
  };

  } // namespace cereal
*/  

#include "gradientboostclassifier_impl.hpp"

#endif
