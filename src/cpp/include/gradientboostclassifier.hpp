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
#include "classifiers.hpp"
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

  GradientBoostClassifier() = default;

  // 1
  // mat	: arma::Mat<double>
  // labels	: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // context	: ClassifierContext::Context
  GradientBoostClassifier(const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset, 
			  const Row<std::size_t>& labels,
			  Context context,
			  const std::string& folderName=std::string{}) :
    CompositeClassifier<ClassifierType>(dataset, labels, context, folderName) {}

  // 2
  // mat	: arma::Mat<double>
  // labels	: arma::Row<double>
  // context	: ClassifierContext::Context
  GradientBoostClassifier(const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& labels,
			  Context context,
			  const std::string& folderName=std::string{}) :
    CompositeClassifier<ClassifierType>(dataset, labels, context, folderName) {}
  
  // 3
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<std::size_t> <- CONVERTED TO Row<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
			  const Row<std::size_t>& labels,
			  const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  Context context,
			  const std::string& folderName=std::string{}) :
    CompositeClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, context, folderName) {}

  // 4
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& labels,
			  const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset_oos,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& labels_oos,
			  Context context,
			  const std::string& folderName=std::string{}) :
    CompositeClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, context, folderName) {}

  // 5
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // colMask		: uvec
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
			  const Row<std::size_t>& labels,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& latestPrediction,
			  const uvec& colMask,
			  Context context,
			  const std::string& folderName=std::string{}) :
    CompositeClassifier<ClassifierType>(dataset, labels, latestPrediction, colMask, context, folderName) {}

  // 6
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
			  const Row<std::size_t>& labels,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& latestPrediction,
			  Context context,
			  const std::string& folderName=std::string{}) :
    CompositeClassifier<ClassifierType>(dataset, labels, latestPrediction, context, folderName) {}

  // 7
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& labels,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& latestPrediction,
			  const uvec& colMask,
			  Context context,
			  const std::string& folderName=std::string{}) :
    CompositeClassifier<ClassifierType>(dataset, labels, latestPrediction, colMask, context, folderName) {}

  // 8
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& labels,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& latestPrediction,
			  Context context,
			  const std::string& folderName=std::string{}) :
    CompositeClassifier<ClassifierType>(dataset, labels, latestPrediction, context, folderName) {}

  // 9
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<std::size_t> <- CONVERTED TO Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
			  const Row<std::size_t>& labels,
			  const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& latestPrediction,
			  const uvec& colMask,
			  Context context,
			  const std::string& folderName=std::string{}) :
    CompositeClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, latestPrediction, colMask, context, folderName) {}

  // 10
  // mat		: arma::Mat<double>
  // labels		: arma::Row<std::size_t> <- CONVERTED TO Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<std::size_t> <- CONVERTED TO Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
			  const Row<std::size_t>& labels,
			  const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset_oos,
			  const Row<std::size_t>& labels_oos,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& latestPrediction,
			  Context context,
			  const std::string& folderName=std::string{}) :
    CompositeClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, latestPrediction, context, folderName) {}

  // 11
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& labels,
			  const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset_oos,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& labels_oos,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& latestPrediction,
			  const uvec& colMask,
			  Context context,
			  const std::string& folderName=std::string{}) :
    CompositeClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, latestPrediction, colMask, context, folderName) {}

  // 12
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ClassifierContext::Context
  GradientBoostClassifier(const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& labels,
			  const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset_oos,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& labels_oos,
			  const Row<typename CompositeClassifier<ClassifierType>::DataType>& latestPrediction,
			  Context context,
			  const std::string& folderName=std::string{}) :
    CompositeClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, latestPrediction, context, folderName) {}

};

#include "gradientboostclassifier_impl.hpp"

using CompositeClassifierDTC  = CompositeClassifier<DecisionTreeClassifier>;
using CompositeClassifierRFC  = CompositeClassifier<RandomForestClassifier>;

/*
// Decorator using directives
using C1 = CompositeClassifier<DC5>;
*/

using GradientBoostClassifierDTC  = GradientBoostClassifier<DecisionTreeClassifier>;
using GradientBoostClassifierRFC  = GradientBoostClassifier<RandomForestClassifier>;

/*
// Decorator using directives
using G1 = GradientBoostClassifier<DC5>;
*/

CEREAL_REGISTER_TYPE(GradientBoostClassifierDTC);
CEREAL_REGISTER_TYPE(GradientBoostClassifierRFC);

/*
// Decorator registrations
CEREAL_REGISTER_TYPE(G1);
*/

CEREAL_REGISTER_TYPE(CompositeClassifierDTC);
CEREAL_REGISTER_TYPE(CompositeClassifierRFC);

/*
// Decorator registrations
CEREAL_REGISTER_TYPE(C1);
*/

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, CompositeClassifierDTC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, CompositeClassifierDTC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, CompositeClassifierRFC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, CompositeClassifierRFC);

/*
// Decorator polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, C1);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, C1);
*/

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, GradientBoostClassifierDTC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, GradientBoostClassifierDTC); 
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, GradientBoostClassifierRFC)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, GradientBoostClassifierRFC);

/*
// Decorator polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, G1);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, G1);
*/


#endif
