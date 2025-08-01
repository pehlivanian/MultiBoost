#ifndef __GRADIENTBOOSTCLASSIFIER_HPP__
#define __GRADIENTBOOSTCLASSIFIER_HPP__

// #define DEBUG() __debug dd{__FILE__, __FUNCTION__, __LINE__};

#include <algorithm>
#include <cassert>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <random>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DP.hpp"
#include "classifier.hpp"
#include "classifier_loss.hpp"
#include "classifiers.hpp"
#include "compositeclassifier.hpp"
#include "model.hpp"
#include "model_traits.hpp"
#include "recursivemodel.hpp"
#include "score2.hpp"
#include "utils.hpp"

using namespace arma;
using namespace mlpack;

using namespace Objectives;
using namespace LossMeasures;
using namespace ModelContext;
using namespace PartitionSize;
using namespace LearningRate;

using namespace IB_utils;

template <typename ClassifierType>
class GradientBoostClassifier : public CompositeClassifier<ClassifierType> {
public:
  GradientBoostClassifier() = default;

  // Common constructor patterns - explicit to avoid template overhead
  GradientBoostClassifier(
      const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
      const Row<std::size_t>& labels,
      Context context,
      const std::string& folderName = std::string{})
      : CompositeClassifier<ClassifierType>(dataset, labels, context, folderName) {}

  GradientBoostClassifier(
      const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
      const Row<typename CompositeClassifier<ClassifierType>::DataType>& labels,
      Context context,
      const std::string& folderName = std::string{})
      : CompositeClassifier<ClassifierType>(dataset, labels, context, folderName) {}

  GradientBoostClassifier(
      const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset,
      const Row<std::size_t>& labels,
      const Mat<typename CompositeClassifier<ClassifierType>::DataType>& dataset_oos,
      const Row<std::size_t>& labels_oos,
      Context context,
      const std::string& folderName = std::string{})
      : CompositeClassifier<ClassifierType>(
            dataset, labels, dataset_oos, labels_oos, context, folderName) {}

  // Template fallback for other patterns
  template <typename... Args>
  explicit GradientBoostClassifier(Args&&... args)
      : CompositeClassifier<ClassifierType>(std::forward<Args>(args)...) {}
};

#include "gradientboostclassifier_impl.hpp"

using CompositeClassifierDTC = CompositeClassifier<DecisionTreeClassifier>;
using CompositeClassifierRFC = CompositeClassifier<RandomForestClassifier>;

using RecursiveModelDTCD = RecursiveModel<double, CompositeClassifierDTC>;
using RecursiveModelDTCF = RecursiveModel<float, CompositeClassifierDTC>;
using RecursiveModelRFCD = RecursiveModel<double, CompositeClassifierRFC>;
using RecursiveModelRFCF = RecursiveModel<float, CompositeClassifierRFC>;

/*
// Decorator using directives
using C1 = CompositeClassifier<DC5>;
*/

using GradientBoostClassifierDTC = GradientBoostClassifier<DecisionTreeClassifier>;
using GradientBoostClassifierRFC = GradientBoostClassifier<RandomForestClassifier>;

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

CEREAL_REGISTER_POLYMORPHIC_RELATION(RecursiveModelDTCD, CompositeClassifierDTC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(RecursiveModelDTCF, CompositeClassifierDTC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(RecursiveModelRFCD, CompositeClassifierRFC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(RecursiveModelRFCF, CompositeClassifierRFC);

/*
// Decorator polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, C1);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, C1);
*/

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, GradientBoostClassifierDTC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, GradientBoostClassifierDTC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, GradientBoostClassifierRFC)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, GradientBoostClassifierRFC);

CEREAL_REGISTER_POLYMORPHIC_RELATION(RecursiveModelDTCD, GradientBoostClassifierDTC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(RecursiveModelDTCF, GradientBoostClassifierDTC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(RecursiveModelRFCD, GradientBoostClassifierRFC)
CEREAL_REGISTER_POLYMORPHIC_RELATION(RecursiveModelRFCF, GradientBoostClassifierRFC);

/*
// Decorator polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, G1);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, G1);
*/

#endif
