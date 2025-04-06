#ifndef __GRADIENTBOOSTMULTICLASSIFIER_HPP__
#define __GRADIENTBOOSTMULTICLASSIFIER_HPP__

// #define DEBUG() __debug dd{__FILE__, __FUNCTION__, __LINE__};
#define DEBUG() ;

#include <algorithm>
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
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "gradientboostclassifier.hpp"
#include "threadpool.hpp"
#include "threadsafequeue.hpp"
#include "utils.hpp"

using namespace ClassifierContext;

namespace MultiClassifierContext {
struct MultiContext {
  MultiContext(bool allVOne, std::size_t steps, std::size_t partitionSize, bool serializeModel)
      : allVOne{allVOne}, steps{steps}, serializeModel{serializeModel} {}
  MultiContext() : allVOne{true} {}
  bool allVOne;
  std::size_t steps;
  bool serializeModel;
};

template <typename ClassifierType>
struct CombinedContext : Context<ClassifierType> {
  CombinedContext(Context<ClassifierType> context, MultiContext overlay)
      : context{context},
        allVOne{overlay.allVOne},
        steps{overlay.steps},
        serializeModel{overlay.serializeModel} {}
  CombinedContext() : Context<ClassifierType>{}, allVOne{false} {}

  Context<ClassifierType> context;
  bool allVOne;
  bool serializeModel;
  std::size_t steps;
};
}  // namespace MultiClassifierContext

template <typename ClassifierType>
class GradientBoostClassClassifier : public GradientBoostClassifier<ClassifierType> {
public:
  using ClassPair = std::pair<std::size_t, std::size_t>;

  using DataType = typename classifier_traits<ClassifierType>::datatype;
  using IntegralLabelType = typename classifier_traits<ClassifierType>::integrallabeltype;
  using Classifier = typename classifier_traits<ClassifierType>::classifier;
  using ClassClassifier = GradientBoostClassClassifier<ClassifierType>;
  using ClassifierList = std::vector<std::unique_ptr<ClassClassifier>>;

  GradientBoostClassClassifier() = default;

  GradientBoostClassClassifier(
      const mat& dataset,
      const Row<std::size_t>& labels,
      Context<ClassifierType> context,
      std::size_t classValue)
      : GradientBoostClassifier<ClassifierType>(dataset, labels, context),
        classValue_{classValue} {}

  GradientBoostClassClassifier(
      const mat& dataset,
      const Row<std::size_t>& labels,
      const mat& dataset_oos,
      const Row<std::size_t>& labels_oos,
      Context<ClassifierType> context,
      std::size_t classValue)
      : GradientBoostClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, context),
        classValue_{classValue} {}

  GradientBoostClassClassifier(
      const mat& dataset,
      const Row<std::size_t>& labels,
      Context<ClassifierType> context,
      ClassPair classValues,
      std::size_t num1,
      std::size_t num2)
      : GradientBoostClassifier<ClassifierType>(dataset, labels, context),
        classValues_{classValues},
        allVOne_{true},
        num1_{num1},
        num2_{num2} {}

  GradientBoostClassClassifier(
      const mat& dataset,
      const Row<std::size_t>& labels,
      const mat& dataset_oos,
      const Row<std::size_t>& labels_oos,
      Context<ClassifierType> context,
      ClassPair classValues,
      std::size_t num1,
      std::size_t num2)
      : GradientBoostClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, context),
        classValues_{classValues},
        allVOne_{true},
        num1_{num1},
        num2_{num2} {}

  GradientBoostClassClassifier(
      const mat& dataset,
      const Row<double>& labels,
      Context<ClassifierType> context,
      std::size_t classValue)
      : GradientBoostClassifier<ClassifierType>(dataset, labels, context),
        classValue_{classValue} {}

  GradientBoostClassClassifier(
      const mat& dataset,
      const Row<double>& labels,
      const mat& dataset_oos,
      const Row<double>& labels_oos,
      Context<ClassifierType> context,
      std::size_t classValue)
      : GradientBoostClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, context),
        classValue_{classValue} {}

  GradientBoostClassClassifier(
      const mat& dataset,
      const Row<double>& labels,
      Context<ClassifierType> context,
      ClassPair classValues,
      std::size_t num1,
      std::size_t num2)
      : GradientBoostClassifier<ClassifierType>(dataset, labels, context),
        classValues_{classValues},
        allVOne_{false},
        num1_{num1},
        num2_{num2} {}

  GradientBoostClassClassifier(
      const mat& dataset,
      const Row<double>& labels,
      const mat& dataset_oos,
      const Row<double>& labels_oos,
      Context<ClassifierType> context,
      ClassPair classValues,
      std::size_t num1,
      std::size_t num2)
      : GradientBoostClassifier<ClassifierType>(dataset, labels, dataset_oos, labels_oos, context),
        classValues_{classValues},
        allVOne_{false},
        num1_{num1},
        num2_{num2} {}

  void Classify_(const mat&, Row<DataType>&) override;

  void info(const mat&);

  void printStats(int stepNum) override {
    DEBUG()

    if (allVOne_) {
      std::cerr << "SUMMARY FOR CLASS: " << classValue_ << std::endl;
    } else {
      std::cerr << "SUMMARY FOR CLASS: (" << classValues_.first << ", " << classValues_.second
                << ")" << std::endl;
    }

    GradientBoostClassifier<ClassifierType>::printStats(stepNum);
  }

  template <class Archive>
  void serialize(Archive& ar) {
    DEBUG()

    ar(cereal::base_class<GradientBoostClassifier<ClassifierType>>(this), classValue_);
    ar(cereal::base_class<GradientBoostClassifier<ClassifierType>>(this), classValues_);
    ar(cereal::base_class<GradientBoostClassifier<ClassifierType>>(this), allVOne_);
    ar(cereal::base_class<GradientBoostClassifier<ClassifierType>>(this), num1_);
    ar(cereal::base_class<GradientBoostClassifier<ClassifierType>>(this), num2_);
  }

private:
  std::size_t classValue_;
  ClassPair classValues_;
  bool allVOne_;

  std::size_t num1_;
  std::size_t num2_;
};

template <typename ClassifierType>
class GradientBoostMultiClassifier : public ClassifierBase<
                                         typename classifier_traits<ClassifierType>::datatype,
                                         typename classifier_traits<ClassifierType>::classifier> {
public:
  using DataType = typename classifier_traits<ClassifierType>::datatype;
  using IntegralLabelType = typename classifier_traits<ClassifierType>::integrallabeltype;
  using Classifier = typename classifier_traits<ClassifierType>::classifier;
  using ClassClassifier = GradientBoostClassClassifier<ClassifierType>;
  using ClassifierList = std::vector<std::unique_ptr<ClassClassifier>>;

  GradientBoostMultiClassifier() = default;

  GradientBoostMultiClassifier(
      const mat& dataset,
      const Row<std::size_t>& labels,
      MultiClassifierContext::CombinedContext<ClassifierType> context)
      : dataset_{dataset},
        labels_{conv_to<Row<double>>::from(labels)},
        hasOOSData_{false},
        steps_{context.steps},
        allVOne_{context.allVOne},
        serializeModel_{context.serializeModel},
        context_{context} {
    contextInit_(std::move(context));
    init_();
  }

  GradientBoostMultiClassifier(
      const mat& dataset,
      const Row<double>& labels,
      MultiClassifierContext::CombinedContext<ClassifierType> context)
      : dataset_{dataset},
        labels_{labels},
        hasOOSData_{false},
        steps_{context.steps},
        allVOne_{context.allVOne},
        serializeModel_{context.serializeModel},
        context_{context} {
    contextInit_(std::move(context));
    init_();
  }

  GradientBoostMultiClassifier(
      const mat& dataset,
      const Row<std::size_t>& labels,
      const mat& dataset_oos,
      const Row<std::size_t>& labels_oos,
      MultiClassifierContext::CombinedContext<ClassifierType> context)
      : dataset_{dataset},
        labels_{conv_to<Row<double>>::from(labels)},
        dataset_oos_{dataset_oos},
        labels_oos_{conv_to<Row<double>>::from(labels_oos)},
        hasOOSData_{true},
        steps_{context.steps},
        allVOne_{context.allVOne},
        serializeModel_{context.serializeModel},
        context_{context} {
    contextInit_(std::move(context));
    init_();
  }

  GradientBoostMultiClassifier(
      const mat& dataset,
      const Row<double>& labels,
      const mat& dataset_oos,
      const Row<double>& labels_oos,
      MultiClassifierContext::CombinedContext<ClassifierType> context)
      : dataset_{dataset},
        labels_{labels},
        dataset_oos_{dataset_oos},
        labels_oos_{labels_oos},
        hasOOSData_{true},
        steps_{context.steps},
        allVOne_{context.allVOne},
        serializeModel_{context.serializeModel},
        context_{context} {
    contextInit_(std::move(context));
    init_();
  }

  void fit();

  void Classify_(const mat&, Row<DataType>&) override;
  void purge() override;

  void printStats(int);

  // 4 Predict methods
  // predict on member dataset; loop through and sum step prediction vectors
  void Predict(Row<DataType>&);
  // predict on subset of dataset defined by uvec; sum step prediction vectors
  void Predict(Row<DataType>&, const uvec&);
  // predict OOS, loop through and call Classify_ on individual classifiers, sum
  void Predict(const mat&, Row<DataType>&, bool = false);

  // overloaded versions of above based based on label datatype
  virtual void Predict(Row<IntegralLabelType>&);
  virtual void Predict(Row<IntegralLabelType>&, const uvec&);
  virtual void Predict(const mat&, Row<IntegralLabelType>&);

  // overloaded versions for archive classifier
  void Predict(std::string, Row<DataType>&, bool = false);
  void Predict(std::string, const mat&, Row<DataType>&, bool = false);

  void deSymmetrize(Row<DataType>&);
  void commit();

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this),
       CEREAL_NVP(classClassifiers_));
    ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this), allVOne_);
  }

private:
  void init_();
  void contextInit_(MultiClassifierContext::CombinedContext<ClassifierType>&&);
  void fit_step(int stepNum);

  std::size_t numClasses_;
  mat dataset_;
  mat dataset_oos_;
  Row<double> labels_;
  Row<double> labels_oos_;
  Row<std::size_t> uniqueVals_;
  MultiClassifierContext::CombinedContext<ClassifierType> context_;
  ClassifierList classClassifiers_;

  std::string indexName_;

  bool hasOOSData_;
  bool allVOne_;
  bool serializeModel_;
  bool symmetrized_;

  std::size_t steps_;
};

using DTC = ClassifierTypes::DecisionTreeClassifierType;
using CTC = ClassifierTypes::DecisionTreeRegressorType;
using RFC = ClassifierTypes::RandomForestClassifierType;

using ClassifierBaseDD = ClassifierBase<double, DTC>;
using GradientBoostClassClassifierD = GradientBoostClassClassifier<DTC>;
using GradientBoostClassifierD = GradientBoostClassifier<DTC>;
using GradientBoostMultiClassifierD = GradientBoostMultiClassifier<DTC>;

// Register each class with cereal
CEREAL_REGISTER_TYPE(GradientBoostClassClassifierD);
CEREAL_REGISTER_TYPE(GradientBoostMultiClassifierD);

CEREAL_REGISTER_POLYMORPHIC_RELATION(GradientBoostClassifierD, GradientBoostClassClassifierD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDD, GradientBoostMultiClassifierD);

#include "gradientboostmulticlassifier_impl.hpp"

#endif
