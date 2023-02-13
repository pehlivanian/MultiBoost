#ifndef __GRADIENTBOOSTMULTICLASSIFIER_HPP__
#define __GRADIENTBOOSTMULTICLASSIFIER_HPP__

#include <vector>
#include <memory>
#include <algorithm>
#include <utility>

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

#include "threadpool.hpp"
#include "threadsafequeue.hpp"
#include "gradientboostclassifier.hpp"

namespace MultiClassifierContext {
  struct MultiContext {
    MultiContext(bool allVOne) :
      allVOne{allVOne} 
    {}
    MultiContext() :
      allVOne{true}
    {}
    bool allVOne;
  };

  struct CombinedContext : ClassifierContext::Context{
    CombinedContext(ClassifierContext::Context context, MultiContext overlay) : 
      context{context},
      allVOne{overlay.allVOne} 
    {}
    CombinedContext() : 
      ClassifierContext::Context{},
      allVOne{false} 
    {}
      
    ClassifierContext::Context context;
    bool allVOne;
      
  };
}


template<typename ClassifierType>
class GradientBoostClassClassifier : public GradientBoostClassifier<ClassifierType> {
public:

  using ClassPair = std::pair<std::size_t, std::size_t>;

  using DataType = typename classifier_traits<ClassifierType>::datatype;
  using IntegralLabelType = typename classifier_traits<ClassifierType>::integrallabeltype;
  using Classifier = typename classifier_traits<ClassifierType>::classifier;
  using ClassClassifier = GradientBoostClassClassifier<ClassifierType>;
  using ClassifierList = std::vector<std::unique_ptr<ClassClassifier>>;

  GradientBoostClassClassifier() = default;
  GradientBoostClassClassifier(const mat& dataset,
			       const Row<std::size_t>& labels,
			       ClassifierContext::Context context,
			       std::size_t classValue) :
    GradientBoostClassifier<ClassifierType>(dataset, labels, context),
    classValue_{classValue}
  {}
  
  GradientBoostClassClassifier(const mat& dataset,
			       const Row<std::size_t>& labels,
			       ClassifierContext::Context context,
			       ClassPair classValues,
			       std::size_t num1,
			       std::size_t num2) :
    GradientBoostClassifier<ClassifierType>(dataset, labels, context),
    classValues_{classValues},
    allVOne_{true},
    num1_{num1},
    num2_{num2}
  {}

  GradientBoostClassClassifier(const mat& dataset,
			       const Row<double>& labels,
			       ClassifierContext::Context context,
			       std::size_t classValue) :
    GradientBoostClassifier<ClassifierType>(dataset, labels, context),
    classValue_{classValue}
  {}
    
  GradientBoostClassClassifier(const mat& dataset,
			       const Row<double>& labels,
			       ClassifierContext::Context context,
			       ClassPair classValues,
			       std::size_t num1,
			       std::size_t num2) :
    GradientBoostClassifier<ClassifierType>(dataset, labels, context),
    classValues_{classValues},
    allVOne_{false},
    num1_{num1},
    num2_{num2}
  {}
  
  void Classify_(const mat&, Row<DataType>&) override;
  
  void info(const mat&);

  void printStats(int stepNum) override { 

    if (allVOne_) {
      std::cerr << "SUMMARY FOR CLASS: " << classValue_ << std::endl;
    }
    else {
      std::cerr << "SUMMARY FOR CLASS: (" << classValues_.first
		<< ", " << classValues_.second 
		<< ")" << std::endl;
    }
    GradientBoostClassifier<ClassifierType>::printStats(stepNum);
    
  }
  
  template<class Archive>
  void serialize(Archive &ar) {
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

template<typename ClassifierType>
class GradientBoostMultiClassifier : public ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
							   typename classifier_traits<ClassifierType>::classifier> {
public:

  using DataType = typename classifier_traits<ClassifierType>::datatype;
  using IntegralLabelType = typename classifier_traits<ClassifierType>::integrallabeltype;
  using Classifier = typename classifier_traits<ClassifierType>::classifier;
  using ClassClassifier = GradientBoostClassClassifier<ClassifierType>;
  using ClassifierList = std::vector<std::unique_ptr<ClassClassifier>>;

  GradientBoostMultiClassifier() = default;
  GradientBoostMultiClassifier(const mat& dataset,
			       const Row<std::size_t>& labels,
			       MultiClassifierContext::CombinedContext context) :
    dataset_{dataset},
    labels_{conv_to<Row<double>>::from(labels)},    
    allVOne_{context.allVOne},
    context_{context}
  {
    if (hasOOSData_ = context.hasOOSData) {
      dataset_oos_ = context.dataset_oos;
      labels_oos_ = conv_to<Row<double>>::from(context.labels_oos);
    }
    init_(); 
  }

  GradientBoostMultiClassifier(const mat& dataset,
			       const Row<double>& labels,
			       MultiClassifierContext::CombinedContext context) :
    dataset_{dataset},
    labels_{labels},
    allVOne_{context.allVOne},
    context_{context} 
  { 
    if (hasOOSData_ = context.hasOOSData) {
      dataset_oos_ = context.dataset_oos;
      labels_oos_ = conv_to<Row<double>>::from(context.labels_oos);
    }
    init_(); 
  }

  void fit();
  
  void Classify_(const mat&, Row<DataType>&) override;
  void purge() override;

  // 4 Predict methods
  // predict on member dataset; loop through and sum step prediction vectors
  void Predict(Row<DataType>&);
  // predict on subset of dataset defined by uvec; sum step prediction vectors
  void Predict(Row<DataType>&, const uvec&);
  // predict OOS, loop through and call Classify_ on individual classifiers, sum
  void Predict(const mat&, Row<DataType>&, bool=false);

  // overloaded versions for archive classifier
  // predict on member dataset from archive
  void Predict(std::string, Row<DataType>&, bool=false);
  // prediction OOS, loop through and call Classify_ on individual classifiers, sum
  void Predict(std::string, const mat&, Row<DataType>&, bool=false);

  void deSymmetrize(Row<DataType>&);

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this), CEREAL_NVP(classClassifiers_));
    ar(cereal::base_class<ClassifierBase<DataType, Classifier>>(this), allVOne_);
  }

private:
  void init_();

  std::size_t numClasses_;
  mat dataset_;
  mat dataset_oos_;
  Row<double> labels_;
  Row<double> labels_oos_;
  Row<std::size_t> uniqueVals_;
  MultiClassifierContext::CombinedContext context_;
  ClassifierList classClassifiers_;

  std::string indexName_;

  bool hasOOSData_;
  bool allVOne_;
  bool serialize_;
  bool symmetrized_;


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
