#ifndef __GRADIENTBOOSTMULTICLASSIFIER_HPP__
#define __GRADIENTBOOSTMULTICLASSIFIER_HPP__

#include <vector>
#include <memory>
#include <utility>

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
			       ClassPair classValues) :
    GradientBoostClassifier<ClassifierType>(dataset, labels, context),
    classValues_{classValues},
    allVOne_{true}
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
			       ClassPair classValues) :
    GradientBoostClassifier<ClassifierType>(dataset, labels, context),
    classValues_{classValues},
    allVOne_{false}
  {}
  
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

private:
  std::size_t classValue_;
  ClassPair classValues_; 
  bool allVOne_;
};

template<typename ClassifierType>
class GradientBoostMultiClassifier : public ClassifierBase<typename classifier_traits<ClassifierType>::datatype,
							   typename classifier_traits<ClassifierType>::classifier> {
public:

  using DataType = typename classifier_traits<ClassifierType>::datatype;
  using IntegralLabelType = typename classifier_traits<ClassifierType>::integrallabeltype;
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
  void Predict(const mat&, Row<DataType>&);
  
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

  // overloaded versions of above based based on label datatype
  void Predict(Row<IntegralLabelType>&);
  void Predict(Row<IntegralLabelType>&, const uvec&);
  void Predict(const mat&, Row<IntegralLabelType>&);

private:
  void init_();

  std::size_t numClasses_;
  mat dataset_;
  mat dataset_oos_;
  Row<double> labels_;
  Row<double> labels_oos_;
  MultiClassifierContext::CombinedContext context_;
  ClassifierList classClassifiers_;

  bool hasOOSData_;
  bool allVOne_;

};

#include "gradientboostmulticlassifier_impl.hpp"

#endif
