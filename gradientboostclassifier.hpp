#ifndef __GRADIENTBOOSTCLASSIFIER_HPP__
#define __GRADIENTBOOSTCLASSIFIER_HPP__

#include <list>
#include <memory>
#include <vector>
#include <random>
#include <unordered_map>

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>

#include "dataset.hpp"
#include "decision_tree.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

template<typename DataType>
class ClassifierBase {
public:
  ClassifierBase() = default;
  ~ClassifierBase() = default;

  virtual void Classify_(const mat& dataset, Row<std::size_t>& labels) = 0;
};

template<typename DataType>
class DecisionTreeClassifier : public ClassifierBase<DataType> {
public:
  using Classifier = DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  
  DecisionTreeClassifier(const mat& dataset, Row<std::size_t>& labels) :
    classifier_{dataset, labels, 7, 10, 3} {}
    
  void Classify_(const mat& dataset, Row<std::size_t>& labels) {
    classifier_.Classify(dataset, labels);
  }
private:
  Classifier classifier_;
};

template<typename DataType>
class LeafOnlyClassifier : public ClassifierBase<DataType> {
public:
  LeafOnlyClassifier(Row<std::size_t> leaves) :
    leaves_{leaves} {}

  void Classify_(const mat& dataset, Row<std::size_t>& labels) {
    labels = leaves_;
  }
private:
  Row<std::size_t> leaves_;
};

template<typename DataType>
class GradientBoostClassifier {
public:

  using Partition = std::vector<int>;
  using PartitionList = std::vector<Partition>;
  using RegularizationList = std::vector<DataType>;
  // using ClassifierList = std::vector<std::unique_ptr<ClassifierBase<DataType>>>;
  using ClassifierList = std::vector<std::unique_ptr<DecisionTreeClassifier<DataType>>>;
  using Leaves = Row<double>;
  using LeavesValues = std::vector<Leaves>;
  using LeavesMap = std::unordered_map<double, int>;
  using Prediction = Row<std::size_t>;
  using PredictionList = std::vector<Prediction>;
  
  GradientBoostClassifier<DataType>(const mat& dataset, const Row<std::size_t>& labels, int steps) : 
    steps_{steps},
    dataset_{dataset},
    labels_{labels},
    current_classifier_ind_{0} { init_(); }
  std::pair<std::unique_ptr<ClassifierBase<DataType>>, Row<size_t>> predict(const Row<double>& labels);

private:
  void init_();
  LeavesMap relabel(const Row<double>&, Row<std::size_t>&);
  void fit();

  
  void setNextClassifier(const ClassifierBase<DataType>&);
  int steps_;
  mat dataset_;
  Row<std::size_t> labels_;

  Row<int> rowMask_;

  int current_classifier_ind_;

  LeavesValues leaves_;
  ClassifierList classifiers_;
  PartitionList partitions_;
  PredictionList predictions_;
  RegularizationList regularizations_;

};

#include "gradientboostclassifier_impl.hpp"

#endif
