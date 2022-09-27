#ifndef __DECISION_TREE_HPP__
#define __DECISION_TREE_HPP__

#include <vector>

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>


using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

class ClassifierModel {
public:
  ClassifierModel() = default;
  virtual void predict() = 0;
  void init_() { predict(); }
};

class DecisionTreeModel : public ClassifierModel
{
public:
  DecisionTreeModel(mat dataset, 
		    Row<size_t> labels,
		    std::size_t minimumLeafSize=10,
		    size_t maximumDepth=100) : 
    dataset_{dataset}, 
    labels_{labels},
    model_{dataset, 
	labels, 
	minimumLeafSize=minimumLeafSize,
	maximumDepth=maximumDepth}
	
  { init_(); }
  
  void predict() override { model_.Classify(dataset_, predictions_); }

private:
  DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit,
	       AllDimensionSelect, true> model_;
  mat dataset_;
  Row<size_t> labels_;
  Row<size_t> predictions_;
};

class LeafOnlyTreeModel : public ClassifierModel {
public:
  LeafOnlyTreeModel(const Row<size_t>& leaves) : 
    leaves_{leaves},
    model_{leaves} {}

  void classify(mat, Row<size_t>);
  
private:
  class LeafOnlyTree {
  public:
    LeafOnlyTree(Row<size_t> leaves) : leaves_{leaves} {}
    void Classify(Row<size_t>& predictions) {
      predictions = leaves_; }
  private:
    Row<size_t> leaves_;
  };  
  void predict() override { model_.Classify(predictions_); }
private:
  Row<size_t> leaves_;
  Row<size_t> predictions_;
  LeafOnlyTree model_;
};

#endif
