#include <unordered_map>

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>


using namespace arma;
using namespace mlpack;

using row_d = Row<double>;
using row_t = Row<std::size_t>;

std::unordered_map<std::size_t, double> encode(const row_d& labels_d,
				       row_t& labels_t) {
  std::unordered_map<std::size_t, double> leavesMap;
  row_d uniqueVals = unique(labels_d);
  for (auto it=uniqueVals.begin(); it!=uniqueVals.end(); ++it) {
    uvec ind = find(labels_d == *(it));
    std::size_t equiv = std::distance(it, uniqueVals.end());
    labels_t.elem(ind).fill(equiv);
    leavesMap.insert(std::make_pair(static_cast<std::size_t>(equiv), (*it)));
  }
  return leavesMap;
}

void decode(const row_t& labels_t, 
	    row_d& labels_d, 
	    std::unordered_map<std::size_t, double>& leavesMap) {
  row_t uniqueVals = unique(labels_t);
  for (auto it=uniqueVals.begin(); it!=uniqueVals.end(); ++it) {
    uvec ind = find(labels_t == *(it));
    double equiv = leavesMap[*it];
    labels_d.elem(ind).fill(equiv);
  }
}

auto main() -> int {

  mat dataset = mat(10, 100, arma::fill::randu);
  row_d labels_d(100);

  for (size_t i=0; i<25; ++i) {
    dataset(3,i) = i;
    labels_d[i] = .214;
  }
  for (size_t i=25; i<50; ++i) {
    dataset(3,i) = i;
    labels_d[i] = 47.88;
  }
  for (size_t i=50; i<75; ++i) {
    dataset(3,i) = i;
    labels_d[i] = 2.1;
  }
  for (size_t i=75; i<100; ++i) {
    dataset(3,i) = i;
    labels_d[i] = 14.14;
  }

  /************************************************/
  /* DecisionTreeRegressor requires double labels */
  /************************************************/
  DecisionTreeRegressor<> reg(dataset, labels_d, 1, 0.0);
  /************************************************/
  /* DecisionTreeRegressor requires double labels */
  /************************************************/


  row_d prediction_d;
  reg.Predict(dataset, prediction_d);
  
  row_t labels_t(100);
  std::unordered_map<std::size_t, double> leavesMap = encode(labels_d, labels_t);

  /**********************************************/
  /* DecisionTree requires long unsigned labels */
  /**********************************************/
  DecisionTree<> cla(dataset, labels_t, 5, 1, 0.);
  /**********************************************/
  /* DecisionTree requires long unsigned labels */
  /**********************************************/

  
  row_t prediction_t;
  row_d prediction_t_to_d(100);
  cla.Classify(dataset, prediction_t);
  decode(prediction_t, prediction_t_to_d, leavesMap);

  for (std::size_t i=0; i<100; ++i) {
    std::cout << labels_d[i] << " : " << labels_t[i] << " :: " << prediction_t_to_d[i] << " : " << prediction_t[i] << std::endl;
  }

  return 0;
}
