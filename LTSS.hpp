#ifndef __LTSS_HPP__
#define __LTSS_HPP__

#include <list>
#include <utility>
#include <vector>
#include <limits>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <memory>

#include "utils.hpp"
#include "port_utils.hpp"
#include "score2.hpp"

using namespace Utils;
using namespace Objectives;

template<typename DataType>
class LTSSSolver {
public:
  LTSSSolver(std::vector<DataType> a,
	     std::vector<DataType> b,
	     objective_fn parametric_dist=objective_fn::Gaussian
	     ) :
    n_{static_cast<int>(a.size())},
    a_{a},
    b_{b},
    parametric_dist_{parametric_dist}
  { _init(); }
	     
  LTSSSolver(int n,
	     std::vector<DataType> a,
	     std::vector<DataType> b,
	     objective_fn parametric_dist=objective_fn::Gaussian
	     ) :
    n_{n},
    a_{a},
    b_{b},
    parametric_dist_{parametric_dist}
  { _init(); }

  std::vector<int> priority_sortind_;
  std::vector<int> get_optimal_subset_extern() const;
  DataType get_optimal_score_extern() const;

private:
  int n_;
  std::vector<DataType> a_;
  std::vector<DataType> b_;
  DataType optimal_score_;
  std::vector<int> subset_;
  objective_fn parametric_dist_;
  std::unique_ptr<ParametricContext<DataType>> context_;

  void _init() { create(); optimize(); }
  void create();
  void createContext();
  void optimize();

  void sort_by_priority(std::vector<DataType>&, std::vector<DataType>&);
  DataType compute_score(int, int);
};

#include "LTSS_impl.hpp"

#endif
