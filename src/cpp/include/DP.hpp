#ifndef __DP_HPP__
#define __DP_HPP__

#include <list>
#include <algorithm>
#include <utility>
#include <vector>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <string>
#include <exception>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <math.h>

// No use for eigen lib
#undef EIGEN

#if EIGEN
#if (!IS_CXX_11) && !(__cplusplus == 201103L)
  #include <Eigen/Dense>
  using namespace Eigen;
#else
  #include "port_utils.hpp"
#endif
#endif

#include "utils.hpp"
#include "score2.hpp"

#define UNUSED(expr) do { (void)(expr); } while (0)

using namespace Objectives;
using namespace IB_utils;

template<typename DataType>
class DPSolver {
  using all_scores = std::pair<std::vector<std::vector<int> >, DataType>;
  using all_part_scores = std::vector<all_scores>;
public:

  DPSolver() = default;

  DPSolver(std::vector<DataType> a,
	   std::vector<DataType> b,
	   int T,
	   objective_fn parametric_dist=objective_fn::Gaussian,
	   bool risk_partitioning_objective=false,
	   bool use_rational_optimization=false,
	   DataType gamma=0.,
	   int reg_power=1.,
	   bool sweep_down=false,
	   bool find_optimal_t=false
	   ) :
    n_{static_cast<int>(a.size())},
    T_{T},
    a_{a},
    b_{b},
    optimal_score_{0.},
    parametric_dist_{parametric_dist},
    risk_partitioning_objective_{risk_partitioning_objective},
    use_rational_optimization_{use_rational_optimization},
    gamma_{gamma},
    reg_power_{reg_power},
    sweep_down_{sweep_down},
    find_optimal_t_{find_optimal_t},
    optimal_num_clusters_OLS_{0}    
  { _init(); }

  DPSolver(int n,
	   int T,
	   std::vector<DataType> a,
	   std::vector<DataType> b,
	   objective_fn parametric_dist=objective_fn::Gaussian,
	   bool risk_partitioning_objective=false,
	   bool use_rational_optimization=false,
	   DataType gamma=0.,
	   int reg_power=1.,
	   bool sweep_down=false,
	   bool find_optimal_t=false
	   ) :
    n_{n},
    T_{T},
    a_{a},
    b_{b},
    optimal_score_{0.},
    parametric_dist_{parametric_dist},
    risk_partitioning_objective_{risk_partitioning_objective},
    use_rational_optimization_{use_rational_optimization},
    gamma_{gamma},
    reg_power_{reg_power},
    sweep_down_{sweep_down},
    find_optimal_t_{find_optimal_t},
    optimal_num_clusters_OLS_{0}
  
  { _init(); }

  std::vector<std::vector<int> > get_optimal_subsets_extern() const;
  DataType get_optimal_score_extern() const;
  std::vector<DataType> get_score_by_subset_extern() const;
  all_part_scores get_all_subsets_and_scores_extern() const;
  int get_optimal_num_clusters_OLS_extern() const;
  void print_maxScore_();
  void print_nextStart_();
    
private:
  int n_;
  int T_;
  std::vector<DataType> a_;
  std::vector<DataType> b_;
  std::vector<std::vector<DataType> > maxScore_, maxScore_sec_;
  std::vector<std::vector<int> > nextStart_, nextStart_sec_;
  std::vector<int> priority_sortind_;
  DataType optimal_score_;
  std::vector<std::vector<int> > subsets_;
  std::vector<DataType> score_by_subset_;
  objective_fn parametric_dist_;
  bool risk_partitioning_objective_;
  bool use_rational_optimization_;
  DataType gamma_;
  int reg_power_;
  bool sweep_down_;
  bool find_optimal_t_;
  all_part_scores subsets_and_scores_;
  int optimal_num_clusters_OLS_;
  std::unique_ptr<ParametricContext<DataType>> context_;

  void create();
  void createContext();
  void create_multiple_clustering_case();
  all_scores optimize_for_fixed_S(int);
  void optimize();
  void optimize_multiple_clustering_case();
  void sort_by_priority(std::vector<DataType>&, std::vector<DataType>&);
  void reorder_subsets(std::vector<std::vector<int> >&, std::vector<DataType>&);
  DataType compute_score(int, int);
  DataType compute_ambient_score(DataType, DataType);
#if EIGEN
  void find_optimal_t();
#endif
  void _init() { 
    create();
    optimize();
  }
};

#include "DP_impl.hpp"

#endif
