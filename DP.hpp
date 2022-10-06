#ifndef __DP_HPP__
#define __DP_HPP__

#include <list>
#include <utility>
#include <vector>
#include <iostream>
#include <iterator>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <math.h>

#if (!IS_CXX_11) && !(__cplusplus == 201103L)
  #include <Eigen/Dense>
  using namespace Eigen;
#else
  #include "port_utils.hpp"
#endif

#include "score.hpp"
#include "LTSS.hpp"

#define UNUSED(expr) do { (void)(expr); } while (0)

using namespace Objectives;

class DPSolver {
  using all_scores = std::pair<std::vector<std::vector<int> >, float>;
  using all_part_scores = std::vector<all_scores>;
public:
  DPSolver(std::vector<float> a,
	   std::vector<float> b,
	   int T,
	   objective_fn parametric_dist=objective_fn::Gaussian,
	   bool risk_partitioning_objective=false,
	   bool use_rational_optimization=false,
	   float gamma=0.,
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
	   std::vector<float> a,
	   std::vector<float> b,
	   objective_fn parametric_dist=objective_fn::Gaussian,
	   bool risk_partitioning_objective=false,
	   bool use_rational_optimization=false,
	   float gamma=0.,
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
  float get_optimal_score_extern() const;
  std::vector<float> get_score_by_subset_extern() const;
  all_part_scores get_all_subsets_and_scores_extern() const;
  int get_optimal_num_clusters_OLS_extern() const;
  void print_maxScore_();
  void print_nextStart_();
    
private:
  int n_;
  int T_;
  std::vector<float> a_;
  std::vector<float> b_;
  std::vector<std::vector<float> > maxScore_, maxScore_sec_;
  std::vector<std::vector<int> > nextStart_, nextStart_sec_;
  std::vector<int> priority_sortind_;
  float optimal_score_;
  std::vector<std::vector<int> > subsets_;
  std::vector<float> score_by_subset_;
  objective_fn parametric_dist_;
  bool risk_partitioning_objective_;
  bool use_rational_optimization_;
  float gamma_;
  int reg_power_;
  bool sweep_down_;
  bool find_optimal_t_;
  all_part_scores subsets_and_scores_;
  int optimal_num_clusters_OLS_;
  std::unique_ptr<ParametricContext> context_;
  std::unique_ptr<LTSSSolver> LTSSSolver_;
  
  void create();
  void createContext();
  void create_multiple_clustering_case();
  all_scores optimize_for_fixed_S(int);
  void optimize();
  void optimize_multiple_clustering_case();
  void sort_by_priority(std::vector<float>&, std::vector<float>&);
  void reorder_subsets(std::vector<std::vector<int> >&, std::vector<float>&);
  float compute_score(int, int);
  float compute_ambient_score(float, float);
  void find_optimal_t();
  void _init() { 
    create();
    optimize();
  }
};


#endif
