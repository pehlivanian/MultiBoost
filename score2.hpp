#ifndef __SCORE2_HPP__
#define __SCORE2_HPP__

#include <vector>
#include <list>
#include <limits>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <exception>
#include <immintrin.h>
#include <string.h>
#include <thread>


#define UNUSED(expr) do { (void)(expr); } while (0)

namespace Objectives {
  enum class objective_fn { Gaussian = 0, 
			    Poisson = 1, 
			    RationalScore = 2 };

  struct optimizationFlagException : public std::exception {
   const char* what() const throw () {
    return "Optimized version not implemented";
  };
 };


  template<typename DataType>
  class ParametricContext {

  public:
    ParametricContext(std::vector<DataType> a, 
		      std::vector<DataType> b, 
		      int n, 
		      bool risk_partitioning_objective,
		      bool use_rational_optimization,
		      std::string name
		      ) :
      a_{a},
      b_{b},
      n_{n},
      risk_partitioning_objective_{risk_partitioning_objective},
      use_rational_optimization_{use_rational_optimization},
      name_{name}

    { init_(); }

    virtual ~ParametricContext() = default;

    void init_();
    virtual void compute_partial_sums_AVX256();
    virtual void compute_partial_sums_parallel();
    virtual void compute_scores_parallel();
    virtual void compute_scores();

    virtual DataType compute_score_multclust(int, int) = 0;
    virtual DataType compute_score_multclust_optimized(int, int) = 0;
    virtual DataType compute_score_riskpart(int, int) = 0;
    virtual DataType compute_score_riskpart_optimized(int, int) = 0;

    virtual DataType compute_ambient_score_multclust(DataType, DataType) = 0;
    virtual DataType compute_ambient_score_riskpart(DataType, DataType) = 0;

    std::string getName() const { return name_; }
    bool getRiskPartitioningObjective() const { return risk_partitioning_objective_; }
    bool getUseRationalOptimization() const { return use_rational_optimization_; }

    DataType compute_score(int, int);
    DataType compute_ambient_score(DataType, DataType);

    std::vector<std::vector<DataType>> get_partial_sums_a() const { return a_sums_; }
    std::vector<std::vector<DataType>> get_partial_sums_b() const { return b_sums_; }

    std::vector<std::vector<DataType>> get_scores() const { return partialSums_; }
    DataType get_score(int i, int j) const { return partialSums_[i][j]; }

  protected:
    std::vector<DataType> a_;
    std::vector<DataType> b_;
    std::vector<std::vector<DataType>> partialSums_;
    int n_;
    std::vector<std::vector<DataType> > a_sums_;
    std::vector<std::vector<DataType> > b_sums_;
    bool risk_partitioning_objective_;
    bool use_rational_optimization_;
    std::string name_;

  };
  
  template<typename DataType>
  class PoissonContext : public ParametricContext<DataType> {
  public:
    PoissonContext(std::vector<DataType> a, 
		   std::vector<DataType> b, 
		   int n, 
		   bool risk_partitioning_objective,
		   bool use_rational_optimization) : ParametricContext<DataType>(a,
										 b,
										 n,
										 risk_partitioning_objective,
										 use_rational_optimization,
										 "Poisson")
    {}
  
    DataType compute_score_multclust(int, int) override;
    DataType compute_score_riskpart(int, int) override;
    DataType compute_ambient_score_multclust(DataType, DataType) override;
    DataType compute_ambient_score_riskpart(DataType, DataType) override;
    DataType compute_score_riskpart_optimized(int, int) override;
    DataType compute_score_multclust_optimized(int, int) override;

  private:
    std::vector<DataType> a_;
    std::vector<DataType> b_;
    std::vector<std::vector<DataType>> partialSums_;
    int n_;
    std::vector<std::vector<DataType> > a_sums_;
    std::vector<std::vector<DataType> > b_sums_;

  };

  template<typename DataType>
  class GaussianContext : public ParametricContext<DataType> {
  public:
    GaussianContext(std::vector<DataType> a, 
		    std::vector<DataType> b, 
		    int n, 
		    bool risk_partitioning_objective,
		    bool use_rational_optimization) : ParametricContext<DataType>(a,
									b,
									n,
									risk_partitioning_objective,
									use_rational_optimization,
									"Gaussian")
    {}
  
    DataType compute_score_multclust(int, int) override;
    DataType compute_score_riskpart(int, int) override;
    DataType compute_ambient_score_multclust(DataType, DataType) override;
    DataType compute_ambient_score_riskpart(DataType, DataType) override;
    DataType compute_score_multclust_optimized(int, int) override;
    DataType compute_score_riskpart_optimized(int, int) override;

  private:
    std::vector<DataType> a_;
    std::vector<DataType> b_;
    std::vector<std::vector<DataType>> partialSums_;
    int n_;
    std::vector<std::vector<DataType> > a_sums_;
    std::vector<std::vector<DataType> > b_sums_;

  };

  template<typename DataType>
  class RationalScoreContext : public ParametricContext<DataType> {
    // This class doesn't correspond to any regular exponential family,
    // it is used to define ambient functions on the partition polytope
    // for targeted applications - quadratic approximations to loss, for
    // XGBoost, e.g.
  private:
    std::vector<DataType> a_;
    std::vector<DataType> b_;
    std::vector<std::vector<DataType>> partialSums_;
    int n_;
    std::vector<std::vector<DataType> > a_sums_;
    std::vector<std::vector<DataType> > b_sums_;

  public:
    RationalScoreContext(std::vector<DataType> a,
			 std::vector<DataType> b,
			 int n,
			 bool risk_partitioning_objective,
			 bool use_rational_optimization) : ParametricContext<DataType>(a,
									     b,
									     n,
									     risk_partitioning_objective,
									     use_rational_optimization,
									     "RationalScore")
    {}

    DataType compute_score_multclust(int, int) override;
    DataType compute_score_riskpart(int, int) override;
    DataType compute_score_riskpart_optimized(int, int) override;
    DataType compute_score_multclust_optimized(int, int) override;
    DataType compute_ambient_score_multclust(DataType, DataType) override;
    DataType compute_ambient_score_riskpart(DataType, DataType) override;

  };

} // namespace Objectives

#include "score2_impl.hpp"


#endif
