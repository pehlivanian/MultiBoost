#ifndef __SCORE_HPP__
#define __SCORE_HPP__

#include <vector>
#include <list>
#include <limits>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <exception>


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


  class ParametricContext {
  protected:
    std::vector<double> a_;
    std::vector<double> b_;
    int n_;
    std::vector<std::vector<double> > a_sums_;
    std::vector<std::vector<double> > b_sums_;
    bool risk_partitioning_objective_;
    bool use_rational_optimization_;
    std::string name_;

  public:
    ParametricContext(std::vector<double> a, 
		      std::vector<double> b, 
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

    {}

    virtual ~ParametricContext() = default;

    virtual void compute_partial_sums() {};

    virtual double compute_score_multclust(int, int) = 0;
    virtual double compute_score_multclust_optimized(int, int) = 0;
    virtual double compute_score_riskpart(int, int) = 0;
    virtual double compute_score_riskpart_optimized(int, int) = 0;

    virtual double compute_ambient_score_multclust(double, double) = 0;
    virtual double compute_ambient_score_riskpart(double, double) = 0;

    std::string getName() const { return name_; }
    bool getRiskPartitioningObjective() const { return risk_partitioning_objective_; }
    bool getUseRationalOptimization() const { return use_rational_optimization_; }

    double compute_score(int i, int j) {
      if (risk_partitioning_objective_) {
	if (use_rational_optimization_) {
	  return compute_score_riskpart_optimized(i, j);
	}
	else {
	  return compute_score_riskpart(i, j);
	}
      }
      else {
	if (use_rational_optimization_) {
	  return compute_score_multclust_optimized(i, j);
	}
	else {
	  return compute_score_multclust(i, j);
	}
      }
    }
    
    double compute_ambient_score(double a, double b) {
      if (risk_partitioning_objective_) {
	return compute_ambient_score_riskpart(a, b);
      }
      else {
	return compute_ambient_score_multclust(a, b);
      }
    }
  };
  
  class PoissonContext : public ParametricContext {

  public:
    PoissonContext(std::vector<double> a, 
		   std::vector<double> b, 
		   int n, 
		   bool risk_partitioning_objective,
		   bool use_rational_optimization) : ParametricContext(a,
								       b,
								       n,
								       risk_partitioning_objective,
								       use_rational_optimization,
								       "Poisson")
    { if (use_rational_optimization) {
	compute_partial_sums();
      }
    }
  
    double compute_score_multclust(int i, int j) override {    
      double C = std::accumulate(a_.cbegin()+i, a_.cbegin()+j, 0.);
      double B = std::accumulate(b_.cbegin()+i, b_.cbegin()+j, 0.);
      return (C>B)? C*std::log(C/B) + B - C : 0.;
    }

    double compute_score_riskpart(int i, int j) override {
      double C = std::accumulate(a_.cbegin()+i, a_.cbegin()+j, 0.);
      double B = std::accumulate(b_.cbegin()+i, b_.cbegin()+j, 0.);
      return C*std::log(C/B);
    }
    
    double compute_ambient_score_multclust(double a, double b) override {
      return (a>b)? a*std::log(a/b) + b - a : 0.;
    }

    double compute_ambient_score_riskpart(double a, double b) override {
      return a*std::log(a/b);
    }  

    void compute_partial_sums() override {
      double a_cum;
      a_sums_ = std::vector<std::vector<double> >(n_, std::vector<double>(n_+1, std::numeric_limits<double>::lowest()));
      b_sums_ = std::vector<std::vector<double> >(n_, std::vector<double>(n_+1, std::numeric_limits<double>::lowest()));
    
      for (int i=0; i<n_; ++i) {
	a_sums_[i][i] = 0.;
	b_sums_[i][i] = 0.;
      }

      for (int i=0; i<n_; ++i) {
	a_cum = -a_[i-1];
	for (int j=i+1; j<=n_; ++j) {
	  a_cum += a_[j-2];
	  a_sums_[i][j] = a_sums_[i][j-1] + a_[j-1];
	  b_sums_[i][j] = b_sums_[i][j-1] + b_[j-1];
	}
      }  
    }

    double compute_score_riskpart_optimized(int i, int j) override {
      double score = a_sums_[i][j]*std::log(a_sums_[i][j]/b_sums_[i][j]);
      return score;
    }
    
    double compute_score_multclust_optimized(int i, int j) override {
      double score = (a_sums_[i][j] > b_sums_[i][j]) ? a_sums_[i][j]*std::log(a_sums_[i][j]/b_sums_[i][j]) + b_sums_[i][j] - a_sums_[i][j]: 0.;
      return score;
    }
    
  };

  class GaussianContext : public ParametricContext {

  public:
    GaussianContext(std::vector<double> a, 
		    std::vector<double> b, 
		    int n, 
		    bool risk_partitioning_objective,
		    bool use_rational_optimization) : ParametricContext(a,
									b,
									n,
									risk_partitioning_objective,
									use_rational_optimization,
									"Gaussian")
    { if (use_rational_optimization) {
	compute_partial_sums();
      }
    }
  
    double compute_score_multclust(int i, int j) override {
      double C = std::accumulate(a_.cbegin()+i, a_.cbegin()+j, 0.);
      double B = std::accumulate(b_.cbegin()+i, b_.cbegin()+j, 0.);
      return (C>B)? .5*(std::pow(C,2)/B + B) - C : 0.;
    }
  
    double compute_score_riskpart(int i, int j) override {
      double C = std::accumulate(a_.cbegin()+i, a_.cbegin()+j, 0.);
      double B = std::accumulate(b_.cbegin()+i, b_.cbegin()+j, 0.);
      return C*C/2./B;
    }

    double compute_ambient_score_multclust(double a, double b) override {
      return (a>b)? .5*(std::pow(a,2)/b + b) - a : 0.;
    }

    double compute_ambient_score_riskpart(double a, double b) override {
      return a*a/2./b;
    }

    void compute_partial_sums() override {
      double a_cum;
      a_sums_ = std::vector<std::vector<double> >(n_, std::vector<double>(n_+1, std::numeric_limits<double>::lowest()));
      b_sums_ = std::vector<std::vector<double> >(n_, std::vector<double>(n_+1, std::numeric_limits<double>::lowest()));
    
      for (int i=0; i<n_; ++i) {
	a_sums_[i][i] = 0.;
	b_sums_[i][i] = 0.;
      }

      for (int i=0; i<n_; ++i) {
	a_cum = -a_[i-1];
	for (int j=i+1; j<=n_; ++j) {
	  a_cum += a_[j-2];
	  a_sums_[i][j] = a_sums_[i][j-1] + a_[j-1];
	  b_sums_[i][j] = b_sums_[i][j-1] + b_[j-1];
	}
      }  
    }

    double compute_score_multclust_optimized(int i, int j) override {
      double score = (a_sums_[i][j] > b_sums_[i][j]) ? .5*(std::pow(a_sums_[i][j], 2)/b_sums_[i][j] + b_sums_[i][j]) - a_sums_[i][j] : 0.;
      return score;
    }
    
    double compute_score_riskpart_optimized(int i, int j) override {
      double score = a_sums_[i][j]*a_sums_[i][j]/2./b_sums_[i][j];
      return score;
    }

  };

  class RationalScoreContext : public ParametricContext {
    // This class doesn't correspond to any regular exponential family,
    // it is used to define ambient functions on the partition polytope
    // for targeted applications - quadratic approximations to loss, for
    // XGBoost, e.g.

  public:
    RationalScoreContext(std::vector<double> a,
			 std::vector<double> b,
			 int n,
			 bool risk_partitioning_objective,
			 bool use_rational_optimization) : ParametricContext(a,
									     b,
									     n,
									     risk_partitioning_objective,
									     use_rational_optimization,
									     "RationalScore")
    { if (use_rational_optimization) {
	compute_partial_sums();
      }
    }

    void compute_partial_sums() override {
      double a_cum;
      a_sums_ = std::vector<std::vector<double> >(n_, std::vector<double>(n_+1, std::numeric_limits<double>::lowest()));
      b_sums_ = std::vector<std::vector<double> >(n_, std::vector<double>(n_+1, std::numeric_limits<double>::lowest()));
    
      for (int i=0; i<n_; ++i) {
	a_sums_[i][i] = 0.;
	b_sums_[i][i] = 0.;
      }

      for (int i=0; i<n_; ++i) {
	a_cum = -a_[i-1];
	for (int j=i+1; j<=n_; ++j) {
	  a_cum += a_[j-2];
	  a_sums_[i][j] = a_sums_[i][j-1] + (2*a_cum + a_[j-1])*a_[j-1];
	  b_sums_[i][j] = b_sums_[i][j-1] + b_[j-1];
	}
      }  
    }
  
    double compute_score_multclust(int i, int j) override {
      double score = std::pow(std::accumulate(a_.cbegin()+i, a_.cbegin()+j, 0.), 2) /
	std::accumulate(b_.cbegin()+i, b_.cbegin()+j, 0.);
      return score;
    }
  
    double compute_score_riskpart(int i, int j) override {
      return compute_score_multclust(i, j);
    }

    double compute_score_riskpart_optimized(int i, int j) override {
      return compute_score_multclust_optimized(i, j);
    }

    double compute_score_multclust_optimized(int i, int j) override {
      double score = a_sums_[i][j] / b_sums_[i][j];
      return score;
    }

    double compute_ambient_score_multclust(double a, double b) override {
      return a*a/b;
    }

    double compute_ambient_score_riskpart(double a, double b) override {
      return a*a/b;
    }

  };

} // namespace Objectives


#endif
