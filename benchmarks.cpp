#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>

#include <mlpack/core.hpp>

#include "utils.hpp"
#include "gradientboostclassifier.hpp"
#include "score2.hpp"
#include "DP.hpp"

using namespace IB_utils;
using namespace arma;
using namespace Objectives;

// 3-step process:
// 1 sort a, b by priority
// 2. compute a_sums, b_sums as cached versions of compute_score
// 3. compute partialSums matrix as straight cached version of 
// 
//   score_function(i,j) = $\Sum_{i \in \left{ i, \dots, j\right}\F\left( x_i, y_i\right)$
//                       = partialSums[i][j]

template<typename T>
class ContextFixture : public benchmark::Fixture {
public:
  using Context = RationalScoreContext<T>;

  const unsigned int N = 1<<13;

  Context context;
  std::vector<T> a;
  std::vector<T> b;

  ContextFixture() {

    a.resize(N), b.resize(N);
    compute_ab(a, b);

    bool risk_partitioning_objective=true, use_rational_optimization=true;

    context = RationalScoreContext<T>(a, 
				      b, 
				      N,
				      risk_partitioning_objective,
				      use_rational_optimization);
  }
  
  ~ContextFixture() = default;

private:
  void compute_ab(std::vector<T>& a, std::vector<T>& b) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_real_distribution<T> dista(-10., 10.);
    std::uniform_real_distribution<T> distb(0., 1.);
    
    auto gena = [&dista, &mersenne_engine]() { return dista(mersenne_engine); };
    auto genb = [&distb, &mersenne_engine]() { return distb(mersenne_engine); };
    
    std::generate(a.begin(), a.end(), gena);
    std::generate(b.begin(), b.end(), genb);    
  }
};

// score interface we wish to benchmark
// void __compute_partial_sums__() { compute_partial_sums(); }
// void __compute_partial_sums_AVX256__() { compute_partial_sums_AVX256(); }
// void __compute_partial_sums_parallel__() { compute_partial_sums_parallel(); }
// void __compute_scores__() { compute_scores(); }
// void __compute_scores_parallel__() { compute_scores_parallel(); }
// T __compute_score__(int i, int j) { return compute_score(i, j); }
// T __compute_ambient_score__(int i, int j) { return compute_ambient_score(i, j); }

BENCHMARK_TEMPLATE_DEFINE_F(ContextFixture, BM_float_compute_partial_sums_serial, float)(benchmark::State& state) {
  
  for (auto _ : state) {
    context.__compute_partial_sums__();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(ContextFixture, BM_float_compute_partial_sums_AVX256, float)(benchmark::State& state) {
  
  for (auto _ : state) {
    context.__compute_partial_sums_AVX256__();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(ContextFixture, BM_float_compute_partial_sums_parallel, float)(benchmark::State& state) {
  
  for (auto _ : state) {
    context.__compute_partial_sums_parallel__();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(ContextFixture, BM_float_compute_scores_serial, float)(benchmark::State& state) {

  for (auto _ : state) {
    context.__compute_scores__();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(ContextFixture, BM_float_compute_scores_AVX256, float)(benchmark::State& state) {
  
  for (auto _ : state) {
    context.__compute_scores_AVX256__();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(ContextFixture, BM_float_compute_scores_parallel, float)(benchmark::State& state) {

  for (auto _ : state) {
    context.__compute_scores_parallel__();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(ContextFixture, BM_double_create_DPSolver, double)(benchmark::State& state) {
  int n = 1<<11, T = 500;
  bool risk_partitioning_objective = true;
  bool use_rational_optimization = true;
  bool sweep_down = false;
  double gamma = 0.;
  double reg_power=1.;
  bool find_optimal_t = false;

  DPSolver<double> dp;
  std::vector<std::vector<int>> subsets;

  for (auto _ : state) {
    benchmark::DoNotOptimize(dp = DPSolver(n, T, a, b,
					   objective_fn::RationalScore,
					   risk_partitioning_objective,
					   use_rational_optimization,
					   gamma,
					   reg_power,
					   sweep_down,
					   find_optimal_t));

    benchmark::DoNotOptimize(subsets = dp.get_optimal_subsets_extern());
  }

}

// Tests of armadillo primitives
void BM_colMask_arma_generation(benchmark::State& state) {

  const unsigned int N = state.range(0);
  const unsigned int n = 100;

  for (auto _ : state) {
      uvec r = sort(randperm(N, n));
  }

}

void BM_colMask_stl_generation1(benchmark::State& state) {
  
  const unsigned int N = state.range(0);
  const unsigned int n = 100;

  for (auto _ : state) {
    uvec r = PartitionUtils::sortedSubsample1(N, n);
  }
}

void BM_colMask_stl_generation2(benchmark::State& state) {
  
  const unsigned int N = state.range(0);
  const unsigned int n = 100;

  for (auto _ : state) {
    uvec r = PartitionUtils::sortedSubsample2(N, n);
  }
}

// DP solver benchmarks
BENCHMARK_REGISTER_F(ContextFixture, BM_float_compute_partial_sums_serial);
BENCHMARK_REGISTER_F(ContextFixture, BM_float_compute_partial_sums_AVX256);
BENCHMARK_REGISTER_F(ContextFixture, BM_float_compute_partial_sums_parallel);
BENCHMARK_REGISTER_F(ContextFixture, BM_float_compute_scores_serial);
BENCHMARK_REGISTER_F(ContextFixture, BM_float_compute_scores_AVX256);
BENCHMARK_REGISTER_F(ContextFixture, BM_float_compute_scores_parallel);

// DPSolver benchmarks
BENCHMARK_REGISTER_F(ContextFixture, BM_double_create_DPSolver);

// armadillo benchmarks
unsigned long N = (1<<12);

BENCHMARK(BM_colMask_arma_generation)->Arg(N);
BENCHMARK(BM_colMask_stl_generation1)->Arg(N);
BENCHMARK(BM_colMask_stl_generation2)->Arg(N);

BENCHMARK_MAIN();

