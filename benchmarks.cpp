#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>

#include "score2.hpp"

using namespace Objectives;

// 3-step process:
// 1 sort a, b by priority
// 2. compute a_sums, b_sums as cached versions of compute_score
// 3. compute partialSums matrix as straight cached version of 
// 
//   score_function(i,j) = $\Sum_{i \in \left{ i, \dots, j\right}\F\left( x_i, y_i\right)$
//                       = partialSums[i][j]

template<typename T>
T compute_ab(std::vector<T>& a, std::vector<T>& b) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<T> dista(-10., 10.);
  std::uniform_real_distribution<T> distb(0., 1.);

  auto gena = [&dista, &mersenne_engine]() { return dista(mersenne_engine); };
  auto genb = [&distb, &mersenne_engine]() { return distb(mersenne_engine); };

  std::generate(a.begin(), a.end(), gena);
  std::generate(b.begin(), b.end(), genb);
  
  return static_cast<T>(31);
}

// score interface we wish to benchmark
// void __compute_partial_sums__() { compute_partial_sums(); }
// void __compute_partial_sums_AVX256__() { compute_partial_sums_AVX256(); }
// void __compute_partial_sums_parallel__() { compute_partial_sums_parallel(); }
// void __compute_scores__() { compute_scores(); }
// void __compute_scores_parallel__() { compute_scores_parallel(); }
// T __compute_score__(int i, int j) { return compute_score(i, j); }
// T __compute_ambient_score__(int i, int j) { return compute_ambient_score(i, j); }

void BM_float_compute_partial_sums_serial(benchmark::State& state) {
  using T = float;
  using Context = RationalScoreContext<T>;

  const unsigned int N = state.range(0);
  std::vector<T> a(N), b(N);
  std::vector<std::vector<T>> a_sums, b_sums;
  bool risk_partitioning_objective=true, use_rational_optimization=true;

  Context context = RationalScoreContext<T>(a, 
					    b, 
					    N,
					    risk_partitioning_objective,
					    use_rational_optimization);
  
  compute_ab(a, b);
  
  for (auto _ : state) {
    context.__compute_partial_sums__();
  }
}

void BM_float_compute_partial_sums_AVX256(benchmark::State& state) {
  using T = float;
  using Context = RationalScoreContext<T>;

  const unsigned int N = state.range(0);
  std::vector<T> a(N), b(N);
  std::vector<std::vector<T>> a_sums, b_sums;
  bool risk_partitioning_objective=true, use_rational_optimization=true;

  Context context = RationalScoreContext<T>(a, 
					    b, 
					    N,
					    risk_partitioning_objective,
					    use_rational_optimization);
  
  compute_ab(a, b);
  
  for (auto _ : state) {
    context.__compute_partial_sums_AVX256__();
  }
}

void BM_float_compute_partial_sums_parallel(benchmark::State& state) {
  using T = float;
  using Context = RationalScoreContext<T>;

  const unsigned int N = state.range(0);
  std::vector<T> a(N), b(N);
  std::vector<std::vector<T>> a_sums, b_sums;
  bool risk_partitioning_objective=true, use_rational_optimization=true;

  Context context = RationalScoreContext<T>(a, 
					    b, 
					    N,
					    risk_partitioning_objective,
					    use_rational_optimization);
  
  compute_ab(a, b);
  
  for (auto _ : state) {
    context.__compute_partial_sums_parallel__();
  }
}

unsigned long long N = (1<<13);


BENCHMARK(BM_float_compute_partial_sums_serial)->Arg(N);
BENCHMARK(BM_float_compute_partial_sums_AVX256)->Arg(N);
BENCHMARK(BM_float_compute_partial_sums_parallel)->Arg(N);

BENCHMARK_MAIN();

