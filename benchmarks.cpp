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
class ContextFixture : public benchmark::Fixture {
public:
  using Context = RationalScoreContext<T>;

  Context context;
  const unsigned int N = 1<<13;

  ContextFixture() {

    bool risk_partitioning_objective=true, use_rational_optimization=true;

    std::vector<T> a(N), b(N);    
    compute_ab(a, b);

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

unsigned long long M = (1<<12);

BENCHMARK_REGISTER_F(ContextFixture, BM_float_compute_partial_sums_serial)->Arg(M);
BENCHMARK_REGISTER_F(ContextFixture, BM_float_compute_partial_sums_AVX256)->Arg(M);
BENCHMARK_REGISTER_F(ContextFixture, BM_float_compute_partial_sums_parallel)->Arg(M);

BENCHMARK_MAIN();

