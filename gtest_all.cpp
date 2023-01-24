#include <gtest/gtest.h>

#include <vector>
#include <random>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

#include "score2.hpp"
#include "DP.hpp"


TEST(DPSolverTest, TestAVXMatchesSerial) {
  using namespace Objectives;

  int n = 500;
  int numTrials = 1000;
  std::vector<bool> trials(numTrials);
  
  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dista(-10., 10.), distb(0., 10.);
  std::uniform_int_distribution<int> distRow(0, n-1);
  std::uniform_int_distribution<int> distCol(0, n);

  std::vector<float> a(n), b(n);
  for (auto &el : a)
    el = dista(gen);
  for (auto &el : b)
    el = distb(gen);

  for (auto _ : trials) {
    RationalScoreContext<float>* context = new RationalScoreContext{a, b, n, false, true};
    context->__compute_partial_sums__();
    auto a_sums_serial = context->get_partial_sums_a();
    auto b_sums_serial = context->get_partial_sums_b();

    context->__compute_partial_sums_AVX256__();
    auto a_sums_AVX = context->get_partial_sums_a();
    auto b_sums_AVX = context->get_partial_sums_b();
    
    int ind1 = distRow(gen);
    int ind2 = distCol(gen);
    
    ASSERT_EQ(a_sums_serial[ind1][ind2], a_sums_AVX[ind2][ind1]);
    ASSERT_EQ(b_sums_serial[ind1][ind2], b_sums_AVX[ind2][ind1]);
  }
}


TEST(DPSolverTest, TestAVXPartialSumsMatchSerialPartialSums) {
  using namespace Objectives;

  int n = 100;
  int numTrials = 1000;
  std::vector<bool> trials(numTrials);
  
  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dista(-10., 10.), distb(0., 10.);
  std::uniform_int_distribution<int> distRow(0, n-1);
  std::uniform_int_distribution<int> distCol(0, n);

  std::vector<float> a(n), b(n);
  for (auto &el : a)
    el = dista(gen);
  for (auto &el : b)
    el = distb(gen);

  for (auto _ : trials) {
    RationalScoreContext<float>* context = new RationalScoreContext{a, b, n, false, true};
    
    context->__compute_partial_sums__();
    auto a_sums_serial = context->get_partial_sums_a();
    auto b_sums_serial = context->get_partial_sums_b();

    auto partialSums_serial = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.));
    
    for (int i=0; i<n; ++i) {
      for (int j=i; j<n; ++j) {
	partialSums_serial[i][j] = context->__compute_score__(i, j);
      }
    }

    context->__compute_partial_sums_AVX256__();
    auto a_sums_AVX = context->get_partial_sums_a();
    auto b_sums_AVX = context->get_partial_sums_b();
    
    auto partialSums_AVX = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.));    

    for(int i=0; i<n; ++i) {
      for (int j=0; j<=i; ++j) {
	partialSums_AVX[j][i] = context->__compute_score__(i, j);
      }
    }
    
    context->__compute_partial_sums_parallel__();
    auto a_sums_parallel = context->get_partial_sums_a();
    auto b_sums_parallel = context->get_partial_sums_b();
    
    auto partialSums_parallel = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.));

    for (int i=0; i<n; ++i) {
      for (int j=i; j<n; ++j) {
	partialSums_parallel[i][j] = context->__compute_score__(i, j);
      }
    }
    
    int ind1 = distRow(gen);
    int ind2 = distCol(gen);
    
    ASSERT_EQ(a_sums_serial[ind1][ind2], a_sums_AVX[ind2][ind1]);
    ASSERT_EQ(b_sums_serial[ind1][ind2], b_sums_AVX[ind2][ind1]);
    ASSERT_EQ(a_sums_serial[ind1][ind2], a_sums_parallel[ind1][ind2]);
    ASSERT_EQ(b_sums_serial[ind1][ind2], b_sums_parallel[ind1][ind2]);
    
    
    int numSamples = 1000;
    std::vector<bool> samples(numSamples);

    for (auto _ : samples) {
      int ind1_ = distRow(gen);
      int ind2_ = distRow(gen);
      if (ind1_ == ind2_)
	continue;
      if (ind1_ >= ind2_)
	std::swap(ind1_, ind2_);

      ASSERT_EQ(partialSums_serial[ind1_][ind2_], partialSums_AVX[ind1_][ind2_]);
      ASSERT_EQ(partialSums_serial[ind1_][ind2_], partialSums_parallel[ind1_][ind2_]);
    }
  }
}

TEST(DPSolverTest, TestParallelScoresMatchSerialScores) {
  using namespace Objectives;

  int n = 100;
  int numTrials = 1000;
  std::vector<bool> trials(numTrials);
  
  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dista(-10., 10.), distb(0., 10.);
  std::uniform_int_distribution<int> distRow(0, n-1);
  std::uniform_int_distribution<int> distCol(0, n);

  std::vector<float> a(n), b(n);
  for (auto &el : a)
    el = dista(gen);
  for (auto &el : b)
    el = distb(gen);

  for (auto _ : trials) {
    RationalScoreContext<float>* context = new RationalScoreContext{a, b, n, false, true};
    
    context->__compute_partial_sums__();
    auto a_sums_serial = context->get_partial_sums_a();
    auto b_sums_serial = context->get_partial_sums_b();

    auto partialSums_serial = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.));
    
    for (int i=0; i<n; ++i) {
      for (int j=i; j<n; ++j) {
	partialSums_serial[i][j] = context->__compute_score__(i, j);
      }
    }
    context->__compute_scores_parallel__();    
    auto partialSums_serial_from_context = context->get_scores();

    context->__compute_partial_sums_parallel__();
    auto a_sums_parallel = context->get_partial_sums_a();
    auto b_sums_parallel = context->get_partial_sums_b();
    
    auto partialSums_parallel = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.));

    for (int i=0; i<n; ++i) {
      for (int j=i; j<n; ++j) {
	partialSums_parallel[i][j] = context->__compute_score__(i, j);
      }
    }
    
    int ind1 = distRow(gen);
    int ind2 = distCol(gen);
    
    ASSERT_EQ(a_sums_serial[ind1][ind2], a_sums_parallel[ind1][ind2]);
    ASSERT_EQ(b_sums_serial[ind1][ind2], b_sums_parallel[ind1][ind2]);    
    
    int numSamples = 1000;
    std::vector<bool> samples(numSamples);

    for (auto _ : samples) {
      int ind1_ = distRow(gen);
      int ind2_ = distRow(gen);
      if (ind1_ == ind2_)
	continue;

      ASSERT_EQ(partialSums_serial[ind1_][ind2_], partialSums_parallel[ind1_][ind2_]);
      ASSERT_EQ(partialSums_serial[ind1_][ind2_], partialSums_serial_from_context[ind1_][ind2_]);
      ASSERT_EQ(partialSums_parallel[ind1_][ind2_], partialSums_serial_from_context[ind1_][ind2_]);
    }
  }
}


auto main(int argc, char **argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
