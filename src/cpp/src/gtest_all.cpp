#include <gtest/gtest.h>

#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)

#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/process/child.hpp>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <regex>
#include <vector>

// Force gtest symbols to be available before other includes
using ::testing::TestWithParam;
using ::testing::Values;

#include "DP.hpp"
#include "classifiers.hpp"
#include "gradientboostclassifier.hpp"
#include "gradientboostregressor.hpp"
#include "path_utils.hpp"
#include "replay.hpp"
#include "score2.hpp"
#include "utils.hpp"

namespace {
using DataType = Model_Traits::model_traits<DecisionTreeClassifier>::datatype;
}

using namespace boost::process;

using namespace IB_utils;
using namespace ModelContext;
using namespace ClassifierTypes;
using namespace RegressorTypes;

using dataset_t = Mat<DataType>;
using dataset_d = Mat<DataType>;
using dataset_regress_t = Mat<double>;
using dataset_regress_d = Mat<double>;
using labels_t = Row<std::size_t>;
using labels_d = Row<DataType>;
using labels_regress_t = Row<double>;
using labels_regress_d = Row<double>;

class DPSolverTestFixture : public ::testing::TestWithParam<objective_fn> {};

std::vector<std::string> tokenize(const std::string& s, const char* c) {
  std::vector<std::string> r;

  std::regex ws_re(c);
  std::copy(
      std::sregex_token_iterator(s.begin(), s.end(), ws_re, -1),
      std::sregex_token_iterator{},
      std::back_inserter(r));

  return r;
}

void sort_by_priority(std::vector<float>& a, std::vector<float>& b) {
  std::vector<int> ind(a.size());
  std::iota(ind.begin(), ind.end(), 0);

  std::stable_sort(
      ind.begin(), ind.end(), [&a, &b](int i, int j) { return (a[i] / b[i]) < (a[j] / b[j]); });
  std::vector<float> a_s, b_s;
  for (auto i : ind) {
    a_s.push_back(a[i]);
    b_s.push_back(b[i]);
  }

  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());
}

void sort_partition(std::vector<std::vector<int>>& v) {
  std::sort(v.begin(), v.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
    return (a.size() < b.size()) ||
           ((a.size() == b.size()) &&
            (a.at(std::distance(a.begin(), std::min_element(a.begin(), a.end()))) <
             b.at(std::distance(b.begin(), std::min_element(b.begin(), b.end())))));
  });
}

float rational_obj(std::vector<float> a, std::vector<float> b, int start, int end) {
  if (start == end) return 0.;
  float den = 0., num = 0.;
  for (int ind = start; ind < end; ++ind) {
    num += a[ind];
    den += b[ind];
  }
  return num * num / den;
}

std::vector<float> form_levels(int num_true_clusters, float epsilon) {
  std::vector<float> r;
  r.resize(num_true_clusters);
  float delta =
      ((2 - epsilon / num_true_clusters) - epsilon / num_true_clusters) / (num_true_clusters - 1);
  for (int i = 0; i < num_true_clusters; ++i) {
    r[i] = epsilon / static_cast<float>(num_true_clusters) + i * delta;
  }
  return r;
}

std::vector<int> form_splits(int n, int numMixed) {
  std::vector<int> r(numMixed);
  int splitInd = n / numMixed;
  int resid = n - numMixed * splitInd;
  for (int i = 0; i < numMixed; ++i) r[i] = splitInd;
  for (int i = 0; i < resid; ++i) r[i] += 1;
  std::partial_sum(r.begin(), r.end(), r.begin(), std::plus<float>());
  return r;
}

std::vector<float> mixture_gaussian_dist(
    int n, const std::vector<float> b, int numMixed, float sigma, float epsilon) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};

  std::vector<float> a(n);

  std::vector<int> splits = form_splits(n, numMixed);
  std::vector<float> levels = form_levels(numMixed, epsilon);

  for (int i = 0; i < n; ++i) {
    int ind = 0;
    while (i >= splits[ind]) {
      ++ind;
    }
    std::normal_distribution<float> dista(levels[ind] * b[i], sigma);
    a[i] = static_cast<float>(dista(mersenne_engine));
  }

  return a;
}

std::vector<float> mixture_poisson_dist(
    int n, const std::vector<float>& b, int numMixed, float epsilon) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};

  std::vector<float> a(n);

  std::vector<int> splits = form_splits(n, numMixed);
  std::vector<float> levels = form_levels(numMixed, epsilon);

  for (int i = 0; i < n; ++i) {
    int ind = 0;
    while (i >= splits[ind]) {
      ++ind;
    }
    std::poisson_distribution<int> dista(levels[ind] * b[i]);
    a[i] = static_cast<float>(dista(mersenne_engine));
  }

  return a;
}

float mixture_of_uniforms(int n) {
  int bin = 1;
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<float> distmixer(0., 1.);
  std::uniform_real_distribution<float> dista(0., 1. / static_cast<float>(n));

  float mixer = distmixer(mersenne_engine);

  while (bin < n) {
    if (mixer < static_cast<float>(bin) / static_cast<float>(n)) break;
    ++bin;
  }
  return dista(mersenne_engine) + static_cast<float>(bin) - 1.;
}

void loadClassifierDatasets(dataset_t& dataset, labels_t& labels) {
  if (!data::Load(IB_utils::resolve_test_data_path("sonar_X.csv"), dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load(IB_utils::resolve_test_data_path("sonar_y.csv"), labels))
    throw std::runtime_error("Could not load file");
}

void loadRegressorDatasets(dataset_regress_t& dataset, labels_regress_d& labels) {
  if (!data::Load(IB_utils::resolve_test_data_path("Regression/1193_BNG_lowbwt_X.csv"), dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load(IB_utils::resolve_test_data_path("Regression/1193_BNG_lowbwt_y.csv"), labels))
    throw std::runtime_error("Could not load file");
}

void exec(std::string cmd) {
  const char* cmd_c_str = cmd.c_str();
  FILE* pipe = popen(cmd_c_str, "r");
  if (!pipe) throw std::runtime_error("popen() failed!");
  pclose(pipe);
}

TEST(DPSolverTest, TestUnsortedIndWorksAsARMAIndexer) {
  int n = 500;
  int numTrials = 1000;
  std::vector<bool> trials(numTrials);

  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dista(-10., 10.), distb(0., 10.);

  float eps = std::numeric_limits<float>::epsilon();

  for (auto _ : trials) {
    (void)_;

    std::vector<float> a(n), b(n);
    for (auto& el : a) el = dista(gen);
    for (auto& el : b) el = distb(gen);

    auto dp = DPSolver(n, 10, a, b, objective_fn::Gaussian, true, true);
    auto opt = dp.get_optimal_subsets_extern();

    rowvec a_arma = arma::conv_to<rowvec>::from(a);
    rowvec b_arma = arma::conv_to<rowvec>::from(b);

    for (size_t i = 0; i < opt.size(); ++i) {
      auto subset = opt[i];
      // unsorted ratio
      uvec ind1 = arma::conv_to<uvec>::from(subset);
      float res1 = sum(a_arma(ind1)) / sum(b_arma(ind1));

      // sorted ratio
      std::sort(subset.begin(), subset.end());
      uvec ind2 = arma::conv_to<uvec>::from(subset);
      float res2 = sum(a_arma(ind2)) / sum(b_arma(ind2));

      ASSERT_LT(fabs(res1 - res2), eps);
    }
  }
}

TEST(DPSolverTest, TestCachedScoresMatchAcrossMethods) {
  using namespace Objectives;

  std::size_t n = 500;
  int numTrials = 1000;
  std::vector<bool> trials(numTrials);

  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dista(-10., 10.), distb(0., 10.);
  std::uniform_int_distribution<int> distRow(0, n - 1);
  std::uniform_int_distribution<int> distCol(0, n);

  for (auto _ : trials) {
    (void)_;

    std::vector<float> a(n), b(n);
    for (auto& el : a) el = dista(gen);
    for (auto& el : b) el = distb(gen);

    RationalScoreContext<float>* context_serial = new RationalScoreContext{a, b, n, false, true};
    context_serial->__compute_partial_sums__();
    auto a_sums_serial = context_serial->get_partial_sums_a();
    auto b_sums_serial = context_serial->get_partial_sums_b();

    RationalScoreContext<float>* context_parallel = new RationalScoreContext{a, b, n, false, true};
    context_parallel->__compute_partial_sums_parallel__();
    auto a_sums_parallel = context_parallel->get_partial_sums_a();
    auto b_sums_parallel = context_parallel->get_partial_sums_b();

    int ind1 = distRow(gen);
    int ind2 = distCol(gen);

    // ASSERT_EQ(a_sums_serial[ind1][ind2], a_sums_AVX[ind2][ind1]);
    // ASSERT_EQ(b_sums_serial[ind1][ind2], b_sums_AVX[ind2][ind1]);
    ASSERT_EQ(a_sums_serial[ind1][ind2], a_sums_parallel[ind1][ind2]);
    ASSERT_EQ(b_sums_serial[ind1][ind2], b_sums_parallel[ind1][ind2]);
  }
}

TEST(DPSolverTest, TestRationalScoreContextComputeScoreMethoDs) {
  using namespace Objectives;

  std::size_t n = 100;
  int numTrials = 100;
  std::vector<bool> trials(numTrials);

  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dista(-10., 10.), distb(0., 10.);
  std::uniform_int_distribution<int> distRow(0, n - 1);
  std::uniform_int_distribution<int> distCol(0, n);

  std::vector<float> a(n), b(n);
  for (auto& el : a) el = dista(gen);
  for (auto& el : b) el = distb(gen);

  for (auto trial : trials) {
    UNUSED(trial);

    RationalScoreContext<float>* context_serial =
        new RationalScoreContext<float>{a, b, n, false, true};
    // RationalScoreContext<float>* context_AVX      = new RationalScoreContext<float>{a, b, n,
    // false, true };
    RationalScoreContext<float>* context_parallel =
        new RationalScoreContext<float>{a, b, n, false, true};

    context_serial->__compute_scores__();
    // context_AVX->__compute_scores_AVX256__();
    context_parallel->__compute_scores_parallel__();

    auto a_sums_serial = context_serial->get_partial_sums_a();
    auto b_sums_serial = context_serial->get_partial_sums_b();

    // auto a_sums_AVX      = context_AVX->get_partial_sums_a();
    // auto b_sums_AVX      = context_AVX->get_partial_sums_b();

    auto a_sums_parallel = context_parallel->get_partial_sums_a();
    auto b_sums_parallel = context_parallel->get_partial_sums_b();

    auto partialSums_serial = context_serial->get_partial_sums();
    // auto partialSums_AVX      = context_serial->get_partial_sums();
    auto partialSums_parallel = context_parallel->get_partial_sums();

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j <= n; ++j) {
        // ASSERT_EQ(context_serial->get_score(i,j), context_AVX->get_score(i,j));
        ASSERT_EQ(context_serial->get_score(i, j), context_parallel->get_score(i, j));
      }
    }
  }
}

TEST(DPSolverTest, TestCachedScoresMatchExternalScores) {
  using namespace Objectives;

  std::size_t n = 100;
  int numTrials = 1000;
  std::vector<bool> trials(numTrials);

  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dista(-10., 10.), distb(0., 10.);
  std::uniform_int_distribution<int> distRow(0, n - 1);
  std::uniform_int_distribution<int> distCol(0, n);

  std::vector<float> a(n), b(n);
  for (auto& el : a) el = dista(gen);
  for (auto& el : b) el = distb(gen);

  for (auto _ : trials) {
    (void)_;

    PoissonContext<float>* context_serial = new PoissonContext{a, b, n, false, true};

    context_serial->__compute_partial_sums__();
    auto a_sums_serial = context_serial->get_partial_sums_a();
    auto b_sums_serial = context_serial->get_partial_sums_b();

    auto partialSums_serial = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.));

    // Based on cached a_sums_, b_sums_ from above
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i; j < n; ++j) {
        partialSums_serial[i][j] = context_serial->__compute_score__(i, j);
      }
    }

    PoissonContext<float>* context_AVX = new PoissonContext{a, b, n, false, true};

    context_AVX->__compute_partial_sums_AVX256__();
    auto a_sums_AVX = context_AVX->get_partial_sums_a();
    auto b_sums_AVX = context_AVX->get_partial_sums_b();

    auto partialSums_AVX = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.));

    // Based on cached a_sums_, b_sums_ from above
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        partialSums_AVX[j][i] = context_AVX->__compute_score__(i, j);
      }
    }

    PoissonContext<float>* context_parallel = new PoissonContext{a, b, n, false, true};

    context_parallel->__compute_partial_sums_parallel__();
    auto a_sums_parallel = context_parallel->get_partial_sums_a();
    auto b_sums_parallel = context_parallel->get_partial_sums_b();

    auto partialSums_parallel = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.));

    // Based on cached a_sums_, b_sums_ from above
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i; j < n; ++j) {
        partialSums_parallel[i][j] = context_parallel->__compute_score__(i, j);
      }
    }

    int ind1 = distRow(gen);
    int ind2 = distCol(gen);

    // a_sums, b_sums transposed for AVX case...
    ASSERT_EQ(a_sums_serial[ind1][ind2], a_sums_AVX[ind2][ind1]);
    ASSERT_EQ(b_sums_serial[ind1][ind2], b_sums_AVX[ind2][ind1]);
    ASSERT_EQ(a_sums_serial[ind1][ind2], a_sums_parallel[ind1][ind2]);
    ASSERT_EQ(b_sums_serial[ind1][ind2], b_sums_parallel[ind1][ind2]);

    int numSamples = 1000;
    std::vector<bool> samples(numSamples);

    for (auto _ : samples) {
      (void)_;

      int ind1_ = distRow(gen);
      int ind2_ = distRow(gen);
      if (ind1_ == ind2_) continue;
      if (ind1_ >= ind2_) std::swap(ind1_, ind2_);

      // ... but scores aren't
      ASSERT_EQ(partialSums_serial[ind1_][ind2_], partialSums_AVX[ind1_][ind2_]);
      ASSERT_EQ(partialSums_serial[ind1_][ind2_], partialSums_parallel[ind1_][ind2_]);
    }
  }
}

/*
// TODO: Parameterized test disabled due to namespace conflict with cereal/rapidjson
TEST_P(DPSolverTestFixture, TestConsecutiveProperty) {
  // Case (n,T) = (100,20)
  int n = 100, T = 20;

  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(1., 10.);

  std::vector<float> a(n), b(n);

  objective_fn objective = GetParam();
  for (size_t i = 0; i < 5; ++i) {
    for (auto& el : a) el = dist(gen);
    for (auto& el : b) el = dist(gen);

    // Presort
    sort_by_priority(a, b);

    auto dp = DPSolver(n, T, a, b, objective, false, true);
    auto opt = dp.get_optimal_subsets_extern();

    int sum;
    std::vector<int> v;

    for (auto& list : opt) {
      if (list.size() > 1) {
        v.resize(list.size());
        std::adjacent_difference(list.begin(), list.end(), v.begin());
        sum = std::accumulate(v.begin() + 1, v.end(), 0);
      }
    }

    // We ignored the first element as adjacent_difference has unintuitive
    // result for first element
    ASSERT_EQ(sum, v.size() - 1);
  }
}
*/

TEST(DPSolverTest, TestOptimalityWithRandomPartitionsRationalScore) {
  const int NUM_CASES = 100, NUM_SUB_CASES = 100;

  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_int_distribution<int> distn(100, 1000);
  std::uniform_real_distribution<float> dista(1., 10.);
  std::uniform_real_distribution<float> distb(1., 10.);

  std::vector<float> a, b;

  for (int case_num = 0; case_num < NUM_CASES; ++case_num) {
    int n = distn(gen);
    std::uniform_int_distribution<int> distT(2, n - 1);
    int T = distT(gen);

    a.resize(n);
    b.resize(n);

    for (auto& el : a) el = dista(gen);
    for (auto& el : b) el = distb(gen);

    objective_fn objective = objective_fn::RationalScore;

    auto dp = DPSolver(n, T, a, b, objective, true, true);

    auto subsets = dp.get_optimal_subsets_extern();
    float cum_opt_score = 0.;
    for (const auto& subset : subsets) {
      float cum_a = 0., cum_b = 0.;
      for (const auto& el : subset) {
        cum_a += a[el];
        cum_b += b[el];
      }
      cum_opt_score += cum_a * cum_a / cum_b;
    }

    auto dp_score = dp.get_optimal_score_extern();

    for (int subcase_num = 0; subcase_num < NUM_SUB_CASES; ++subcase_num) {
      int last_pt = 0;
      std::vector<int> div_pts;
      div_pts.push_back(last_pt);

      auto T_tmp = T;
      while (T_tmp > 1) {
        std::uniform_int_distribution<int> distdivpt(last_pt, n - 1);
        div_pts.push_back(distdivpt(gen));
        --T_tmp;
      }
      div_pts.push_back(n);
      std::sort(div_pts.begin(), div_pts.end());

      float cum_rnd_score = 0.;
      auto dp_a = dp.get_a();
      auto dp_b = dp.get_b();
      for (std::size_t i = 0; i < div_pts.size() - 1; ++i) {
        cum_rnd_score += dp.__compute_score__(div_pts[i], div_pts[i + 1]);
      }

      ASSERT_LT(fabs(dp_score - cum_opt_score), std::numeric_limits<float>::epsilon());
      ASSERT_LE(cum_rnd_score, dp_score);
    }
  }
}

/*
// TODO: Parameterized test disabled due to namespace conflict with cereal/rapidjson
TEST_P(DPSolverTestFixture, TestOptimalityWithRandomPartitionsSmallT) {
  int NUM_CASES = 1000, NUM_SUBCASES = 500, T = 3;

  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_int_distribution<int> distn(5, 50);
  std::uniform_real_distribution<float> dista(1., 10.);
  std::uniform_real_distribution<float> distb(1., 10.);

  std::vector<float> a, b;

  for (int case_num = 0; case_num < NUM_CASES; ++case_num) {
    int n = distn(gen);
    a.resize(n);
    b.resize(n);

    for (auto& el : a) el = dista(gen);
    for (auto& el : b) el = distb(gen);

    sort_by_priority(a, b);

    ASSERT_GE(n, 5);
    ASSERT_LE(n, 100);

    objective_fn objective = GetParam();

    auto dp = DPSolver(n, T, a, b, objective, true, true);
    auto dp_opt = dp.get_optimal_subsets_extern();
    auto scores = dp.get_score_by_subset_extern();

    for (int subcase_num = 0; subcase_num < NUM_SUBCASES; ++subcase_num) {
      std::uniform_int_distribution<int> distm(5, n);

      int m1 = distm(gen), m21;
      int m2 = distm(gen), m22;
      if ((m1 == m2) || (m1 == n) || (m2 == n)) continue;
      m21 = std::min(m1, m2);
      m22 = std::max(m1, m2);

      std::unique_ptr<ParametricContext<float>> context;

      switch (objective) {
        case objective_fn::Gaussian:
          context = std::make_unique<GaussianContext<float>>(a, b, n, true, false);
        case objective_fn::Poisson:
          context = std::make_unique<PoissonContext<float>>(a, b, n, true, false);
        case objective_fn::RationalScore:
          context = std::make_unique<RationalScoreContext<float>>(a, b, n, true, false);
      }

      context->__compute_partial_sums__();

      float rand_score, dp_score;
      rand_score = context->__compute_score__(0, m21) + context->__compute_score__(m21, m22) +
                   context->__compute_score__(m22, n);
      dp_score = context->__compute_score__(dp_opt[0][0], 1 + dp_opt[0][dp_opt[0].size() - 1]) +
                 context->__compute_score__(dp_opt[1][0], 1 + dp_opt[1][dp_opt[1].size() - 1]) +
                 context->__compute_score__(dp_opt[2][0], 1 + dp_opt[2][dp_opt[2].size() - 1]);

      if ((dp_score - rand_score) > std::numeric_limits<float>::epsilon()) {
        ASSERT_LE(rand_score, dp_score);
      }
    }
  }
}
*/

TEST(DPSolverTest, TestParallelScoresMatchSerialScores) {
  using namespace Objectives;

  std::size_t n = 100;
  int numTrials = 1000;
  std::vector<bool> trials(numTrials);

  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dista(-10., 10.), distb(0., 10.);
  std::uniform_int_distribution<int> distRow(0, n - 1);
  std::uniform_int_distribution<int> distCol(0, n);

  std::vector<float> a(n), b(n);
  for (auto& el : a) el = dista(gen);
  for (auto& el : b) el = distb(gen);

  for (auto _ : trials) {
    (void)_;

    GaussianContext<float>* context = new GaussianContext{a, b, n, false, true};

    context->__compute_partial_sums__();
    auto a_sums_serial = context->get_partial_sums_a();
    auto b_sums_serial = context->get_partial_sums_b();

    auto partialSums_serial = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.));

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i; j < n; ++j) {
        partialSums_serial[i][j] = context->__compute_score__(i, j);
      }
    }
    context->__compute_scores_parallel__();
    auto partialSums_serial_from_context = context->get_scores();

    context->__compute_partial_sums_parallel__();
    auto a_sums_parallel = context->get_partial_sums_a();
    auto b_sums_parallel = context->get_partial_sums_b();

    auto partialSums_parallel = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.));

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i; j < n; ++j) {
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
      (void)_;

      int ind1_ = distRow(gen);
      int ind2_ = distRow(gen);
      if (ind1_ == ind2_) continue;

      ASSERT_EQ(partialSums_serial[ind1_][ind2_], partialSums_parallel[ind1_][ind2_]);
      ASSERT_EQ(partialSums_serial[ind1_][ind2_], partialSums_serial_from_context[ind1_][ind2_]);
      ASSERT_EQ(partialSums_parallel[ind1_][ind2_], partialSums_serial_from_context[ind1_][ind2_]);
    }
  }
}

TEST(DPSolverTest, TestBaselines) {
  std::vector<float> a{0.0212651,   -0.20654906, -0.20654906, -0.20654906, -0.20654906, 0.0212651,
                       -0.20654906, 0.0212651,   -0.20654906, 0.0212651,   -0.20654906, 0.0212651,
                       -0.20654906, -0.06581402, 0.0212651,   0.03953075,  -0.20654906, 0.16200014,
                       0.0212651,   -0.20654906, 0.20296943,  -0.18828341, -0.20654906, -0.20654906,
                       -0.06581402, -0.20654906, 0.16200014,  0.03953075,  -0.20654906, -0.20654906,
                       0.03953075,  0.20296943,  -0.20654906, 0.0212651,   0.20296943,  -0.20654906,
                       0.0212651,   0.03953075,  -0.20654906, 0.03953075};
  std::vector<float> b{0.22771114, 0.21809504, 0.21809504, 0.21809504, 0.21809504, 0.22771114,
                       0.21809504, 0.22771114, 0.21809504, 0.22771114, 0.21809504, 0.22771114,
                       0.21809504, 0.22682739, 0.22771114, 0.22745816, 0.21809504, 0.2218354,
                       0.22771114, 0.21809504, 0.218429,   0.219738,   0.21809504, 0.21809504,
                       0.22682739, 0.21809504, 0.2218354,  0.22745816, 0.21809504, 0.21809504,
                       0.22745816, 0.218429,   0.21809504, 0.22771114, 0.218429,   0.21809504,
                       0.22771114, 0.22745816, 0.21809504, 0.22745816};

  std::vector<std::vector<int>> expected = {
      {1, 2, 3, 4, 6, 8, 10, 12, 16, 19, 21, 22, 23, 25, 28, 29, 32, 35, 38},
      {13, 24},
      {0, 5, 7, 9, 11, 14, 15, 18, 27, 30, 33, 36, 37, 39},
      {17, 26},
      {20, 31, 34}};

  std::vector<float> a1{
      2.26851454, 2.86139335, 5.51314769, 6.84829739, 6.96469186, 7.1946897, 9.80764198, 4.2310646};
  std::vector<float> b1{
      3.43178016,
      3.92117518,
      7.29049707,
      7.37995406,
      4.80931901,
      4.38572245,
      3.98044255,
      0.59677897};

  auto dp1 = DPSolver(8, 3, a1, b1, objective_fn::Poisson, false, true);
  auto opt1 = dp1.get_optimal_subsets_extern();

  auto dp = DPSolver(40, 5, a, b, objective_fn::Gaussian, true, true);
  auto opt = dp.get_optimal_subsets_extern();

  for (size_t i = 0; i < expected.size(); ++i) {
    auto expected_subset = expected[i], opt_subset = opt[i];
    std::sort(opt_subset.begin(), opt_subset.end());
    ASSERT_EQ(expected_subset.size(), opt_subset.size());
    for (size_t j = 0; j < expected_subset.size(); ++j) {
      ASSERT_EQ(expected_subset[j], opt_subset[j]);
    }
  }
}

TEST(GradientBoostClassifierTest, TestAggregateClassifierRecursiveReplay) {
  std::vector<bool> trials = {false, true};
  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;

  loadClassifierDatasets(dataset, labels);
  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.2);

  Context context{};

  context.loss = classifierLossFunction::BinomialDeviance;
  context.childPartitionSize = std::vector<std::size_t>{11, 5};
  context.childNumSteps = std::vector<std::size_t>{21, 2};
  context.childLearningRate = std::vector<double>{.001, .001};
  context.childActivePartitionRatio = std::vector<double>{0.33, 0.54};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.activePartitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = 1.;  // .75
  context.serializeModel = true;

  context.serializationWindow = 100;

  using T = GradientBoostClassifier<DecisionTreeClassifier>;

  for (auto recursive : trials) {
    context.recursiveFit = recursive;
    float eps = std::numeric_limits<float>::epsilon();

    // Fit classifier
    T classifier, newClassifier, secondClassifier;
    context.serializeModel = true;
    classifier = T(trainDataset, trainLabels, context);
    classifier.fit();

    // Predict IS with live classifier fails due to serialization...
    Row<DataType> liveTrainPrediction;
    EXPECT_THROW(
        classifier.Predict(trainDataset, liveTrainPrediction),
        predictionAfterClearedModelException);

    // Use latestPrediction_ instead
    classifier.Predict(liveTrainPrediction);

    // Get index, fldr
    std::string indexName = classifier.getIndexName();
    boost::filesystem::path fldr = classifier.getFldr();

    // Use replay to predict IS based on archive classifier
    Row<DataType> archiveTrainPrediction;
    Replay<DataType, DecisionTreeClassifier>::Classify(
        indexName, trainDataset, archiveTrainPrediction, fldr);

    for (std::size_t i = 0; i < liveTrainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(liveTrainPrediction[i] - archiveTrainPrediction[i]), eps);

    // Predict OOS with live classifier
    Row<DataType> liveTestPrediction;
    context.serializeModel = false;
    context.serializePrediction = false;
    secondClassifier = T(trainDataset, trainLabels, context);
    secondClassifier.fit();
    secondClassifier.Predict(testDataset, liveTestPrediction);

    // Use replay to predict OOS based on archive classifier
    Row<DataType> archiveTestPrediction;
    Replay<DataType, DecisionTreeClassifier>::Classify(
        indexName, testDataset, archiveTestPrediction, fldr, true);

    for (std::size_t i = 0; i < liveTestPrediction.size(); ++i) {
      ASSERT_LE(fabs(liveTestPrediction[i] - archiveTestPrediction[i]), eps);
    }
  }
}

TEST(GradientBoostClassifierTest, TestInSamplePredictionMatchesLatestPrediction) {
  std::vector<bool> trials = {false, true};

  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;

  loadClassifierDatasets(dataset, labels);
  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.2);

  Context context{};

  context.loss = classifierLossFunction::BinomialDeviance;
  context.childPartitionSize = std::vector<std::size_t>{11, 5};
  context.childNumSteps = std::vector<std::size_t>{5, 2};
  context.childLearningRate = std::vector<double>{.001, .001};
  context.childActivePartitionRatio = std::vector<double>{0.5, 0.2};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.activePartitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45;  // .75
  context.serializeModel = false;

  using T = GradientBoostClassifier<DecisionTreeClassifier>;

  for (auto recursive : trials) {
    context.recursiveFit = recursive;
    float eps = std::numeric_limits<float>::epsilon();

    // Fit classifier
    T classifier, newClassifier;
    classifier = T(trainDataset, trainLabels, testDataset, testLabels, context);
    classifier.fit();

    // IS prediction - live classifier
    Row<DataType> liveTrainPrediction;
    classifier.Predict(liveTrainPrediction);

    // IS lastestPrediction_ - archive classifier
    Row<DataType> latestPrediction = classifier.getLatestPrediction();

    for (std::size_t i = 0; i < liveTrainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(liveTrainPrediction[i] - latestPrediction[i]), eps);
  }
}

TEST(GradientBoostClassifierTest, TestAggregateClassifierRecursiveRoundTrips) {
  std::vector<bool> trials = {true};

  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;

  loadClassifierDatasets(dataset, labels);
  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.2);

  Context context{};

  context.loss = classifierLossFunction::BinomialDeviance;
  context.childPartitionSize = std::vector<std::size_t>{11, 5};
  context.childNumSteps = std::vector<std::size_t>{5, 2};
  context.childLearningRate = std::vector<double>{.001, .001};
  context.childActivePartitionRatio = std::vector<double>{0.5, 0.2};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.activePartitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45;  // .75
  context.recursiveFit = false;

  using T = GradientBoostClassifier<DecisionTreeClassifier>;

  for (auto recursive : trials) {
    context.recursiveFit = recursive;

    T classifier, newClassifier;
    classifier = T(trainDataset, trainLabels, testDataset, testLabels, context);
    classifier.fit();

    std::string fileName = classifier.write();
    auto tokens = strSplit(fileName, '_');
    ASSERT_EQ(tokens[0], "CLS");
    fileName = strJoin(tokens, '_', 1);
    classifier.read(newClassifier, fileName);

    Row<DataType> trainPrediction, trainNewPrediction, testPrediction, testNewPrediction;
    classifier.Predict(testDataset, testPrediction);
    classifier.Predict(trainDataset, trainPrediction);
    newClassifier.Predict(testDataset, testNewPrediction);
    newClassifier.Predict(trainDataset, trainNewPrediction);

    ASSERT_EQ(testPrediction.n_elem, testNewPrediction.n_elem);
    ASSERT_EQ(trainPrediction.n_elem, trainNewPrediction.n_elem);

    float eps = std::numeric_limits<float>::epsilon();
    for (std::size_t i = 0; i < testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i] - testNewPrediction[i]), eps);
    for (std::size_t i = 0; i < trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i] - trainNewPrediction[i]), eps);
  }
}

TEST(GradientBoostClassifierTest, TestChildSerializationRoundTrips) {
  int numTrials = 1;
  std::vector<bool> trials(numTrials);

  std::size_t numClasses = 2;
  std::size_t minLeafSize = 1;
  double minimumGainSplit = 0.;
  std::size_t maxDepth = 10;

  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;

  loadClassifierDatasets(dataset, labels);
  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.2);

  using T = DecisionTreeClassifierType;
  using IArchiveType = cereal::BinaryInputArchive;
  using OArchiveType = cereal::BinaryOutputArchive;

  for (auto _ : trials) {
    (void)_;

    T classifier, newClassifier;
    classifier = T(trainDataset, trainLabels, numClasses, minLeafSize, minimumGainSplit, maxDepth);

    std::string fileName =
        dumps<T, IArchiveType, OArchiveType>(classifier, SerializedType::CLASSIFIER);
    auto tokens = strSplit(fileName, '_');
    ASSERT_EQ(tokens[0], "CLS");
    fileName = strJoin(tokens, '_', 1);

    loads<T, IArchiveType, OArchiveType>(newClassifier, fileName);

    ASSERT_EQ(classifier.NumChildren(), newClassifier.NumChildren());
    ASSERT_EQ(classifier.NumClasses(), newClassifier.NumClasses());

    Row<std::size_t> trainPrediction, trainNewPrediction, testPrediction, testNewPrediction;
    classifier.Classify(testDataset, testPrediction);
    classifier.Classify(trainDataset, trainPrediction);
    newClassifier.Classify(testDataset, testNewPrediction);
    newClassifier.Classify(trainDataset, trainNewPrediction);

    ASSERT_EQ(testPrediction.n_elem, testNewPrediction.n_elem);
    ASSERT_EQ(trainPrediction.n_elem, trainNewPrediction.n_elem);

    float eps = std::numeric_limits<float>::epsilon();
    for (std::size_t i = 0; i < testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i] - testNewPrediction[i]), eps);
    for (std::size_t i = 0; i < trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i] - trainNewPrediction[i]), eps);
  }
}

TEST(GradientBoostClassifierTest, TestAggregateClassifierNonRecursiveRoundTrips) {
  int numTrials = 1;
  std::vector<bool> trials(numTrials);

  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;

  loadClassifierDatasets(dataset, labels);
  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.2);

  Context context{};

  context.loss = classifierLossFunction::BinomialDeviance;
  context.childPartitionSize = std::vector<std::size_t>{10};
  context.childNumSteps = std::vector<std::size_t>{8};
  context.childLearningRate = std::vector<double>{.001};
  context.childActivePartitionRatio = std::vector<double>{0.2, 0.3};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.activePartitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45;  // .75
  context.recursiveFit = false;

  using T = GradientBoostClassifier<DecisionTreeClassifier>;

  for (auto _ : trials) {
    (void)_;

    T classifier, newClassifier;
    classifier = T(trainDataset, trainLabels, testDataset, testLabels, context);
    classifier.fit();

    std::string fileName = classifier.write();
    auto tokens = strSplit(fileName, '_');
    ASSERT_EQ(tokens[0], "CLS");
    fileName = strJoin(tokens, '_', 1);

    classifier.read(newClassifier, fileName);

    Row<DataType> trainPrediction, trainNewPrediction, testPrediction, testNewPrediction;
    classifier.Predict(testDataset, testPrediction);
    classifier.Predict(trainDataset, trainPrediction);
    newClassifier.Predict(testDataset, testNewPrediction);
    newClassifier.Predict(trainDataset, trainNewPrediction);

    ASSERT_EQ(testPrediction.n_elem, testNewPrediction.n_elem);
    ASSERT_EQ(trainPrediction.n_elem, trainNewPrediction.n_elem);

    float eps = std::numeric_limits<float>::epsilon();
    for (std::size_t i = 0; i < testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i] - testNewPrediction[i]), eps);
    for (std::size_t i = 0; i < trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i] - trainNewPrediction[i]), eps);
  }
}

TEST(GradientBoostRegressorTest, TestContextWrittenWithCorrectValues) {
  using CerealT = Context;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  Context context_archive;

  std::string fileName = "./ctx_cls.dat";
  std::string cmd = IB_utils::resolve_path("build/create_context_regressor ");

  cmd += "--loss 0 ";
  cmd += "--steps 4122 --symmetrizeLabels false ";
  cmd += "--childPartitionSize 1000 500 250 125 10 1 --childNumSteps 100 10 5 2 1 1 ";
  cmd += "--childMinLeafSize 10 10 5 2 1 ";
  cmd += "--fileName ./ctx_cls.dat";

  exec(cmd);

  loads<CerealT, CerealIArch, CerealOArch>(context_archive, fileName);

  // User set values
  ASSERT_EQ(std::get<regressorLossFunction>(context_archive.loss), regressorLossFunction::MSE);
  ASSERT_EQ(context_archive.steps, 4122);
  ASSERT_EQ(context_archive.symmetrizeLabels, false);

  std::vector<std::size_t> targetPartitionSize = {1000, 500, 250, 125, 10, 1};
  std::vector<std::size_t> targetNumSteps = {100, 10, 5, 2, 1, 1};
  std::vector<std::size_t> targetMinLeafSize = {10, 10, 5, 2, 1};

  for (std::size_t i = 0; i < context_archive.childPartitionSize.size(); ++i) {
    ASSERT_EQ(context_archive.childPartitionSize[i], targetPartitionSize[i]);
  }

  for (std::size_t i = 0; i < context_archive.childNumSteps.size(); ++i) {
    ASSERT_EQ(context_archive.childNumSteps[i], targetNumSteps[i]);
  }

  for (std::size_t i = 0; i < context_archive.childMinLeafSize.size(); ++i) {
    ASSERT_EQ(context_archive.childMinLeafSize[i], targetMinLeafSize[i]);
  }

  // Default values
  ASSERT_EQ(context_archive.recursiveFit, true);
  ASSERT_EQ(context_archive.rowSubsampleRatio, 1.);
  ASSERT_EQ(context_archive.colSubsampleRatio, .25);
}

TEST(GradientBoostClassifierTest, TestContextWrittenWithCorrectValues) {
  using CerealT = Context;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  Context context_archive;

  std::string fileName = "ctx_reg.dat";
  std::string cmd = IB_utils::resolve_path("build/create_context_classifier ");

  cmd += "--loss 7 ";
  cmd += "--learningRate .01 --steps 1010 --symmetrizeLabels true ";
  cmd += "--childMinLeafSize 10 10 5 2 1 ";
  cmd += "--fileName ctx_reg.dat";

  exec(cmd);

  loads<CerealT, CerealIArch, CerealOArch>(context_archive, fileName);

  // User set values
  ASSERT_EQ(
      std::get<classifierLossFunction>(context_archive.loss), classifierLossFunction::SquareLoss);
  ASSERT_EQ(context_archive.steps, 1010);

  std::vector<std::size_t> targetMinLeafSize = {10, 10, 5, 2, 1};
  for (std::size_t i = 0; i < context_archive.childMinLeafSize.size(); ++i) {
    ASSERT_EQ(context_archive.childMinLeafSize[i], targetMinLeafSize[i]);
  }

  // Default values
  ASSERT_EQ(context_archive.symmetrizeLabels, true);
  ASSERT_EQ(context_archive.recursiveFit, true);
  ASSERT_EQ(context_archive.rowSubsampleRatio, 1.);
  ASSERT_EQ(context_archive.colSubsampleRatio, .25);
}

TEST(GradientBoostClassifierTest, TestContextReadWrite) {
  std::size_t minLeafSize = 1;
  double minimumGainSplit = 0.;
  std::size_t maxDepth = 10;
  std::size_t partitionSize = 10;

  Context context{}, context_archive;

  context.loss = classifierLossFunction::BinomialDeviance;
  context.activePartitionRatio = .25;
  context.steps = 21;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45;  // .75
  context.recursiveFit = false;
  context.minLeafSize = minLeafSize;
  context.maxDepth = maxDepth;
  context.minimumGainSplit = minimumGainSplit;

  std::string binFileName = "gtest__Context.dat";

  using CerealT = Context;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  dumps<CerealT, CerealIArch, CerealOArch>(context, binFileName);
  loads<CerealT, CerealIArch, CerealOArch>(context_archive, binFileName);

  ASSERT_EQ(
      std::get<classifierLossFunction>(context_archive.loss),
      classifierLossFunction::BinomialDeviance);
  ASSERT_EQ(context_archive.loss, context.loss);
}

TEST(GradientBoostClassifierTest, TestWritePrediction) {
  std::vector<bool> trials = {false, true};
  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;

  loadClassifierDatasets(dataset, labels);
  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.2);

  Context context{};

  context.loss = classifierLossFunction::BinomialDeviance;
  context.childPartitionSize = std::vector<std::size_t>{11, 6};
  context.childNumSteps = std::vector<std::size_t>{4, 4};
  context.childLearningRate = std::vector<double>{.001, .001};
  context.childActivePartitionRatio = std::vector<double>{0.2, 0.1};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.activePartitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45;  // .75
  context.serializeModel = true;
  context.serializePrediction = true;
  context.serializeColMask = true;

  context.serializationWindow = 100;

  using T = GradientBoostClassifier<DecisionTreeClassifier>;

  for (auto recursive : trials) {
    context.recursiveFit = recursive;

    float eps = std::numeric_limits<float>::epsilon();

    // Fit classifier
    T classifier, newClassifier;
    classifier = T(trainDataset, trainLabels, testDataset, testLabels, context);
    classifier.fit();

    // Predict IS with live classifier fails due to serialization...
    Row<DataType> liveTrainPrediction;
    EXPECT_THROW(
        classifier.Predict(trainDataset, liveTrainPrediction),
        predictionAfterClearedModelException);

    // Use latestPrediction_ instead
    classifier.Predict(liveTrainPrediction);

    // Use replay to predict IS based on archive classifier
    Row<DataType> archiveTrainPrediction1, archiveTrainPrediction2;
    std::string indexName = classifier.getIndexName();
    boost::filesystem::path fldr = classifier.getFldr();

    // Method 1
    Replay<DataType, DecisionTreeClassifier>::Classify(
        indexName, trainDataset, archiveTrainPrediction1, fldr);

    // Method 2
    Replay<DataType, DecisionTreeClassifier>::Classify(indexName, archiveTrainPrediction2, fldr);

    for (std::size_t i = 0; i < liveTrainPrediction.n_elem; ++i) {
      ASSERT_LE(fabs(liveTrainPrediction[i] - archiveTrainPrediction1[i]), eps);
      ASSERT_LE(fabs(liveTrainPrediction[i] - archiveTrainPrediction2[i]), eps);
    }
  }
}

TEST(GradientBoostRegressorTest, TestPredictionRoundTrip) {
  std::vector<bool> trials = {false};
  dataset_regress_d dataset, trainDataset, testDataset;
  labels_regress_d labels, trainLabels, testLabels;

  loadRegressorDatasets(dataset, labels);
  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.85);

  Context context{};

  context.loss = regressorLossFunction::MSE;
  context.childPartitionSize = std::vector<std::size_t>{11};
  context.childNumSteps = std::vector<std::size_t>{21};
  context.childLearningRate = {1., 1., 1.};
  context.childActivePartitionRatio = std::vector<double>{0.1, 0.1};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.activePartitionRatio = .25;
  context.steps = 35;
  context.quietRun = true;
  context.symmetrizeLabels = false;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = 1.;  // .75
  context.serializeModel = true;
  context.serializePrediction = true;

  context.serializationWindow = 11;

  using T = GradientBoostRegressor<DecisionTreeRegressorRegressor>;

  for (auto recursive : trials) {
    context.recursiveFit = recursive;

    double eps = std::numeric_limits<double>::epsilon();

    Row<double> prediction, archivePrediction, newPrediciton, secondPrediction;

    // Fit regressor
    T regressor;
    regressor = T(trainDataset, trainLabels, testDataset, testLabels, context);
    regressor.fit();
    regressor.Predict(prediction);

    std::string indexName = regressor.getIndexName();
    boost::filesystem::path fldr = regressor.getFldr();

    readPrediction(indexName, archivePrediction, fldr);

    for (std::size_t i = 0; i < prediction.n_elem; ++i) {
      ASSERT_LE(fabs(prediction[i] - archivePrediction[i]), eps);
      ASSERT_LE(fabs(prediction[i] - archivePrediction[i]), eps);
    }

    // We have archive prediction for an intermiediate point [1..22..106]
    // Create a regressor over the entire period and fit
    context.childNumSteps = std::vector<std::size_t>{35};
    context.steps = 35;
    T secondRegressor;
    secondRegressor = T(trainDataset, trainLabels, testDataset, testLabels, context);
    secondRegressor.fit();
    secondRegressor.Predict(secondPrediction);

    // Compare with 24 steps from archivePrediction
    context.childNumSteps = std::vector<std::size_t>{14};
    context.steps = 35;
    T archiveRegressor =
        T(trainDataset, trainLabels, testDataset, testLabels, archivePrediction, context);
    archiveRegressor.fit();
    archiveRegressor.Predict(archivePrediction);

    for (std::size_t i = 0; i < secondPrediction.size(); ++i) {
      ASSERT_LE(fabs(secondPrediction[i] - archivePrediction[i]), eps);
    }
  }
}

TEST(GradientBoostClassifierTest, TestPredictionRoundTrip) {
  std::vector<bool> trials = {false};
  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;

  loadClassifierDatasets(dataset, labels);
  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.2);

  Context context{};

  context.loss = classifierLossFunction::BinomialDeviance;
  context.childPartitionSize = std::vector<std::size_t>{11};
  context.childNumSteps = std::vector<std::size_t>{114};
  context.childLearningRate = std::vector<double>{.001};
  context.childActivePartitionRatio = std::vector<double>{0.4, 0.2};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.activePartitionRatio = .25;
  context.quietRun = true;
  context.steps = 214;
  context.symmetrizeLabels = true;
  context.quietRun = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = 1.;  // .75
  context.serializeModel = true;
  context.serializePrediction = true;
  context.serializeColMask = false;

  context.serializationWindow = 100;

  using T = GradientBoostClassifier<DecisionTreeClassifier>;

  for (auto recursive : trials) {
    context.recursiveFit = recursive;

    double eps = std::numeric_limits<double>::epsilon();

    Row<DataType> prediction, archivePrediction, newPrediction, secondPrediction;

    // Fit classifier
    T classifier;
    classifier = T(trainDataset, trainLabels, testDataset, testLabels, context);
    classifier.fit();

    // Test classifier prediction matches reloaded latestPrediction from same classifier
    classifier.Predict(prediction);

    std::string indexName = classifier.getIndexName();
    boost::filesystem::path fldr = classifier.getFldr();

    readPrediction(indexName, archivePrediction, fldr);

    for (std::size_t i = 0; i < prediction.n_elem; ++i) {
      ASSERT_LE(fabs(prediction[i] - archivePrediction[i]), eps);
    }

    // We have archive prediction for an intermediate point [1..114..214]
    // Create a classifier over the entire period and fit
    context.childNumSteps = std::vector<std::size_t>{214};
    context.steps = 214;
    T secondClassifier;
    secondClassifier = T(trainDataset, trainLabels, testDataset, testLabels, context);
    secondClassifier.fit();
    secondClassifier.Predict(secondPrediction);

    // Compare with 100 steps from archivePrediction
    context.childNumSteps = std::vector<std::size_t>{100};
    context.steps = 214;
    T archiveClassifier =
        T(trainDataset, trainLabels, testDataset, testLabels, archivePrediction, context);
    archiveClassifier.fit();
    archiveClassifier.Predict(archivePrediction);

    if (false)
      eps = 1.5;
    else
      eps = std::numeric_limits<float>::epsilon();

    for (std::size_t i = 0; i < secondPrediction.size(); ++i) {
      ASSERT_LE(fabs(secondPrediction[i] - archivePrediction[i]), eps);
    }
  }
}

TEST(GradientBoostRegressorTest, TestChildSerializationRoundTrips) {
  int numTrials = 1;
  std::vector<bool> trials(numTrials);

  std::size_t minLeafSize = 1;
  double minimumGainSplit = 0.;
  std::size_t maxDepth = 10;

  dataset_regress_d dataset, trainDataset, testDataset;
  labels_regress_d labels, trainLabels, testLabels;

  loadRegressorDatasets(dataset, labels);
  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.9);

  using T = DecisionTreeRegressorRegressorType;
  using IArchiveType = cereal::BinaryInputArchive;
  using OArchiveType = cereal::BinaryOutputArchive;

  for (auto _ : trials) {
    (void)_;

    T regressor, newRegressor;
    regressor = T(trainDataset, trainLabels, minLeafSize, minimumGainSplit, maxDepth);

    std::string fileName =
        dumps<T, IArchiveType, OArchiveType>(regressor, SerializedType::REGRESSOR);
    auto tokens = strSplit(fileName, '_');
    ASSERT_EQ(tokens[0], "REG");
    fileName = strJoin(tokens, '_', 1);

    loads<T, IArchiveType, OArchiveType>(newRegressor, fileName);

    ASSERT_EQ(regressor.NumChildren(), newRegressor.NumChildren());

    Row<double> trainPrediction, trainNewPrediction, testPrediction, testNewPrediction;
    regressor.Predict(testDataset, testPrediction);
    regressor.Predict(trainDataset, trainPrediction);
    newRegressor.Predict(testDataset, testNewPrediction);
    newRegressor.Predict(trainDataset, trainNewPrediction);

    ASSERT_EQ(testPrediction.n_elem, testNewPrediction.n_elem);
    ASSERT_EQ(trainPrediction.n_elem, trainNewPrediction.n_elem);

    float eps = std::numeric_limits<float>::epsilon();
    for (std::size_t i = 0; i < testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i] - testNewPrediction[i]), eps);
    for (std::size_t i = 0; i < trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i] - trainNewPrediction[i]), eps);
  }
}

TEST(GradientBoostRegressorTest, TestInSamplePredictionMatchesLatestPrediction) {
  std::vector<bool> trials = {false, true};

  dataset_regress_d dataset, trainDataset, testDataset;
  labels_regress_d labels, trainLabels, testLabels;

  loadRegressorDatasets(dataset, labels);
  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.9);

  Context context{};

  context.loss = regressorLossFunction::MSE;
  context.childPartitionSize = std::vector<std::size_t>{11, 5, 2};
  context.childNumSteps = std::vector<std::size_t>{14, 1, 2};
  context.childLearningRate = std::vector<double>{1., 1., 1.};
  context.childActivePartitionRatio = std::vector<double>{0.1, 0.1, 0.1};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0., 0.};
  context.activePartitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45;  // .75
  context.serializeModel = false;

  using T = GradientBoostRegressor<DecisionTreeRegressorRegressor>;

  for (auto recursive : trials) {
    context.recursiveFit = recursive;
    float eps = std::numeric_limits<float>::epsilon();

    // Fit classifier
    T regressor;
    regressor = T(trainDataset, trainLabels, testDataset, testLabels, context);
    regressor.fit();

    // IS prediction - live regressor
    Row<double> liveTrainPrediction;
    regressor.Predict(liveTrainPrediction);

    // IS lastestPrediction_ - archive regressor
    Row<double> latestPrediction = regressor.getLatestPrediction();

    for (std::size_t i = 0; i < liveTrainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(liveTrainPrediction[i] - latestPrediction[i]), eps);
  }
}

TEST(GradientBoostRegressorTest, TestAggregateRegressorRecursiveReplay) {
  std::vector<bool> trials = {true, false};
  dataset_regress_d dataset, trainDataset, testDataset;
  labels_regress_d labels, trainLabels, testLabels;

  loadRegressorDatasets(dataset, labels);

  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.995);

  Context context{};

  context.loss = regressorLossFunction::MSE;
  context.childPartitionSize = std::vector<std::size_t>{11, 5, 2};
  context.childNumSteps = std::vector<std::size_t>{1, 1, 1};
  context.childLearningRate = std::vector<double>{1., 1., 1.};
  context.childActivePartitionRatio = std::vector<double>{0.1, 0.67, 0.77};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0., 0.};
  context.activePartitionRatio = .25;
  context.quietRun = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = 1.;  // .75
  context.serializeModel = true;

  context.serializationWindow = 2;

  using T = GradientBoostRegressor<DecisionTreeRegressorRegressor>;

  for (auto recursive : trials) {
    context.recursiveFit = recursive;
    double eps = std::numeric_limits<float>::epsilon();

    // Fit regressor
    T regressor, newRegressor, secondRegressor;
    context.serializeModel = true;
    regressor = T(trainDataset, trainLabels, context);
    regressor.fit();

    // Predict IS with live regressor fails due to serialization...
    Row<double> liveTrainPrediction;
    EXPECT_THROW(
        regressor.Predict(trainDataset, liveTrainPrediction), predictionAfterClearedModelException);

    // Use latestPrediction_ instead
    regressor.Predict(liveTrainPrediction);

    auto latestPrediction = regressor.getLatestPrediction();
    for (std::size_t i = 0; i < trainLabels.n_elem; ++i) {
      ASSERT_LE(fabs(liveTrainPrediction[i] - latestPrediction[i]), eps);
    }

    // Get index
    std::string indexName = regressor.getIndexName();
    boost::filesystem::path fldr = regressor.getFldr();
    // Use replay to predict IS based on archive regressor
    Row<double> archiveTrainPrediction;
    Replay<double, DecisionTreeRegressorRegressor>::Predict(
        indexName, trainDataset, archiveTrainPrediction, fldr);

    for (std::size_t i = 0; i < liveTrainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(liveTrainPrediction[i] - archiveTrainPrediction[i]), eps);

    // Predict OOS with live regressor
    Row<double> liveTestPrediction;
    context.serializeModel = false;
    context.serializePrediction = false;
    secondRegressor = T(trainDataset, trainLabels, context);
    secondRegressor.fit();
    secondRegressor.Predict(testDataset, liveTestPrediction);

    // Use replay to predict OOS based on archive classifier
    Row<double> archiveTestPrediction;
    Replay<double, DecisionTreeRegressorRegressor>::Predict(
        indexName, testDataset, archiveTestPrediction, fldr);

    for (std::size_t i = 0; i < liveTestPrediction.size(); ++i) {
      ASSERT_LE(fabs(liveTestPrediction[i] - archiveTestPrediction[i]), eps);
    }
  }
}

TEST(GradientBoostRegressorTest, TestAggregateRegressorNonRecursiveRoundTrips) {
  int numTrials = 1;
  std::vector<bool> trials(numTrials);

  dataset_regress_d dataset, trainDataset, testDataset;
  labels_regress_d labels, trainLabels, testLabels;

  loadRegressorDatasets(dataset, labels);
  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.85);

  Context context{};

  context.loss = regressorLossFunction::MSE;
  context.childPartitionSize = std::vector<std::size_t>{10, 2};
  context.childNumSteps = std::vector<std::size_t>{14, 2};
  context.childLearningRate = std::vector<double>{1., 1.};
  context.childActivePartitionRatio = std::vector<double>{0.2, 0.2};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.activePartitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = false;
  context.removeRedundantLabels = false;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45;  // .75
  context.recursiveFit = false;

  using T = GradientBoostRegressor<DecisionTreeRegressorRegressor>;

  for (auto _ : trials) {
    (void)_;

    T regressor, newRegressor;
    regressor = T(trainDataset, trainLabels, testDataset, testLabels, context);
    regressor.fit();

    std::string fileName = regressor.write();
    auto tokens = strSplit(fileName, '_');
    ASSERT_EQ(tokens[0], "REG");
    fileName = strJoin(tokens, '_', 1);

    regressor.read(newRegressor, fileName);

    Row<double> trainPrediction, trainNewPrediction, testPrediction, testNewPrediction;
    regressor.Predict(testDataset, testPrediction);
    regressor.Predict(trainDataset, trainPrediction);
    newRegressor.Predict(testDataset, testNewPrediction);
    newRegressor.Predict(trainDataset, trainNewPrediction);

    ASSERT_EQ(testPrediction.n_elem, testNewPrediction.n_elem);
    ASSERT_EQ(trainPrediction.n_elem, trainNewPrediction.n_elem);

    double eps = std::numeric_limits<double>::epsilon();
    for (std::size_t i = 0; i < testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i] - testNewPrediction[i]), eps);
    for (std::size_t i = 0; i < trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i] - trainNewPrediction[i]), eps);
  }
}

TEST(UtilsTest, TestPartitionSubsampling1) {
  const int N = 1000;
  const int n = 500;

  uvec inds = PartitionUtils::sortedSubsample1(N, n);

  ASSERT_EQ(inds.n_elem, n);
}

TEST(UtilsTest, TestPartitionSubsampling2) {
  const int N = 1000;
  const int n = 500;

  uvec inds = PartitionUtils::sortedSubsample2(N, n);

  ASSERT_EQ(inds.n_elem, n);
}

TEST(GradientBoostRegressorTest, TestPerfectInSampleFit) {
  std::vector<bool> recursive{false, true};

  Mat<double> dataset;
  Row<double> labels, prediction;

  int rows = 1000, cols = 20;

  dataset = Mat<double>(rows, cols, fill::randu);
  labels = Row<double>(cols);

  for (std::size_t i = 0; i < 4; ++i) {
    dataset(444, i) = i;
    labels[i] = 14.;
  }
  for (std::size_t i = 4; i < 8; ++i) {
    dataset(444, i) = i;
    labels[i] = -1.45;
  }
  for (std::size_t i = 8; i < 12; ++i) {
    dataset(444, i) = i;
    labels[i] = -2.077;
  }
  for (std::size_t i = 12; i < 16; ++i) {
    dataset(444, i) = i;
    labels[i] = 2.077;
  }
  for (std::size_t i = 16; i < 20; ++i) {
    dataset(444, i) = i;
    labels[i] = 2.24;
  }

  for (auto recursive_ : recursive) {
    Context context{};

    context.loss = regressorLossFunction::MSE;
    context.childPartitionSize = std::vector<std::size_t>{10, 4};
    context.childNumSteps = std::vector<std::size_t>{5, 6};
    context.childLearningRate = std::vector<double>{1., 1.};
    context.childActivePartitionRatio = std::vector<double>{0.3, 0.2};
    context.childMinLeafSize = std::vector<std::size_t>{1, 1};
    context.childMaxDepth = std::vector<std::size_t>{5, 5};
    context.childMinimumGainSplit = std::vector<double>{0., 0.};
    context.activePartitionRatio = .25;
    context.steps = 1000;
    context.symmetrizeLabels = true;
    context.serializationWindow = 1000;
    context.removeRedundantLabels = false;
    context.rowSubsampleRatio = 1.;
    context.colSubsampleRatio = 1.;  // .75
    context.recursiveFit = recursive_;
    context.serializeModel = false;
    context.serializePrediction = false;
    context.serializeDataset = false;
    context.serializeLabels = false;
    context.serializationWindow = 1000;

    auto regressor =
        GradientBoostRegressor<DecisionTreeRegressorRegressor>(dataset, labels, context);
    regressor.fit();
    regressor.Predict(prediction);

    const double trainError = err(prediction, labels);

    ASSERT_EQ(trainError, 0.);
  }
}

TEST(GradientBoostRegressorTest, TestOutofSampleFit) {
  std::vector<bool> recursive{false, true};

  Mat<double> dataset, dataset_oos;
  Row<double> labels, labels_oos, prediction, prediction_oos;

  int rows = 50, cols = 20;

  dataset = Mat<double>(rows, cols, fill::randu);
  labels = Row<double>(cols);

  dataset_oos = Mat<double>(rows, cols, fill::randu);
  labels_oos = Row<double>(cols);

  for (std::size_t i = 0; i < 4; ++i) {
    dataset(44, i) = i;
    dataset_oos(44, i) = 2 * i;
    labels[i] = 2.;
    labels_oos[i] = 2 * 2.;
  }
  for (std::size_t i = 4; i < 8; ++i) {
    dataset(44, i) = i;
    dataset_oos(44, i) = 2 * i;
    labels[i] = 4.;
    labels_oos[i] = 2 * 4.;
  }
  for (std::size_t i = 8; i < 12; ++i) {
    dataset(44, i) = i;
    dataset_oos(44, i) = 2 * i;
    labels[i] = 6.;
    labels_oos[i] = 2 * 6.;
  }
  for (std::size_t i = 12; i < 16; ++i) {
    dataset(44, i) = i;
    dataset_oos(44, i) = 2 * i;
    labels[i] = 8.;
    labels_oos[i] = 2 * 8.;
  }
  for (std::size_t i = 16; i < 20; ++i) {
    dataset(44, i) = i;
    dataset_oos(44, i) = 2 * i;
    labels[i] = 10.;
    labels_oos[i] = 2 * 10.;
  }

  for (auto recursive_ : recursive) {
    Context context{};

    context.loss = regressorLossFunction::MSE;
    context.childPartitionSize = std::vector<std::size_t>{20, 4};
    context.childNumSteps = std::vector<std::size_t>{5, 6};
    context.childLearningRate = std::vector<double>{1., 1.};
    context.childActivePartitionRatio = std::vector<double>{0.2, 0.3};
    context.childMinLeafSize = std::vector<std::size_t>{1, 1};
    context.childMaxDepth = std::vector<std::size_t>{5, 5};
    context.childMinimumGainSplit = std::vector<double>{0., 0.};
    context.activePartitionRatio = .25;
    context.steps = 1000;
    context.symmetrizeLabels = true;
    context.serializationWindow = 1000;
    context.removeRedundantLabels = false;
    context.rowSubsampleRatio = 1.;
    context.colSubsampleRatio = 1.;  // .75
    context.recursiveFit = recursive_;
    context.serializeModel = false;
    context.serializePrediction = false;
    context.serializeDataset = false;
    context.serializeLabels = false;
    context.serializationWindow = 1000;

    auto regressor =
        GradientBoostRegressor<DecisionTreeRegressorRegressor>(dataset, labels, context);

    regressor.fit();
    regressor.Predict(prediction);

    const double trainError = err(prediction, labels);

    ASSERT_EQ(trainError, 0.);

    regressor.Predict(dataset_oos, prediction_oos);

    for (std::size_t i = 0; i < labels.n_elem; ++i) {
      ASSERT_EQ(labels[i], prediction[i]);
    }
  }
}

TEST(GradientBoostRegressorTest, DISABLED_TestIncrementalRegressorScript) {
  // use folder to determine pwd
  boost::filesystem::path folder("../data/");

  ipstream pipe_stream;
  char dataset_name_train[50] = "Regression/606_fri_c2_1000_10_train";
  char dataset_name_test[50] = "Regression/606_fri_c2_1000_10_test";

  char rg_ex[50];
  char cmd[500];
  sprintf(rg_ex, "\\[%s\\]\\sOOS[\\s]*:[\\s]*.*:[\\s]+\\((.*)\\)", dataset_name_test);

  // Set data directory to test_data directory in the project root
  std::string script_path = IB_utils::resolve_path("src/script/incremental_regressor_fit.sh");
  sprintf(
      cmd,
      "%s 2 10 10 1 1 0.01 0.01 0.5 0.5 0 0 1 1 0 0 %s 10 1 1 1 1 1 -1 1 .2",
      script_path.c_str(),
      dataset_name_train);

  std::array<float, 11> rsquared = {
      0.0433046,
      0.0870988,
      0.118315,
      0.14776,
      0.174561,
      0.189737,
      0.208556,
      0.225737,
      0.24033,
      0.253228,
      0.253228};
  std::string cmd_str{cmd};

  child c(cmd_str, std_out > pipe_stream);

  std::regex ws_re{rg_ex};
  std::smatch sm;
  std::string line;
  std::size_t cnt = 0;

  while (pipe_stream && std::getline(pipe_stream, line) && !line.empty()) {
    if (std::regex_match(line, ws_re)) {
      std::regex_match(line.cbegin(), line.cend(), sm, ws_re);
      std::vector<std::string> res = tokenize(sm[1], ", ");
      ASSERT_EQ(std::stof(res[0]), rsquared[cnt]);
      cnt += 1;
    }
  }
}

TEST(GradientBoostClassifierTest, DISABLED_TestIncrementalClassifierScript) {
  // use folder to determine pwd
  boost::filesystem::path folder{"../data/"};

  ipstream pipe_stream;
  const std::string dataset_name_train = "buggyCrx_train";
  const std::string dataset_name_test = "buggyCrx_test";

  // Set data directory to test_data directory in the project root
  std::string data_dir = "TEST_DATA_DIR=" + IB_utils::resolve_test_data_path("");
  std::string script_path = IB_utils::resolve_path("src/script/incremental_classifier_fit.sh");

  std::string cmd_str =
      script_path +
      " 18 800 250 500 100 250 20 100 75 50 40 35 25 20 "
      "10 7 4 2 1 1 1 1 1 1 1 2 1 3 1 2 1 1 1 1 1 1 1 0.00015 0.00015 0.00015 0.0001 0.0001 0.0001 "
      "0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0002 0.0002 0.0002 0.35 "
      "0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 " +
      dataset_name_train + " 10 8 2.4 1 1 1 4 -4 0";

  child c(cmd_str, std_out > pipe_stream);

  std::string line;
  std::size_t cnt = 0;

  std::array<float, 11> errors = {44, 44, 44, 25, 21, 16, 16, 16, 16, 15, 15};
  std::array<float, 11> precision = {
      .56, .56, .56, .701299, .739726, .794118, .80303, .80303, .80303, .815385, .815385};
  std::array<float, 11> recall = {
      1., 1., 1., .964286, .964286, .964286, .946429, .946429, .946429, .946429, .946429};
  std::array<float, 11> F1 = {
      .717949,
      .717949,
      .717949,
      .81203,
      .837209,
      .870968,
      .868852,
      .868852,
      .868852,
      .876033,
      .876033};
  float imbalance = 0.9604;

  while (pipe_stream && std::getline(pipe_stream, line) && !line.empty()) {
    // Look for lines containing OOS results, but skip header lines
    if (line.find("[buggyCrx_test] OOS:") != std::string::npos &&
        line.find("error, precision, recall, F1, imbalance") == std::string::npos) {
      // Find the parentheses containing the values
      size_t start = line.find("(");
      size_t end = line.find(")", start);
      if (start != std::string::npos && end != std::string::npos && start < end) {
        std::string values_str = line.substr(start + 1, end - start - 1);
        std::vector<std::string> res = tokenize(values_str, ", ");
        if (res.size() >= 5 && cnt < errors.size()) {
          try {
            float error_val = std::stof(res[0]);
            float precision_val = std::stof(res[1]);
            float recall_val = std::stof(res[2]);
            float f1_val = std::stof(res[3]);
            float imbalance_val = std::stof(res[4]);

            ASSERT_EQ(error_val, errors[cnt]);
            ASSERT_EQ(precision_val, precision[cnt]);
            ASSERT_EQ(recall_val, recall[cnt]);
            ASSERT_EQ(f1_val, F1[cnt++]);
            ASSERT_EQ(imbalance_val, imbalance);
          } catch (const std::exception& e) {
            std::cerr << "Error parsing values from line: " << line << std::endl;
            std::cerr << "Values string: " << values_str << std::endl;
            std::cerr << "Exception: " << e.what() << std::endl;
            throw;
          }
        }
      }
    }
  }

  c.wait();

  ASSERT_EQ(1, 1);
}

// TODO: Parameterized tests disabled due to namespace conflict with cereal/rapidjson
// The original code was:
// INSTANTIATE_TEST_SUITE_P(DPSolverTests, DPSolverTestFixture,
//     ::testing::Values(objective_fn::Gaussian, objective_fn::Poisson,
//     objective_fn::RationalScore));

auto main(int argc, char** argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
