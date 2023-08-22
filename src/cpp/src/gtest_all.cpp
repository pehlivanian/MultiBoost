#include <gtest/gtest.h>

#include <vector>
#include <random>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <memory>

#include "utils.hpp"
#include "score2.hpp"
#include "DP.hpp"
#include "gradientboostclassifier.hpp"
#include "gradientboostregressor.hpp"
#include "replay.hpp"

using namespace IB_utils;
using namespace ModelContext;
using namespace ClassifierTypes;
using namespace RegressorTypes;

using dataset_t = Mat<double>;
using dataset_d = Mat<double>;
using labels_t = Row<std::size_t>;
using labels_d = Row<double>;

class DPSolverTestFixture : public ::testing::TestWithParam<objective_fn> {
};

INSTANTIATE_TEST_SUITE_P(DPSolverTests,
			 DPSolverTestFixture,
			 ::testing::Values(
					   objective_fn::Gaussian,
					   objective_fn::Poisson,
					   objective_fn::RationalScore
					   )
			 );
void sort_by_priority(std::vector<float>& a, std::vector<float>& b) {
  std::vector<int> ind(a.size());
  std::iota(ind.begin(), ind.end(), 0);
  
  std::stable_sort(ind.begin(), ind.end(),
		   [&a, &b](int i, int j) {
		     return (a[i]/b[i]) < (a[j]/b[j]);
		   });
  std::vector<float> a_s, b_s;
  for (auto i : ind) {
    a_s.push_back(a[i]);
    b_s.push_back(b[i]);
  }

  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());
	    
}

void sort_partition(std::vector<std::vector<int> > &v) {
  std::sort(v.begin(), v.end(),
	    [](const std::vector<int>& a, const std::vector<int>& b) {
	      return (a.size() < b.size()) || 
		((a.size() == b.size()) && 
		 (a.at(std::distance(a.begin(), std::min_element(a.begin(), a.end()))) <
		  b.at(std::distance(b.begin(), std::min_element(b.begin(), b.end())))));
	    });
}

float rational_obj(std::vector<float> a, std::vector<float> b, int start, int end) {
  if (start == end)
    return 0.;
  float den = 0., num = 0.;
  for (int ind=start; ind<end; ++ind) {
    num += a[ind];
    den += b[ind];
  }
  return num*num/den;
}

std::vector<float> form_levels(int num_true_clusters, float epsilon) {
  std::vector<float> r; r.resize(num_true_clusters);
  float delta = ((2-epsilon/num_true_clusters) - epsilon/num_true_clusters)/(num_true_clusters - 1);
  for(int i=0; i<num_true_clusters; ++i) {
    r[i] = epsilon/static_cast<float>(num_true_clusters) + i*delta;
  }
  return r;
}

std::vector<int> form_splits(int n, int numMixed) {
  std::vector<int> r(numMixed);
  int splitInd = n/numMixed;
  int resid = n - numMixed*splitInd;
  for (int i=0; i<numMixed; ++i)
    r[i] = splitInd;
  for (int i=0; i<resid; ++i)
    r[i]+=1;
  std::partial_sum(r.begin(), r.end(), r.begin(), std::plus<float>());
  return r;
}

std::vector<float> mixture_gaussian_dist(int n,
					 const std::vector<float> b,
					 int numMixed,
					 float sigma,
					 float epsilon) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};

  std::vector<float> a(n);

  std::vector<int> splits = form_splits(n, numMixed);
  std::vector<float> levels = form_levels(numMixed, epsilon);

  for (int i=0; i<n; ++i) {
    int ind = 0;
    while (i >= splits[ind]) {
      ++ind;
    }
    std::normal_distribution<float> dista(levels[ind]*b[i], sigma);
    a[i] = static_cast<float>(dista(mersenne_engine));
  }

  return a;
}

std::vector<float> mixture_poisson_dist(int n, 
					const std::vector<float>& b, 
					int numMixed, 
					float epsilon) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};

  std::vector<float> a(n);
  
  std::vector<int> splits = form_splits(n, numMixed);
  std::vector<float> levels = form_levels(numMixed, epsilon);

  for (int i=0; i<n; ++i) {
    int ind = 0;
    while (i >= splits[ind]) {
      ++ind;
    }
    std::poisson_distribution<int> dista(levels[ind]*b[i]);
    a[i] = static_cast<float>(dista(mersenne_engine));
  }

  return a;

}

float mixture_of_uniforms(int n) {
  int bin = 1;
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<float> distmixer(0., 1.);
  std::uniform_real_distribution<float> dista(0., 1./static_cast<float>(n));
  
  float mixer = distmixer(mersenne_engine);

  while (bin < n) {
    if (mixer < static_cast<float>(bin)/static_cast<float>(n))
      break;
    ++bin;
  }
  return dista(mersenne_engine) + static_cast<float>(bin)-1.;
}


void loadClassifierDatasets(dataset_t& dataset, labels_t& labels) {
  if (!data::Load("/home/charles/Data/sonar_X.csv", dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/sonar_y.csv", labels))
    throw std::runtime_error("Could not load file");
}

void loadRegressorDatasets(dataset_t& dataset, labels_d& labels) {
  if (!data::Load("/home/charles/Data/Regression/1193_BNG_lowbwt_X.csv", dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/Regression/1193_BNG_lowbwt_y.csv", labels))
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
    for (auto &el : a)
      el = dista(gen);
    for (auto &el : b)
      el = distb(gen);

    auto dp = DPSolver(n, 10, a, b, objective_fn::Gaussian, true, true);
    auto opt = dp.get_optimal_subsets_extern();
  
    rowvec a_arma = arma::conv_to<rowvec>::from(a);
    rowvec b_arma = arma::conv_to<rowvec>::from(b);

    for (size_t i=0; i<opt.size(); ++i) {
      auto subset = opt[i];
      // unsorted ratio
      uvec ind1 = arma::conv_to<uvec>::from(subset);
      float res1 = sum(a_arma(ind1))/sum(b_arma(ind1));
      
      // sorted ratio
      std::sort(subset.begin(), subset.end());
      uvec ind2 = arma::conv_to<uvec>::from(subset);
      float res2 = sum(a_arma(ind2))/sum(b_arma(ind2));

      ASSERT_LT(fabs(res1-res2), eps);
      
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
  std::uniform_int_distribution<int> distRow(0, n-1);
  std::uniform_int_distribution<int> distCol(0, n);

  for (auto _ : trials) {

    (void)_;

    std::vector<float> a(n), b(n);
    for (auto &el : a)
      el = dista(gen);
    for (auto &el : b)
      el = distb(gen);

    RationalScoreContext<float>* context_serial = new RationalScoreContext{a, b, n, false, true};
    context_serial->__compute_partial_sums__();
    auto a_sums_serial = context_serial->get_partial_sums_a();
    auto b_sums_serial = context_serial->get_partial_sums_b();

    RationalScoreContext<float>* context_AVX = new RationalScoreContext{a, b, n, false, true};
    context_AVX->__compute_partial_sums_AVX256__();
    auto a_sums_AVX = context_AVX->get_partial_sums_a();
    auto b_sums_AVX = context_AVX->get_partial_sums_b();

    RationalScoreContext<float>* context_parallel = new RationalScoreContext{a, b, n, false, true};
    context_parallel->__compute_partial_sums_parallel__();
    auto a_sums_parallel = context_parallel->get_partial_sums_a();
    auto b_sums_parallel = context_parallel->get_partial_sums_b();
    
    int ind1 = distRow(gen);
    int ind2 = distCol(gen);
    
    ASSERT_EQ(a_sums_serial[ind1][ind2], a_sums_AVX[ind2][ind1]);
    ASSERT_EQ(b_sums_serial[ind1][ind2], b_sums_AVX[ind2][ind1]);
    ASSERT_EQ(a_sums_serial[ind1][ind2], a_sums_parallel[ind1][ind2]);
    ASSERT_EQ(b_sums_serial[ind1][ind2], b_sums_parallel[ind1][ind2]);

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
  std::uniform_int_distribution<int> distRow(0, n-1);
  std::uniform_int_distribution<int> distCol(0, n);

  std::vector<float> a(n), b(n);
  for (auto &el : a)
    el = dista(gen);
  for (auto &el : b)
    el = distb(gen);

  for (auto _ : trials) {

    (void)_;

    PoissonContext<float>* context_serial = new PoissonContext{a, b, n, false, true};
    
    context_serial->__compute_partial_sums__();
    auto a_sums_serial = context_serial->get_partial_sums_a();
    auto b_sums_serial = context_serial->get_partial_sums_b();

    auto partialSums_serial = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.));

    // Based on cached a_sums_, b_sums_ from above    
    for (int i=0; i<n; ++i) {
      for (int j=i; j<n; ++j) {
	partialSums_serial[i][j] = context_serial->__compute_score__(i, j);
      }
    }

    PoissonContext<float>* context_AVX = new PoissonContext{a, b, n, false, true};

    context_AVX->__compute_partial_sums_AVX256__();
    auto a_sums_AVX = context_AVX->get_partial_sums_a();
    auto b_sums_AVX = context_AVX->get_partial_sums_b();
    
    auto partialSums_AVX = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.));    

    // Based on cached a_sums_, b_sums_ from above
    for(int i=0; i<n; ++i) {
      for (int j=0; j<=i; ++j) {
	partialSums_AVX[j][i] = context_AVX->__compute_score__(i, j);
      }
    }
    
    PoissonContext<float>* context_parallel = new PoissonContext{a, b, n, false, true};

    context_parallel->__compute_partial_sums_parallel__();
    auto a_sums_parallel = context_parallel->get_partial_sums_a();
    auto b_sums_parallel = context_parallel->get_partial_sums_b();
    
    auto partialSums_parallel = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.));

    // Based on cached a_sums_, b_sums_ from above
    for (int i=0; i<n; ++i) {
      for (int j=i; j<n; ++j) {
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
      if (ind1_ == ind2_)
	continue;
      if (ind1_ >= ind2_)
	std::swap(ind1_, ind2_);

      // ... but scores aren't
      ASSERT_EQ(partialSums_serial[ind1_][ind2_], partialSums_AVX[ind1_][ind2_]);
      ASSERT_EQ(partialSums_serial[ind1_][ind2_], partialSums_parallel[ind1_][ind2_]);
    }
  }
}

TEST_P(DPSolverTestFixture, TestConsecutiveProperty) {
  // Case (n,T) = (100,20)
  int n = 100, T = 20;
  
  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(1., 10.);

  std::vector<float> a(n), b(n);

  objective_fn objective = GetParam();
  for (size_t i=0; i<5; ++i) {
    for (auto &el : a)
      el = dist(gen);
    for (auto &el : b)
      el = dist(gen);
    
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
	sum = std::accumulate(v.begin()+1, v.end(), 0);
      }
    }
    
    // We ignored the first element as adjacent_difference has unintuitive
    // result for first element
    ASSERT_EQ(sum, v.size()-1);
  }
}

TEST_P(DPSolverTestFixture, TestOptimalityWithRandomPartitions) {
  int NUM_CASES = 1000, NUM_SUBCASES = 500, T = 3;

  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_int_distribution<int> distn(5, 50);
  std::uniform_real_distribution<float> dista( 1., 10.);
  std::uniform_real_distribution<float> distb( 1., 10.);

  std::vector<float> a, b;

  for (int case_num=0; case_num<NUM_CASES; ++case_num) {

    int n = distn(gen);
    a.resize(n); b.resize(n);

    for (auto &el : a)
      el = dista(gen);
    for (auto &el : b)
      el = distb(gen);
  
    sort_by_priority(a, b);

    ASSERT_GE(n, 5);
    ASSERT_LE(n, 100);

    objective_fn objective = GetParam();

    auto dp = DPSolver(n, T, a, b, objective, true, true);
    auto dp_opt = dp.get_optimal_subsets_extern();
    auto scores = dp.get_score_by_subset_extern();

    for (int subcase_num=0; subcase_num<NUM_SUBCASES; ++subcase_num) {
      std::uniform_int_distribution<int> distm(5, n);
      
      int m1 = distm(gen), m21;
      int m2 = distm(gen), m22;
      if ((m1 == m2) || (m1 == n) || (m2 == n))
	continue;
      m21 = std::min(m1, m2);
      m22 = std::max(m1, m2);
      
      std::unique_ptr<ParametricContext<float>> context;

      switch (objective) {
      case objective_fn::Gaussian :
	context = std::make_unique<GaussianContext<float>>(a, b, n, true, false);
      case objective_fn::Poisson :
	context = std::make_unique<PoissonContext<float>>(a, b, n, true, false);
      case objective_fn::RationalScore :
	context = std::make_unique<RationalScoreContext<float>>(a, b, n, true, false);
      }

      context->__compute_partial_sums__();
	
      float rand_score, dp_score;
      rand_score = context->__compute_score__(0, m21) + 
	context->__compute_score__(m21, m22) + 
	context->__compute_score__(m22, n);
      dp_score = context->__compute_score__(dp_opt[0][0], 1+dp_opt[0][dp_opt[0].size()-1]) + 
	context->__compute_score__(dp_opt[1][0], 1+dp_opt[1][dp_opt[1].size()-1]) + 
	context->__compute_score__(dp_opt[2][0], 1+dp_opt[2][dp_opt[2].size()-1]);

      if ((dp_score - rand_score) > std::numeric_limits<float>::epsilon()) {
	ASSERT_LE(rand_score, dp_score);
      }
    }
  }
}

TEST(DPSolverTest, TestParallelScoresMatchSerialScores) {
  using namespace Objectives;

  std::size_t n = 100;
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

    (void)_;

    GaussianContext<float>* context = new GaussianContext{a, b, n, false, true};
    
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
      
      (void)_;

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

TEST(DPSolverTest, TestBaselines ) {

  std::vector<float> a{0.0212651 , -0.20654906, -0.20654906, -0.20654906, -0.20654906,
      0.0212651 , -0.20654906,  0.0212651 , -0.20654906,  0.0212651 ,
      -0.20654906,  0.0212651 , -0.20654906, -0.06581402,  0.0212651 ,
      0.03953075, -0.20654906,  0.16200014,  0.0212651 , -0.20654906,
      0.20296943, -0.18828341, -0.20654906, -0.20654906, -0.06581402,
      -0.20654906,  0.16200014,  0.03953075, -0.20654906, -0.20654906,
      0.03953075,  0.20296943, -0.20654906,  0.0212651 ,  0.20296943,
      -0.20654906,  0.0212651 ,  0.03953075, -0.20654906,  0.03953075};
  std::vector<float> b{0.22771114, 0.21809504, 0.21809504, 0.21809504, 0.21809504,
      0.22771114, 0.21809504, 0.22771114, 0.21809504, 0.22771114,
      0.21809504, 0.22771114, 0.21809504, 0.22682739, 0.22771114,
      0.22745816, 0.21809504, 0.2218354 , 0.22771114, 0.21809504,
      0.218429  , 0.219738  , 0.21809504, 0.21809504, 0.22682739,
      0.21809504, 0.2218354 , 0.22745816, 0.21809504, 0.21809504,
      0.22745816, 0.218429  , 0.21809504, 0.22771114, 0.218429  ,
      0.21809504, 0.22771114, 0.22745816, 0.21809504, 0.22745816};

  std::vector<std::vector<int> > expected = {
    {1, 2, 3, 4, 6, 8, 10, 12, 16, 19, 21, 22, 23, 25, 28, 29, 32, 35, 38},
    {13, 24}, 
    {0, 5, 7, 9, 11, 14, 15, 18, 27, 30, 33, 36, 37, 39},
    {17, 26}, 
    {20, 31, 34}
  };

  std::vector<float> a1{2.26851454, 2.86139335, 5.51314769, 6.84829739, 6.96469186, 7.1946897,
      9.80764198, 4.2310646};
  std::vector<float> b1{3.43178016, 3.92117518, 7.29049707, 7.37995406, 4.80931901, 4.38572245,
      3.98044255, 0.59677897};

  auto dp1 = DPSolver(8, 3, a1, b1, objective_fn::Poisson, false, true);
  auto opt1 = dp1.get_optimal_subsets_extern();
  
  auto dp = DPSolver(40, 5, a, b, objective_fn::Gaussian, true, true);
  auto opt = dp.get_optimal_subsets_extern();

  for (size_t i=0; i<expected.size(); ++i) {
    auto expected_subset = expected[i], opt_subset = opt[i];
    std::sort(opt_subset.begin(), opt_subset.end());
    ASSERT_EQ(expected_subset.size(), opt_subset.size());
    for(size_t j=0; j<expected_subset.size(); ++j) {
      ASSERT_EQ(expected_subset[j], opt_subset[j]);
    }
  }
}


TEST(GradientBoostClassifierTest, TestAggregateClassifierRecursiveReplay) {
  
  std::vector<bool> trials = {false, true};
  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;


  loadClassifierDatasets(dataset, labels);
  data::Split(dataset, 
	      labels,
	      trainDataset,
	      testDataset,
	      trainLabels,
	      testLabels, 0.2);

  Context context{};
  
  context.loss = lossFunction::BinomialDeviance;
  context.childPartitionSize = std::vector<std::size_t>{11, 5};
  context.childNumSteps = std::vector<std::size_t>{21, 2};
  context.childLearningRate = std::vector<double>{.001, .001};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.partitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = 1.; // .75
  context.serializeModel = true;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED;
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;

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
    Row<double> liveTrainPrediction;
    EXPECT_THROW(classifier.Predict(trainDataset, liveTrainPrediction), 
		 predictionAfterClearedClassifiersException );

    // Use latestPrediction_ instead
    classifier.Predict(liveTrainPrediction);

    // Get index, fldr
    std::string indexName = classifier.getIndexName();
    boost::filesystem::path fldr = classifier.getFldr();

    // Use replay to predict IS based on archive classifier
    Row<double> archiveTrainPrediction;
    Replay<double, DecisionTreeClassifier>::Classify(indexName, trainDataset, archiveTrainPrediction, fldr);

    for (std::size_t i=0; i<liveTrainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(liveTrainPrediction[i]-archiveTrainPrediction[i]), eps);

    // Predict OOS with live classifier
    Row<double> liveTestPrediction;
    context.serializeModel = false;
    context.serializePrediction = false;
    secondClassifier = T(trainDataset, trainLabels, context);
    secondClassifier.fit();
    secondClassifier.Predict(testDataset, liveTestPrediction);

    // Use replay to predict OOS based on archive classifier
    Row<double> archiveTestPrediction;
    Replay<double, DecisionTreeClassifier>::Classify(indexName, testDataset, archiveTestPrediction, fldr, true);

    for (std::size_t i=0; i<liveTestPrediction.size(); ++i) {
      ASSERT_LE(fabs(liveTestPrediction[i]-archiveTestPrediction[i]), eps);
    }

  }

}

TEST(GradientBoostClassifierTest, TestInSamplePredictionMatchesLatestPrediction) {
  std::vector<bool> trials = {false, true};

  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;

  loadClassifierDatasets(dataset, labels);
  data::Split(dataset,
	      labels,
	      trainDataset,
	      testDataset,
	      trainLabels,
	      testLabels, 0.2);

  Context context{};
  
  context.loss = lossFunction::BinomialDeviance;
  context.childPartitionSize = std::vector<std::size_t>{11, 5};
  context.childNumSteps = std::vector<std::size_t>{5, 2};
  context.childLearningRate = std::vector<double>{.001, .001};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.partitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45; // .75
  context.serializeModel = false;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED;
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;

  using T = GradientBoostClassifier<DecisionTreeClassifier>;

  for (auto recursive : trials) {
    
    context.recursiveFit = recursive;
    float eps = std::numeric_limits<float>::epsilon();

    // Fit classifier
    T classifier, newClassifier;
    classifier = T(trainDataset, trainLabels, testDataset, testLabels, context);
    classifier.fit();

    // IS prediction - live classifier
    Row<double> liveTrainPrediction;
    classifier.Predict(liveTrainPrediction);
    
    // IS lastestPrediction_ - archive classifier
    Row<double> latestPrediction = classifier.getLatestPrediction();

    for (std::size_t i=0; i<liveTrainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(liveTrainPrediction[i]-latestPrediction[i]), eps);
  }
  
}

TEST(GradientBoostClassifierTest, TestAggregateClassifierRecursiveRoundTrips) {

  std::vector<bool> trials = {true};
  
  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;
  
  loadClassifierDatasets(dataset, labels);
  data::Split(dataset, 
	      labels, 
	      trainDataset, 
	      testDataset, 
	      trainLabels, 
	      testLabels, 0.2);

  Context context{};
  
  context.loss = lossFunction::BinomialDeviance;
  context.childPartitionSize = std::vector<std::size_t>{11, 5};
  context.childNumSteps = std::vector<std::size_t>{5, 2};
  context.childLearningRate = std::vector<double>{.001, .001};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.partitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45; // .75
  context.recursiveFit = false;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED;
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;

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

    Row<double> trainPrediction, trainNewPrediction, testPrediction, testNewPrediction;
    classifier.Predict(testDataset, testPrediction);
    classifier.Predict(trainDataset, trainPrediction);
    newClassifier.Predict(testDataset, testNewPrediction);
    newClassifier.Predict(trainDataset, trainNewPrediction);
    
    ASSERT_EQ(testPrediction.n_elem, testNewPrediction.n_elem);
    ASSERT_EQ(trainPrediction.n_elem, trainNewPrediction.n_elem);
    
    float eps = std::numeric_limits<float>::epsilon();
    for (std::size_t i=0; i<testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i]-testNewPrediction[i]), eps);
    for (std::size_t i=0; i<trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i]-trainNewPrediction[i]), eps);

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
  data::Split(dataset, 
	      labels, 
	      trainDataset, 
	      testDataset, 
	      trainLabels, 
	      testLabels, 0.2);

  using T = DecisionTreeClassifierType;
  using IArchiveType = cereal::BinaryInputArchive;
  using OArchiveType = cereal::BinaryOutputArchive;

  for (auto _ : trials) {

    (void)_;

    T classifier, newClassifier;
    classifier = T(trainDataset, 
		   trainLabels,
		   numClasses,
		   minLeafSize,
		   minimumGainSplit,
		   maxDepth);

    std::string fileName = dumps<T, IArchiveType, OArchiveType>(classifier, SerializedType::CLASSIFIER);
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
    for (std::size_t i=0; i<testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i]-testNewPrediction[i]), eps);
    for (std::size_t i=0; i<trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i]-trainNewPrediction[i]), eps);

  }
}

TEST(GradientBoostClassifierTest, TestAggregateClassifierNonRecursiveRoundTrips) {
  
  int numTrials = 1;
  std::vector<bool> trials(numTrials);
  
  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;

  loadClassifierDatasets(dataset, labels);
  data::Split(dataset, 
	      labels, 
	      trainDataset, 
	      testDataset, 
	      trainLabels, 
	      testLabels, 0.2);

  Context context{};
  
  context.loss = lossFunction::BinomialDeviance;
  context.childPartitionSize = std::vector<std::size_t>{10};
  context.childNumSteps = std::vector<std::size_t>{8};
  context.childLearningRate = std::vector<double>{.001};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.partitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45; // .75
  context.recursiveFit = false;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED;
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;

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

    Row<double> trainPrediction, trainNewPrediction, testPrediction, testNewPrediction;
    classifier.Predict(testDataset, testPrediction);
    classifier.Predict(trainDataset, trainPrediction);
    newClassifier.Predict(testDataset, testNewPrediction);
    newClassifier.Predict(trainDataset, trainNewPrediction);
    
    ASSERT_EQ(testPrediction.n_elem, testNewPrediction.n_elem);
    ASSERT_EQ(trainPrediction.n_elem, trainNewPrediction.n_elem);
    
    float eps = std::numeric_limits<float>::epsilon();
    for (std::size_t i=0; i<testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i]-testNewPrediction[i]), eps);
    for (std::size_t i=0; i<trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i]-trainNewPrediction[i]), eps);

  }
}

TEST(GradientBoostRegressorTest, TestContextWrittenWithCorrectValues) {

  using CerealT = Context;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;
  
  Context context_archive;

  std::string fileName = "./ctx_cls.dat";
  std::string cmd = "/home/charles/src/C++/sandbox/Inductive-Boost/build/createContext ";

  cmd += "--loss 0 --partitionSize 41 --partitionRatio .25 ";
  cmd += "--partitionSizeMethod 0 --learningRateMethod 0 ";
  cmd += "--learningRate 1. --steps 4122 --symmetrizeLabels false ";
  cmd += "--childPartitionSize 1000 500 250 125 10 1 --childNumSteps 100 10 5 2 1 1 ";
  cmd += "--childMinLeafSize 10 10 5 2 1 ";
  cmd += "--fileName ./ctx_cls.dat";

  exec(cmd);

  loads<CerealT, CerealIArch, CerealOArch>(context_archive, fileName);

  // Uset set values
  ASSERT_EQ(context_archive.loss, lossFunction::MSE);
  ASSERT_EQ(context_archive.partitionSize, 41);
  ASSERT_EQ(context_archive.steps, 4122);
  ASSERT_EQ(context_archive.partitionRatio, .25);
  ASSERT_EQ(context_archive.partitionSizeMethod, PartitionSize::PartitionSizeMethod::FIXED);
  ASSERT_EQ(context_archive.learningRateMethod, LearningRate::LearningRateMethod::FIXED);
  ASSERT_EQ(context_archive.symmetrizeLabels, false);
  
  std::vector<std::size_t> targetPartitionSize = {1000, 500, 250, 125, 10, 1};
  std::vector<std::size_t> targetNumSteps = {100, 10, 5, 2, 1, 1};
  std::vector<std::size_t> targetMinLeafSize = {10, 10, 5, 2, 1};

  for (std::size_t i=0; i<context_archive.childPartitionSize.size(); ++i) {
    ASSERT_EQ(context_archive.childPartitionSize[i], targetPartitionSize[i]);
  }

  for (std::size_t i=0; i<context_archive.childNumSteps.size(); ++i) {
    ASSERT_EQ(context_archive.childNumSteps[i], targetNumSteps[i]);
  }

  for (std::size_t i=0; i<context_archive.childMinLeafSize.size(); ++i) {
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
  std::string cmd = "/home/charles/src/C++/sandbox/Inductive-Boost/build/createContext ";

  cmd += "--loss 1 --partitionSize 6 --partitionRatio .25 ";
  cmd += "--partitionSizeMethod 1 --learningRateMethod 2 ";
  cmd += "--learningRate .01 --steps 1010 --symmetrizeLabels true ";
  cmd += "--childMinLeafSize 10 10 5 2 1 ";
  cmd += "--fileName ctx_reg.dat";

  exec(cmd);

  loads<CerealT, CerealIArch, CerealOArch>(context_archive, fileName);

  // User set values
  ASSERT_EQ(context_archive.loss, lossFunction::BinomialDeviance);
  ASSERT_EQ(context_archive.partitionSize, 6);
  ASSERT_EQ(context_archive.steps, 1010);
  ASSERT_EQ(context_archive.partitionRatio, .25);
  ASSERT_EQ(context_archive.partitionSizeMethod, PartitionSize::PartitionSizeMethod::FIXED_PROPORTION);
  ASSERT_EQ(context_archive.learningRateMethod, LearningRate::LearningRateMethod::DECREASING);

  std::vector<std::size_t> targetMinLeafSize = {10, 10, 5, 2, 1};
  for (std::size_t i=0; i<context_archive.childMinLeafSize.size(); ++i) {
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
  
  context.loss = lossFunction::BinomialDeviance;
  context.partitionSize = partitionSize;
  context.partitionRatio = .25;
  context.learningRate = .01;
  context.steps = 21;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45; // .75
  context.recursiveFit = false;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED;
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;
  context.minLeafSize = minLeafSize;
  context.maxDepth = maxDepth;
  context.minimumGainSplit = minimumGainSplit;

  std::string binFileName = "gtest__Context.dat";
  
  using CerealT = Context;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  dumps<CerealT, CerealIArch, CerealOArch>(context, binFileName);
  loads<CerealT, CerealIArch, CerealOArch>(context_archive, binFileName);

  ASSERT_EQ(context_archive.loss, lossFunction::BinomialDeviance);
  ASSERT_EQ(context_archive.loss, context.loss);

  ASSERT_EQ(context_archive.partitionSize, 10);
  ASSERT_EQ(context_archive.partitionSize, context.partitionSize);

  ASSERT_EQ(context_archive.partitionSizeMethod, PartitionSize::PartitionSizeMethod::FIXED);
  ASSERT_EQ(context_archive.partitionSizeMethod, context.partitionSizeMethod);

}

TEST(GradientBoostClassifierTest, TestWritePrediction) {

  std::vector<bool> trials = {false, true};
  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;

  loadClassifierDatasets(dataset, labels);
  data::Split(dataset, 
	      labels,
	      trainDataset,
	      testDataset,
	      trainLabels,
	      testLabels, 0.2);

  Context context{};
  
  context.loss = lossFunction::BinomialDeviance;
  context.childPartitionSize = std::vector<std::size_t>{11, 6};
  context.childNumSteps = std::vector<std::size_t>{4, 4};
  context.childLearningRate = std::vector<double>{.001, .001};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.partitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45; // .75
  context.serializeModel = true;
  context.serializePrediction = true;
  context.serializeColMask = true;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED;
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;

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
    Row<double> liveTrainPrediction;
    EXPECT_THROW(classifier.Predict(trainDataset, liveTrainPrediction), predictionAfterClearedClassifiersException );

    // Use latestPrediction_ instead
    classifier.Predict(liveTrainPrediction);
    
    // Use replay to predict IS based on archive classifier
    Row<double> archiveTrainPrediction1, archiveTrainPrediction2;
    std::string indexName = classifier.getIndexName();
    boost::filesystem::path fldr = classifier.getFldr();

    // Method 1
    Replay<double, DecisionTreeClassifier>::Classify(indexName, trainDataset, archiveTrainPrediction1, fldr);

    // Method 2
    Replay<double, DecisionTreeClassifier>::Classify(indexName, archiveTrainPrediction2, fldr);

    for (std::size_t i=0; i<liveTrainPrediction.n_elem; ++i) {
      ASSERT_LE(fabs(liveTrainPrediction[i]-archiveTrainPrediction1[i]), eps);
      ASSERT_LE(fabs(liveTrainPrediction[i]-archiveTrainPrediction2[i]), eps);
    }

  }

}

TEST(GradientBoostRegressorTest, TestPredictionRoundTrip) {

  std::vector<bool> trials = {false};
  dataset_d dataset, trainDataset, testDataset;
  labels_d labels, trainLabels, testLabels;

  loadRegressorDatasets(dataset, labels);
  data::Split(dataset,
	      labels,
	      trainDataset,
	      testDataset,
	      trainLabels,
	      testLabels, 0.85);

  Context context{};
  
  context.loss = lossFunction::MSE;
  context.childPartitionSize = std::vector<std::size_t>{11};
  context.childNumSteps = std::vector<std::size_t>{21};
  context.childLearningRate = {1., 1., 1.};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.partitionRatio = .25;
  context.baseSteps = 35;
  context.quietRun = true;
  context.symmetrizeLabels = false;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = 1.; // .75
  context.serializeModel = true;
  context.serializePrediction = true;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED;
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;

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
    
    for (std::size_t i=0; i<prediction.n_elem; ++i) {
      ASSERT_LE(fabs(prediction[i]-archivePrediction[i]), eps);
      ASSERT_LE(fabs(prediction[i]-archivePrediction[i]), eps);
    }

    // We have archive prediction for an intermiediate point [1..22..106]
    // Create a regressor over the entire period and fit
    context.childNumSteps = std::vector<std::size_t>{35};
    context.baseSteps = 35;
    T secondRegressor;
    secondRegressor = T(trainDataset, trainLabels, testDataset, testLabels, context);
    secondRegressor.fit();
    secondRegressor.Predict(secondPrediction);

    // Compare with 24 steps from archivePrediction
    context.childNumSteps = std::vector<std::size_t>{14};
    context.baseSteps = 35;
    T archiveRegressor = T(trainDataset, trainLabels, testDataset, testLabels, archivePrediction, context);
    archiveRegressor.fit();
    archiveRegressor.Predict(archivePrediction);

    for (std::size_t i=0; i<secondPrediction.size(); ++i) {
      ASSERT_LE(fabs(secondPrediction[i]-archivePrediction[i]), eps);
    }
    
  }

}

TEST(GradientBoostClassifierTest, TestPredictionRoundTrip) {

  std::vector<bool> trials = {false};
  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;

  loadClassifierDatasets(dataset, labels);
  data::Split(dataset, 
	      labels,
	      trainDataset,
	      testDataset,
	      trainLabels,
	      testLabels, 0.2);

  Context context{};
  
  context.loss = lossFunction::BinomialDeviance;
  context.childPartitionSize = std::vector<std::size_t>{11};
  context.childNumSteps = std::vector<std::size_t>{114};
  context.childLearningRate = std::vector<double>{.001};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.partitionRatio = .25;
  context.quietRun = true;
  context.baseSteps = 214;
  context.symmetrizeLabels = true;
  context.quietRun = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = 1.; // .75
  context.serializeModel = true;
  context.serializePrediction = true;
  context.serializeColMask = false;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED;
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;

  context.serializationWindow = 100;

  using T = GradientBoostClassifier<DecisionTreeClassifier>;

  for (auto recursive : trials) {

    context.recursiveFit = recursive;

    double eps = std::numeric_limits<double>::epsilon();

    Row<double> prediction, archivePrediction, newPrediction, secondPrediction;

    // Fit classifier
    T classifier;
    classifier = T(trainDataset, trainLabels, testDataset, testLabels, context);
    classifier.fit();


    // Test classifier prediction matches reloaded latestPrediction from same classifier
    classifier.Predict(prediction);

    std::string indexName = classifier.getIndexName();
    boost::filesystem::path fldr = classifier.getFldr();
    
    readPrediction(indexName, archivePrediction, fldr);

    for (std::size_t i=0; i<prediction.n_elem; ++i) {
      ASSERT_LE(fabs(prediction[i]-archivePrediction[i]), eps);
    }

    // We have archive prediction for an intermediate point [1..114..214]
    // Create a classifier over the entire period and fit
    context.childNumSteps = std::vector<std::size_t>{214};
    context.baseSteps = 214;
    T secondClassifier;
    secondClassifier = T(trainDataset, trainLabels, testDataset, testLabels, context);
    secondClassifier.fit();
    secondClassifier.Predict(secondPrediction);

    // Compare with 100 steps from archivePrediction
    context.childNumSteps = std::vector<std::size_t>{100};
    context.baseSteps = 214;
    T archiveClassifier = T(trainDataset, trainLabels, testDataset, testLabels, archivePrediction, context);
    archiveClassifier.fit();
    archiveClassifier.Predict(archivePrediction);

    if (false)
      eps = 1.5;
    else
      eps = std::numeric_limits<float>::epsilon();

    for (std::size_t i=0; i<secondPrediction.size(); ++i) {
      ASSERT_LE(fabs(secondPrediction[i]-archivePrediction[i]), eps);
    }
  
  }
  
}

TEST(GradientBoostRegressorTest, TestChildSerializationRoundTrips) {

  int numTrials = 1;
  std::vector<bool> trials(numTrials);

  std::size_t minLeafSize = 1;
  double minimumGainSplit = 0.;
  std::size_t maxDepth = 10;

  dataset_d dataset, trainDataset, testDataset;
  labels_d labels, trainLabels, testLabels;

  loadRegressorDatasets(dataset, labels);
  data::Split(dataset, 
	      labels, 
	      trainDataset, 
	      testDataset, 
	      trainLabels, 
	      testLabels, 0.9);

  using T = DecisionTreeRegressorRegressorType;
  using IArchiveType = cereal::BinaryInputArchive;
  using OArchiveType = cereal::BinaryOutputArchive;

  for (auto _ : trials) {

    (void)_;

    T regressor, newRegressor;
    regressor = T(trainDataset, 
		  trainLabels,
		  minLeafSize,
		  minimumGainSplit,
		  maxDepth);

    std::string fileName = dumps<T, IArchiveType, OArchiveType>(regressor, SerializedType::REGRESSOR);
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
    for (std::size_t i=0; i<testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i]-testNewPrediction[i]), eps);
    for (std::size_t i=0; i<trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i]-trainNewPrediction[i]), eps);

  }
}


TEST(GradientBoostRegressorTest, TestInSamplePredictionMatchesLatestPrediction) {

  std::vector<bool> trials = {false, true};

  dataset_d dataset, trainDataset, testDataset;
  labels_d labels, trainLabels, testLabels;

  loadRegressorDatasets(dataset, labels);
  data::Split(dataset,
	      labels,
	      trainDataset,
	      testDataset,
	      trainLabels,
	      testLabels, 0.9);

  Context context{};
  
  context.loss = lossFunction::MSE;
  context.childPartitionSize = std::vector<std::size_t>{11, 5, 2};
  context.childNumSteps = std::vector<std::size_t>{14, 1, 2};
  context.childLearningRate = std::vector<double>{1., 1., 1.};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0., 0.};
  context.partitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45; // .75
  context.serializeModel = false;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED;
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;

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

    for (std::size_t i=0; i<liveTrainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(liveTrainPrediction[i]-latestPrediction[i]), eps);
  }
  
}

TEST(GradientBoostRegressorTest, TestAggregateRegressorRecursiveReplay) {
  
  std::vector<bool> trials = {true, false};
  dataset_d dataset, trainDataset, testDataset;
  labels_d labels, trainLabels, testLabels;

  loadRegressorDatasets(dataset, labels);

  data::Split(dataset, 
	      labels,
	      trainDataset,
	      testDataset,
	      trainLabels,
	      testLabels, 0.95);

  Context context{};
  
  context.loss = lossFunction::MSE;
  context.childPartitionSize = std::vector<std::size_t>{11, 5, 2};
  context.childNumSteps = std::vector<std::size_t>{14, 1, 2};
  context.childLearningRate = std::vector<double>{1., 1., 1.};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0., 0.};
  context.partitionRatio = .25;
  context.quietRun = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = 1.; // .75
  context.serializeModel = true;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED;
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;
  
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
    EXPECT_THROW(regressor.Predict(trainDataset, liveTrainPrediction),
		 predictionAfterClearedClassifiersException);

    // Use latestPrediction_ instead
    regressor.Predict(liveTrainPrediction);

    // Get index
    std::string indexName = regressor.getIndexName();
    boost::filesystem::path fldr = regressor.getFldr();
    // Use replay to predict IS based on archive regressor
    Row<double> archiveTrainPrediction;
    Replay<double, DecisionTreeRegressorRegressor>::Predict(indexName, 
							    trainDataset, 
							    archiveTrainPrediction,
							    fldr);
  
    for (std::size_t i=0; i<liveTrainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(liveTrainPrediction[i]-archiveTrainPrediction[i]), eps);

    // Predict OOS with live regressor
    Row<double> liveTestPrediction;
    context.serializeModel = false;
    context.serializePrediction = false;
    secondRegressor = T(trainDataset, trainLabels, context);
    secondRegressor.fit();
    secondRegressor.Predict(testDataset, liveTestPrediction);

    // Use replay to predict OOS based on archive classifier
    Row<double> archiveTestPrediction;
    Replay<double, DecisionTreeRegressorRegressor>::Predict(indexName, 
							    testDataset, 
							    archiveTestPrediction,
							    fldr);

    for (std::size_t i=0; i<liveTestPrediction.size(); ++i) {
      ASSERT_LE(fabs(liveTestPrediction[i]-archiveTestPrediction[i]), eps);
    }
  
  }
  
}

TEST(GradientBoostRegressorTest, TestAggregateRegressorNonRecursiveRoundTrips) {
  
  int numTrials = 1;
  std::vector<bool> trials(numTrials);
  
  dataset_d dataset, trainDataset, testDataset;
  labels_d labels, trainLabels, testLabels;

  loadRegressorDatasets(dataset, labels);
  data::Split(dataset, 
	      labels, 
	      trainDataset, 
	      testDataset, 
	      trainLabels, 
	      testLabels, 0.85);

  Context context{};
  
  context.loss = lossFunction::MSE;
  context.childPartitionSize = std::vector<std::size_t>{10, 2};
  context.childNumSteps = std::vector<std::size_t>{14, 2};
  context.childLearningRate = std::vector<double>{1., 1.};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1};
  context.childMaxDepth = std::vector<std::size_t>{5, 5};
  context.childMinimumGainSplit = std::vector<double>{0., 0.};
  context.partitionRatio = .25;
  context.quietRun = true;
  context.symmetrizeLabels = false;
  context.removeRedundantLabels = false;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45; // .75
  context.recursiveFit = false;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED;
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;

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
    for (std::size_t i=0; i<testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i]-testNewPrediction[i]), eps);
    for (std::size_t i=0; i<trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i]-trainNewPrediction[i]), eps);

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
  
  int rows=1000, cols=20;
  
  dataset = Mat<double>(rows, cols, fill::randu);
  labels = Row<double>(cols);

  for (std::size_t i=0; i<4; ++i) {
    dataset(444,i) = i;
    labels[i] = 14.;
  }
  for (std::size_t i=4; i<8; ++i) {
    dataset(444,i) = i;
    labels[i] = -1.45;
  }
  for (std::size_t i=8; i<12; ++i) {
    dataset(444,i) = i;
    labels[i] = -2.077;
  }
  for (std::size_t i=12; i<16; ++i) {
    dataset(444,i) = i;
    labels[i] = 2.077;
  }
  for (std::size_t i=16; i<20; ++i) {
    dataset(444,i) = i;
    labels[i] = 2.24;
  }

  for (auto recursive_ : recursive) {
    Context context{};
    
    context.loss = lossFunction::MSE;
    context.childPartitionSize = std::vector<std::size_t>{10, 4};
    context.childNumSteps = std::vector<std::size_t>{5, 6};
    context.childLearningRate = std::vector<double>{1., 1.};
    context.childMinLeafSize = std::vector<std::size_t>{1, 1};
    context.childMaxDepth = std::vector<std::size_t>{5, 5};
    context.childMinimumGainSplit = std::vector<double>{0., 0.};    
    context.partitionRatio = .25;
    context.baseSteps = 1000;
    context.symmetrizeLabels = true;
    context.serializationWindow = 1000;
    context.removeRedundantLabels = false;
    context.rowSubsampleRatio = 1.;
    context.colSubsampleRatio = 1.; // .75
    context.recursiveFit = recursive_;
    context.serializeModel = false;
    context.serializePrediction = false;
    context.serializeDataset = false;
    context.serializeLabels = false;
    context.serializationWindow = 1000;
    context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED; // INCREASING
    context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;    // DECREASING
    context.stepSizeMethod = StepSize::StepSizeMethod::LOG;	
    
    auto regressor = GradientBoostRegressor<DecisionTreeRegressorRegressor>(dataset,
									    labels,
									    context);
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
  
  int rows=50, cols=20;
  
  dataset = Mat<double>(rows, cols, fill::randu);
  labels = Row<double>(cols);

  dataset_oos = Mat<double>(rows, cols, fill::randu);
  labels_oos = Row<double>(cols);

  for (std::size_t i=0; i<4; ++i) {
    dataset(44,i) = i;
    dataset_oos(44,i) = i;
    labels[i] = 2.;
    labels_oos[i] = 4.;
  }
  for (std::size_t i=4; i<8; ++i) {
    dataset(44,i) = i;
    dataset_oos(44,i) = i;
    labels[i] = 4.;
    labels_oos[i] = 8.;
  }
  for (std::size_t i=8; i<12; ++i) {
    dataset(44,i) = i;
    dataset_oos(44,i) = i;
    labels[i] = 6.;
    labels_oos[i] = 12.;
  }
  for (std::size_t i=12; i<16; ++i) {
    dataset(44,i) = i;
    dataset_oos(44,i) = i;
    labels[i] = 8.;
    labels_oos[i] = 16.;
  }
  for (std::size_t i=16; i<20; ++i) {
    dataset(44,i) = i;
    dataset_oos(44,i) = i;
    labels[i] = 10.;
    labels_oos[i] = 20.;
  }

  for (auto recursive_ : recursive) {
    Context context{};
    
    context.loss = lossFunction::MSE;
    context.childPartitionSize = std::vector<std::size_t>{10, 4};
    context.childNumSteps = std::vector<std::size_t>{5, 6};
    context.childLearningRate = std::vector<double>{1., 1.};
    context.childMinLeafSize = std::vector<std::size_t>{1, 1};
    context.childMaxDepth = std::vector<std::size_t>{5, 5};
    context.childMinimumGainSplit = std::vector<double>{0., 0.};
    context.partitionRatio = .25;
    context.baseSteps = 1000;
    context.symmetrizeLabels = true;
    context.serializationWindow = 1000;
    context.removeRedundantLabels = false;
    context.rowSubsampleRatio = 1.;
    context.colSubsampleRatio = 1.; // .75
    context.recursiveFit = recursive_;
    context.serializeModel = false;
    context.serializePrediction = false;
    context.serializeDataset = false;
    context.serializeLabels = false;
    context.serializationWindow = 1000;
    context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED; // INCREASING
    context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;    // DECREASING
    context.stepSizeMethod = StepSize::StepSizeMethod::LOG;	
    
    auto regressor = GradientBoostRegressor<DecisionTreeRegressorRegressor>(dataset,
									    labels,
									    context);

    regressor.fit();
    regressor.Predict(prediction);

    const double trainError = err(prediction, labels);
    
    ASSERT_EQ(trainError, 0.);

    regressor.Predict(dataset_oos, prediction_oos);
    
    for (std::size_t i=0; i<labels.n_elem; ++i) {
      ASSERT_EQ(labels[i], prediction[i]);
    }
  }
  
}

auto main(int argc, char **argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
