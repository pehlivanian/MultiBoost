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

using namespace IB_utils;
using namespace ClassifierContext;
using namespace ClassifierTypes;

using dataset_t = Mat<double>;
using labels_t = Row<std::size_t>;

void loadDatasets(dataset_t& dataset, labels_t& labels) {
  if (!data::Load("/home/charles/Data/sonar_X.csv", dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/sonar_y.csv", labels))
    throw std::runtime_error("Could not load file");
}

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

TEST(GradientBoostClassifierTest, TestAggregateClassifierNonRecursiveRoundTrips) {
  
  int numTrials = 1;
  std::vector<bool> trials(numTrials);

  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;

  loadDatasets(dataset, labels);
  data::Split(dataset, 
	      labels, 
	      trainDataset, 
	      testDataset, 
	      trainLabels, 
	      testLabels, 0.2);

  std::size_t minLeafSize = 1;
  double minimumGainSplit = 0.;
  std::size_t maxDepth = 10;
  std::size_t partitionSize = 10;

  Context context{};
  
  context.loss = lossFunction::BinomialDeviance;
  context.partitionSize = partitionSize;
  context.partitionRatio = .25;
  context.learningRate = .01;
  context.steps = 17;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45; // .75
  context.recursiveFit = false;
  context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED;
  context.learningRateMethod = LearningRate::RateMethod::FIXED;
  context.minLeafSize = 1;
  context.maxDepth = 10;
  context.minimumGainSplit = 0.;
  context.hasOOSData = true;
  context.dataset_oos = testDataset;
  context.labels_oos = conv_to<Row<double>>::from(testLabels);

  using T = GradientBoostClassifier<DecisionTreeClassifier>;
  using IArchiveType = cereal::BinaryInputArchive;
  using OArchiveType = cereal::BinaryOutputArchive;

  for (auto _ : trials) {
    
    T classifier, newClassifier;
    classifier = T(trainDataset, trainLabels, context);
    classifier.fit();

    std::string fileName = classifier.write();
    classifier.read(newClassifier, fileName);

    Row<std::size_t> trainPrediction, trainNewPrediction, testPrediction, testNewPrediction;
    classifier.Predict(testDataset, testPrediction);
    classifier.Predict(trainDataset, trainPrediction);
    newClassifier.Predict(testDataset, testNewPrediction);
    newClassifier.Predict(trainDataset, trainNewPrediction);
    
    ASSERT_EQ(testPrediction.n_elem, testNewPrediction.n_elem);
    ASSERT_EQ(trainPrediction.n_elem, trainNewPrediction.n_elem);
    
    float eps = std::numeric_limits<float>::epsilon();
    for (int i=0; i<testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i]-testNewPrediction[i]), eps);
    for (int i=0; i<trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i]-trainNewPrediction[i]), eps);

  }
}

TEST(GradientBoostClassifierTest, TestAggregateClassifierRecursiveRoundTrips) {

  std::vector<bool> trials = {true};
  
  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;
  
  loadDatasets(dataset, labels);
  data::Split(dataset, 
	      labels, 
	      trainDataset, 
	      testDataset, 
	      trainLabels, 
	      testLabels, 0.2);

  std::size_t minLeafSize = 1;
  double minimumGainSplit = 0.;
  std::size_t maxDepth = 10;
  std::size_t partitionSize = 10;

  Context context{};
  
  context.loss = lossFunction::BinomialDeviance;
  context.partitionSize = partitionSize;
  context.partitionRatio = .25;
  context.learningRate = .01;
  context.steps = 55;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45; // .75
  context.recursiveFit = false;
  context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED;
  context.learningRateMethod = LearningRate::RateMethod::FIXED;
  context.minLeafSize = 1;
  context.maxDepth = 10;
  context.minimumGainSplit = 0.;
  context.hasOOSData = true;
  context.dataset_oos = testDataset;
  context.labels_oos = conv_to<Row<double>>::from(testLabels);

  using T = GradientBoostClassifier<DecisionTreeClassifier>;
  using IArchiveType = cereal::BinaryInputArchive;
  using OArchiveType = cereal::BinaryOutputArchive;

  for (auto recursive : trials) {
  
    context.recursiveFit = recursive;
    
    T classifier, newClassifier;
    classifier = T(trainDataset, trainLabels, context);
    classifier.fit();

    std::string fileName = classifier.write();
    classifier.read(newClassifier, fileName);

    Row<std::size_t> trainPrediction, trainNewPrediction, testPrediction, testNewPrediction;
    classifier.Predict(testDataset, testPrediction);
    classifier.Predict(trainDataset, trainPrediction);
    newClassifier.Predict(testDataset, testNewPrediction);
    newClassifier.Predict(trainDataset, trainNewPrediction);
    
    ASSERT_EQ(testPrediction.n_elem, testNewPrediction.n_elem);
    ASSERT_EQ(trainPrediction.n_elem, trainNewPrediction.n_elem);
    
    float eps = std::numeric_limits<float>::epsilon();
    for (int i=0; i<testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i]-testNewPrediction[i]), eps);
    for (int i=0; i<trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i]-trainNewPrediction[i]), eps);

  }
}

TEST(GradientBoostClassifierTest, TestChildSerializationRoundTrips) {

  int numTrials = 1;
  std::vector<bool> trials(numTrials);

  std::size_t minLeafSize = 1;
  double minimumGainSplit = 0.;
  std::size_t maxDepth = 10;
  std::size_t partitionSize = 10;

  dataset_t dataset, trainDataset, testDataset;
  labels_t labels, trainLabels, testLabels;

  loadDatasets(dataset, labels);
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
    using ClassifierType = DecisionTree<>;

    T classifier, newClassifier;
    classifier = T(trainDataset, 
		   trainLabels,
		   partitionSize,
		   minLeafSize,
		   minimumGainSplit,
		   maxDepth);

    std::string fileName = dumps<T, IArchiveType, OArchiveType>(classifier);
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
    for (int i=0; i<testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i]-testNewPrediction[i]), eps);
    for (int i=0; i<trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i]-trainNewPrediction[i]), eps);

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
