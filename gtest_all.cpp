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
#include "replay.hpp"

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

void exec(std::string cmd) {
  const char* cmd_c_str = cmd.c_str();
  FILE* pipe = popen(cmd_c_str, "r");
  if (!pipe) throw std::runtime_error("popen() failed!");
  pclose(pipe);
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
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45; // .75
  context.recursiveFit = false;
  context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED;
  context.learningRateMethod = LearningRate::RateMethod::FIXED;
  context.minLeafSize = 1;
  context.maxDepth = 10;
  context.minimumGainSplit = 0.;

  using T = GradientBoostClassifier<DecisionTreeClassifier>;
  using IArchiveType = cereal::BinaryInputArchive;
  using OArchiveType = cereal::BinaryOutputArchive;

  for (auto _ : trials) {
    
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
    for (int i=0; i<testPrediction.n_elem; ++i)
      ASSERT_LE(fabs(testPrediction[i]-testNewPrediction[i]), eps);
    for (int i=0; i<trainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(trainPrediction[i]-trainNewPrediction[i]), eps);

  }
}

TEST(GradientBoostClassifierTest, TestAggregateClassifierRecursiveReplay) {
  
  std::vector<bool> trials = {false, true};
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
  std::size_t maxDepth = 5;
  std::size_t partitionSize = 11;

  Context context{};
  
  context.loss = lossFunction::BinomialDeviance;
  context.partitionSize = partitionSize;
  context.partitionRatio = .25;
  context.learningRate = .001;
  context.steps = 214;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = 1.; // .75
  context.serialize = true;
  context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED;
  context.learningRateMethod = LearningRate::RateMethod::FIXED;
  context.minLeafSize = minLeafSize;
  context.maxDepth = maxDepth;
  context.minimumGainSplit = minimumGainSplit;

  context.serializationWindow = 100;

  using T = GradientBoostClassifier<DecisionTreeClassifier>;
  using IArchiveType = cereal::BinaryInputArchive;
  using OArchiveType = cereal::BinaryOutputArchive;

  for (auto recursive : trials) {
  
    context.recursiveFit = recursive;
    float eps = std::numeric_limits<float>::epsilon();


    // Fit classifier
    T classifier, newClassifier, secondClassifier;
    context.serialize = true;
    classifier = T(trainDataset, trainLabels, context);
    classifier.fit();

    // Predict IS with live classifier fails due to serialization...
    Row<double> liveTrainPrediction;
    EXPECT_THROW(classifier.Predict(trainDataset, liveTrainPrediction), predictionAfterClearedClassifiersException );

    // Use latestPrediction_ instead
    classifier.Predict(liveTrainPrediction);

    // Get index
    std::string indexName = classifier.getIndexName();
    // Use replay to predict IS based on archive classifier
    Row<double> archiveTrainPrediction;
    Replay<double, DecisionTreeClassifier>::Classify(indexName, trainDataset, archiveTrainPrediction);

    for (int i=0; i<liveTrainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(liveTrainPrediction[i]-archiveTrainPrediction[i]), eps);

    // Predict OOS with live classifier
    Row<double> liveTestPrediction;
    context.serialize = false;
    context.serializePrediction = false;
    secondClassifier = T(trainDataset, trainLabels, context);
    secondClassifier.fit();
    secondClassifier.Predict(testDataset, liveTestPrediction);

    // Use replay to predict OOS based on archive classifier
    Row<double> archiveTestPrediction;
    Replay<double, DecisionTreeClassifier>::Classify(indexName, testDataset, archiveTestPrediction, true);

    for (int i=0; i<liveTestPrediction.size(); ++i)
      ASSERT_LE(fabs(liveTestPrediction[i]-archiveTestPrediction[i]), eps);

  }

}

TEST(GradientBoostClassifierTest, TestInSamplePredictionMatchesLatestPrediction) {
  std::vector<bool> trials = {false, true};

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
  std::size_t maxDepth = 5;
  std::size_t partitionSize = 11;

  Context context{};
  
  context.loss = lossFunction::BinomialDeviance;
  context.partitionSize = partitionSize;
  context.partitionRatio = .25;
  context.learningRate = .001;
  context.steps = 514;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45; // .75
  context.serialize = false;
  context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED;
  context.learningRateMethod = LearningRate::RateMethod::FIXED;
  context.minLeafSize = minLeafSize;
  context.maxDepth = maxDepth;
  context.minimumGainSplit = minimumGainSplit;

  using T = GradientBoostClassifier<DecisionTreeClassifier>;
  using IArchiveType = cereal::BinaryInputArchive;
  using OArchiveType = cereal::BinaryOutputArchive;


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

    for (int i=0; i<liveTrainPrediction.n_elem; ++i)
      ASSERT_LE(fabs(liveTrainPrediction[i]-latestPrediction[i]), eps);
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
  context.steps = 21;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45; // .75
  context.recursiveFit = false;
  context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED;
  context.learningRateMethod = LearningRate::RateMethod::FIXED;
  context.minLeafSize = minLeafSize;
  context.maxDepth = maxDepth;
  context.minimumGainSplit = minimumGainSplit;

  using T = GradientBoostClassifier<DecisionTreeClassifier>;
  using IArchiveType = cereal::BinaryInputArchive;
  using OArchiveType = cereal::BinaryOutputArchive;

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

TEST(GradientBoostClassifierTest, TestContextWrittenWithCorrectValues) {

  std::size_t minLeafSize = 1;
  double minimumGainSplit = 0.;
  std::size_t maxDepth = 10;
  std::size_t partitionSize = 10;

  Context context{}, context_archive;
  
  std::string fileName = "__Context_gtest.dat";
  std::string cmd = "/home/charles/src/C++/sandbox/Inductive-Boost/build/createContext ";

  cmd += "--loss 1 --partitionSize 6 --partitionRatio .25 ";
  cmd += "--partitionSizeMethod 1 --learningRateMethod 2 ";
  cmd += "--learningRate .01 --steps 1010 --symmetrizeLabels true ";
  cmd += "--fileName __Context_gtest.dat";

  exec(cmd);

  readBinary<Context>(fileName, context_archive);

  // User set values
  ASSERT_EQ(context_archive.loss, lossFunction::BinomialDeviance);
  ASSERT_EQ(context_archive.partitionSize, 6);
  ASSERT_EQ(context_archive.steps, 1010);
  ASSERT_EQ(context_archive.partitionRatio, .25);
  ASSERT_EQ(context_archive.partitionSizeMethod, PartitionSize::SizeMethod::FIXED_PROPORTION);
  ASSERT_EQ(context_archive.learningRateMethod, LearningRate::RateMethod::DECREASING);

  // Default values
  ASSERT_EQ(context_archive.minLeafSize, 1);
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
  context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED;
  context.learningRateMethod = LearningRate::RateMethod::FIXED;
  context.minLeafSize = minLeafSize;
  context.maxDepth = maxDepth;
  context.minimumGainSplit = minimumGainSplit;

  std::string binFileName = "gtest__Context.dat";
  
  writeBinary<Context>(binFileName, context);
  readBinary<Context>(binFileName, context_archive);

  ASSERT_EQ(context_archive.loss, lossFunction::BinomialDeviance);
  ASSERT_EQ(context_archive.loss, context.loss);

  ASSERT_EQ(context_archive.partitionSize, 10);
  ASSERT_EQ(context_archive.partitionSize, context.partitionSize);

  ASSERT_EQ(context_archive.partitionSizeMethod, PartitionSize::SizeMethod::FIXED);
  ASSERT_EQ(context_archive.partitionSizeMethod, context.partitionSizeMethod);

}

TEST(GradientBoostClassifierTest, TestWritePrediction) {

  std::vector<bool> trials = {false, true};
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
  std::size_t maxDepth = 5;
  std::size_t partitionSize = 11;

  Context context{};
  
  context.loss = lossFunction::BinomialDeviance;
  context.partitionSize = partitionSize;
  context.partitionRatio = .25;
  context.learningRate = .001;
  context.steps = 114;
  context.quietRun = true;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .45; // .75
  context.serialize = true;
  context.serializePrediction = true;
  context.serializeColMask = true;
  context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED;
  context.learningRateMethod = LearningRate::RateMethod::FIXED;
  context.minLeafSize = minLeafSize;
  context.maxDepth = maxDepth;
  context.minimumGainSplit = minimumGainSplit;

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

    // Method 1
    Replay<double, DecisionTreeClassifier>::Classify(indexName, trainDataset, archiveTrainPrediction1);

    // Method 2
    Replay<double, DecisionTreeClassifier>::Classify(indexName, archiveTrainPrediction2);

    for (int i=0; i<liveTrainPrediction.n_elem; ++i) {
      ASSERT_LE(fabs(liveTrainPrediction[i]-archiveTrainPrediction1[i]), eps);
      ASSERT_LE(fabs(liveTrainPrediction[i]-archiveTrainPrediction2[i]), eps);
    }

  }

}

TEST(GradientBoostClassifierTest, TestPredictionRoundTrip) {

  std::vector<bool> trials = {false};
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
  std::size_t maxDepth = 5;
  std::size_t partitionSize = 11;

  Context context{};
  
  context.loss = lossFunction::BinomialDeviance;
  context.partitionSize = partitionSize;
  context.partitionRatio = .25;
  context.learningRate = .001;
  context.steps = 114;
  context.quietRun = true;
  context.baseSteps = 214;
  context.symmetrizeLabels = true;
  context.quietRun = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = 1.; // .75
  context.serialize = true;
  context.serializePrediction = true;
  context.serializeColMask = false;
  context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED;
  context.learningRateMethod = LearningRate::RateMethod::FIXED;
  context.minLeafSize = minLeafSize;
  context.maxDepth = maxDepth;
  context.minimumGainSplit = minimumGainSplit;

  context.serializationWindow = 100;

  using T = GradientBoostClassifier<DecisionTreeClassifier>;

  for (auto recursive : trials) {

    context.recursiveFit = recursive;

    float eps;

    Row<double> prediction, archivePrediction, newPrediction, secondPrediction;

    // Fit classifier
    T classifier;
    classifier = T(trainDataset, trainLabels, testDataset, testLabels, context);
    classifier.fit();


    // Test classifier prediction matches reloaded latestPrediction from same classifier
    classifier.Predict(prediction);

    std::string indexName = classifier.getIndexName();
    
    Replay<double, DecisionTreeClassifier>::readPrediction(indexName, archivePrediction);

    for (int i=0; i<prediction.n_elem; ++i) {
      ASSERT_LE(fabs(prediction[i]-archivePrediction[i]), eps);
      ASSERT_LE(fabs(prediction[i]-archivePrediction[i]), eps);
    }

    // We have archive prediction for an intermediate point [1..114..214]
    // Create a classifier over the entire period and fit
    context.steps = 214;
    context.baseSteps = 214;
    T secondClassifier;
    secondClassifier = T(trainDataset, trainLabels, testDataset, testLabels, context);
    secondClassifier.fit();
    secondClassifier.Predict(secondPrediction);

    // Compare with 100 steps from archivePrediction
    context.steps = 100;
    context.baseSteps = 214;
    T archiveClassifier = T(trainDataset, trainLabels, testDataset, testLabels, archivePrediction, context);
    archiveClassifier.fit();
    archiveClassifier.Predict(archivePrediction);

    if (recursive)
      eps = 1.5;
    else
      eps = std::numeric_limits<float>::epsilon();

    for (int i=0; i<secondPrediction.size(); ++i) {
      ASSERT_LE(fabs(secondPrediction[i]-archivePrediction[i]), eps);
    }
  
  }
  
}

TEST(GradientBoostClassifierTest, TestIncrementalContextContent) {
  
  std::string fileName = "__CTX_TEST_EtxetnoC7txetnoCreifissa.cxt";
  Context context;

  readBinary<Context>(fileName, context);

  ASSERT_EQ(context.loss, lossFunction::Synthetic);
  ASSERT_EQ(context.partitionSize, 6);
  ASSERT_EQ(context.partitionRatio, .25);
  ASSERT_EQ(context.learningRate, .0001);
  ASSERT_EQ(context.steps, 1000);
  ASSERT_EQ(context.baseSteps, 10000);
  ASSERT_EQ(context.symmetrizeLabels, true);
  ASSERT_EQ(context.removeRedundantLabels, false);
  ASSERT_EQ(context.quietRun, true);
  ASSERT_EQ(context.rowSubsampleRatio, 1.);
  ASSERT_EQ(context.colSubsampleRatio, .25);
  ASSERT_EQ(context.recursiveFit, true);
  ASSERT_EQ(context.serialize, true);
  ASSERT_EQ(context.serializePrediction, true);
  ASSERT_EQ(context.partitionSizeMethod, PartitionSize::SizeMethod::FIXED);
  ASSERT_EQ(context.learningRateMethod, LearningRate::RateMethod::FIXED);
  ASSERT_EQ(context.minLeafSize, 1);
  ASSERT_EQ(context.maxDepth, 10);
  ASSERT_EQ(context.minimumGainSplit, 0.);
  ASSERT_EQ(context.serializationWindow, 500);
	    
}

auto main(int argc, char **argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
