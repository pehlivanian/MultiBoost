#include "titanic_driver.hpp"

namespace {
  using DataType = Model_Traits::classifier_traits<DecisionTreeClassifier>::datatype;
}

using namespace ModelContext;
using namespace PartitionSize;
using namespace StepSize;
using namespace LearningRate;

auto main() -> int {

  using namespace arma;
  using namespace mlpack;
  using namespace std;

  using namespace ClassifierLossMeasures;
  using namespace IB_utils;

  Mat<DataType> trainDataset, testDataset;
  Row<std::size_t> trainLabels, testLabels;
  Row<DataType> testPrediction;

  if (!data::Load("/home/charles/Data/titanic_train_X.csv", trainDataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/titanic_train_y.csv", trainLabels))
    throw std::runtime_error("Could not load file");

  std::cout << "TRAIN DATASET: (" << trainDataset.n_cols << " x " 
	    << trainDataset.n_rows << ")" << std::endl;
  std::cout << "TEST DATASET: (" << testDataset.n_cols << " x " 
	    << testDataset.n_rows << ")" << std::endl;
  
  
  Context context{};

  // context.loss = classifierLossFunction::Savage;
  // context.loss = classifierLossFunction::BinomialDeviance;
  // context.loss = classifierLossFunction::MSE;
  // context.loss = classifierLossFunction::Exp;
  // context.loss = classifierLossFunction::Arctan;
  context.loss = classifierLossFunction::Synthetic;
  context.partitionSize = 12;
  context.partitionRatio = .25;
  context.learningRate = .0001;
  context.steps = 200;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .25; // .75
  context.recursiveFit = true;
  context.serializeModel = false;
  context.serializationWindow = 500;
  context.partitionSizeMethod = PartitionSizeMethod::FIXED; // INCREASING
  context.learningRateMethod = LearningRateMethod::FIXED;   // DECREASING
  context.minLeafSize = 1;
  context.maxDepth = 10;
  context.minimumGainSplit = 0.;

  using classifier = GradientBoostClassifier<DecisionTreeClassifier>;
  auto c = std::make_unique<classifier>(trainDataset, 
					trainLabels, 
					testDataset,
					testLabels,
					context);
  
  c->fit();
  std::string indexName = c->getIndexName();

  
  if (!data::Load("/home/charles/Data/titanic_test_X.csv", testDataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/titanic_test_y.csv", testLabels))
    throw std::runtime_error("Could not load file");

  Replay<DataType, DecisionTreeClassifier>::Classify(indexName, testDataset, testPrediction);
  
  std::cout << "HERE" << std::endl;

  return 0;
 
}
