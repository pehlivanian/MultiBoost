#include "single_class_pmlb_driver.hpp"

namespace {
  using DataType = Model_Traits::model_traits<DecisionTreeClassifier>::datatype;
}

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;
using namespace std;

using namespace LossMeasures;
using namespace ModelContext;
using namespace IB_utils;

auto main() -> int {

  Mat<DataType> dataset, trainDataset, testDataset;
  // Mat<double> dataset, trainDataset, testDataset;
  Row<std::size_t> labels, trainLabels, testLabels;
  Row<std::size_t> trainPrediction, testPrediction;

  if (!data::Load("/home/charles/Data/diabetes_X.csv", dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/diabetes_y.csv", labels))
    throw std::runtime_error("Could not load file");

  data::Split(dataset, 
	      labels, 
	      trainDataset, 
	      testDataset, 
	      trainLabels, 
	      testLabels, 0.2);
  std::cout << "TRAIN DATASET: (" << trainDataset.n_cols << " x " 
	    << trainDataset.n_rows << ")" << std::endl;
  std::cout << "TEST DATASET:  (" << testDataset.n_cols << " x " 
	    << testDataset.n_rows << ")" << std::endl;
  
  
  Context context{};
  // context.loss = lossFunction::Savage;
  context.loss = lossFunction::BinomialDeviance;
  // context.loss = lossFunction::LogLoss;
  // context.loss = lossFunction::MSE;
  // context.loss = lossFunction::Exp;
  // context.loss = lossFunction::Arctan;
  // context.loss = lossFunction::Synthetic;
  // context.loss = lossFunction::SyntheticVar1;
  // context.loss = lossFunction::SyntheticVar2;
  context.childPartitionSize = std::vector<std::size_t>{100, 50, 20, 10, 1};
  context.childNumSteps = std::vector<std::size_t>{100, 2, 4, 2, 1};
  context.childLearningRate = std::vector<double>{.001, .001, .001, .001, .001, .001};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1, 1, 1, 1};
  context.childMaxDepth = std::vector<std::size_t>{10, 10, 10, 10, 10};
  context.childMinimumGainSplit = std::vector<double>{0., 0., 0., 0., 0.};
  context.partitionRatio = .25;
  context.baseSteps = 1000;
  context.symmetrizeLabels = true;
  context.serializationWindow = 1000;
  context.removeRedundantLabels = false;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .25; // .75
  context.recursiveFit = true;
  context.serializeModel = false;
  context.serializePrediction = false;
  context.serializeDataset = false;
  context.serializeLabels = false;
  context.serializationWindow = 1;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED; // INCREASING
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;    // DECREASING
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;	

  using classifier = GradientBoostClassifier<DecisionTreeClassifier>;
  auto c = std::make_unique<classifier>(trainDataset, 
					trainLabels,
					testDataset,
					testLabels,
					context);
  
  c->fit();

  c->Predict(trainDataset, trainPrediction);
  c->Predict(testDataset, testPrediction);

  const double trainError = err(trainPrediction, trainLabels);
  const double testError = err(testPrediction, testLabels);

  std::cout << "TRAINING ERROR: " << trainError << "%." << std::endl;
  std::cout << "TEST ERROR    : " << testError << "%." << std::endl;

  return 0;
}
