#include "single_class_pmlb_driver.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;
using namespace std;

using namespace LossMeasures;
using namespace ModelContext;
using namespace IB_utils;

auto main(int argc, char **argv) -> int {

  Mat<double> dataset, trainDataset, testDataset;
  Row<std::size_t> labels, trainLabels, testLabels;
  Row<std::size_t> trainPrediction, testPrediction;

  if (!data::Load("/home/charles/Data/magic_X.csv", dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/magic_y.csv", labels))
    throw std::runtime_error("Could not load file");

  data::Split(dataset, 
	      labels, 
	      trainDataset, 
	      testDataset, 
	      trainLabels, 
	      testLabels, 0.8);
  std::cout << "TRAIN DATASET: (" << trainDataset.n_cols << " x " 
	    << trainDataset.n_rows << ")" << std::endl;
  std::cout << "TEST DATASET:  (" << testDataset.n_cols << " x " 
	    << testDataset.n_rows << ")" << std::endl;
  
  
  Context context{};
  // context.loss = lossFunction::Savage;
  // context.loss = lossFunction::BinomialDeviance;
  // context.loss = lossFunction::MSE;
  // context.loss = lossFunction::Exp;
  // context.loss = lossFunction::Arctan;
  context.loss = lossFunction::Synthetic;
  // context.loss = lossFunction::SyntheticVar1;
  // context.loss = lossFunction::SyntheticVar2;
  context.partitionSize = 10;
  context.partitionRatio = .25;
  context.learningRate = .0001;
  context.steps = 1000;
  context.baseSteps = 1000;
  context.symmetrizeLabels = true;
  context.serializationWindow = 1000;
  context.removeRedundantLabels = false;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .25; // .75
  context.recursiveFit = true;
  context.serialize = false;
  context.serializePrediction = false;
  context.serializeDataset = false;
  context.serializeLabels = false;
  context.serializationWindow = 1000;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED; // INCREASING
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;    // DECREASING
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;	
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

  c->Predict(trainDataset, trainPrediction);
  c->Predict(testDataset, testPrediction);

  const double trainError = err(trainPrediction, trainLabels);
  const double testError = err(testPrediction, testLabels);

  std::cout << "TRAINING ERROR: " << trainError << "%." << std::endl;
  std::cout << "TEST ERROR    : " << testError << "%." << std::endl;

  return 0;
}