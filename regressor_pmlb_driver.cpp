#include "regressor_pmlb_driver.hpp"

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
  Row<double> labels, trainLabels, testLabels;
  Row<double> trainPrediction, testPrediction;

  if (!data::Load("/home/charles/Data/Regression/1193_BNG_lowbwt_X.csv", dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/Regression/1193_BNG_lowbwt_y.csv", labels))
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
  context.loss = lossFunction::MSE;
  // context.loss = lossFunction::Exp;
  // context.loss = lossFunction::Arctan;
  // context.loss = lossFunction::Synthetic;
  // context.loss = lossFunction::SyntheticVar1;
  // context.loss = lossFunction::SyntheticVar2;
  context.partitionSize = 10;
  context.partitionRatio = 1.;
  context.learningRate = 1.;
  context.steps = 100;
  context.baseSteps = 1000;
  context.symmetrizeLabels = true;
  context.serializationWindow = 1000;
  context.removeRedundantLabels = false;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .25; // .75
  context.recursiveFit = false;
  context.serialize = false;
  context.serializePrediction = false;
  context.serializeDataset = false;
  context.serializeLabels = false;
  context.serializationWindow = 100;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED; // INCREASING
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;    // DECREASING
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;	
  context.minLeafSize = 1;
  context.maxDepth = 10;
  context.minimumGainSplit = 0.;

  using regressor = GradientBoostRegressor<DecisionTreeRegressorRegressor>;
  auto c = std::make_unique<regressor>(trainDataset, 
					trainLabels,
					testDataset,
					testLabels,
					context);
  
  c->fit();

  c->Predict(trainDataset, trainPrediction);
  c->Predict(testDataset, testPrediction);

  const double trainError = err(trainPrediction, trainLabels);
  const double testError = err(testPrediction, testLabels);
  const double trainLoss = c->loss(trainPrediction, trainLabels);
  const double testLoss = c->loss(testPrediction, testLabels);

  std::cout << "TRAINING ERROR: " << trainError << "%." << std::endl;
  std::cout << "TEST ERROR    : " << testError << "%." << std::endl;
  std::cout << "TRAINING LOSS:  " << trainLoss << std::endl;
  std::cout << "TEST LOSS:      " << testLoss << std::endl;

  std::cout << "RESULTS" << std::endl;
  for (std::size_t i=0; i<10; ++i) {
    std::cout << trainLabels[i] << " : " << trainPrediction[i] << " :: "
	      << testLabels[i] << " : " << testPrediction[i] 
	      << std::endl;
    
  }

  return 0;
}
