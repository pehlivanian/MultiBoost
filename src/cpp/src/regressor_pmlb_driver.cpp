#include "regressor_pmlb_driver.hpp"

using namespace arma;
using namespace mlpack;
using namespace std;

using namespace RegressorLossMeasures;
using namespace ModelContext;
using namespace IB_utils;

auto main() -> int {
  Mat<double> dataset, trainDataset, testDataset;
  Row<double> labels, trainLabels, testLabels;
  Row<double> trainPrediction, testPrediction;

  /*
  if (!data::Load("/home/charles/Data/Regression/1193_BNG_lowbwt_X.csv", dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/Regression/1193_BNG_lowbwt_y.csv", labels))
    throw std::runtime_error("Could not load file");
  */
  if (!data::Load(
          "/home/charles/Data/tabular_benchmark/Regression/Mixed/"
          "Mercedes_Benz_Greener_Manufacturing_X.csv",
          dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load(
          "/home/charles/Data/tabular_benchmark/Regression/Mixed/"
          "Mercedes_Benz_Greener_Manufacturing_y.csv",
          labels))
    throw std::runtime_error("Could not load file");

  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.1);
  std::cout << "TRAIN DATASET: (" << trainDataset.n_cols << " x " << trainDataset.n_rows << ")"
            << std::endl;
  std::cout << "TEST DATASET:  (" << testDataset.n_cols << " x " << testDataset.n_rows << ")"
            << std::endl;

  Context context{};
  // context.loss = regressorLossFunction::Savage;
  // context.loss = regressorLossFunction::BinomialDeviance;
  context.loss = regressorLossFunction::MSE;
  // context.loss = regressorLossFunction::Exp;
  // context.loss = regressorLossFunction::Arctan;
  // context.loss = regressorLossFunction::Synthetic;
  // context.loss = regressorLossFunction::SyntheticVar1;
  // context.loss = regressorLossFunction::SyntheticVar2;
  context.activePartitionRatio = 1.;
  context.steps = 1;
  context.symmetrizeLabels = false;
  context.serializationWindow = 10;
  context.removeRedundantLabels = false;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = 1.;  // .75
  context.recursiveFit = true;
  context.quietRun = false;
  context.serializeModel = false;
  context.serializePrediction = false;
  context.serializeDataset = false;
  context.serializeLabels = false;
  context.serializationWindow = 1;
  // context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED; // INCREASING
  // context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;    // DECREASING
  // context.stepSizeMethod = StepSize::StepSizeMethod::LOG;
  context.childPartitionSize = std::vector<std::size_t>{3789};
  context.childNumSteps = std::vector<std::size_t>{1};
  context.childLearningRate = std::vector<double>{.5};
  context.childMinLeafSize = std::vector<std::size_t>{1};
  context.childMaxDepth = std::vector<std::size_t>{10};
  context.childMinimumGainSplit = std::vector<double>{0.};

  using regressor = GradientBoostRegressor<DecisionTreeRegressorRegressor>;
  auto c = std::make_unique<regressor>(trainDataset, trainLabels, testDataset, testLabels, context);

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
  for (std::size_t i = 0; i < 10; ++i) {
    std::cout << trainLabels[i] << " : " << trainPrediction[i] << " :: " << testLabels[i] << " : "
              << testPrediction[i] << std::endl;
  }

  return 0;
}
