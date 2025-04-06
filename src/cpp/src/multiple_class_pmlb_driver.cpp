#include "multiple_class_pmlb_driver.hpp"

using namespace arma;
using namespace mlpack;
using namespace std;

using namespace ClassifierLossMeasures;
using namespace IB_utils;

auto main(int argc, char** argv) -> int {
  Mat<double> dataset, trainDataset, testDataset;
  Row<std::size_t> labels, trainLabels, testLabels;
  Row<std::size_t> trainPrediction, testPrediction;

  if (!data::Load("/home/charles/Data/1029_LEV_X.csv", dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/1029_LEV_y.csv", labels))
    throw std::runtime_error("Could not load file");

  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.2);

  std::cout << "TRAIN DATASET: (" << trainDataset.n_cols << " x " << trainDataset.n_rows << ")"
            << std::endl;
  std::cout << "TRAIN LABELS:  (" << trainLabels.n_cols << " x " << trainLabels.n_rows << ")"
            << std::endl;
  std::cout << "TEST DATASET:  (" << testDataset.n_cols << " x " << testDataset.n_rows << ")"
            << std::endl;
  std::cout << "TEST LABELS:   (" << testLabels.n_cols << " x " << testLabels.n_rows << ")"
            << std::endl;

  ClassifierContext::Context context{};
  MultiClassifierContext::MultiContext multiContext{};
  MultiClassifierContext::CombinedContext combinedContext{};

  // context.loss = classifierLossFunction::Savage;
  // context.loss = classifierLossFunction::BinomialDeviance;
  // context.loss = classifierLossFunction::MSE;
  // context.loss = classifierLossFunction::Exp;
  // context.loss = classifierLossFunction::Arctan;
  context.loss = classifierLossFunction::Synthetic;
  context.partitionSize = 6;
  context.partitionRatio = .25;
  context.learningRate = .001;
  context.steps = 500;
  context.symmetrizeLabels = true;
  context.removeRedundantLabels = false;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .25;  // .75
  context.recursiveFit = true;
  context.serializeModel = false;
  context.serializationWindow = 100;
  context.partitionSizeMethod = PartitionSize::PartitionSizeMethod::FIXED;  // INCREASING
  context.learningRateMethod = LearningRate::LearningRateMethod::FIXED;     // DECREASING
  context.stepSizeMethod = StepSize::StepSizeMethod::LOG;
  context.minLeafSize = 1;
  context.maxDepth = 10;
  context.minimumGainSplit = 0.;

  multiContext.allVOne = false;
  multiContext.steps = 2;

  combinedContext.context = context;
  combinedContext.allVOne = multiContext.allVOne;
  combinedContext.steps = multiContext.steps;

  using classifier = GradientBoostMultiClassifier<DecisionTreeClassifier>;
  auto c = std::make_unique<classifier>(
      trainDataset, trainLabels, testDataset, testLabels, combinedContext);

  c->fit();

  c->Predict(trainDataset, trainPrediction);
  c->Predict(testDataset, testPrediction);

  const double trainError = err(trainPrediction, trainLabels);
  const double testError = err(testPrediction, testLabels);

  std::cout << "TRAINING ERROR: " << trainError << "%." << std::endl;
  std::cout << "TEST ERROR    : " << testError << "%." << std::endl;

  return 0;
}
