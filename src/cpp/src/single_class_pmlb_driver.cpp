#include "single_class_pmlb_driver.hpp"
#include "path_utils.hpp"

namespace {
using DataType = Model_Traits::model_traits<DecisionTreeClassifier>::datatype;
}

using namespace arma;
using namespace mlpack;
using namespace std;

using namespace ClassifierLossMeasures;
using namespace ModelContext;
using namespace IB_utils;

auto main() -> int {
  Mat<DataType> dataset, trainDataset, testDataset;
  // Mat<double> dataset, trainDataset, testDataset;
  Row<std::size_t> labels, trainLabels, testLabels;
  Row<std::size_t> trainPrediction, testPrediction;

  if (!data::Load(resolve_data_path("diabetes_X.csv"), dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load(resolve_data_path("diabetes_y.csv"), labels))
    throw std::runtime_error("Could not load file");

  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.2);
  std::cout << "TRAIN DATASET: (" << trainDataset.n_cols << " x " << trainDataset.n_rows << ")"
            << std::endl;
  std::cout << "TEST DATASET:  (" << testDataset.n_cols << " x " << testDataset.n_rows << ")"
            << std::endl;

  Context context{};
  // context.loss = classifierLossFunction::Savage;
  context.loss = classifierLossFunction::BinomialDeviance;
  // context.loss = classifierLossFunction::LogLoss;
  // context.loss = classifierLossFunction::MSE;
  // context.loss = classifierLossFunction::Exp;
  // context.loss = classifierLossFunction::Arctan;
  // context.loss = classifierLossFunction::Synthetic;
  // context.loss = classifierLossFunction::SyntheticVar1;
  // context.loss = classifierLossFunction::SyntheticVar2;
  // context.loss = classifierLossFunction::CrossEntropyLoss;
  context.childPartitionSize = std::vector<std::size_t>{100, 50, 20, 10, 1};
  context.childNumSteps = std::vector<std::size_t>{100, 2, 4, 2, 1};
  context.childLearningRate = std::vector<double>{.001, .001, .001, .001, .001, .001};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1, 1, 1, 1};
  context.childMaxDepth = std::vector<std::size_t>{10, 10, 10, 10, 10};
  context.childMinimumGainSplit = std::vector<double>{0., 0., 0., 0., 0.};
  context.activePartitionRatio = .25;
  context.steps = 1000;
  context.symmetrizeLabels = true;
  context.serializationWindow = 1000;
  context.removeRedundantLabels = false;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .25;  // .75
  context.recursiveFit = true;
  context.serializeModel = false;
  context.serializePrediction = false;
  context.serializeDataset = false;
  context.serializeLabels = false;
  context.serializationWindow = 1;

  using classifier = GradientBoostClassifier<DecisionTreeClassifier>;
  auto c =
      std::make_unique<classifier>(trainDataset, trainLabels, testDataset, testLabels, context);

  c->fit();

  c->Predict(trainDataset, trainPrediction);
  c->Predict(testDataset, testPrediction);

  const double trainError = err(trainPrediction, trainLabels);
  const double testError = err(testPrediction, testLabels);

  std::cout << "TRAINING ERROR: " << trainError << "%." << std::endl;
  std::cout << "TEST ERROR    : " << testError << "%." << std::endl;

  return 0;
}
