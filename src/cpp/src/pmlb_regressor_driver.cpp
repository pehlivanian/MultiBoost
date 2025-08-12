#include "regressor_pmlb_driver.hpp"
#include "path_utils.hpp"

namespace {
using DataType = Model_Traits::model_traits<DecisionTreeRegressorRegressor>::datatype;

// Custom NegativeFeedback regressor with beta = 0.15
class NegativeFeedbackRegressorBeta15 : public NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> {
public:
  using Args = typename Model_Traits::model_traits<DecisionTreeRegressorRegressor>::modelArgs;
  using DataType = typename Model_Traits::model_traits<DecisionTreeRegressorRegressor>::datatype;

  // Default constructor - this is what gets called during decorator pattern
  NegativeFeedbackRegressorBeta15() : NegativeFeedbackRegressor(0.15f, 5) {
    // std::cout << "DEBUG: NegativeFeedbackRegressorBeta15 default constructor called" << std::endl;
  }

  // Constructor with dataset parameters - should delegate to base that properly initializes base classes
  NegativeFeedbackRegressorBeta15(const Mat<DataType>& dataset, Row<DataType>& labels,
                        std::size_t minLeafSize, double minGainSplit, std::size_t maxDepth)
      : NegativeFeedbackRegressor(dataset, labels, minLeafSize, minGainSplit, maxDepth) {
    // std::cout << "DEBUG: NegativeFeedbackRegressorBeta15 dataset constructor called" << std::endl;
    // Override beta and iterations after base construction
    beta_ = 0.15f;
    iterations_ = 5;
  }

  // Constructor with weights - should delegate to base that properly initializes base classes 
  NegativeFeedbackRegressorBeta15(const Mat<DataType>& dataset, Row<DataType>& labels,
                        Row<DataType>& weights,
                        std::size_t minLeafSize, double minGainSplit, std::size_t maxDepth)
      : NegativeFeedbackRegressor(dataset, labels, weights, minLeafSize, minGainSplit, maxDepth) {
    // std::cout << "DEBUG: NegativeFeedbackRegressorBeta15 weighted constructor called" << std::endl;
    // Override beta and iterations after base construction
    beta_ = 0.05f;
    iterations_ = 10;
  }

  static Args _args(const Model_Traits::AllRegressorArgs& p) { 
    return DecisionTreeRegressorRegressor::_args(p); 
  }
};
}

// Model traits specialization for the custom class
namespace Model_Traits {
template <>
struct model_traits<NegativeFeedbackRegressorBeta15> {
  using datatype = model_traits<DecisionTreeRegressorRegressor>::datatype;
  using integrallabeltype = model_traits<DecisionTreeRegressorRegressor>::integrallabeltype;
  using model = model_traits<DecisionTreeRegressorRegressor>::model;
  using modelArgs = model_traits<DecisionTreeRegressorRegressor>::modelArgs;
};
}

using namespace arma;
using namespace mlpack;
using namespace std;

using namespace RegressorLossMeasures;
using namespace ModelContext;
using namespace IB_utils;

auto main() -> int {
  Mat<DataType> dataset, trainDataset, testDataset;
  Row<DataType> labels, trainLabels, testLabels;
  Row<DataType> trainPrediction, testPrediction;

  // Try to load a standard regression dataset
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
  context.loss = regressorLossFunction::MSE;
  context.childPartitionSize = std::vector<std::size_t>{100, 50, 20, 10, 1};
  context.childNumSteps = std::vector<std::size_t>{1500, 2, 4, 2, 1};  // Changed to 50 steps as requested
  context.childLearningRate = std::vector<double>{.001, .001, .001, .001, .001, .001};
  context.childMinLeafSize = std::vector<std::size_t>{1, 1, 1, 1, 1};
  context.childMaxDepth = std::vector<std::size_t>{10, 10, 10, 10, 10};
  context.childMinimumGainSplit = std::vector<double>{0., 0., 0., 0., 0.};
  context.childActivePartitionRatio = std::vector<double>{.25, .25, .25, .25, .25};
  context.steps = 1000;
  context.symmetrizeLabels = false;
  context.serializationWindow = 10;  // Show progress every 10 steps
  context.removeRedundantLabels = false;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .25;
  context.recursiveFit = true;
  context.serializeModel = false;
  context.serializePrediction = false;
  context.serializeDataset = false;
  context.serializeLabels = false;
  context.quietRun = false;  // Enable detailed iterative output

  std::cout << "\n=== Testing Standard DecisionTreeRegressor ===" << std::endl;
  std::cout << "Configuring standard GradientBoostRegressor with " << context.steps << " steps..." << std::endl;
  std::cout << "Serialization window: " << context.serializationWindow << " (progress shown every " << context.serializationWindow << " steps)" << std::endl;
  
  // Test 1: Standard DecisionTreeRegressor
  context.childNumSteps = {10, 2, 4, 2, 1};
  using standardRegressor = GradientBoostRegressor<DecisionTreeRegressorRegressor>;
  auto standard_regressor = std::make_unique<standardRegressor>(trainDataset, trainLabels, testDataset, testLabels, context);

  std::cout << "\nStarting iterative fitting process for Standard Regressor..." << std::endl;
  standard_regressor->fit();

  Row<DataType> standardTrainPrediction, standardTestPrediction;
  standard_regressor->Predict(trainDataset, standardTrainPrediction);
  standard_regressor->Predict(testDataset, standardTestPrediction);

  const double standardTrainError = err(standardTrainPrediction, trainLabels);
  const double standardTestError = err(standardTestPrediction, testLabels);
  const double standardTrainLoss = standard_regressor->loss(standardTrainPrediction, trainLabels);
  const double standardTestLoss = standard_regressor->loss(standardTestPrediction, testLabels);

  std::cout << "STANDARD REGRESSOR RESULTS:" << std::endl;
  std::cout << "TRAINING ERROR: " << standardTrainError << "%." << std::endl;
  std::cout << "TEST ERROR    : " << standardTestError << "%." << std::endl;
  std::cout << "TRAINING LOSS : " << standardTrainLoss << std::endl;
  std::cout << "TEST LOSS     : " << standardTestLoss << std::endl;

  std::cout << "\n=== Testing NegativeFeedback Decorated Regressor ===" << std::endl;
  std::cout << "Configuring NegativeFeedback GradientBoostRegressor (beta=0.15, iterations=5) with " << context.steps << " steps..." << std::endl;
  std::cout << "Serialization window: " << context.serializationWindow << " (progress shown every " << context.serializationWindow << " steps)" << std::endl;

  // Test 2: NegativeFeedback decorated regressor with beta=0.15
  context.childNumSteps = {50000, 2, 4, 2, 1};
  using decoratedRegressor = GradientBoostRegressor<NegativeFeedbackRegressorBeta15>;
  auto decorated_regressor = std::make_unique<decoratedRegressor>(trainDataset, trainLabels, testDataset, testLabels, context);

  std::cout << "\nStarting iterative fitting process for NegativeFeedback Decorated Regressor..." << std::endl;
  decorated_regressor->fit();

  Row<DataType> decoratedTrainPrediction, decoratedTestPrediction;
  decorated_regressor->Predict(trainDataset, decoratedTrainPrediction);
  decorated_regressor->Predict(testDataset, decoratedTestPrediction);

  const double decoratedTrainError = err(decoratedTrainPrediction, trainLabels);
  const double decoratedTestError = err(decoratedTestPrediction, testLabels);
  const double decoratedTrainLoss = decorated_regressor->loss(decoratedTrainPrediction, trainLabels);
  const double decoratedTestLoss = decorated_regressor->loss(decoratedTestPrediction, testLabels);

  std::cout << "NEGATIVE FEEDBACK REGRESSOR RESULTS (beta=0.15, iterations=5):" << std::endl;
  std::cout << "TRAINING ERROR: " << decoratedTrainError << "%." << std::endl;
  std::cout << "TEST ERROR    : " << decoratedTestError << "%." << std::endl;
  std::cout << "TRAINING LOSS : " << decoratedTrainLoss << std::endl;
  std::cout << "TEST LOSS     : " << decoratedTestLoss << std::endl;

  std::cout << "\n=== COMPARISON ===" << std::endl;
  std::cout << "Performance Improvement:" << std::endl;
  std::cout << "Train Error: " << standardTrainError << " -> " << decoratedTrainError 
            << " (change: " << (decoratedTrainError - standardTrainError) << "%)" << std::endl;
  std::cout << "Test Error:  " << standardTestError << " -> " << decoratedTestError 
            << " (change: " << (decoratedTestError - standardTestError) << "%)" << std::endl;
  std::cout << "Train Loss:  " << standardTrainLoss << " -> " << decoratedTrainLoss 
            << " (change: " << (decoratedTrainLoss - standardTrainLoss) << ")" << std::endl;
  std::cout << "Test Loss:   " << standardTestLoss << " -> " << decoratedTestLoss 
            << " (change: " << (decoratedTestLoss - standardTestLoss) << ")" << std::endl;

  std::cout << "\n=== SAMPLE PREDICTIONS ===" << std::endl;
  std::cout << "First 10 predictions comparison:" << std::endl;
  std::cout << "Index\tTrue\tStandard\tDecorated" << std::endl;
  for (std::size_t i = 0; i < std::min(static_cast<std::size_t>(10), static_cast<std::size_t>(testLabels.n_elem)); ++i) {
    std::cout << i << "\t" << testLabels[i] << "\t" 
              << standardTestPrediction[i] << "\t\t" << decoratedTestPrediction[i] << std::endl;
  }

  return 0;
}
