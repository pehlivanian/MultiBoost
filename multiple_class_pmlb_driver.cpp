#include "multiple_class_pmlb_driver.hpp"


using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;
using namespace std;

using namespace LossMeasures;
using namespace IB_utils;

auto main(int argc, char **argv) -> int {

  Mat<double> dataset, trainDataset, testDataset;
  Row<std::size_t> labels, trainLabels, testLabels;
  Row<std::size_t> trainPrediction, testPrediction;

  if (!data::Load("/home/charles/Data/1028_SWD_X.csv", dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/1028_SWD_y.csv", labels))
    throw std::runtime_error("Could not load file");

  data::Split(dataset, 
	      labels, 
	      trainDataset, 
	      testDataset, 
	      trainLabels, 
	      testLabels, 0.2);


  std::cout << "TRAIN DATASET: (" << trainDataset.n_cols << " x " 
	    << trainDataset.n_rows << ")" << std::endl;
  std::cout << "TRAIN LABELS:  (" << trainLabels.n_cols << " x "
	    << trainLabels.n_rows << ")" << std::endl;
  std::cout << "TEST DATASET:  (" << testDataset.n_cols << " x " 
	    << testDataset.n_rows << ")" << std::endl;
  std::cout << "TEST LABELS:   (" << testLabels.n_cols << " x "
	    << testLabels.n_rows << ")" << std::endl;


  ClassifierContext::Context context{};
  MultiClassifierContext::MultiContext multiContext{};
  MultiClassifierContext::CombinedContext combinedContext{};


  // context.loss = lossFunction::Savage;
  // context.loss = lossFunction::BinomialDeviance;
  context.loss = lossFunction::MSE;
  // context.loss = lossFunction::Exp;
  // context.loss = lossFunction::Arctan;
  // context.loss = lossFunction::Synthetic;
  context.partitionSize = 24;
  context.partitionRatio = .25;
  context.learningRate = .001;
  context.steps = 1000;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .25; // .75
  context.recursiveFit = false;
  context.serialize = false;
  context.serializationWindow = 1000;
  context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED; // INCREASING
  context.learningRateMethod = LearningRate::RateMethod::FIXED;   // DECREASING
  context.minLeafSize = 1;
  context.maxDepth = 10;
  context.minimumGainSplit = 0.;
  context.hasOOSData = true;
  context.dataset_oos = testDataset;
  context.labels_oos = conv_to<Row<double>>::from(testLabels);

  multiContext.allVOne = false;

  combinedContext.context = context;
  combinedContext.allVOne = multiContext.allVOne;

  using classifier = GradientBoostMultiClassifier<DecisionTreeClassifier>;
  auto c = std::make_unique<classifier>(trainDataset, 
					trainLabels, 
					combinedContext);

  c->fit();

  /*
    c->Predict(trainDataset, trainPrediction);
    c->Predict(testDataset, testPrediction);
    
    const double trainError = err(trainPrediction, trainLabels);
    const double testError = err(testPrediction, testLabels);
    
    std::cout << "TRAINING ERROR: " << trainError << "%." << std::endl;
    std::cout << "TEST ERROR    : " << testError << "%." << std::endl;  
  */
  
  return 0;
}
