#include "utils.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::data;
using namespace mlpack::util;

using namespace IB_utils;

auto main(int argc, char **argv) -> int {

  Mat<double> dataset, trainDataset, testDataset;
  Row<std::size_t> labels, trainLabels, testLabels;
  Row<std::size_t> trainPrediction, testPrediction;

  if (!data::Load("/home/charles/Data/titanic_train_X.csv", dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/titanic_train_y.csv", labels))
    throw std::runtime_error("Could not load file");

  data::Split(dataset, 
	      labels, 
	      trainDataset, 
	      testDataset, 
	      trainLabels, 
	      testLabels, 0.2);

  std::cout << "TRAIN DATASET: (" << trainDataset.n_cols << " x " 
	    << trainDataset.n_rows << ")" << std::endl;
  std::cout << "TEST DATASET: (" << testDataset.n_cols << " x " 
	    << testDataset.n_rows << ")" << std::endl;

  ClassifierContext::Context context{};
  
  // context.loss = lossFunction::Savage;
  // context.loss = lossFunction::BinomialDeviance;
  // context.loss = lossFunction::MSE;
  // context.loss = lossFunction::Exp;
  // context.loss = lossFunction::Arctan;
  context.loss = lossFunction::Synthetic;
  context.partitionSize = 6;
  context.partitionRatio = .25;
  context.learningRate = .0001;
  context.steps = 10000;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .25; // .75
  context.recursiveFit = true;
  context.serialize = false;
  context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED; // INCREASING
  context.learningRateMethod = LearningRate::RateMethod::FIXED;   // DECREASING
  context.minLeafSize = 1;
  context.maxDepth = 10;
  context.minimumGainSplit = 0.;
  context.hasOOSData = true;
  context.dataset_oos = testDataset;
  context.labels_oos = conv_to<Row<double>>::from(testLabels);

  std::string fileName = "context.dat";
  writeBinary<ClassifierContext::Context>(fileName, context);


  return 0;
}
