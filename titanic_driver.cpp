#include "titanic_driver.hpp"


auto main(int argc, char **argv) -> int {

  using namespace arma;
  using namespace mlpack;
  using namespace mlpack::tree;
  using namespace mlpack::data;
  using namespace mlpack::util;
  using namespace std;

  using namespace LossMeasures;
  using namespace IB_utils;

  Mat<double> trainDataset, testDataset;
  Row<std::size_t> trainLabels, testLabels;
  Row<double> testPrediction;

  if (!data::Load("/home/charles/Data/titanic_train_X.csv", trainDataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/titanic_train_y.csv", trainLabels))
    throw std::runtime_error("Could not load file");

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
  context.partitionSize = 12;
  context.partitionRatio = .25;
  context.learningRate = .0001;
  context.steps = 200;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .25; // .75
  context.recursiveFit = true;
  context.serialize = false;
  context.serializationWindow = 500;
  context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED; // INCREASING
  context.learningRateMethod = LearningRate::RateMethod::FIXED;   // DECREASING
  context.minLeafSize = 1;
  context.maxDepth = 10;
  context.minimumGainSplit = 0.;
  context.hasOOSData = false;
  context.dataset_oos = testDataset;
  context.labels_oos = conv_to<Row<double>>::from(testLabels);


  using classifier = GradientBoostClassifier<DecisionTreeClassifier>;
  auto c = std::make_unique<classifier>(trainDataset, 
					trainLabels, 
					context);
  
  c->fit();
  std::string indexName = c->getIndexName();

  
  if (!data::Load("/home/charles/Data/titanic_test_X.csv", testDataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/titanic_test_y.csv", testLabels))
    throw std::runtime_error("Could not load file");

  Replay<double, DecisionTreeClassifier>::Predict(indexName, testDataset, testPrediction);
  
  std::cout << "HERE" << std::endl;

  return 0;
 
}