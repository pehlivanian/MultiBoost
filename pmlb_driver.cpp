#include "pmlb_driver.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;
using namespace std;

using namespace LossMeasures;

auto main(int argc, char **argv) -> int {

  /* 
     Old way
     
     std::string Xpath = "/home/charles/Data/test_X.csv";
     std::string ypath = "/home/charles/Data/test_y.csv";
     auto df = DataSet<float>(Xpath, ypath, false);
     std::size_t ind1 = 4, ind2 = 2;
     
     auto shape = df.shape();
     
     std::cout << "COMPLETE." << std::endl;
     std::cout << "SIZE: (" << shape.first << ", " 
     << shape.second << ")" << std::endl;
     std::cout << "df[" << ind1 << "][" << ind2
     << "]: " << df[ind1][ind2] << std::endl;
     
     auto splitter = SplitProcessor<float>(.8);
  
     df.accept(splitter);
  */

  /*
    The mlpack way

  */

  /*
    uvec rowMask = linspace<uvec>(0, -1+dataset.n_rows, dataset.n_rows);
    uvec colMask = linspace<uvec>(0, -1+10000, 10000);
    dataset = dataset.submat(rowMask, colMask);
    labels = labels.submat(zeros<uvec>(1), colMask);
    
  */

  /*
    uvec rowMask = linspace<uvec>(0, -1+dataset.n_rows, dataset.n_rows);
    uvec colMask = linspace<uvec>(0, -1+1000, 1000);
    dataset = dataset.submat(rowMask, colMask);
    labels = labels.submat(zeros<uvec>(1), colMask);
  */  

  Mat<double> dataset, trainDataset, testDataset;
  Row<std::size_t> labels, trainLabels, testLabels, trainPrediction, testPrediction;

  if (!data::Load("/home/charles/Data/cleve_X.csv", dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load("/home/charles/Data/cleve_y.csv", labels))
    throw std::runtime_error("Could not load file");
  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, 0.2);
  
  ClassifierContext::Context context{};
  context.loss = lossFunction::BinomialDeviance;
  // context.loss = lossFunction::MSE;
  context.partitionSize = 4;
  context.partitionRatio = .25;
  context.learningRate = .01;
  context.steps = 10000;
  context.symmetrizeLabels = true;
  context.rowSubsampleRatio = 1.;
  context.colSubsampleRatio = .5; // .75
  context.recursiveFit = true;
  context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED;
  context.learningRateMethod = LearningRate::RateMethod::DECREASING;
  context.minLeafSize = 1;
  context.maxDepth = 10;
  context.minimumGainSplit = 0.;
  context.hasOOSData = true;
  context.dataset_oos = testDataset;
  context.labels_oos = conv_to<Row<double>>::from(testLabels);


  auto gradientBoostClassifier = GradientBoostClassifier<DecisionTreeClassifier>(trainDataset, 
										 trainLabels, 
										 context);

  gradientBoostClassifier.fit();
  gradientBoostClassifier.Predict(trainDataset, trainPrediction);
  gradientBoostClassifier.Predict(testDataset, testPrediction);
    
  const double trainError = accu(trainPrediction != trainLabels) * 100. / trainLabels.n_elem;
  const double testError = accu(testPrediction != testLabels) * 100. / testLabels.n_elem;
  std::cout << "TRAINING ERROR: " << trainError << "%." << std::endl;
  std::cout << "TEST ERROR    : " << testError << "%." << std::endl;

  return 0;
}
