#include "incremental_driver.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

using namespace IB_utils;

using namespace boost::program_options;

auto main(int argc, char **argv) -> int {

  std::string dataName;
  std::string contextFileName;
  std::string indexName;

  // ClassifierContext::Context context;

  options_description desc("Options");
  desc.add_options()
    ("help,h", "Help screen")
    ("contextFileName",	value<std::string>(&contextFileName),	"contextFileName")
    ("dataName",	value<std::string>(&dataName),		"dataName");

  variables_map vm;
    
  try {
    store(parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << "Context creator helper" << std::endl
		<< desc << std::endl;

    }
    notify(vm);
	  
  }
  catch (const error& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    std::cerr << desc << std::endl;
  }
  
  // Get context
  // readBinary<ClassifierContext::Context>(contextFileName, context);

  // Get data
  std::string path = "/home/charles/Data/";
  std::string XPath = path + dataName + "_X.csv";
  std::string yPath = path + dataName + "_y.csv";

  Mat<double> dataset, trainDataset, testDataset;
  Row<std::size_t> labels, trainLabels, testLabels;
  Row<std::size_t> trainPrediction, testPrediction;

  if (!data::Load(XPath, dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load(yPath, labels))
    throw std::runtime_error("Could not load file");

  data::Split(dataset, 
	      labels, 
	      trainDataset, 
	      testDataset, 
	      trainLabels, 
	      testLabels, 0.2);
  std::cout << "TRAIN DATASET: (" << trainDataset.n_cols << " x " 
	    << trainDataset.n_rows << ")" << std::endl;
  std::cout << "TEST DATASET:  (" << testDataset.n_cols << " x " 
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
  context.removeRedundantLabels = false;
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

  // Create classifier
  using classifier = GradientBoostClassifier<DecisionTreeClassifier>;
  auto c = std::make_unique<classifier>(trainDataset, 
					trainLabels,
					testDataset,
					testLabels,
					context);

  // Fit
  c->fit();

  // Get indexName
  indexName = c->getIndexName();

  std::cout << indexName << std::endl;

  return 0;
}
