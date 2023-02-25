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
  bool quietRun = true;
  bool warmStart = false;
  bool mergeIndexFiles = false;
  Row<double> prediction;

  ClassifierContext::Context context;

  options_description desc("Options");
  desc.add_options()
    ("help,h", "Help screen")
    ("contextFileName",	value<std::string>(&contextFileName),	"contextFileName")
    ("dataName",	value<std::string>(&dataName),		"dataName")
    ("quietRun",	value<bool>(&quietRun),			"quietRun")
    ("warmStart",	value<bool>(&warmStart),		"warmStart")
    ("mergeIndexFiles",	value<bool>(&mergeIndexFiles),		"mergeIndexFiles")
    ("indexName",	value<std::string>(&indexName),		"indexName");

  variables_map vm;
    
  try {
    store(parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << "Context creator helper" << std::endl
		<< desc << std::endl;

    }
    notify(vm);
	  
  }
  catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    std::cerr << desc << std::endl;
  }
  
  // Get context
  readBinary<ClassifierContext::Context>(contextFileName, context);
  context.quietRun = quietRun;

  // Get data
  std::string absPath = "/home/charles/Data/";
  std::string XPath = absPath + dataName + "_X.csv";
  std::string yPath = absPath + dataName + "_y.csv";

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
  // std::cout << "TRAIN DATASET: (" << trainDataset.n_cols << " x " 
  // 	    << trainDataset.n_rows << ")" << std::endl;
  // std::cout << "TEST DATASET:  (" << testDataset.n_cols << " x " 
  //	    << testDataset.n_rows << ")" << std::endl;
  
  // Create classifier
  // Get prediction if warmStart
  using classifier = GradientBoostClassifier<DecisionTreeClassifier>;
  using CPtr = std::unique_ptr<classifier>;
  CPtr c;

  if (warmStart) {
       Replay<double, DecisionTreeClassifier>::readPrediction(indexName, prediction);
       c = std::make_unique<classifier>(trainDataset,
					trainLabels,
					testDataset,
					testLabels,
					prediction,
					context);
  } else {
    c = std::make_unique<classifier>(trainDataset, 
				     trainLabels,
				     testDataset,
				     testLabels,
				     context);
  }

  // Fit
  c->fit();

  // Get indexName
  std::string indexNameNew = c->getIndexName();

  // Combine information in index
  if (mergeIndexFiles)
    mergeIndices(indexName, indexNameNew);

  std::cout << indexNameNew << std::endl;

  return 0;
}
