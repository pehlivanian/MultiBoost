#include "incremental_classify.hpp"

namespace {
  using DataType = Model_Traits::model_traits<DecisionTreeClassifier>::datatype;
}

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

using namespace ModelContext;
using namespace IB_utils;

using CerealT = Context;
using CerealIArch = cereal::BinaryInputArchive;
using CerealOArch = cereal::BinaryOutputArchive;

using namespace boost::program_options;

const std::string DELIM = ";";

auto main(int argc, char **argv) -> int {

  std::string dataName;
  std::string contextFileName;
  std::string indexName;
  std::string folderName = "";
  bool quietRun = true;
  bool warmStart = false;
  bool mergeIndexFiles = false;
  double splitRatio = .2;
  Row<DataType> prediction;

  Context context;

  options_description desc("Options");
  desc.add_options()
    ("help,h", "Help screen")
    ("contextFileName",	value<std::string>(&contextFileName),	"contextFileName")
    ("dataName",	value<std::string>(&dataName),		"dataName")
    ("splitRatio",	value<double>(&splitRatio),		"splitRatio")
    ("quietRun",	value<bool>(&quietRun),			"quietRun")
    ("warmStart",	value<bool>(&warmStart),		"warmStart")
    ("mergeIndexFiles",	value<bool>(&mergeIndexFiles),		"mergeIndexFiles")
    ("indexName",	value<std::string>(&indexName),		"indexName")
    ("folderName",	value<std::string>(&folderName),	"folderName");

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
    std::cerr << "ERROR [INCREMENTAL_CLASSIFY]: " << e.what() << std::endl;
    std::cerr << desc << std::endl;
  }
  
  // Get context; no subdirectory for initial read
  // classifier will persist to digest subdirectory
  loads<CerealT, CerealIArch, CerealOArch>(context, contextFileName);

  context.quietRun = quietRun;

  // Get data
  std::string absPath = "/home/charles/Data/";
  std::string XPath = absPath + dataName + "_X.csv";
  std::string yPath = absPath + dataName + "_y.csv";

  Mat<DataType> dataset, trainDataset, testDataset;
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
	      testLabels, 
	      splitRatio);

  /*
    std::cerr << "TRAIN DATASET: (" << trainDataset.n_cols << " x " 
    << trainDataset.n_rows << ")" << std::endl;
    std::cerr << "TEST DATASET:  (" << testDataset.n_cols << " x " 
    << testDataset.n_rows << ")" << std::endl;
  */
  
  // Create classifier
  // Get prediction if warmStart
  using classifier = GradientBoostClassifier<DecisionTreeClassifier>;
  using CPtr = std::unique_ptr<classifier>;
  CPtr c;

  if (warmStart) {
    readPrediction(indexName, prediction, folderName);
    c = std::make_unique<classifier>(trainDataset,
				     trainLabels,
				     testDataset,
				     testLabels,
				     prediction,
				     context,
				     folderName);
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
  boost::filesystem::path fldr = c->getFldr();

  // Combine information in index
  if (mergeIndexFiles)
    mergeIndices(indexName, indexNameNew, fldr, true);

  if (warmStart) {
    std::cout << indexNameNew << std::endl;
  } else {
    std::cout << indexNameNew << DELIM
	      << fldr.string() << std::endl;
  }

  return 0;
}
