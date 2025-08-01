#include "incremental_predict.hpp"

#include "path_utils.hpp"

using namespace arma;
using namespace mlpack;

using namespace ModelContext;
using namespace IB_utils;

const std::string DELIM = ";";

using namespace boost::program_options;

bool strEndsWith(const std::string& a, const std::string& b) {
  if (b.size() > a.size()) return false;
  return std::equal(a.begin() + a.size() - b.size(), a.end(), b.begin());
}

auto main(int argc, char** argv) -> int {
  std::string dataName;
  std::string contextFileName;
  std::string indexName;
  std::string folderName = "";
  bool quietRun = true;
  bool warmStart = false;
  bool mergeIndexFiles = false;
  double splitRatio = .2;
  Row<double> prediction;

  Context context;

  options_description desc("Options");
  desc.add_options()("help,h", "Help screen")(
      "contextFileName", value<std::string>(&contextFileName), "contextFileName")(
      "dataName", value<std::string>(&dataName), "dataName")(
      "splitRatio", value<double>(&splitRatio), "splitRatio")(
      "quietRun", value<bool>(&quietRun), "quietRun")(
      "warmStart", value<bool>(&warmStart), "warmStart")(
      "mergeIndexFiles", value<bool>(&mergeIndexFiles), "mergeIndexFiles")(
      "indexName", value<std::string>(&indexName), "indexName")(
      "folderName", value<std::string>(&folderName), "folderName");

  variables_map vm;

  try {
    store(parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << "Incremental predict helper" << std::endl << desc << std::endl;
    }
    notify(vm);

  } catch (const std::exception& e) {
    std::cerr << "ERROR [INCREMENTAL_PREDICT]: " << e.what() << std::endl;
    std::cerr << desc << std::endl;
  }

  // Get context
  if (strEndsWith(contextFileName, ".json")) {
    loads<Context, cereal::JSONInputArchive, cereal::JSONOutputArchive>(context, contextFileName);

  } else {
    loads<Context, cereal::BinaryInputArchive, cereal::BinaryOutputArchive>(
        context, contextFileName);
  }

  context.quietRun = quietRun;

  // Get data - resolve_data_path now handles S3 via Python helper
  std::string XPath = IB_utils::resolve_data_path(dataName + "_X.csv");
  std::string yPath = IB_utils::resolve_data_path(dataName + "_y.csv");

  Mat<double> dataset, trainDataset, testDataset;
  Row<double> labels, trainLabels, testLabels;
  Row<double> trainPrediction, testPrediction;

  if (!data::Load(XPath, dataset)) throw std::runtime_error("Could not load file");
  if (!data::Load(yPath, labels)) throw std::runtime_error("Could not load file");

  data::Split(dataset, labels, trainDataset, testDataset, trainLabels, testLabels, splitRatio);

  if (false && !warmStart) {
    std::cerr << "TRAIN DATASET: (" << trainDataset.n_cols << " x " << trainDataset.n_rows << ")"
              << std::endl;
    std::cerr << "TEST DATASET:  (" << testDataset.n_cols << " x " << testDataset.n_rows << ")"
              << std::endl;
  }

  // Create regressor
  // Get prediction if warmStart
  using regressor = GradientBoostRegressor<DecisionTreeRegressorRegressor>;
  using RPtr = std::unique_ptr<regressor>;
  RPtr c;

  if (warmStart) {
    readPrediction(indexName, prediction, folderName);
    c = std::make_unique<regressor>(
        trainDataset, trainLabels, testDataset, testLabels, prediction, context, folderName);
  } else {
    c = std::make_unique<regressor>(trainDataset, trainLabels, testDataset, testLabels, context);
  }

  // Fit
  c->fit();

  // Get indexName
  std::string indexNameNew = c->getIndexName();
  boost::filesystem::path fldr = c->getFldr();

  // Combine information in index
  if (mergeIndexFiles) mergeIndices(indexName, indexNameNew, fldr, true);

  if (warmStart) {
    std::cout << indexNameNew << std::endl;
  } else {
    std::cout << indexNameNew << DELIM << fldr.string() << std::endl;
  }

  // S3 cleanup is handled by Python API if needed

  return 0;
}
