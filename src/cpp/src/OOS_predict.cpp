#include "OOS_predict.hpp"

using namespace boost::program_options;

auto main(int argc, char **argv) -> int {
  
  std::string dataName;
  std::string indexName;
  std::string folderName = "";

  options_description desc("Options");
  desc.add_options()
    ("help,h", "Help screen")
    ("dataName",	value<std::string>(&dataName),		"dataName")
    ("indexName",	value<std::string>(&indexName),		"indexName")
    ("folderName",	value<std::string>(&folderName),	"folderName");

  variables_map vm;

  try {
    store(parse_command_line(argc, argv, desc), vm);
    
    if (vm.count("help")) {
      std::cout << "Predict OOS helper" << std::endl
		<< desc << std::endl;
    }
    notify(vm);
  }
  catch (const std::exception& e) {
    std::cerr << "ERROR [OOS_PREDICT]: " << e.what() << std::endl;
    std::cerr << desc << std::endl;
  }

  // Get data
  std::string absPath = "/home/charles/Data/";
  std::string XPath = absPath + dataName + "_X.csv";
  std::string yPath = absPath + dataName + "_y.csv";

  Mat<double> dataset;
  Row<double> labels;
  Row<double> prediction, predictionStep;

  if (!data::Load(XPath, dataset))
    throw std::runtime_error("Could not load file");
  if (!data::Load(yPath, labels))
    throw std::runtime_error("Could not load file");

  prediction = zeros<Row<double>>(dataset.n_cols);
  predictionStep = zeros<Row<double>>(dataset.n_cols);

  // Deserialize archived regressor
  boost::filesystem::path fldr{folderName};
  std::vector<std::string> fileNames;

  using R = GradientBoostRegressor<DecisionTreeRegressorRegressor>;
  std::unique_ptr<R> regressorArchive = std::make_unique<R>();

  // Predict OOS
  readIndex(indexName, fileNames, fldr);

  for (auto & fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "REG") {
      fileName = strJoin(tokens, '_', 1);
      read(*regressorArchive, fileName, fldr);
      regressorArchive->Predict(dataset, predictionStep);
      prediction += predictionStep;
    }
  }

  // Create summary stats
  auto mn = mean(labels);
  auto num = sum(pow((labels - prediction), 2));
  auto den = sum(pow((labels - mn), 2));
  double r_squared = 1. - (num/den);
  
  std::cout << "(r_squared) : " << r_squared << std::endl;
  
  return 0;
}
