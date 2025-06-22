#include "OOS_classify.hpp"

#include "path_utils.hpp"

namespace {
using DataType = Model_Traits::model_traits<DecisionTreeClassifier>::datatype;
}

using namespace boost::program_options;

void desymmetrize(Row<DataType>& prediction, double a, double b) {
  prediction = (sign(prediction) - b) / a;
}

auto main(int argc, char** argv) -> int {
  std::string dataName;
  std::string indexName;
  std::string folderName = "";
  std::string prefixStr = "";

  options_description desc("Options");
  desc.add_options()("help,h", "Help screen")(
      "dataName", value<std::string>(&dataName), "dataName")(
      "indexName", value<std::string>(&indexName), "indexName")(
      "folderName", value<std::string>(&folderName), "folderName")(
      "prefixStr", value<std::string>(&prefixStr), "prefixStr");

  variables_map vm;

  try {
    store(parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << "Predict OOS helper" << std::endl << desc << std::endl;
    }
    notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "ERROR [OOS_PREDICT]: " << e.what() << std::endl;
    std::cerr << desc << std::endl;
  }

  // Get data
  std::string XPath = IB_utils::resolve_data_path(dataName + "_X.csv");
  std::string yPath = IB_utils::resolve_data_path(dataName + "_y.csv");

  Mat<DataType> dataset;
  Row<DataType> labels;
  Row<DataType> prediction, predictionStep;

  if (!data::Load(XPath, dataset)) throw std::runtime_error("Could not load file");
  if (!data::Load(yPath, labels)) throw std::runtime_error("Could not load file");

  prediction = zeros<Row<DataType>>(dataset.n_cols);
  predictionStep = zeros<Row<DataType>>(dataset.n_cols);

  // Deserialize archived classifier
  boost::filesystem::path fldr{folderName};
  std::vector<std::string> fileNames;

  using C = GradientBoostClassifier<DecisionTreeClassifier>;
  std::unique_ptr<C> classifierArchive = std::make_unique<C>();

  std::pair<double, double> ab;

  // Classifier OOS
  readIndex(indexName, fileNames, fldr);

  for (auto& fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "CLS") {
      fileName = strJoin(tokens, '_', 1);
      read(*classifierArchive, fileName, fldr);
      classifierArchive->Predict(dataset, predictionStep, true);
      ab = classifierArchive->getAB();
      prediction += predictionStep;
    }
  }

  desymmetrize(prediction, ab.first, ab.second);

  double error = err(prediction, labels);

  // Symmetrize labels for remaining metrics
  classifierArchive->symmetrizeLabels(labels);
  classifierArchive->symmetrizeLabels(prediction);

  Row<int> labels_i = conv_to<Row<int>>::from(labels);
  Row<int> prediction_i = conv_to<Row<int>>::from(prediction);

  auto [prec, recall, F1] = precision(labels_i, prediction_i);

  double imb = imbalance(labels);

  std::cout << prefixStr << " OOS: (error, precision, recall, F1, imbalance) : (" << error << ", "
            << prec << ", " << recall << ", " << F1 << ", " << imb << ")" << std::endl;

  return 0;
}
