#include "incremental_predict.hpp"

using namespace boost::program_options;

auto main(int argc, char **argv) -> int {

  using Cls = GradientBoostClassifier<DecisionTreeClassifier>;

  std::string datasetFileName;
  std::string regressorFileName;
  std::string outFileName;

  options_description desc("Options");
  desc.add_options()
    ("help,h", "Help screen")
    ("datasetFileName",		value<std::string>(&datasetFileName),		"datasetFileName")
    ("regressorFileName",	value<std::string>(&regressorFileName),		"regressorFileName")
    ("outFileName",		value<std::string>(&outFileName),		"outFileName");

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

  Row<double> prediction;
  Replay<double, DecisionTreeRegressorRegressor>::PredictStep(regressorFileName,
							      datasetFileName,
							      outFileName);

  return 0;
}
