#include "stepwise_classify.hpp"

using namespace boost::program_options;

auto main(int argc, char **argv) -> int {

  std::string indexFileName;

  options_description desc("Options");
  desc.add_options()
    ("help,h", "Help screen")
    ("indexFileName",		value<std::string>(&indexFileName),	"indexFileName");

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
    std::cerr << "ERROR [STEPWISE_CLASSIFY]: " << e.what() << std::endl;
    std::cerr << desc << std::endl;
  }

  Row<double> prediction_oos, labels_oos;
  Replay<double, DecisionTreeClassifier>::ClassifyStepwise(indexFileName, prediction_oos, labels_oos, true);

  const double testError = err(prediction_oos, labels_oos);

  std::cout << "TEST ERROR    : " << testError << "%." << std::endl;



  return 0;
}
