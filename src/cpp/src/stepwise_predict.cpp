#include "stepwise_predict.hpp"

using namespace boost::program_options;

auto main(int argc, char **argv) -> int {

  std::string indexFileName;
  std::string folderName;

  options_description desc("Options");
  desc.add_options()
    ("help,h", "Help screen")
    ("indexFileName",		value<std::string>(&indexFileName),	"indexFileName")
    ("folderName",		value<std::string>(&folderName),	"folderName");

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
    std::cerr << "ERROR [STEPWISE_PREDICT]: " << e.what() << std::endl;
    std::cerr << desc << std::endl;
  }

  Row<double> prediction_oos, labels_oos;
  const auto [error_OOS, 
	      rSquared_OOS,
	      error_IS,
	      rSquared_IS] = Replay<double, DecisionTreeRegressorRegressor>::PredictStepwise(indexFileName, 
												     prediction_oos, 
												     labels_oos, 
												     false, 
												     true,
												     folderName);
  std::cout << "OOS: (error, r_squared) : ("
	    << error_OOS.value_or(-1.) << ", "
	    << rSquared_OOS.value_or(-1.) << ")" << std::endl;
  std::cout << "IS:  (error, r_squared) : ("
	    << error_IS.value_or(-1.) << ", "
	    << rSquared_IS.value_or(-1.) << ")" << std::endl;
    
    
  return 0;
}
