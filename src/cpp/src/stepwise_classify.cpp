#include "stepwise_classify.hpp"

namespace {
  using DataType = Model_Traits::classifier_traits<DecisionTreeClassifier>::datatype;
}

using namespace boost::program_options;

auto main(int argc, char **argv) -> int {

  std::string indexFileName;
  std::string folderName;
  std::string prefixStr = "";

  options_description desc("Options");
  desc.add_options()
    ("help,h", "Help screen")
    ("indexFileName",		value<std::string>(&indexFileName),	"indexFileName")
    ("folderName",		value<std::string>(&folderName),	"folderName")
    ("prefixStr",		value<std::string>(&prefixStr),		"prefixStr");

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

  Row<DataType> prediction_oos, labels_oos;
  const auto [error_OOS,
	      precision_OOS,
	      recall_OOS,
	      F1_OOS,
	      imbalance_OOS,
	      error_IS,
	      precision_IS,
	      recall_IS,
	      F1_IS,
	      imbalance_IS] = Replay<DataType, DecisionTreeClassifier>::ClassifyStepwise(indexFileName, 
										   prediction_oos, 
										   labels_oos, 
										   true, 
										   false,
										   true,
										   folderName);

  std::cout << prefixStr << " OOS : (error, precision, recall, F1, imbalance) : ("
	    << error_OOS.value_or(-1.) << ", "
	    << precision_OOS.value_or(-1.) << ", "
	    << recall_OOS.value_or(-1.) << ", "
	    << F1_OOS.value_or(-1.) << ", "
	    << imbalance_OOS.value_or(-1.) << ")" << std::endl;
  std::cout << prefixStr << " IS :  (error, precision, recall, F1, imbalance) : ("
	    << error_IS.value_or(-1.) << ", "
	    << precision_IS.value_or(-1.) << ", "
	    << recall_IS.value_or(-1.) << ", "
	    << F1_IS.value_or(-1.) << ", "
	    << imbalance_IS.value_or(-1.) << ")" << std::endl;


  return 0;
}
