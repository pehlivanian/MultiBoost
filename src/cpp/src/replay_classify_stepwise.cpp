#include "replay_classify_stepwise.hpp"

namespace {
  using DataType = Model_Traits::classifier_traits<DecisionTreeClassifier>::datatype;
}

using namespace boost::program_options;

auto main(int argc, char **argv) -> int {

  std::string datasetFileName;
  std::string classifierFileName;
  std::string outFileName;
  std::string folderName;

  options_description desc("Options");
  desc.add_options()
    ("help,h", "Help screen")
    ("datasetFileName",		value<std::string>(&datasetFileName),		"datasetFileName")
    ("classifierFileName",	value<std::string>(&classifierFileName),	"classifierFileName")
    ("outFileName",		value<std::string>(&outFileName),		"outFileName")
    ("folderName",		value<std::string>(&folderName),		"folderName");

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

  Row<DataType> prediction;
  Replay<DataType, DecisionTreeClassifier>::ClassifyStep(classifierFileName,
							 datasetFileName,
							 outFileName,
							 false,
							 folderName);
  
  return 0;
}
