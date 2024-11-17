#include <iostream>
#include <vector>
#include <string>
#include <iterator>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "utils.hpp"
#include "regressor_loss.hpp"

using namespace IB_utils;
using namespace ModelContext;
using namespace LossMeasures;

using namespace boost::program_options;

std::string path_(std::string inputString) {
    std::string fileName;
    for (auto it = inputString.rbegin(); it != inputString.rend() &&
	   fileName.size() != 24; ++it)
      {
	if (std::isalnum(*it))
	  fileName.push_back(*it);
      }
    return fileName;
}

namespace std {

  std::istream& operator>>(std::istream& in, regressorLossFunction& loss) {
    int token;
    in >> token;
    switch (token) {
    case(0):
      loss = regressorLossFunction::MSE;
      break;
    case (1):
      loss = regressorLossFunction::SyntheticRegLoss;
      break;
    case (2):
      loss = regressorLossFunction::LogLoss;
      break;
    case (3):
      loss = regressorLossFunction::RegressorPowerLoss;
      break;
    default:
      in.setstate(std::ios_base::failbit);
    }
    
    return in;
  }

};


auto main(int argc, char **argv) -> int {

  /* TYPICAL CASE
     ============
     context.loss = lossFunction::Synthetic;
     context.partitionSize = 6;
     context.partitionRatio = .25;
     context.learningRate = .0001;
     context.steps = 10000;
     context.baseSteps = 10000;
     context.symmetrizeLabels = true;
     context.rowSubsampleRatio = 1.;
     context.colSubsampleRatio = .25; // .75
     context.recursiveFit = true;
     context.serializeModel = false;
     context.minLeafSize = 1;
     context.maxDepth = 10;
     context.minimumGainSplit = 0.;
  */

  regressorLossFunction			loss			  = regressorLossFunction::MSE;
  float					lossPower		  = -1.0;
  bool					clamp_gradient		  = false;
  double				upper_val		  = 0.;
  double				lower_val		  = 0.;
  int					steps			  = 10000;
  int					baseSteps		  = 10000;
  bool					symmetrizeLabels	  = true;
  bool					removeRedundantLabels	  = true;
  bool					quietRun		  = false;
  bool					useWeights		  = false;
  double				rowSubsampleRatio	  = 1.;
  double				colSubsampleRatio	  = .25;
  bool					recursiveFit		  = true;
  double				activePartitionRatio	  = 0.25;
  std::vector<std::size_t>		childPartitionSize	  = std::vector<std::size_t>{};
  std::vector<std::size_t>		childNumSteps		  = std::vector<std::size_t>{};
  std::vector<double>			childLearningRate	  = std::vector<double>{};
  std::vector<double>			childActivePartitionRatio = std::vector<double>{};
  std::vector<std::size_t>		childMinLeafSize	  = std::vector<std::size_t>{};
  std::vector<std::size_t>		childMaxDepth		  = std::vector<std::size_t>{};
  std::vector<double>			childMinimumGainSplit	  = std::vector<double>{};
  std::size_t				numTrees		  = 10;
  bool					serializeModel		  = false;
  bool					serializePrediction	  = false;
  bool					serializeColMask	  = false;
  bool					serializeDataset	  = false;
  bool					serializeLabels		  = false;
  std::size_t				serializationWindow	  = 1000;
  std::size_t				depth			  = 0;

  std::string				fileName		  = path_(typeid(Context).name());

  options_description desc("Options");
  desc.add_options()
    ("help,h", "Help screen")
    ("loss",			value<regressorLossFunction>(&loss),				"loss")
    ("lossPower",		value<float>(&lossPower),					"lossPower")
    ("clamp_gradient",		value<bool>(&clamp_gradient),					"clamp_gradient")
    ("upper_val",		value<double>(&upper_val),					"upper_val")
    ("lower_val",		value<double>(&lower_val),					"lower_val")
    ("steps",			value<int>(&steps),						"steps")
    ("baseSteps",		value<int>(&baseSteps),						"baseSteps")
    ("symmetrizeLabels",	value<bool>(&symmetrizeLabels),					"symmetrizeLabels")
    ("removeRedundantLabels",	value<bool>(&removeRedundantLabels),				"removeRedundantLabels")
    ("quietRun",		value<bool>(&quietRun),						"quietRun")
    ("useWeights",		value<bool>(&useWeights),					"useWeights")
    ("rowSubsampleRatio",	value<double>(&rowSubsampleRatio),				"rowSubsampleRatio")
    ("colSubsampleRatio",	value<double>(&colSubsampleRatio),				"colSubsampleRatio")
    ("recursiveFit",		value<bool>(&recursiveFit),					"recursiveFit")
    ("childPartitionSize",	value<std::vector<std::size_t>>(&childPartitionSize)->multitoken(),		"childPartitionSize")
    ("childNumSteps",		value<std::vector<std::size_t>>(&childNumSteps)->multitoken(),		"childNumSteps")
    ("childLearningRate",	value<std::vector<double>>(&childLearningRate)->multitoken(),	"childLearningRate")
    ("childActivePartitionRatio", value<std::vector<double>>(&childActivePartitionRatio)->multitoken(), "childActivePartitionRatio")
    ("childMinLeafSize",	value<std::vector<std::size_t>>(&childMinLeafSize)->multitoken(),	"childMinLeafSize")
    ("childMaxDepth",		value<std::vector<std::size_t>>(&childMaxDepth)->multitoken(),	"childMaxDepth")
    ("childMinimumGainSplit",	value<std::vector<double>>(&childMinimumGainSplit)->multitoken(),	"childMinimumGainSplit")
    ("numTrees",		value<std::size_t>(&numTrees),					"numTrees")
    ("serializeModel",		value<bool>(&serializeModel),					"serializeModel")
    ("serializePrediction",	value<bool>(&serializePrediction),				"serializePrediction")
    ("serializeColMask",	value<bool>(&serializeColMask),					"serializeColMask")
    ("serializeDataset",	value<bool>(&serializeDataset),					"serializeDataset")
    ("serializeLabels",		value<bool>(&serializeLabels),					"serializeLabels")
    ("serializationWindow",	value<std::size_t>(&serializationWindow),			"serializationWindow")
    ("depth",			value<std::size_t>(&depth),					"depth")
    ("fileName",		value<std::string>(&fileName),					"fileName for Context");


    variables_map vm;
  
  try {
    store(parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << "Context creator helper" << std::endl
		<< desc << std::endl;

    }
    notify(vm);
	  
  } 
  catch (const error& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    std::cerr << desc << std::endl;
  }

  Context context{};

  context.loss		= loss;
  context.lossPower	= lossPower;
  context.clamp_gradient = clamp_gradient;
  context.upper_val = upper_val;
  context.lower_val = lower_val;
  context.activePartitionRatio = activePartitionRatio;
  context.steps = steps;
  context.baseSteps = baseSteps;
  context.symmetrizeLabels = symmetrizeLabels;
  context.removeRedundantLabels = removeRedundantLabels;
  context.quietRun = quietRun;
  context.useWeights = useWeights;
  context.rowSubsampleRatio = rowSubsampleRatio;
  context.colSubsampleRatio = colSubsampleRatio;
  context.recursiveFit = recursiveFit;
  context.childPartitionSize = childPartitionSize;
  context.childNumSteps = childNumSteps;
  context.childLearningRate = childLearningRate;
  context.childActivePartitionRatio = childActivePartitionRatio;
  context.childMinLeafSize = childMinLeafSize;
  context.childMaxDepth = childMaxDepth;
  context.childMinimumGainSplit = childMinimumGainSplit;
  context.numTrees = numTrees;
  context.serializeModel = serializeModel;
  context.serializePrediction = serializePrediction;
  context.serializeDataset = serializeDataset;
  context.serializeLabels = serializeLabels;
  context.serializeColMask = serializeColMask;
  context.serializationWindow = serializationWindow;
  context.depth = depth;

  // Serialize
  using CerealT = Context;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;

  boost::filesystem::path fldr{"./"};
  dumps<CerealT, CerealIArch, CerealOArch>(context, fileName, fldr);

  std::cout << "Context archive: " << fileName << std::endl;
    
  // To deserialize
  // Context context_archive;
  // loads<CerealT, CerealIArch, CerealOArch>(context_archive, fileName, fldr);


  // Create json archive
  std::string fileNameJSON = fileName + ".json";
  dumps<Context, cereal::JSONInputArchive, cereal::JSONOutputArchive>(context, fileNameJSON, fldr);  

  Context context_json;
  loads<Context, cereal::JSONInputArchive, cereal::JSONOutputArchive>(context_json, fileNameJSON, fldr);
  dumps<CerealT, CerealIArch, CerealOArch>(context_json, fileName, fldr);

  return 0;
}
