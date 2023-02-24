#include <iostream>
#include <vector>
#include <string>
#include <iterator>

#include <boost/program_options.hpp>

#include "utils.hpp"
#include "loss.hpp"
#include "gradientboostclassifier.hpp"

using namespace IB_utils;
using namespace ClassifierContext;
using namespace LossMeasures;

using namespace boost::program_options;

std::string path(std::string inputString) {
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
  std::istream& operator>>(std::istream& in, lossFunction& loss) {
    int token;
    in >> token;
    switch (token) {
    case (0):
      loss = lossFunction::MSE;
      break;
    case (1):
      loss = lossFunction::BinomialDeviance;
      break;
    case (2):
      loss = lossFunction::Savage;
      break;
    case (3):
      loss = lossFunction::Exp;
      break;
    case (4):
      loss = lossFunction::Arctan;
      break;
    case (5):
      loss = lossFunction::Synthetic;
      break;
    default:
      in.setstate(std::ios_base::failbit);
    }
    return in;
  }

  std::istream& operator>>(std::istream& in, PartitionSize::SizeMethod& meth) {
    int token;
    in >> token;
    switch (token) {
    case (0):
      meth = PartitionSize::SizeMethod::FIXED;
      break;
    case (1):
      meth = PartitionSize::SizeMethod::FIXED_PROPORTION;
      break;
    case (2):
      meth = PartitionSize::SizeMethod::DECREASING;
      break;
    case (3):
      meth = PartitionSize::SizeMethod::INCREASING;
      break;
    case (4):
      meth = PartitionSize::SizeMethod::RANDOM;
      break;
    case (5):
      meth = PartitionSize::SizeMethod::MULTISCALE;
      break;
    default:
      in.setstate(std::ios_base::failbit);
    }
    return in;
  }

  std::istream& operator>>(std::istream& in, LearningRate::RateMethod& meth) {
    int token;
    in >> token;
    switch (token) {
    case (0):
      meth = LearningRate::RateMethod::FIXED;
      break;
    case (1):
      meth = LearningRate::RateMethod::INCREASING;
      break;
    case (2):
      meth = LearningRate::RateMethod::DECREASING;
      break;
    default:
      in.setstate(std::ios_base::failbit);
    }
    return in;
  }
};


auto main(int argc, char **argv) -> int {

  using T = Context;

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
     context.serialize = false;
     context.partitionSizeMethod = PartitionSize::SizeMethod::FIXED; // INCREASING
     context.learningRateMethod = LearningRate::RateMethod::FIXED;   // DECREASING
     context.minLeafSize = 1;
     context.maxDepth = 10;
     context.minimumGainSplit = 0.;
  */

  lossFunction			loss=lossFunction::Synthetic;
  std::size_t			partitionSize		= 6;
  double			partitionRatio		= .25;
  double			learningRate		= .0001;
  int				steps			= 10000;
  int				baseSteps		= 10000;
  bool				symmetrizeLabels	= true;
  bool				removeRedundantLabels	= true;
  bool				quietRun		= false;
  double			rowSubsampleRatio	= 1.;
  double			colSubsampleRatio	= .25;
  bool				recursiveFit		= true;
  PartitionSize::SizeMethod	partitionSizeMethod	= PartitionSize::SizeMethod::FIXED;
  LearningRate::RateMethod	learningRateMethod	= LearningRate::RateMethod::FIXED;
  std::size_t			minLeafSize		= 1;
  double			minimumGainSplit	= 0.;
  std::size_t			maxDepth		= 10;
  std::size_t			numTrees		= 10;
  bool				serialize		= false;
  bool				serializePrediction	= false;
  bool				serializeColMask	= false;
  bool				serializeDataset	= false;
  bool				serializeLabels		= false;
  std::size_t			serializationWindow	= 1000;
  std::string fileName = path(typeid(T).name());

  options_description desc("Options");
  desc.add_options()
    ("help,h", "Help screen")
    ("loss",			value<lossFunction>(&loss),				"loss")
    ("partitionSize",		value<std::size_t>(&partitionSize),			"partitionSize")
    ("partitionRatio",		value<double>(&partitionRatio),				"partitionRatio")
    ("learningRate",		value<double>(&learningRate),				"learningRate")
    ("steps",			value<int>(&steps),					"steps")
    ("baseSteps",		value<int>(&baseSteps),					"baseSteps")
    ("symmetrizeLabels",	value<bool>(&symmetrizeLabels),				"symmetrizeLabels")
    ("removeRedundantLabels",	value<bool>(&removeRedundantLabels),			"removeRedundantLabels")
    ("quietRun",		value<bool>(&quietRun),					"quietRun")
    ("rowSubsampleRatio",	value<double>(&rowSubsampleRatio),			"rowSubsampleRatio")
    ("colSubsampleRatio",	value<double>(&colSubsampleRatio),			"colSubsampleRatio")
    ("recursiveFit",		value<bool>(&recursiveFit),				"recursiveFit")
    ("partitionSizeMethod",	value<PartitionSize::SizeMethod>(&partitionSizeMethod), "partitionSizeMethod")
    ("learningRateMethod",	value<LearningRate::RateMethod>(&learningRateMethod),	"learningRateMethod")
    ("minLeafSize",		value<std::size_t>(&minLeafSize),			"minLeafSize")
    ("minimumGainSplit",	value<double>(&minimumGainSplit),			"minimumGainSplit")
    ("maxDepth",		value<std::size_t>(&maxDepth),				"maxDepth")
    ("numTrees",		value<std::size_t>(&numTrees),				"numTrees")
    ("serialize",		value<bool>(&serialize),				"serialize")
    ("serializePrediction",	value<bool>(&serializePrediction),			"serializePrediction")
    ("serializeColMask",	value<bool>(&serializeColMask),				"serializeColMask")
    ("serializeDataset",	value<bool>(&serializeDataset),				"serializeDataset")
    ("serializeLabels",		value<bool>(&serializeLabels),				"serializeLabels")
    ("serializationWindow",	value<std::size_t>(&serializationWindow),		"serializationWindow")
    ("fileName",		value<std::string>(&fileName),				"fileName for Context");


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

  ClassifierContext::Context context{};

  context.loss		= loss;
  context.partitionSize = partitionSize;
  context.partitionRatio = partitionRatio;
  context.learningRate = learningRate;
  context.steps = steps;
  context.baseSteps = baseSteps;
  context.symmetrizeLabels = symmetrizeLabels;
  context.removeRedundantLabels = removeRedundantLabels;
  context.quietRun = quietRun;
  context.rowSubsampleRatio = rowSubsampleRatio;
  context.colSubsampleRatio = colSubsampleRatio;
  context.recursiveFit = recursiveFit;
  context.partitionSizeMethod = partitionSizeMethod;
  context.learningRateMethod = learningRateMethod;
  context.minLeafSize = minLeafSize;
  context.minimumGainSplit = minimumGainSplit;
  context.maxDepth = maxDepth;
  context.numTrees = numTrees;
  context.serialize = serialize;
  context.serializePrediction = serializePrediction;
  context.serializeDataset = serializeDataset;
  context.serializeLabels = serializeLabels;
  context.serializeColMask = serializeColMask;
  context.serializationWindow = serializationWindow;

  writeBinary<Context>(fileName, context);

  std::cout << "Context archive: " << fileName << std::endl;

  return 0;
}
