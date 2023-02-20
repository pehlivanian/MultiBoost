#include <iostream>
#include <vector>
#include <iterator>

#include "utils.hpp"
#include "gradientboostclassifier.hpp"

using namespace IB_utils;
using namespace ClassifierContext;

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


auto main(int argc, char **argv) -> int {

  using T = Context;

  std::string fileName = path(typeid(T).name());

  ClassifierContext::Context context{};
  // context.loss = lossFunction::Savage;
  // context.loss = lossFunction::BinomialDeviance;
  // context.loss = lossFunction::MSE;
  // context.loss = lossFunction::Exp;
  // context.loss = lossFunction::Arctan;
  context.loss = lossFunction::Synthetic;
  context.partitionSize = 6;
  context.partitionRatio = .25;
  context.learningRate = .0001;
  context.steps = 10000;
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

  writeBinary<Context>(fileName, context);

  std::cout << "Context archive: " << fileName << std::endl;

  return 0;
}
