#ifndef __CONTEXTMANAGER_IMPL_HPP__
#define __CONTEXTMANAGER_IMPL_HPP__

template<typename ClassifierType>
void
ContextManager::contextInit(CompositeClassifier<ClassifierType>& c, const Context& context) {
  
  c.loss_			= context.loss;
  c.lossPower_			= context.lossPower;
  c.clamp_gradient_		= context.clamp_gradient;
  c.upper_val_			= context.upper_val;
  c.lower_val_			= context.lower_val;

  c.partitionSize_		= context.childPartitionSize[0];
  c.steps_			= context.childNumSteps[0];
  c.learningRate_		= context.childLearningRate[0];
  c.activePartitionRatio_	= context.childActivePartitionRatio[0];

  c.minLeafSize_		= context.childMinLeafSize[0];
  c.maxDepth_			= context.childMaxDepth[0];
  c.minimumGainSplit_		= context.childMinimumGainSplit[0];
  
  c.baseSteps_			= context.baseSteps;
  c.symmetrized_		= context.symmetrizeLabels;
  c.removeRedundantLabels_	= context.removeRedundantLabels;
  c.quietRun_			= context.quietRun;
  c.useWeights_			= context.useWeights;
  c.row_subsample_ratio_	= context.rowSubsampleRatio;
  c.col_subsample_ratio_	= context.colSubsampleRatio;
  c.recursiveFit_		= context.recursiveFit;

  c.childPartitionSize_		= context.childPartitionSize;
  c.childNumSteps_		= context.childNumSteps;
  c.childLearningRate_		= context.childLearningRate;
  c.childActivePartitionRatio_	= context.childActivePartitionRatio;

  c.childMinLeafSize_		= context.childMinLeafSize;
  c.childMaxDepth_		= context.childMaxDepth;
  c.childMinimumGainSplit_	= context.childMinimumGainSplit;

  c.numTrees_			= context.numTrees;
  c.serializeModel_		= context.serializeModel;
  c.serializePrediction_	= context.serializePrediction;
  c.serializeColMask_		= context.serializeColMask;
  c.serializeDataset_		= context.serializeDataset;
  c.serializeLabels_		= context.serializeLabels;
  c.serializationWindow_	= context.serializationWindow;

  c.depth_			= context.depth;

}

template<typename ClassifierType>
void
ContextManager::childContext(Context& context, const CompositeClassifier<ClassifierType>& c) {

  auto [partitionSize, 
	stepSize, 
	learningRate, 
	activePartitionRatio] = computeChildPartitionInfo(c);
  auto [maxDepth, 
	minLeafSize, 
	minimumGainSplit] = computeChildModelInfo(c);

  context.loss			= c.loss_;
  context.lossPower		= c.lossPower_;
  context.clamp_gradient	= c.clamp_gradient_;
  context.upper_val		= c.upper_val_;
  context.lower_val		= c.lower_val_;
  context.baseSteps		= c.baseSteps_;
  context.symmetrizeLabels	= false;
  context.removeRedundantLabels	= true;
  context.rowSubsampleRatio	= c.row_subsample_ratio_;
  context.colSubsampleRatio	= c.col_subsample_ratio_;
  context.recursiveFit		= true;
  context.useWeights		= c.useWeights_;
  context.numTrees		= c.numTrees_;

  context.partitionSize		= partitionSize+1;
  context.steps			= stepSize;
  context.learningRate		= learningRate;
  context.activePartitionRatio	= activePartitionRatio;

  context.maxDepth		= maxDepth;
  context.minLeafSize		= minLeafSize;
  context.minimumGainSplit	= minimumGainSplit;

  auto it = std::find(c.childPartitionSize_.cbegin(), c.childPartitionSize_.cend(), partitionSize);
  auto ind = std::distance(c.childPartitionSize_.cbegin(), it);

  // Must ensure that ind > 0; this may happen if partition size is the same through 2 steps
  ind = ind > 0 ? ind : 1;

  context.childPartitionSize	= std::vector<std::size_t>(c.childPartitionSize_.cbegin()+ind,
							   c.childPartitionSize_.cend());
  context.childNumSteps		= std::vector<std::size_t>(c.childNumSteps_.cbegin()+ind,
							   c.childNumSteps_.cend());
  context.childLearningRate	= std::vector<double>(c.childLearningRate_.cbegin()+ind,
						      c.childLearningRate_.cend());
  context.childActivePartitionRatio = std::vector<double>(c.childActivePartitionRatio_.cbegin()+ind,
							  c.childActivePartitionRatio_.cend());
  // Model args
  context.childMinLeafSize	= std::vector<std::size_t>(c.childMinLeafSize_.cbegin()+ind,
							   c.childMinLeafSize_.cend());
  context.childMaxDepth		= std::vector<std::size_t>(c.childMaxDepth_.cbegin()+ind,
							   c.childMaxDepth_.cend());
  context.childMinimumGainSplit	= std::vector<double>(c.childMinimumGainSplit_.cbegin()+ind,
						      c.childMinimumGainSplit_.cend());

  context.depth			= c.depth_ + 1;

}


template<typename ClassifierType>
auto
ContextManager::computeChildPartitionInfo(const CompositeClassifier<ClassifierType>& c) -> childPartitionInfo {
  return std::make_tuple(c.childPartitionSize_[1],
			 c.childNumSteps_[1],
			 c.childLearningRate_[1],
			 c.childActivePartitionRatio_[1]);
}

template<typename ClassifierType>
auto 
ContextManager::computeChildModelInfo(const CompositeClassifier<ClassifierType>& c) -> childModelInfo {
  return std::make_tuple(c.childMaxDepth_[1],
			 c.childMinLeafSize_[1],
			 c.childMinimumGainSplit_[1]);
}

#endif
