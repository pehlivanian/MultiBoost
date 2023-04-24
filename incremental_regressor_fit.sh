#!/bin/bash

REGRESSOR=DecisionTreeRegressorRegressor
PATH=/home/charles/src/C++/sandbox/Inductive-Boost/build/
EXEC_CC=${PATH}createContext

CONTEXT_PATH_RUN1=__CTX_RUN1_EtxetnoC7txetnoCrosserge.cxt
CONTEXT_PATH_RUNS=__CTX_RUNS_EtxetnoC7txetnoCrosserge.cxt

STEPS=100
BASESTEPS=1000
LEARNINGRATE=1.
RECURSIVE_FIT=true
PARTITION_SIZE=100
MINLEAFSIZE=1
MINGAINSPLIT=0.
MAXDEPTH=10
LOSS_FN=0
COLSUBSAMPLE_RATIO=.85
DATANAME=1193_BNG_lowbwt

((ITERS=$BASESTEPS / $STEPS))

# Predict OOS
EXEC_PRED=${PATH}stepwise_predict

# create context for first run
$EXEC_CC \
--loss $LOSS_FN \
--partitionSize $PARTITION_SIZE \
--partitionRatio .25 \
--learningRate $LEARNINGRATE \
--steps $STEPS \
--baseSteps $BASESTEPS \
--symmetrizeLabels false \
--removeRedundantLabels false \
--quietRun true \
--rowSubsampleRatio 1. \
--colSubsampleRatio $COLSUBSAMPLE_RATIO \
--recursiveFit $RECURSIVE_FIT \
--serialize true \
--serializePrediction true \
--serializeDataset true \
--serializeLabels true \
--partitionSizeMethod 0 \
--learningRateMethod 0 \
--stepSizeMethod 0 \
--minLeafSize $MINLEAFSIZE \
--maxDepth $MAXDEPTH \
--minimumGainSplit $MINGAINSPLIT \
--serializationWindow 1000 \
--fileName $CONTEXT_PATH_RUN1

# create context for subsequent runs
$EXEC_CC \
--loss $LOSS_FN \
--partitionSize $PARTITION_SIZE \
--partitionRatio .25 \
--learningRate $LEARNINGRATE \
--steps $STEPS \
--baseSteps $BASESTEPS \
--symmetrizeLabels false \
--removeRedundantLabels false \
--quietRun true \
--rowSubsampleRatio 1. \
--colSubsampleRatio $COLSUBSAMPLE_RATIO \
--recursiveFit $RECURSIVE_FIT \
--serialize true \
--serializePrediction true \
--serializeDataset false \
--serializeLabels false \
--partitionSizeMethod 0 \
--learningRateMethod 0 \
--stepSizeMethod 0 \
--minLeafSize $MINLEAFSIZE \
--maxDepth $MAXDEPTH \
--minimumGainSplit $MINGAINSPLIT \
--serializationWindow 1000 \
--fileName $CONTEXT_PATH_RUNS
