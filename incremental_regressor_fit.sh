#!/bin/bash

REGRESSOR=DecisionTreeRegressorRegressor
PATH=/home/charles/src/C++/sandbox/Inductive-Boost/build/
EXEC_CC=${PATH}createContext

CONTEXT_PATH_RUN1=__CTX_RUN1_EtxetnoC7txetnoCrosserge.cxt
CONTEXT_PATH_RUNS=__CTX_RUNS_EtxetnoC7txetnoCrosserge.cxt

STEPS=100
BASESTEPS=10000
LEARNINGRATE=.01
RECURSIVE_FIT=true
PARTITION_SIZE=400
MINLEAFSIZE=1
MINGAINSPLIT=0.
MAXDEPTH=10
LOSS_FN=0
COLSUBSAMPLE_RATIO=.1
DATANAME=1193_BNG_lowbwt
SPLITRATIO=0.99

((ITERS=$BASESTEPS / $STEPS))

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

# Incremental IS regressor fit
EXEC_INC=${PATH}incremental_predict

# Predict OOS for diagnostics
EXEC_PRED_OOS=${PATH}stepwise_predict

# First run
n=1

INDEX_NAME_STEP=$($EXEC_INC \
--contextFileName $CONTEXT_PATH_RUN1 \
--dataName $DATANAME \
--splitRatio $SPLITRATIO \
--mergeIndexFiles false \
--warmStart false)

echo ${n}" : "${INDEX_NAME_STEP}
((n=n+1))

# Predict OOS
$EXEC_PRED_OOS \
--indexFileName $INDEX_NAME_STEP

# Subsequent runs
for (( ; ; ));
do
  if [ $n -eq $ITERS ]; then
    break
  fi

  # Fit step
  INDEX_NAME_STEP=$($EXEC_INC \
  --contextFileName $CONTEXT_PATH_RUNS \
  --dataName $DATANAME \
  --splitRatio $SPLITRATIO \
  --quietRun true \
  --mergeIndexFiles true \
  --warmStart true \
  --indexName $INDEX_NAME_STEP)

  echo ${n}" : "${INDEX_NAME_STEP}

  # Predict OOS
  $EXEC_PRED_OOS \
  --indexFileName $INDEX_NAME_STEP

  ((n=n+1))
done


