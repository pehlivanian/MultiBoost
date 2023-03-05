#!/bin/bash

PATH=/home/charles/src/C++/sandbox/Inductive-Boost/build/
EXEC_CC=${PATH}createContext

CONTEXT_PATH_RUN1=__CTX_RUN1_EtxetnoC7txetnoCreifissa.cxt
CONTEXT_PATH_RUNS=__CTX_RUNS_EtxetnoC7txetnoCreifissa.cxt

STEPS=101

# Predict OOS
EXEC_PRED=${PATH}stepwise_predict

# create context for first run
$EXEC_CC \
--loss 6 \
--partitionSize 24 \
--partitionRatio .15 \
--learningRate .00001 \
--steps 100 \
--baseSteps 10000 \
--symmetrizeLabels true \
--removeRedundantLabels false \
--quietRun true \
--rowSubsampleRatio 1. \
--colSubsampleRatio .85 \
--recursiveFit true \
--serialize true \
--serializePrediction true \
--serializeDataset true \
--serializeLabels true \
--partitionSizeMethod 0 \
--learningRateMethod 0 \
--minLeafSize 1 \
--maxDepth 10 \
--minimumGainSplit 0. \
--serializationWindow 1000 \
--fileName $CONTEXT_PATH_RUN1

# create context for subsequent runs
$EXEC_CC \
--loss 6 \
--partitionSize 24 \
--partitionRatio .15 \
--learningRate .00001 \
--steps 100 \
--baseSteps 10000 \
--symmetrizeLabels true \
--removeRedundantLabels false \
--quietRun true \
--rowSubsampleRatio 1. \
--colSubsampleRatio .85 \
--recursiveFit true \
--serialize true \
--serializePrediction true \
--serializeDataset false \
--serializeLabels false \
--partitionSizeMethod 0 \
--learningRateMethod 0 \
--minLeafSize 1 \
--maxDepth 10 \
--minimumGainSplit 0. \
--serializationWindow 1000 \
--fileName $CONTEXT_PATH_RUNS

EXEC_STEP=${PATH}incremental_driver 

n=1
# First run
INDEX_NAME_STEP=$($EXEC_STEP \
--contextFileName $CONTEXT_PATH_RUN1 \
--dataName titanic_train \
--mergeIndexFiles false \
--warmStart false)

echo ${n}" : "${INDEX_NAME_STEP}
((n=n+1))

# Predict OOS
$EXEC_PRED \
--indexFileName $INDEX_NAME_STEP

# Subsequent runs
for (( ; ; ));
do
  if [ $n -eq $STEPS ]; then
    break
  fi

  # Fit step
  INDEX_NAME_STEP=$($EXEC_STEP \
  --contextFileName $CONTEXT_PATH_RUNS \
  --dataName titanic_train \
  --quietRun true \
  --mergeIndexFiles true \
  --warmStart true \
  --indexName $INDEX_NAME_STEP)

  echo ${n}" : "${INDEX_NAME_STEP}

  # Predict OOS
  $EXEC_PRED \
  --indexFileName $INDEX_NAME_STEP

  ((n=n+1))
done


