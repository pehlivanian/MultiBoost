#!/bin/bash

PATH=/home/charles/src/C++/sandbox/Inductive-Boost/build/
EXEC_CC=${PATH}createContext

CONTEXT_PATH_RUN1=__CTX_RUN1_EtxetnoC7txetnoCreifissa.cxt
CONTEXT_PATH_RUNS=__CTX_RUNS_EtxetnoC7txetnoCreifissa.cxt

$EXEC_CC \
--loss 5 \
--partitionSize 6 \
--partitionRatio .25 \
--learningRate .0001 \
--steps 1000 \
--baseSteps 10000 \
--symmetrizeLabels true \
--removeRedundantLabels false \
--quietRun true \
--rowSubsampleRatio 1. \
--colSubsampleRatio .25 \
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
--serializationWindow 500 \
--fileName $CONTEXT_PATH_RUN1

EXEC_STEP=${PATH}incremental_driver 

# First run
INDEX_NAME_STEP=$($EXEC_STEP \
--contextFileName $CONTEXT_PATH_RUN1 \
--dataName titanic_train \
--warmStart false)

$EXEC_CC \
--loss 5 \
--partitionSize 6 \
--partitionRatio .25 \
--learningRate .0001 \
--steps 1000 \
--baseSteps 10000 \
--symmetrizeLabels true \
--removeRedundantLabels false \
--quietRun true \
--rowSubsampleRatio 1. \
--colSubsampleRatio .25 \
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
--serializationWindow 500 \
--fileName $CONTEXT_PATH_RUNS

n=2
for (( ; ; ));
do
  if [ $n -eq 11 ]; then
    break
  fi

  # Subsequent runs
  INDEX_NAME_STEP=$($EXEC_STEP \
  --contextFileName $CONTEXT_PATH_RUNS \
  --dataName titanic_train \
  --quietRun true \
  --warmStart true \
  --indexName $INDEX_NAME_STEP)

  echo ${n}" : "${INDEX_NAME_STEP}
  ((n=n+1))
done



