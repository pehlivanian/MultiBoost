#!/bin/bash

PATH=/home/charles/src/C++/sandbox/Inductive-Boost/build/
EXEC_CC=${PATH}createContext

CONTEXT_PATH_RUN1=__CTX_RUN1_EtxetnoC7txetnoCreifissa.cxt
CONTEXT_PATH_RUNS=__CTX_RUNS_EtxetnoC7txetnoCreifissa.cxt

STEPS=10
BASESTEPS=1000
LEARNINGRATE=.0001
RECURSIVE_FIT=true
PARTITION_SIZE=100
MINLEAFSIZE=1
MAXDEPTH=10
LOSS_FN=5
COLSUBSAMPLE_RATIO=.85
DATANAME=titanic_train

# STEPS=10
# BASESTEPS=1000
# LEARNINGRATE=.0001
# RECURSIVE_FIT=true
# PARTITION_SIZE=100
# MINLEAFSIZE=2
# MAXDEPTH=10
# LOSS_FN=5
# COLSUBSAMPLE_RATIO=.75
# DATANAME=Hill_Valley_with_noise

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
--symmetrizeLabels true \
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
--minimumGainSplit 0. \
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
--symmetrizeLabels true \
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
--minimumGainSplit 0. \
--serializationWindow 1000 \
--fileName $CONTEXT_PATH_RUNS

EXEC_STEP=${PATH}incremental_driver 

n=1
# First run
INDEX_NAME_STEP=$($EXEC_STEP \
--contextFileName $CONTEXT_PATH_RUN1 \
--dataName $DATANAME \
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
  if [ $n -eq $ITERS ]; then
    break
  fi

  # Fit step
  INDEX_NAME_STEP=$($EXEC_STEP \
  --contextFileName $CONTEXT_PATH_RUNS \
  --dataName $DATANAME \
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


