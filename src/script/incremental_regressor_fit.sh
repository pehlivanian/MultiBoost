#!/bin/bash

DELIM=';'
REGRESSOR=DecisionTreeRegressorRegressor
PATH=/home/charles/src/C++/sandbox/Inductive-Boost/build/

# Context creation
EXEC_CC=${PATH}createContext

# Incremental IS regressor fit
EXEC_INC=${PATH}incremental_predict

# Predict OOS for diagnostics
EXEC_PRED_OOS=${PATH}stepwise_predict

declare -i childpartitionsize
declare -i childnumsteps
declare -f childlearningrate
declare -i childmaxdepth
declare -i childminleafsize
declare -f childminimumgainsplit
declare -a num_args
declare -a dataname
declare -a loss_fn
declare -a recursivefit

dataname=""
basesteps=""
colsubsample_ratio=""

while (( $# )); do
  num_args=$1; shift

  counter=0
  while (( counter++ < ${num_args} )); do
    childpartitionsize[${#childpartitionsize[@]}]=$1; shift
  done

  counter=0
  while (( counter++ < ${num_args} )); do
    childnumsteps[${#childnumsteps[@]}]=$1; shift
  done

  counter=0
  while (( counter++ < ${num_args} )); do
    childlearningrate[${#childlearningrate[@]}]=$1; shift
  done

  counter=0
  while (( counter++ < ${num_args} )); do
    childmaxdepth[${#childmaxdepth[@]}]=$1; shift
  done

  counter=0
  while (( counter++ < ${num_args} )); do
    childminleafsize[${#childminleafsize[@]}]=$1; shift
  done

  counter=0
  while (( counter++ < ${num_args} )); do
    childminimumgainsplit[${#childminimumgainsplit[@]}]=$1; shift
  done

  dataname+=$1; shift
  basesteps+=$1; shift
  loss_fn+=$1; shift
  colsubsample_ratio+=$1; shift
  recursivefit+=$1; shift

done

STEPS=1
SPLITRATIO=0.2

CONTEXT_PATH_RUN1=__CTX_RUN1_EtxetnoC7txetnoCrosserge.cxt
CONTEXT_PATH_RUNS=__CTX_RUNS_EtxetnoC7txetnoCrosserge.cxt

# CHILDPARTITIONSIZE=(1000 500 250 100 20 10 5 1)
# CHILDNUMSTEPS=(1 1 1 3 5 3 1 1 1)
# CHILDLEARNINGRATE=(.001 .001 .002 .002 .003 .003 .004 .004)
# CHILDMAXDEPTH=(20 20 20 20 20 10 5 5)
# CHILDMINLEAFSIZE=(1 1 1 1 1 50 100 100)
# CHILDMINIMUMGAINSPLIT=(0. 0. 0. 0. 0. 0. 0. 0.)

# # CHILDPARTITIONSIZE=(2000 1000 500 250 100 50 25 10 5 1)
# CHILDPARTITIONSIZE=(2 1 5 2 1 5 2 1 5 1)
# CHILDNUMSTEPS=(1 1 2 2 3 4 3 2 2 1)
# CHILDLEARNINGRATE=(.0225 .0225 .025 .025 .0375 .0375 .04 .04 .05 .05)
# # CHILDLEARNINGRATE=(.01 .01 .01 .01 .01 .01 .01 .01 .01 .01)
# CHILDMAXDEPTH=(20 20 20 20 20 20 10 10 5 5)
# CHILDMINLEAFSIZE=(1 1 1 1 1 1 25 50 100 100)
# CHILDMINIMUMGAINSPLIT=(0. 0. 0. 0. 0. 0. 0. 0. 0. 0.)
# STEPS=1
# BASESTEPS=500
# RECURSIVE_FIT=true
# MINLEAFSIZE=1
# MINGAINSPLIT=0.
# MAXDEPTH=20
# LOSS_FN=0
# COLSUBSAMPLE_RATIO=1.

# DATANAME=tabular_benchmark/Regression/Mixed/Mercedes_Benz_Greener_Manufacturing
# DATANAME=tabular_benchmark/Regression/superconduct
# DATANAME=tabular_benchmark/Regression/wine_quality
# DATANAME=tabular_benchmark/Regression/sulfur
# DATANAME=tabular_benchmark/Regression/houses
# DATANAME=tabular_benchmark/Regression/house_sale
# DATANAME=tabular_benchmark/Regression/MiamiHousing2016
# DATANAME=tabular_benchmark/Regression/Bike_Sharing_Demand
# DATANAME=tabular_benchmark/Regression/elevators
# DATANAME=tabular_benchmark/Regression/house_16H
# DATANAME=tabular_benchmark/Regression/yprop_4_1
SPLITRATIO=0.2

((ITERS=$basesteps / $STEPS))
PREFIX="["${dataname}"]"

# create context for first run
$EXEC_CC \
--loss ${loss_fn} \
--partitionRatio .25 \
--childPartitionSize ${childpartitionsize[@]} \
--childNumSteps ${childnumsteps[@]} \
--childLearningRate ${childlearningrate[@]} \
--baseSteps ${basesteps} \
--symmetrizeLabels false \
--removeRedundantLabels false \
--quietRun true \
--rowSubsampleRatio 1. \
--colSubsampleRatio ${colsubsample_ratio} \
--recursiveFit ${recursivefit} \
--serializeModel true \
--serializePrediction true \
--serializeDataset true \
--serializeLabels true \
--partitionSizeMethod 0 \
--learningRateMethod 0 \
--stepSizeMethod 0 \
--childMinLeafSize ${childminleafsize[@]} \
--childMaxDepth ${childmaxdepth[@]} \
--childMinimumGainSplit ${childminimumgainsplit[@]} \
--serializationWindow 10 \
--fileName $CONTEXT_PATH_RUN1

# create context for subsequent runs
$EXEC_CC \
--loss ${loss_fn} \
--partitionRatio .25 \
--childPartitionSize ${childpartitionsize[@]} \
--childNumSteps ${childnumsteps[@]} \
--childLearningRate ${childlearningrate[@]} \
--baseSteps ${basesteps} \
--symmetrizeLabels false \
--removeRedundantLabels false \
--quietRun true \
--rowSubsampleRatio 1. \
--colSubsampleRatio ${colsubsample_ratio} \
--recursiveFit ${recursivefit} \
--serializeModel true \
--serializePrediction true \
--serializeDataset false \
--serializeLabels false \
--partitionSizeMethod 0 \
--learningRateMethod 0 \
--stepSizeMethod 0 \
--childMinLeafSize ${childminleafsize[@]} \
--childMaxDepth ${childmaxdepth[@]} \
--childMinimumGainSplit ${childminimumgainsplit[@]} \
--serializationWindow 10 \
--fileName $CONTEXT_PATH_RUNS

# First run
n=1

STEP_INFO=$($EXEC_INC \
--contextFileName $CONTEXT_PATH_RUN1 \
--dataName ${dataname} \
--splitRatio $SPLITRATIO \
--mergeIndexFiles false \
--warmStart false)

set -- "$STEP_INFO"
IFS=$DELIM; declare -a res=($*)
arg0="${res[0]}"
arg1="${res[1]}"
INDEX_NAME_STEP=$arg0
FOLDER_STEP=$arg1

echo ${PREFIX}" FOLDER: "${FOLDER_STEP}
echo ${PREFIX}" INDEX: "${INDEX_NAME_STEP}
echo ${PREFIX}" ITER: 1"


/bin/mv ${CONTEXT_PATH_RUN1} ${FOLDER_STEP}
/bin/mv ${CONTEXT_PATH_RUNS} ${FOLDER_STEP}

((n=n+1))

# Predict OOS
$EXEC_PRED_OOS \
--indexFileName $INDEX_NAME_STEP \
--folderName $FOLDER_STEP \
--prefixStr $PREFIX

# Subsequent runs
for (( ; ; ));
do
  if [ $n -ge $ITERS ]; then
    break
  fi

  # Fit step
  INDEX_NAME_STEP=$($EXEC_INC \
  --contextFileName ${FOLDER_STEP}/$CONTEXT_PATH_RUNS \
  --dataName ${dataname} \
  --splitRatio $SPLITRATIO \
  --quietRun true \
  --mergeIndexFiles true \
  --warmStart true \
  --indexName $INDEX_NAME_STEP \
  --folderName $FOLDER_STEP)

  echo ${PREFIX}" ITER: ${n}"

  # Predict OOS
  $EXEC_PRED_OOS \
  --indexFileName $INDEX_NAME_STEP \
  --folderName $FOLDER_STEP \
  --prefixStr $PREFIX

  ((n=n+1))
done
