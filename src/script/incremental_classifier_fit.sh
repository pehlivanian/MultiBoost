#!/bin/bash

DELIM=';'
CLASSIFIER=DecisionTreeClassifier
PATH=/home/charles/src/C++/sandbox/Inductive-Boost/build/

# Context creation
EXEC_CC=${PATH}createContext

# Incremental IS classifier fit
EXEC_INC=${PATH}incremental_classify

# Classify OOS for diagnostics
EXEC_PRED_OOS=${PATH}stepwise_classify

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

CONTEXT_PATH_RUN1=__CTX_RUN1_EtxetnoC7txetnoCreifissa.cxt
CONTEXT_PATH_RUNS=__CTX_RUNS_EtxetnoC7txetnoCreifissa.cxt

# CHILDPARTITIONSIZE=(1000 500 250 100 50 10 5 1)
# CHILDNUMSTEPS=(2 2 3 5 5 3 2 1)
# CHILDPARTITIONSIZE=(12 10 8 6 4 3 2 1)
# CHILDNUMSTEPS=(5 5 4 4 3 3 2 1)
# CHILDLEARNINGRATE=(.0001 .0001 .0002 .0002 .0003 .0003 .0004 .0005)
# CHILDMAXDEPTH=(10 10 10 10 10 10 10 10)
# CHILDMINLEAFSIZE=(1 1 1 1 1 1 1 1)
# CHILDMINIMUMGAINSPLIT=(0. 0. 0. 0. 0. 0. 0. 0.)

# DATANAME=/tabular_benchmark/eye_movements
# DATANAME=breast_w
# DATANAME=analcatdata_cyyoung9302
# DATANAME=colic
# DATANAME=credit_a
# DATANAME=credit_g
# DATANAME=diabetes
# DATANAME=australian
# DATANAME=backache
# DATANAME=biomed
# DATANAME=breast_cancer_wisconsin
# DATANAME=breast
# DATANAME=breast_cancer

((ITERS=$basesteps / $STEPS))
PREFIX="["${dataname}"]"

# create context for first run
$EXEC_CC \
--loss ${loss_fn} \
--childPartitionSize ${childpartitionsize[@]} \
--childNumSteps ${childnumsteps[@]} \
--childLearningRate ${childlearningrate[@]} \
--partitionRatio .25 \
--baseSteps ${basesteps} \
--symmetrizeLabels true \
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
--childPartitionSize ${childpartitionsize[@]} \
--childNumSteps ${childnumsteps[@]} \
--childLearningRate ${childlearningrate[@]} \
--partitionRatio .25 \
--baseSteps ${basesteps} \
--symmetrizeLabels true \
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

# echo ${n}" STEPWISE CLASSIFY :: "${DETAILS}

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


# Classify OOS
$EXEC_PRED_OOS \
--indexFileName $INDEX_NAME_STEP \
--folderName $FOLDER_STEP \
--prefixStr $PREFIX

# Subsequent runs
for (( ; ; ));
do
  if [ $n -gt $ITERS ]; then
    break
  fi

  # Fit step
  INDEX_NAME_STEP=$($EXEC_INC \
  --contextFileName ${FOLDER_STEP}/${CONTEXT_PATH_RUNS} \
  --dataName ${dataname} \
  --splitRatio $SPLITRATIO \
  --quietRun true \
  --mergeIndexFiles true \
  --warmStart true \
  --indexName $INDEX_NAME_STEP \
  --folderName $FOLDER_STEP)

  # echo ${n}" : STEPWISE CLASSIFY :: "${INDEX_NAME_STEP}" "${DETAILS}
  echo ${PREFIX}" ITER: ${n}"

  # Classify OOS
  $EXEC_PRED_OOS \
  --indexFileName $INDEX_NAME_STEP \
  --folderName $FOLDER_STEP \
  --prefixStr $PREFIX

  ((n=n+1))
done