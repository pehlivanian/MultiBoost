#!/bin/bash

DELIM=';'
CLASSIFIER=DecisionTreeClassifier
PATH=/home/charles/src/C++/sandbox/Inductive-Boost/build/
EXEC_CC=${PATH}createContext

CONTEXT_PATH_RUN1=__CTX_RUN1_EtxetnoC7txetnoCreifissa.cxt
CONTEXT_PATH_RUNS=__CTX_RUNS_EtxetnoC7txetnoCreifissa.cxt
CHILDPARTITIONSIZE=(1000 500 250 100 50 10 5 1)
CHILDNUMSTEPS=(2 2 3 5 5 3 2 1)
CHILDLEARNINGRATE=(.0001 .0001 .0002 .0002 .0003 .0003 .0004 .0005)
CHILDMAXDEPTH=(10 10 10 10 10 10 10 10)
CHILDMINLEAFSIZE=(1 1 1 1 1 1 1 1)
CHILDMINIMUMGAINSPLIT=(0. 0. 0. 0. 0. 0. 0. 0.)
STEPS=1
BASESTEPS=150
RECURSIVE_FIT=true
LOSS_FN=7
COLSUBSAMPLE_RATIO=.85
# DATANAME=/tabular_benchmark/eye_movements
# DATANAME=breast_w
# DATANAME=analcatdata_cyyoung9302
# DATANAME=colic
# DATANAME=credit_a
# DATANAME=credit_g
# DATANAME=diabetes
DATANAME=australian
# DATANAME=backache
# DATANAME=biomed
# DATANAME=breast_cancer_wisconsin
# DATANAME=breast
# DATANAME=breast_cancer
# DATANAME=analcatdata_boxing1

SPLITRATIO=0.2


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
PREFIX="["${DATANAME}"]"

# create context for first run
$EXEC_CC \
--loss $LOSS_FN \
--childPartitionSize ${CHILDPARTITIONSIZE[@]} \
--childNumSteps ${CHILDNUMSTEPS[@]} \
--childLearningRate ${CHILDLEARNINGRATE[@]} \
--partitionRatio .25 \
--baseSteps $BASESTEPS \
--symmetrizeLabels true \
--removeRedundantLabels false \
--quietRun true \
--rowSubsampleRatio 1. \
--colSubsampleRatio $COLSUBSAMPLE_RATIO \
--recursiveFit $RECURSIVE_FIT \
--serializeModel true \
--serializePrediction true \
--serializeDataset true \
--serializeLabels true \
--partitionSizeMethod 0 \
--learningRateMethod 0 \
--stepSizeMethod 0 \
--childMinLeafSize ${CHILDMINLEAFSIZE[@]} \
--childMaxDepth ${CHILDMAXDEPTH[@]} \
--childMinimumGainSplit ${CHILDMINIMUMGAINSPLIT[@]} \
--serializationWindow 10 \
--fileName $CONTEXT_PATH_RUN1

# create context for subsequent runs
$EXEC_CC \
--loss $LOSS_FN \
--childPartitionSize ${CHILDPARTITIONSIZE[@]} \
--childNumSteps ${CHILDNUMSTEPS[@]} \
--childLearningRate ${CHILDLEARNINGRATE[@]} \
--partitionRatio .25 \
--baseSteps $BASESTEPS \
--symmetrizeLabels true \
--removeRedundantLabels false \
--quietRun true \
--rowSubsampleRatio 1. \
--colSubsampleRatio $COLSUBSAMPLE_RATIO \
--recursiveFit $RECURSIVE_FIT \
--serializeModel true \
--serializePrediction true \
--serializeDataset false \
--serializeLabels false \
--partitionSizeMethod 0 \
--learningRateMethod 0 \
--stepSizeMethod 0 \
--childMinLeafSize ${CHILDMINLEAFSIZE[@]} \
--childMaxDepth ${CHILDMAXDEPTH[@]} \
--childMinimumGainSplit ${CHILDMINIMUMGAINSPLIT[@]} \
--serializationWindow 10 \
--fileName $CONTEXT_PATH_RUNS

# Details
# DETAILS=${INDEX_NAME_STEP}"(Dataset, Rcsv, lrate, parSize) = ("${DATANAME}", "${RECURSIVE_FIT}", "${LEARNINGRATE}", "${PARTITION_SIZE}")"

# Incremental IS classifier fit
EXEC_INC=${PATH}incremental_classify

# Classify OOS for diagnostics
EXEC_PRED_OOS=${PATH}stepwise_classify

# First run
n=1

# echo ${n}" STEPWISE CLASSIFY :: "${DETAILS}

STEP_INFO=$($EXEC_INC \
--contextFileName $CONTEXT_PATH_RUN1 \
--dataName $DATANAME \
--splitRatio $SPLITRATIO \
--mergeIndexFiles false \
--warmStart false)

set -- "$STEP_INFO"
IFS=$DELIM; declare -a res=($*)
arg0="${res[0]}"
arg1="${res[1]}"
INDEX_NAME_STEP=$arg0
FOLDER_STEP=$arg1

echo ${PREFIX}" ITER: 1"
echo ${PREFIX}" FOLDER: "${FOLDER_STEP}
echo ${PREFIX}" INDEX: "${INDEX_NAME_STEP}

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
  if [ $n -eq $ITERS ]; then
    break
  fi

  # Fit step
  INDEX_NAME_STEP=$($EXEC_INC \
  --contextFileName ${FOLDER_STEP}/${CONTEXT_PATH_RUNS} \
  --dataName $DATANAME \
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
