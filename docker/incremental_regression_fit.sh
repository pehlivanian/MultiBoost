#!/bin/bash

# Get project root directory if not set
if [ -z "${IB_PROJECT_ROOT}" ]; then
  # Try to find the project root (the directory containing CMakeLists.txt)
  export IB_PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
fi

# Allow test data directory to be overridden
if [ ! -z "${TEST_DATA_DIR}" ]; then
  export IB_DATA_DIR="${TEST_DATA_DIR}"
fi

DELIM=';'
PATH=${IB_PROJECT_ROOT}/build/

SHOW_OOS=${SHOW_OOS:-0}
USE_WEIGHTS=0

# Context creation
EXEC_CC=${PATH}create_context_regressor
EXEC_CC_JSON=${PATH}create_context_from_JSON_regressor

# Incremental IS regressor fit
EXEC_INC=${PATH}incremental_predict

# Predict OOS for diagnostics
EXEC_PRED_OOS=${PATH}stepwise_predict


STEPS=1

CONTEXT_PATH_RUN1=__CTX_RUN1_EtxetnoC7txetnoCrosserge.cxt
CONTEXT_PATH_RUNS=__CTX_RUNS_EtxetnoC7txetnoCrosserge.cxt

if [[ $# -ne 3 ]]; then

	    declare -i childpartitionsize
	    declare -i childnumsteps
	    declare -f childlearningrate
	    declare -f childactivepartitionratio
	    declare -i childmaxdepth
	    declare -i childminleafsize
	    declare -f childminimumgainsplit
	    declare -a num_args
	    declare -a dataname
	    declare -a loss_fn
	    declare -f loss_power
	    declare -a recursivefit
	    declare -a clamp_gradient
	    declare -a upper_val
	    declare -a lower_val
	    declare -a runOnTestDataset

	    dataname=""
	    testdataname=""
	    steps=""
	    colsubsample_ratio=""
	    split_ratio=""

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
			childactivepartitionratio[${#childactivepartitionratio[@]}]=$1; shift
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
		    steps+=$1; shift
		    loss_fn+=$1; shift
		    loss_power+=${1:0}; shift
		    colsubsample_ratio+=$1; shift
		    recursivefit+=$1; shift
		    clamp_gradient+=${1:0}; shift
		    upper_val+=${1:0}; shift
		    lower_val+=${1:0}; shift
		    runOnTestDataset+=${1:0}; shift
		    split_ratio+=$1; shift
		    
		    # Check if optional test dataset name is provided
		    if [[ $# -gt 0 ]]; then
		        testdataname+=$1; shift
		    fi

	    done

	    if [ -z "$clamp_gradient" ]; then
		clamp_gradient=0
		upper_val=0
		lower_val=0
	    fi

	    if [ -z "$split_ratio" ]; then
		split_ratio=0
		test_OOS_each_it=1
	    fi
	    
	    # Check if we should test on custom test dataset at each step
	    # Only enable if testdataname is provided AND showOOSEachStep is true (SHOW_OOS=1)
	    if [ ! -z "$testdataname" ] && [ $SHOW_OOS -eq 1 ]; then
		test_OOS_each_it=1
	    fi


            # create context for first run
	    $EXEC_CC \
		--loss ${loss_fn} \
		--lossPower ${loss_power} \
		--clamp_gradient ${clamp_gradient} \
		--upper_val ${upper_val} \
		--lower_val ${lower_val} \
		--childPartitionSize ${childpartitionsize[@]} \
		--childNumSteps ${childnumsteps[@]} \
		--childLearningRate ${childlearningrate[@]} \
		--childActivePartitionRatio ${childactivepartitionratio[@]} \
		--steps ${steps} \
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
		--childMinLeafSize ${childminleafsize[@]} \
		--childMaxDepth ${childmaxdepth[@]} \
		--childMinimumGainSplit ${childminimumgainsplit[@]} \
		--serializationWindow 10 \
		--fileName $CONTEXT_PATH_RUN1

            # create context for subsequent runs
	    $EXEC_CC \
		--loss ${loss_fn} \
		--lossPower ${loss_power} \
		--clamp_gradient ${clamp_gradient} \
		--upper_val ${upper_val} \
		--lower_val ${lower_val} \
		--childPartitionSize ${childpartitionsize[@]} \
		--childNumSteps ${childnumsteps[@]} \
		--childLearningRate ${childlearningrate[@]} \
		--childActivePartitionRatio ${childactivepartitionratio[@]} \
		--steps ${steps} \
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
		--childMinLeafSize ${childminleafsize[@]} \
		--childMaxDepth ${childmaxdepth[@]} \
		--childMinimumGainSplit ${childminimumgainsplit[@]} \
		--serializationWindow 10 \
		--fileName $CONTEXT_PATH_RUNS
else

    CONTEXT_PATH_RUN1=__CTX_RUN1_EtxetnoC7txetnoCrosserge.json
    CONTEXT_PATH_RUNS=__CTX_RUNS_EtxetnoC7txetnoCrosserge.json

    while (( $# )); do
	    dataname=$1; shift
	    ifilename=$1; shift
	    steps=$1; shift
    done

    $EXEC_CC_JSON \
	--dataname ${dataname} \
	--ifileName ${ifilename} \
	--ofileName1 $CONTEXT_PATH_RUN1 \
	--ofileNamen $CONTEXT_PATH_RUNS

    echo "STEPS = "${steps}

    split_ratio=0
    runOnTestDataset=1

fi

((ITERS=$steps / $STEPS))
PREFIX="["${dataname}"]"


# First run
n=1

STEP_INFO=$($EXEC_INC \
    --contextFileName $CONTEXT_PATH_RUN1 \
    --dataName ${dataname} \
    --splitRatio ${split_ratio} \
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

if [ ! -z "$test_OOS_each_it" ]; then

    # Use custom test dataset name if provided, otherwise derive from train dataset name
    if [ -z "$testdataname" ]; then
        testdataname=`echo ${dataname} | /usr/bin/gawk '{split($0,a,"_train"); print a[1]}'`
        testdataname=${testdataname}"_test"
    fi

    EXEC_TEST_OOS=${PATH}OOS_predict
    PREFIX="["${testdataname}"]"

    $EXEC_TEST_OOS \
	--dataName ${testdataname} \
	--indexName $INDEX_NAME_STEP \
	--folderName $FOLDER_STEP \
	--prefixStr $PREFIX  
else
  # Predict OOS
    if [ $SHOW_OOS -eq 1 ]; then
	$EXEC_PRED_OOS \
	    --indexFileName $INDEX_NAME_STEP \
	    --folderName $FOLDER_STEP \
	    --prefixStr $PREFIX
    fi
fi

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
      --splitRatio ${split_ratio} \
      --quietRun true \
      --mergeIndexFiles true \
      --warmStart true \
      --indexName $INDEX_NAME_STEP \
      --folderName $FOLDER_STEP)

  echo ${PREFIX}" ITER: ${n}"

  if [ ! -z "$test_OOS_each_it" ]; then

      # Use custom test dataset name if provided, otherwise derive from train dataset name
      if [ -z "$testdataname" ]; then
          testdataname=`echo ${dataname} | /usr/bin/gawk '{split($0,a,"_train"); print a[1]}'`
          testdataname=${testdataname}"_test"
      fi

      EXEC_TEST_OOS=${PATH}OOS_predict
      PREFIX="["${testdataname}"]"

      $EXEC_TEST_OOS \
	  --dataName ${testdataname} \
	  --indexName $INDEX_NAME_STEP \
	  --folderName $FOLDER_STEP \
	  --prefixStr $PREFIX  

  else
    # Predict OOS
      if [ $SHOW_OOS -eq 1 ]; then

	  $EXEC_PRED_OOS \
	      --indexFileName $INDEX_NAME_STEP \
	      --folderName $FOLDER_STEP \
	      --prefixStr $PREFIX
      fi
  fi

  ((n=n+1))
done

# Final OOS test prediction
# It is assumed that the {*_train_X.csv, *_train_y.csv} datasets
# have companion {*_test_X.csv, *_test_y.csv} datasets on which
# we test the above fitted archived regressor.

# Note: The proper procedure would be to fit a model on all of
# the _train dataset, we fitted it on the proportion (1 - ${split_ratio})
# above.

if [ $SHOW_OOS -ne 1 ]; then
    if [ ! -z "$runOnTestDataset" ]; then
    # We assume that ${dataname} ends with the pattern r'''_train$'''
    # and we test OOS fit on the dataset with "_test" suffix

    # IS
	EXEC_TEST_OOS=${PATH}OOS_predict
	PREFIX="["${dataname}"]"

	$EXEC_TEST_OOS \
	    --dataName ${dataname} \
	    --indexName $INDEX_NAME_STEP \
	    --folderName $FOLDER_STEP \
	    --prefixStr $PREFIX

    # OOS
	# Use custom test dataset name if provided, otherwise derive from train dataset name
	if [ -z "$testdataname" ]; then
	    testdataname=`echo ${dataname} | /usr/bin/gawk '{split($0,a,"_train"); print a[1]}'`
	    testdataname=${testdataname}"_test"
	fi

	EXEC_TEST_OOS=${PATH}OOS_predict
	PREFIX="["${testdataname}"]"

	$EXEC_TEST_OOS \
	    --dataName ${testdataname} \
	    --indexName $INDEX_NAME_STEP \
	    --folderName $FOLDER_STEP \
	    --prefixStr $PREFIX
    fi
fi

