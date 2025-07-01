#!/bin/bash

# Examples
# 
# /home/charles/src/C++/sandbox/Inductive-Boost/src/script/incremental_classifier_fit.sh 1 250 1 0.01 0.75 0 1 0 buggyCrx_train 10 12 1.56 1 1 1 1 -1 1 .2
## SPLIT RATIO: .2
## RUN ON TEST DATASET: 1
## TEST OOS EACH IT: 
## SHOW OOS: 1
# 
# So IS, OOS information will be shown for each iteration if $SHOW_OOS=1
# 
# /home/charles/src/C++/sandbox/Inductive-Boost/src/script/incremental_classifier_fit.sh 1 250 1 0.01 0.75 0 1 0 buggyCrx_train 10 12 1.56 1 1 1 1 -1 0 .2
## SPLIT RATIO: .2
## RUN ON TEST DATASET: 0
## TEST OOS EACH IT: 
## SHOW OOS: 1
# 
# So IS, OOS information will be shown for each iteration if $SHOW_OOS=1 and after the last
# iteration if $SHOW_OOS -ne 1

DELIM=';'
PATH=${IB_PROJECT_ROOT}/build/

SHOW_OOS=${SHOW_OOS:-0}
USE_WEIGHTS=0

# Context creation
EXEC_CC=${PATH}create_context_classifier
EXEC_CC_JSON=${PATH}create_context_from_JSON_classifier

# Incremental IS classifier fit
EXEC_INC=${PATH}incremental_classify

# Classify OOS for diagnostics
EXEC_PRED_OOS=${PATH}stepwise_classify

STEPS=1

CONTEXT_PATH_RUN1=__CTX_RUN1_EtxetnoC7txetnoCreifissa.cxt
CONTEXT_PATH_RUNS=__CTX_RUNS_EtxetnoC7txetnoCreifissa.cxt


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

	    done
	    
	    use_weights=$USE_WEIGHTS
	    
	    if [ -z "$clamp_gradient" ]; then
		clamp_gradient=0
		upper_val=1000
		lower_val=-1000
	    fi

	    if [ -z "$split_ratio" ]; then
		split_ratio=0
		if [ $SHOW_OOS -eq 1 ]; then
		    test_OOS_each_it=1
		fi
	    elif [ $SHOW_OOS -eq 1 ]; then
		test_OOS_each_it=1
	    fi
	    

	    echo -n $PREFIX" "
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
		--useWeights ${use_weights} \
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
	    
	    echo -n $PREFIX" "
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
		--useWeights ${use_weights} \
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

    CONTEXT_PATH_RUN1=__CTX_RUN1_EtxetnoC7txetnoCreifissa.json
    CONTEXT_PATH_RUNS=__CTX_RUNS_EtxetnoC7txetnoCreifissa.json

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

  testdataname=`echo ${dataname} | /usr/bin/gawk '{split($0,a,"_train"); print a[1]}'`
  testdataname=${testdataname}"_test"

  EXEC_TEST_OOS=${PATH}OOS_classify
  PREFIX="["${testdataname}"]"

  $EXEC_TEST_OOS \
  --dataName ${testdataname} \
  --indexName $INDEX_NAME_STEP \
  --folderName $FOLDER_STEP \
  --prefixStr $PREFIX  
else
  # Classify OOS
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

    testdataname=`echo ${dataname} | /usr/bin/gawk '{split($0,a,"_train"); print a[1]}'`
    testdataname=${testdataname}"_test"

    EXEC_TEST_OOS=${PATH}OOS_classify
    PREFIX="["${testdataname}"]"

    $EXEC_TEST_OOS \
    --dataName ${testdataname} \
    --indexName $INDEX_NAME_STEP \
    --folderName $FOLDER_STEP \
    --prefixStr $PREFIX  

  else
    # Classify OOS
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
# the _train dataset, we fitted it on the proportion (1 - $split_ratio)
# above.

if [ $SHOW_OOS -ne 1 ]; then
  if [ ! -z "$runOnTestDataset" ]; then
    # We assume that ${dataname} ends with the pattern r'''_train$'''
    # and we test OOS fit on the dataset with "_test" suffix

    testdataname=`echo ${dataname} | /usr/bin/gawk '{split($0,a,"_train"); print a[1]}'`
    testdataname=${testdataname}"_test"

    EXEC_TEST_OOS=${PATH}OOS_classify
    PREFIX="["${testdataname}"]"

    $EXEC_TEST_OOS \
    --dataName ${testdataname} \
    --indexName $INDEX_NAME_STEP \
    --folderName $FOLDER_STEP \
    --prefixStr $PREFIX
  fi
fi