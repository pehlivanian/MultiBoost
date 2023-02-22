#!/bin/bash

PATH=/home/charles/src/C++/sandbox/Inductive-Boost/build/
EXEC_STP1=${PATH}createContext

CONTEXT_PATH=__CTX_TEST_EtxetnoC7txetnoCreifissa.cxt

# all distros should support
# CPUInfo=`lscpu | grep -i "Model name" | cut -d : -f 2 | gawk '{$1=$1}; 1'`
# echo $CPUInfo

$EXEC_STP1 \
--loss 5 \
--partitionSize 6 \
--partitionRatio .25 \
--learningRate .0001 \
--steps 1000 \
--baseSteps 10000 \
--symmetrizeLabels true \
--removeRedundantLabels false \
--rowSubsampleRatio 1. \
--colSubsampleRatio .25 \
--recursiveFit true \
--serialize true \
--serializePrediction true \
--partitionSizeMethod 0 \
--learningRateMethod 0 \
--minLeafSize 1 \
--maxDepth 10 \
--minimumGainSplit 0. \
--serializationWindow 500 \
--fileName $CONTEXT_PATH

EXEC_STP2=${PATH}incremental_driver 

$EXEC_STP2 \
--contextFileName $CONTEXT_PATH \
--dataName titanic_train



