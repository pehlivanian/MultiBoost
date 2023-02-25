#!/bin/bash

PATH=/home/charles/src/C++/sandbox/Inductive-Boost/build/

INDEX_FILENAME=$1
EXEC_STEP=${PATH}stepwise_predict

$EXEC_STEP \
--indexFileName $INDEX_FILENAME

# IFS=
# while read -r line; do
#     IFS='_'; read -ra tokens <<< "$line"
#     if [ ${tokens[2]} = "CLS" ]; then
# 	$($EXEC_STEP \
# 	    )
#     fi
#     IFS=
# done < $INDEX_NAME_STEP
