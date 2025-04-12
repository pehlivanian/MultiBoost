#!/bin/bash

# Get absolute path to the project root
export IB_PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
echo "Set IB_PROJECT_ROOT to: $IB_PROJECT_ROOT"

# Set the data directory - change this if your data is stored elsewhere
if [ -z "$IB_DATA_DIR" ]; then
  export IB_DATA_DIR=$HOME/Data
  echo "Set IB_DATA_DIR to: $IB_DATA_DIR"
else
  echo "Using existing IB_DATA_DIR: $IB_DATA_DIR"
fi

# To use this script, run:
# source setup_env.sh