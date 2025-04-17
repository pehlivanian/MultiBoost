#!/bin/bash

# Check if script is being sourced or run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being run directly, create a temporary env file
    ENV_FILE=$(mktemp)
    
    # Get absolute path to the project root
    echo "export IB_PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)" > $ENV_FILE
    
    # Set the data directory - change this if your data is stored elsewhere
    if [ -z "$IB_DATA_DIR" ]; then
      echo "export IB_DATA_DIR=$HOME/Data" >> $ENV_FILE
    fi
    
    # Create execution command that sources the env file
    echo "source $ENV_FILE && rm $ENV_FILE" > $ENV_FILE.cmd
    
    echo "Setting environment variables for Inductive-Boost..."
    bash --rcfile $ENV_FILE.cmd -i
    rm $ENV_FILE.cmd
else
    # Script is being sourced, set variables normally
    # Get absolute path to the project root
    export IB_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    echo "Set IB_PROJECT_ROOT to: $IB_PROJECT_ROOT"
    
    # Set the data directory - change this if your data is stored elsewhere
    if [ -z "$IB_DATA_DIR" ]; then
      export IB_DATA_DIR=$HOME/Data
      echo "Set IB_DATA_DIR to: $IB_DATA_DIR"
    else
      echo "Using existing IB_DATA_DIR: $IB_DATA_DIR"
    fi
fi