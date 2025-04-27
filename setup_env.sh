#!/bin/bash

# Check if script is being sourced or run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being run directly, create a temporary env file
    ENV_FILE=$(mktemp)
    
    # Get absolute path to the project root
    echo "export IB_PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)" > $ENV_FILE
    
    # Set the data directory - check if $HOME/Data exists, use /opt/data if it doesn't
    if [ -z "$IB_DATA_DIR" ]; then
      if [ -d "$HOME/Data" ]; then
        echo "export IB_DATA_DIR=$HOME/Data" >> $ENV_FILE
      else
        echo "export IB_DATA_DIR=/opt/data" >> $ENV_FILE
      fi
    fi
    
    # Also set IB_TEST_DATA_DIR to the same location
    echo "export IB_TEST_DATA_DIR=\$IB_DATA_DIR" >> $ENV_FILE
    
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
    
    # Set the data directory - check if $HOME/Data exists, use /opt/data if it doesn't
    if [ -z "$IB_DATA_DIR" ]; then
      if [ -d "$HOME/Data" ]; then
        export IB_DATA_DIR=$HOME/Data
        echo "Set IB_DATA_DIR to: $IB_DATA_DIR"
      else
        export IB_DATA_DIR=/opt/data
        echo "Set IB_DATA_DIR to: $IB_DATA_DIR (\$HOME/Data not found)"
      fi
    else
      echo "Using existing IB_DATA_DIR: $IB_DATA_DIR"
    fi
    
    # Also set IB_TEST_DATA_DIR to the same location
    export IB_TEST_DATA_DIR=$IB_DATA_DIR
    echo "Set IB_TEST_DATA_DIR to: $IB_TEST_DATA_DIR"
fi