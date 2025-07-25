#!/usr/bin/env python3
"""
REST API server to expose incremental regression fit script as an endpoint.
"""

import json
import subprocess
import tempfile
import os
import sys
import urllib.request
import urllib.parse
import shutil
from pathlib import Path
from flask import Flask, request, jsonify
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

app = Flask(__name__)

# Script path - will be copied from source during docker build
SCRIPT_PATH = "/opt/multiboost/scripts/incremental_regression_fit.sh"

def handle_local_dataset(local_data_path, params):
    """
    Handle local dataset path from mounted volume.
    
    local_data_path should be the mounted path in the container where
    the user's local data files are accessible.
    
    Returns the dataset name to use.
    """
    try:
        # Extract dataset name from path or use provided name
        dataset_name = params.get('datasetName', Path(local_data_path).name)
        
        # The local_data_path should be the mounted directory containing the data files
        # We expect the script to find files like:
        # {local_data_path}/{dataset_name}_train_X.csv
        # {local_data_path}/{dataset_name}_train_y.csv
        # etc.
        
        # Verify the path exists
        local_path = Path(local_data_path)
        if not local_path.exists():
            raise ValueError(f"Local data path {local_data_path} does not exist. Make sure to mount it with -v")
        
        # Set the data directory to the mounted path
        return f"{local_data_path.rstrip('/')}/{dataset_name}"
        
    except Exception as e:
        raise ValueError(f"Failed to handle local dataset: {str(e)}")

def download_s3_dataset(s3_config, train_dataset_name, test_dataset_name=None):
    """
    Download dataset files from S3 to local temporary directory.
    
    s3_config should contain:
    {
        "bucket": "bucket-name",
        "prefix": "path/to/dataset/",  # optional
        "access_key": "AWS_ACCESS_KEY",
        "secret_key": "AWS_SECRET_KEY",
        "region": "us-east-1"  # optional, defaults to us-east-1
    }
    
    train_dataset_name: Name for training dataset (e.g., "synthetic_train")
    test_dataset_name: Optional name for test dataset (e.g., "synthetic_test")
    
    Returns tuple: (train_path, test_path) where test_path is None if no test dataset
    """
    try:
        # Extract S3 configuration
        bucket = s3_config['bucket']
        prefix = s3_config.get('prefix', '').rstrip('/')
        region = s3_config.get('region', 'us-east-1')
        
        # Initialize S3 client - use environment variables or IAM role for credentials
        access_key = s3_config.get('access_key') or os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = s3_config.get('secret_key') or os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if access_key and secret_key:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )
        else:
            # Use default credentials (environment variables, IAM role, etc.)
            s3_client = boto3.client('s3', region_name=region)
        
        # Create temporary directory for dataset
        temp_dir = tempfile.mkdtemp(prefix='s3_dataset_')
        
        # Build file patterns based on provided dataset names
        file_patterns = []
        
        # Add training files
        file_patterns.extend([
            f"{train_dataset_name}_X.csv",
            f"{train_dataset_name}_y.csv"
        ])
        
        # Add test files if test dataset name provided
        if test_dataset_name:
            file_patterns.extend([
                f"{test_dataset_name}_X.csv", 
                f"{test_dataset_name}_y.csv"
            ])
        
        downloaded_files = []
        
        # Download each file that exists
        for pattern in file_patterns:
            s3_key = f"{prefix}/{pattern}" if prefix else pattern
            local_file_path = os.path.join(temp_dir, pattern)
            
            try:
                s3_client.download_file(bucket, s3_key, local_file_path)
                downloaded_files.append(pattern)
                print(f"Downloaded {s3_key} to {local_file_path}")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    # File doesn't exist, skip it
                    continue
                else:
                    raise
        
        if not downloaded_files:
            raise ValueError(f"No dataset files found in S3 bucket {bucket} with prefix {prefix}")
        
        print(f"Successfully downloaded {len(downloaded_files)} files: {downloaded_files}")
        
        # Return paths for train and test datasets
        train_path = f"{temp_dir}/{train_dataset_name}"
        test_path = f"{temp_dir}/{test_dataset_name}" if test_dataset_name else None
        
        return train_path, test_path
        
    except NoCredentialsError:
        raise ValueError("AWS credentials not provided or invalid")
    except ClientError as e:
        raise ValueError(f"S3 error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to download S3 dataset: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "multiboost-regression-fit"})

@app.route('/regression-fit', methods=['POST'])
def run_regression_fit():
    """
    Run incremental regression fit script with JSON parameters.
    
    Expected JSON payload format matches params.json structure:
    {
        "x": {
            "traindataname": "dataset_train",
            "testdataname": "dataset_test",  // Optional: specify separate test dataset
            "steps": 200,
            "recursiveFit": true,
            "useWeights": false,
            ...
        }
    }
    
    Returns the console output of the script execution.
    """
    try:
        # Validate JSON payload
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        payload = request.get_json()
        if not payload or 'x' not in payload:
            return jsonify({"error": "Invalid payload format. Expected 'x' parameter object."}), 400
        
        params = payload['x']
        
        # Handle data sources (local, S3, or built-in)
        dataname = params.get('traindataname', params.get('dataname', '1193_BNG_lowbwt_train'))  # Support both old and new parameter names
        testdataname = params.get('testdataname')  # Optional separate test dataset
        local_data_path = params.get('localDataPath')
        s3_config = params.get('s3Config')
        
        print(f"DEBUG: dataname={dataname}")
        print(f"DEBUG: testdataname={testdataname}")
        print(f"DEBUG: s3_config={s3_config}")
        print(f"DEBUG: local_data_path={local_data_path}")
        
        # Track temporary directories for cleanup
        temp_dirs_to_cleanup = []
        
        # Set environment variables for script
        env = os.environ.copy()
        env['IB_PROJECT_ROOT'] = '/opt/multiboost'
        env['IB_DATA_DIR'] = '/opt/data'
        
        # Handle S3 configuration by downloading data to temp directory
        if s3_config:
            try:
                # Download the dataset from S3
                train_path, test_path = download_s3_dataset(
                    s3_config,
                    dataname,
                    testdataname
                )
                
                # Extract the temp directory from the returned path
                temp_dir = os.path.dirname(train_path)
                temp_dirs_to_cleanup.append(temp_dir)
                
                # Set the data directory to the temp directory for C++ executables
                env['IB_DATA_DIR'] = temp_dir
                print(f"S3 data downloaded to: {temp_dir}")
                print(f"Setting IB_DATA_DIR={temp_dir}")
                
                # Debug: List files in temp directory
                print(f"Files in {temp_dir}: {os.listdir(temp_dir)}")
                
            except Exception as e:
                print(f"Failed to download S3 data: {e}")
                return jsonify({"error": f"S3 download failed: {str(e)}", "success": False}), 500
        elif local_data_path:
            # User specifies local path - assume it's mounted as a volume
            dataname = handle_local_dataset(local_data_path, params)
            params['traindataname'] = dataname
        
        # Create temporary JSON file with parameters
        # The MultiBoost executable expects the JSON structure without the 'x' wrapper
        multiboost_params = params.copy()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(multiboost_params, temp_file, indent=2)
            temp_json_path = temp_file.name
        
        try:
            # Extract parameters (dataname already handled above)
            steps = params.get('steps', 200)
            split_ratio = params.get('split_ratio', 0.1)  # Default to 0.1 to suppress OOS at each step

            # Check if user wants to show IS at each step
            show_is_each_step = params.get('showISEachStep', False)
            # Check if user wants to show OOS at each step
            show_oos_each_step = params.get('showOOSEachStep', False)

            # Set SHOW_IS environment variable based on stepwise IS request
            env['SHOW_IS'] = '1' if show_is_each_step else '0'
            # Set SHOW_OOS environment variable based on stepwise OOS request
            env['SHOW_OOS'] = '1' if show_oos_each_step else '0'
            
            # Set the data directory for regression datasets (only if not S3)
            if not s3_config:
                if 'BNG_lowbwt' in dataname:
                    env['IB_DATA_DIR'] = '/opt/data/Regression'
                elif '/' in dataname and not dataname.startswith('1193_BNG') and not dataname.startswith('sonar'):
                    # Local dataset with full path - preserve the full path
                    env['IB_DATA_DIR'] = '/opt/data'
            
            # Always use command line mode (JSON mode has bugs)
            if True:
                # Extract parameters for command line mode
                child_partition_size = params.get('childPartitionSize', [500, 50])
                child_learning_rate = params.get('childLearningRate', [0.01, 0.01])
                child_active_partition_ratio = params.get('childActivePartitionRatio', [0.5, 0.5])
                child_max_depth = params.get('childMaxDepth', [0, 0])
                child_min_leaf_size = params.get('childMinLeafSize', [1, 1])
                child_min_gain_split = params.get('childMinimumGainSplit', [0.0, 0.0])
                loss_fn = params.get('loss', {}).get('data', params.get('lossFn', 1))
                loss_power = params.get('lossPower', 2.4)
                col_subsample_ratio = params.get('colSubsampleRatio', 1.0)
                recursive_fit = int(params.get('recursiveFit', True))
                clamp_gradient = int(params.get('clamp_gradient', False))
                upper_val = params.get('upper_val', 0.0)
                lower_val = params.get('lower_val', 0.0)
                run_on_test = int(params.get('runOnTestDataset', True))
                
                # Build command line arguments
                child_num_steps = params.get('childNumSteps', [1, 1])
                cmd = [
                    '/bin/bash',
                    SCRIPT_PATH,
                    str(len(child_partition_size)),  # num_args
                ] + [str(x) for x in child_partition_size] + \
                  [str(x) for x in child_num_steps] + \
                  [str(x) for x in child_learning_rate] + \
                  [str(x) for x in child_active_partition_ratio] + \
                  [str(x) for x in child_max_depth] + \
                  [str(x) for x in child_min_leaf_size] + \
                  [str(x) for x in child_min_gain_split] + [
                    dataname,
                    str(steps),
                    str(loss_fn),
                    str(loss_power),
                    str(col_subsample_ratio),
                    str(recursive_fit),
                    str(clamp_gradient),
                    str(upper_val),
                    str(lower_val),
                    str(run_on_test),
                    str(split_ratio)  # This suppresses OOS at each step
                ]
                
                # Add optional test dataset name if specified
                if testdataname:
                    cmd.append(testdataname)
            
            # Execute the script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd='/opt/multiboost',
                env=env,
                timeout=3600  # 1 hour timeout
            )
            
            # Prepare response
            response = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": ' '.join(cmd)
            }
            
            return jsonify(response)
            
        finally:
            # Clean up temporary files and directories
            try:
                os.unlink(temp_json_path)
            except:
                pass
            
            
            # Clean up S3 downloaded datasets
            for temp_dir in temp_dirs_to_cleanup:
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                
    except subprocess.TimeoutExpired:
        return jsonify({
            "error": "Script execution timed out after 1 hour",
            "success": False
        }), 408
        
    except Exception as e:
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False
        }), 500

@app.route('/classifier-fit', methods=['POST'])
def run_classifier_fit():
    """
    Run incremental classifier fit script with JSON parameters.
    
    Expected JSON payload format similar to regression but for classification:
    {
        "x": {
            "traindataname": "dataset_train",
            "testdataname": "dataset_test",  // Optional: specify separate test dataset
            "steps": 200,
            "childPartitionSize": 250,
            "childNumSteps": 1,
            "childLearningRate": 0.01,
            "childActivePartitionRatio": 0.75,
            "childMaxDepth": 0,
            "childMinLeafSize": 1,
            "childMinimumGainSplit": 0.0,
            "numTrees": 10,
            "lossFn": 12,
            "lossPower": 1.56,
            "recursiveFit": true,
            "clampGradient": true,
            "upperVal": 1.0,
            "lowerVal": 1.0,
            "runOnTestDataset": true,
            "splitRatio": 0.2
        }
    }
    
    Returns the console output of the script execution.
    """
    try:
        # Validate JSON payload
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        payload = request.get_json()
        if not payload or 'x' not in payload:
            return jsonify({"error": "Invalid payload format. Expected 'x' parameter object."}), 400
        
        params = payload['x']
        
        # Handle data sources (local, S3, or built-in)
        dataname = params.get('traindataname', params.get('dataname', 'sonar'))  # Support both old and new parameter names
        testdataname = params.get('testdataname')  # Optional separate test dataset
        local_data_path = params.get('localDataPath')
        s3_config = params.get('s3Config')
        
        # Track temporary directories for cleanup
        temp_dirs_to_cleanup = []
        
        # Set environment variables for script
        env = os.environ.copy()
        env['IB_PROJECT_ROOT'] = '/opt/multiboost'
        env['IB_DATA_DIR'] = '/opt/data'
        
        # Handle S3 configuration by downloading data to temp directory
        if s3_config:
            try:
                # Download the dataset from S3
                train_path, test_path = download_s3_dataset(
                    s3_config,
                    dataname,
                    testdataname
                )
                
                # Extract the temp directory from the returned path
                temp_dir = os.path.dirname(train_path)
                temp_dirs_to_cleanup.append(temp_dir)
                
                # Set the data directory to the temp directory for C++ executables
                env['IB_DATA_DIR'] = temp_dir
                print(f"S3 data downloaded to: {temp_dir}")
                print(f"Setting IB_DATA_DIR={temp_dir}")
                
                # Debug: List files in temp directory
                print(f"Files in {temp_dir}: {os.listdir(temp_dir)}")
                
            except Exception as e:
                print(f"Failed to download S3 data: {e}")
                return jsonify({"error": f"S3 download failed: {str(e)}", "success": False}), 500
        elif local_data_path:
            # User specifies local path - assume it's mounted as a volume
            dataname = handle_local_dataset(local_data_path, params)
            params['traindataname'] = dataname
        
        # Create temporary JSON file with parameters
        multiboost_params = params.copy()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(multiboost_params, temp_file, indent=2)
            temp_json_path = temp_file.name
        
        try:
            # Check if user wants to show IS at each step
            show_is_each_step = params.get('showISEachStep', False)
            # Check if user wants to show OOS at each step
            show_oos_each_step = params.get('showOOSEachStep', False)

            # Set SHOW_IS environment variable based on stepwise IS request
            env['SHOW_IS'] = '1' if show_is_each_step else '0'
            # Set SHOW_OOS environment variable based on stepwise OOS request
            env['SHOW_OOS'] = '1' if show_oos_each_step else '0'
            
            # Set the data directory for specific datasets (only if not S3)
            if not s3_config:
                if 'BNG_lowbwt' in dataname:
                    env['IB_DATA_DIR'] = '/opt/data/Regression'
                elif '/' in dataname and not dataname.startswith('1193_BNG') and not dataname.startswith('sonar'):
                    # Local dataset with full path - preserve the full path
                    env['IB_DATA_DIR'] = '/opt/data'
            
            # Extract parameters - handle both arrays and single values
            child_partition_size = params.get('childPartitionSize', [250])
            child_num_steps = params.get('childNumSteps', [1])  
            child_learning_rate = params.get('childLearningRate', [0.01])
            child_active_partition_ratio = params.get('childActivePartitionRatio', [0.75])
            child_max_depth = params.get('childMaxDepth', [0])
            child_min_leaf_size = params.get('childMinLeafSize', [1])
            child_min_gain_split = params.get('childMinimumGainSplit', [0.0])
            
            # Convert single values to arrays if needed
            if not isinstance(child_partition_size, list):
                child_partition_size = [child_partition_size]
            if not isinstance(child_num_steps, list):
                child_num_steps = [child_num_steps]
            if not isinstance(child_learning_rate, list):
                child_learning_rate = [child_learning_rate]
            if not isinstance(child_active_partition_ratio, list):
                child_active_partition_ratio = [child_active_partition_ratio]
            if not isinstance(child_max_depth, list):
                child_max_depth = [child_max_depth]
            if not isinstance(child_min_leaf_size, list):
                child_min_leaf_size = [child_min_leaf_size]
            if not isinstance(child_min_gain_split, list):
                child_min_gain_split = [child_min_gain_split]
            
            num_trees = params.get('numTrees', 10)
            loss_fn = params.get('loss', {}).get('data', params.get('lossFn', 12))
            loss_power = params.get('lossPower', 1.56)
            recursive_fit = 1 if params.get('recursiveFit', True) else 0
            clamp_gradient = 1 if params.get('clampGradient', True) else 0
            upper_val = int(params.get('upper_val', params.get('upperVal', 1)))
            lower_val = int(params.get('lower_val', params.get('lowerVal', 1)))
            run_on_test = 0 if params.get('runOnTestDataset', True) else 1  # Note: 0 means run on test
            split_ratio = params.get('splitRatio', 0.0)  # 0 means no split, run on full dataset
            steps = params.get('steps', 10)
            
            # Build command with multiple child parameters
            cmd = [
                '/bin/bash',
                '/opt/multiboost/scripts/incremental_classifier_fit.sh',
                str(len(child_partition_size))  # num_args for child parameters
            ]
            
            # Add all child parameter arrays
            cmd.extend([str(x) for x in child_partition_size])
            cmd.extend([str(x) for x in child_num_steps])
            cmd.extend([str(x) for x in child_learning_rate])
            cmd.extend([str(x) for x in child_active_partition_ratio])
            cmd.extend([str(x) for x in child_max_depth])
            cmd.extend([str(x) for x in child_min_leaf_size])
            cmd.extend([str(x) for x in child_min_gain_split])
            
            # Add remaining parameters in correct order
            cmd.extend([
                dataname,
                str(steps),
                str(loss_fn),
                str(loss_power),
                str(recursive_fit),
                str(clamp_gradient),
                str(recursive_fit),  # appears twice in working command
                str(upper_val),
                str(lower_val),
                str(run_on_test),
                str(split_ratio)  # Add split_ratio parameter
            ])
            
            # Add optional test dataset name if specified
            if testdataname:
                cmd.append(testdataname)
            
            # Execute the script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd='/opt/multiboost',
                env=env,
                timeout=3600  # 1 hour timeout
            )
            
            # Prepare response
            response = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": ' '.join(cmd)
            }
            
            return jsonify(response)
            
        finally:
            # Clean up temporary files and directories
            try:
                os.unlink(temp_json_path)
            except:
                pass
            
            
            # Clean up S3 downloaded datasets
            for temp_dir in temp_dirs_to_cleanup:
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                
    except subprocess.TimeoutExpired:
        return jsonify({
            "error": "Script execution timed out after 1 hour",
            "success": False
        }), 408
        
    except Exception as e:
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False
        }), 500

@app.route('/available-datasets', methods=['GET'])
def list_datasets():
    """List available datasets in the data directory."""
    try:
        datasets = []
        
        # Check main data directory
        data_dir = Path('/opt/data')
        if data_dir.exists():
            # Look for sonar dataset
            if (data_dir / 'sonar_X.csv').exists() and (data_dir / 'sonar_y.csv').exists():
                datasets.append({
                    "name": "sonar",
                    "path": str(data_dir),
                    "files": ["sonar_X.csv", "sonar_y.csv"]
                })
        
        # Check regression subdirectory
        regression_dir = Path('/opt/data/Regression')
        if regression_dir.exists():
            # Look for BNG_lowbwt datasets
            train_files = list(regression_dir.glob('*_train_X.csv'))
            for train_file in train_files:
                base_name = train_file.name.replace('_train_X.csv', '')
                dataset_name = base_name + '_train'
                
                # Check if corresponding files exist
                expected_files = [
                    f"{base_name}_train_X.csv",
                    f"{base_name}_train_y.csv",
                    f"{base_name}_test_X.csv", 
                    f"{base_name}_test_y.csv"
                ]
                
                existing_files = [f for f in expected_files 
                                if (regression_dir / f).exists()]
                
                if len(existing_files) >= 2:  # At least train files exist
                    datasets.append({
                        "name": dataset_name,
                        "path": str(regression_dir),
                        "files": existing_files
                    })
        
        return jsonify({"datasets": datasets})
        
    except Exception as e:
        return jsonify({"error": f"Error listing datasets: {str(e)}"}), 500

if __name__ == '__main__':
    # Ensure script exists
    if not os.path.exists(SCRIPT_PATH):
        print(f"Error: Script not found at {SCRIPT_PATH}")
        sys.exit(1)
    
    # Start server
    app.run(host='0.0.0.0', port=8002, debug=False)
