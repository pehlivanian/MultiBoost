#!/usr/bin/env python3
"""
REST API server to expose incremental regression fit script as an endpoint.
"""

import json
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

# Script path - will be copied from source during docker build
SCRIPT_PATH = "/opt/multiboost/scripts/incremental_regression_fit.sh"

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
        
        # Create temporary JSON file with parameters
        # The MultiBoost executable expects the JSON structure without the 'x' wrapper
        multiboost_params = params.copy()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(multiboost_params, temp_file, indent=2)
            temp_json_path = temp_file.name
        
        try:
            # Set environment variables for script
            env = os.environ.copy()
            env['IB_PROJECT_ROOT'] = '/opt/multiboost'
            env['IB_DATA_DIR'] = '/opt/data'
            # Set the data directory for regression datasets
            if 'BNG_lowbwt' in params.get('dataname', ''):
                env['IB_DATA_DIR'] = '/opt/data/Regression'
            
            # Extract dataset name from parameters (default to a common test dataset)
            dataname = params.get('dataname', '1193_BNG_lowbwt_train')
            steps = params.get('steps', 200)
            
            # Run the script with JSON parameters
            # The script expects: dataname, json_file_path, steps
            cmd = [
                '/bin/bash',
                SCRIPT_PATH,
                dataname,
                temp_json_path,
                str(steps)
            ]
            
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
            # Clean up temporary file
            try:
                os.unlink(temp_json_path)
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