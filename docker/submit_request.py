#!/usr/bin/env python3
"""
Example script to submit regression fit request using Python requests library
"""

import json
import requests
import time

def submit_regression_request():
    # API endpoint
    url = "http://localhost:8002/regression-fit"
    
    # Method 1: Load from JSON file
    with open('example_params.json', 'r') as f:
        payload = json.load(f)
    
    # Method 2: Define payload directly in Python (alternative)
    # payload = {
    #     "x": {
    #         "dataname": "1193_BNG_lowbwt_train",
    #         "steps": 20,
    #         "recursiveFit": True,
    #         "useWeights": False,
    #         "rowSubsampleRatio": 1.0,
    #         "colSubsampleRatio": 0.8,
    #         "removeRedundantLabels": False,
    #         "symmetrizeLabels": True,
    #         "loss": {
    #             "index": 1,
    #             "data": 1
    #         },
    #         "lossPower": 2.0,
    #         "clamp_gradient": True,
    #         "upper_val": -1.0,
    #         "lower_val": 1.0,
    #         "numTrees": 15,
    #         "depth": 0,
    #         "childPartitionSize": [300, 100],
    #         "childNumSteps": [1, 1],
    #         "childLearningRate": [0.05, 0.02],
    #         "childActivePartitionRatio": [0.6, 0.4],
    #         "childMinLeafSize": [5, 10],
    #         "childMinimumGainSplit": [0.001, 0.001],
    #         "childMaxDepth": [3, 2],
    #         "serializeModel": True,
    #         "serializePrediction": True,
    #         "serializeColMask": False,
    #         "serializeDataset": True,
    #         "serializeLabels": True,
    #         "serializationWindow": 5,
    #         "quietRun": True
    #     }
    # }
    
    # Set headers
    headers = {
        'Content-Type': 'application/json'
    }
    
    print("Submitting regression fit request...")
    print(f"Dataset: {payload['x']['dataname']}")
    print(f"Steps: {payload['x']['steps']}")
    
    try:
        # Submit the request
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=600)  # 10 minute timeout
        end_time = time.time()
        
        # Parse response
        result = response.json()
        
        print(f"\nRequest completed in {end_time - start_time:.2f} seconds")
        print(f"HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            if result.get('success'):
                print("✅ SUCCESS: Regression fit completed!")
                
                # Extract key information from stdout
                stdout = result.get('stdout', '')
                if 'OOS: (r_squared):' in stdout:
                    # Extract r-squared values
                    lines = stdout.split('\n')
                    for line in lines:
                        if 'OOS: (r_squared):' in line:
                            print(f"📊 {line.strip()}")
                
                # Show first few lines of output
                print(f"\n📋 Output preview:")
                print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
                
                # Show any errors (non-fatal)
                stderr = result.get('stderr', '')
                if stderr:
                    print(f"\n⚠️  Warnings/Errors (non-fatal):")
                    print(stderr[:300] + "..." if len(stderr) > 300 else stderr)
                    
            else:
                print("❌ FAILED: Script execution failed")
                print(f"Error: {result.get('error', 'Unknown error')}")
                print(f"STDOUT: {result.get('stdout', '')}")
                print(f"STDERR: {result.get('stderr', '')}")
        else:
            print(f"❌ HTTP Error {response.status_code}")
            print(f"Response: {result}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 10 minutes")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is the container running on port 8002?")
    except Exception as e:
        print(f"❌ Error: {e}")

def check_available_datasets():
    """Check what datasets are available"""
    url = "http://localhost:8002/available-datasets"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            datasets = response.json().get('datasets', [])
            print("📁 Available datasets:")
            for dataset in datasets:
                print(f"  • {dataset['name']} (path: {dataset['path']})")
                print(f"    Files: {', '.join(dataset['files'])}")
        else:
            print(f"❌ Error getting datasets: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("=== MultiBoost Regression API Client ===\n")
    
    # First check available datasets
    check_available_datasets()
    print()
    
    # Submit regression request
    submit_regression_request()