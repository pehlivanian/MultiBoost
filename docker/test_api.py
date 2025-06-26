#!/usr/bin/env python3
"""
Test script for the MultiBoost regression API
"""

import json
import requests
import time

def test_api():
    base_url = "http://localhost:8002"
    
    # Test health check
    print("Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test available datasets
    print("\nTesting available datasets...")
    try:
        response = requests.get(f"{base_url}/available-datasets")
        print(f"Available datasets: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Dataset listing failed: {e}")
    
    # Test regression fit with sample parameters
    print("\nTesting regression fit...")
    
    # Sample payload based on params.json
    payload = {
        "x": {
            "steps": 10,  # Reduced for testing
            "recursiveFit": True,
            "useWeights": False,
            "rowSubsampleRatio": 1.0,
            "colSubsampleRatio": 1.0,
            "removeRedundantLabels": False,
            "symmetrizeLabels": True,
            "loss": {
                "index": 1,
                "data": 1
            },
            "lossPower": 2.4,
            "clamp_gradient": True,
            "upper_val": -1.0,
            "lower_val": 1.0,
            "numTrees": 10,
            "depth": 0,
            "childPartitionSize": [500, 50],
            "childNumSteps": [1, 1],
            "childLearningRate": [0.01, 0.01],
            "childActivePartitionRatio": [0.5, 0.5],
            "childMinLeafSize": [1, 1],
            "childMinimumGainSplit": [0.0, 0.0],
            "childMaxDepth": [0, 0],
            "serializeModel": True,
            "serializePrediction": True,
            "serializeColMask": False,
            "serializeDataset": True,
            "serializeLabels": True,
            "serializationWindow": 10,
            "quietRun": True,
            "dataname": "1193_BNG_lowbwt_train"
        }
    }
    
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f"{base_url}/regression-fit", 
                               json=payload, 
                               headers=headers,
                               timeout=300)  # 5 minute timeout
                               
        print(f"Regression fit: {response.status_code}")
        result = response.json()
        
        if result.get('success'):
            print("SUCCESS: Script executed successfully")
            print("STDOUT:", result.get('stdout', '')[:500])  # First 500 chars
            if result.get('stderr'):
                print("STDERR:", result.get('stderr', '')[:500])
        else:
            print("FAILED: Script execution failed")
            print("Error:", result.get('error', 'Unknown error'))
            print("STDOUT:", result.get('stdout', ''))
            print("STDERR:", result.get('stderr', ''))
            
    except Exception as e:
        print(f"Regression fit request failed: {e}")

if __name__ == "__main__":
    test_api()