# MultiBoost Regression API

This Docker container exposes the MultiBoost incremental regression fit script as a REST API endpoint.

## Building the Container

```bash
cd /home/charles/src/devops/sandbox/MultiBoost/docker
docker build -t multiboost-api .
```

## Running the Container

```bash
docker run -p 8002:8002 multiboost-api
```

The API will be available at `http://localhost:8002`

## API Endpoints

### Health Check
- **URL**: `GET /health`
- **Description**: Check if the service is running
- **Response**: `{"status": "healthy", "service": "multiboost-regression-fit"}`

### Available Datasets
- **URL**: `GET /available-datasets`
- **Description**: List available datasets in the container
- **Response**: JSON list of available datasets

### Regression Fit
- **URL**: `POST /regression-fit`
- **Content-Type**: `application/json`
- **Description**: Run incremental regression fit with provided parameters

#### Request Payload Format

The payload should match the structure in `params.json`:

```json
{
  "x": {
    "steps": 200,
    "recursiveFit": true,
    "useWeights": false,
    "rowSubsampleRatio": 1.0,
    "colSubsampleRatio": 1.0,
    "removeRedundantLabels": false,
    "symmetrizeLabels": true,
    "loss": {
      "index": 1,
      "data": 1
    },
    "lossPower": 2.4,
    "clamp_gradient": true,
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
    "serializeModel": true,
    "serializePrediction": true,
    "serializeColMask": false,
    "serializeDataset": true,
    "serializeLabels": true,
    "serializationWindow": 10,
    "quietRun": true,
    "dataname": "1193_BNG_lowbwt_train"
  }
}
```

#### Response Format

```json
{
  "success": true,
  "returncode": 0,
  "stdout": "Script output...",
  "stderr": "Any error messages...",
  "command": "Command that was executed"
}
```

## Example Usage

### Using curl

```bash
# Health check
curl http://localhost:8002/health

# List datasets
curl http://localhost:8002/available-datasets

# Run regression fit
curl -X POST http://localhost:8002/regression-fit \
  -H "Content-Type: application/json" \
  -d @params.json
```

### Using Python

```python
import requests
import json

# Load parameters
with open('params.json', 'r') as f:
    payload = json.load(f)

# Make request
response = requests.post('http://localhost:8002/regression-fit', json=payload)
result = response.json()

if result['success']:
    print("Script executed successfully!")
    print("Output:", result['stdout'])
else:
    print("Script failed:", result['stderr'])
```

## Testing

A test script is provided to verify the API functionality:

```bash
python3 test_api.py
```

## Available Datasets

The container includes the following datasets:
- `sonar` (sonar_X.csv, sonar_y.csv)
- `1193_BNG_lowbwt_train` (1193_BNG_lowbwt_train_X.csv, 1193_BNG_lowbwt_train_y.csv)
- `1193_BNG_lowbwt_test` (1193_BNG_lowbwt_test_X.csv, 1193_BNG_lowbwt_test_y.csv)

## Notes

- The API runs on port 8002 inside the container
- Script execution has a 1-hour timeout
- Temporary files are automatically cleaned up after execution
- The service runs as a non-root user for security