# MultiBoost Regression API

This Docker container exposes the MultiBoost incremental regression fit script as a REST API endpoint.

## Building the Container

```bash
cd /home/charles/src/devops/sandbox/MultiBoost/docker
docker build -f dockerfile -t multiboost_image .
```

## Running the Container

```bash
docker run --name multiboost_container -v /home/charles/Data:/opt/data -p 8002:8002 multiboost_image
```

## Example runs
```bash
# Classification
curl -X POST http://localhost:8002/classifier-fit -H "Content-Type: application/json" -d @example_params_class.json

# Regression  
curl -X POST http://localhost:8002/regression-fit -H "Content-Type: application/json" -d @example_params_reg.json
```

The API will be available at `http://localhost:8002`

## API Endpoints

### Health Check
```bash
curl -X GET http://localhost:8002/health
```
- **Description**: Check if the service is running
- **Response**: `{"status": "healthy", "service": "multiboost-regression-fit"}`

### Available Datasets
```bash
curl -X GET http://localhost:8002/available-datasets
```
- **Description**: List available datasets in the container
- **Response**: JSON list of available datasets

### Classification Fit
```bash
curl -X POST http://localhost:8002/classifier-fit \
  -H "Content-Type: application/json" \
  -d @example_params_class.json
```
- **Description**: Run incremental classification fit with provided parameters
- **Supports**: Multi-child parameter arrays for complex ensemble configurations

### Regression Fit
```bash
curl -X POST http://localhost:8002/regression-fit \
  -H "Content-Type: application/json" \
  -d @example_params_reg.json
```
- **Description**: Run incremental regression fit with provided parameters

## Using Local Datasets

Mount your data directory to the container and specify dataset paths:

```bash
docker run --name multiboost_container -v /home/charles/Data:/opt/data -p 8002:8002 multiboost_image
```

**Classification datasets:** Use `"dataname": "Classification/dataset_name"`
**Regression datasets:** Use `"dataname": "Regression/dataset_name"`

**Expected file structure:**
```
/home/charles/Data/
├── Classification/
│   ├── dataset_name_X.csv
│   ├── dataset_name_y.csv
│   ├── dataset_name_train_X.csv
│   ├── dataset_name_train_y.csv
│   ├── dataset_name_test_X.csv
│   └── dataset_name_test_y.csv
└── Regression/
    ├── dataset_name_X.csv
    ├── dataset_name_y.csv
    ├── dataset_name_train_X.csv
    ├── dataset_name_train_y.csv
    ├── dataset_name_test_X.csv
    └── dataset_name_test_y.csv
```

## Response Format

All endpoints return:
```json
{
  "success": true,
  "returncode": 0,
  "stdout": "Script output...",
  "stderr": "Any error messages...",
  "command": "Command that was executed"
}
```

## Parameter Examples

### Classification Parameters
```json
{
  "x": {
    "dataname": "buggyCrx_train",
    "steps": 10,
    "childPartitionSize": [800, 250, 500, 100],
    "childNumSteps": [1, 1, 1, 1],
    "childLearningRate": [0.00015, 0.00015, 0.0001, 0.0001],
    "childActivePartitionRatio": [0.35, 0.35, 0.35, 0.35],
    "childMaxDepth": [0, 0, 0, 0],
    "childMinLeafSize": [1, 1, 1, 1],
    "childMinimumGainSplit": [0.0, 0.0, 0.0, 0.0],
    "loss": { "data": 8 },
    "lossPower": 2.4,
    "recursiveFit": true,
    "clamp_gradient": true,
    "upper_val": 4,
    "lower_val": -4,
    "numTrees": 10
  }
}
```

### Regression Parameters
```json
{
  "x": {
    "dataname": "Regression/boston_train",
    "steps": 50,
    "childPartitionSize": [300, 100],
    "childLearningRate": [0.1, 0.1],
    "childActivePartitionRatio": [0.2, 0.2],
    "childMaxDepth": [3, 2],
    "childMinLeafSize": [5, 10],
    "childMinimumGainSplit": [0.001, 0.001],
    "loss": { "data": 1 },
    "lossPower": 2.0,
    "recursiveFit": true,
    "clamp_gradient": true,
    "upper_val": -1.0,
    "lower_val": 1.0,
    "numTrees": 20
  }
}
```

## Notes

- The API runs on port 8002 inside the container
- Script execution has a 1-hour timeout
- Temporary files are automatically cleaned up after execution
- The service runs as a non-root user for security
- Classification supports multi-child parameter arrays for complex ensemble configurations
- Data files are mounted at runtime via volume mounts (no datasets included in container)