# Inductive-Boost

A C++ implementation of gradient boosting algorithms, with an emphasis on flexibility, composability, and performance.

## Setup and Environment

### Environment Variables

This project uses environment variables to locate resources:

- `IB_PROJECT_ROOT`: Path to the project root directory (set automatically by scripts)
- `IB_DATA_DIR`: Path to the data directory (defaults to ~/Data)

You can set these variables manually or use the provided setup script:

```bash
source setup_env.sh
```

### Building the Project

```bash
# Set up environment variables
source setup_env.sh

# Build using CMake
cmake -S . -B build && cmake --build build
```

### Running Tests

```bash
./build/tests/gtest_all
```

### Running Benchmarks

```bash
./build/benchmarks/benchmarks
```

## Project Structure

- `src/cpp`: C++ implementation of core algorithms
- `src/python`: Python utilities for data processing and visualization
- `src/script`: Shell scripts for running experiments
- `src/rust`: Rust utilities and experiments
- `benchmarks`: Performance benchmarks
- `docs`: Documentation

## Path Handling

The project uses relative paths based on environment variables:

- C++ code: Uses the `path_utils.hpp` utility to resolve paths
- Python code: Uses the `path_utils.py` utility to resolve paths
- Shell scripts: Uses the `IB_PROJECT_ROOT` environment variable

This allows the code to run correctly regardless of where the repository is cloned.

## License

This project is for research purposes only.