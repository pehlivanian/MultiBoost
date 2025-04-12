#!/bin/bash

# Get project root directory if not set
if [ -z "${IB_PROJECT_ROOT}" ]; then
  # Try to find the project root (the directory containing CMakeLists.txt)
  export IB_PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
fi

# Format all C++ code in the repository
find "${IB_PROJECT_ROOT}" -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i

echo "Formatted all C++ files with clang-format."
