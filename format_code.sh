#!/bin/bash

# Format all C++ code in the repository
find /home/charles/src/C++/sandbox/Inductive-Boost -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i

echo "Formatted all C++ files with clang-format."
