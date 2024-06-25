#!/usr/bin/env bash

# If project not ready, generate cmake file.
if [[ ! -d build ]]; then
    echo "good"
else
    rm -rf build
fi
cmake -B build
cmake --build build

# Run all testcases. 
# You can comment some lines to disable the run of specific examples.
mkdir -p output
# build/FINAL testcases/scene01.txt output/scene01.bmp
# build/FINAL testcases/scene02.txt output/scene02.bmp
build/FINAL testcases/scene03.txt output/scene03.bmp
# build/FINAL testcases/diamond1.txt output/diamond1.bmp
# build/FINAL testcases/diamond2.txt output/diamond2.bmp
# build/FINAL testcases/diamond3.txt output/diamond3.bmp
