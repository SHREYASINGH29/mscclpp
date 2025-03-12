#!/bin/bash

ROOT=/media/datassd/shreya/
$ROOT/llvm-project/build/bin/clang++ -O3 -x cuda -S -emit-llvm tritonOps.hpp --cuda-gpu-arch=sm_86 -D__NVCC__ --cuda-device-only -std=c++17 -o tritonOps-O3.ll
