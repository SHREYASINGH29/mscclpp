#!/bin/bash

ROOT=/uufs/chpc.utah.edu/common/home/u1419864/vast/msr-project/

$ROOT/llvm-project/build/bin/clang++ -O0 -x cuda -S -emit-llvm tritonOps.hpp --cuda-gpu-arch=sm_86 -D__NVCC__ --cuda-device-only -std=c++17 -o tritonOps-O0.ll
