#!/bin/bash

~/msr-project/llvm-project/build/bin/clang++ -O0 -x cuda -S -emit-llvm relaxed_signal.hpp --cuda-gpu-arch=sm_80
~/msr-project/llvm-project/build/bin/opt relaxed_signal-cuda-nvptx64-nvidia-cuda-sm_80.bc -S -o relaxed_signal_cu_1.ll
