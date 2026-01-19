#!/bin/bash
set -e

nvcc -Xcompiler -march=native -Xcompiler -fopenmp \
     -Xcompiler -O3 \
     -lineinfo --use_fast_math -O3 \
     -o ./main main.cc util/GPU.cu \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart\
    -I/usr/include/opencv4 \
    -L/usr/lib \
    -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
    -Xcompiler -Wno-error=deprecated-enum-enum-conversion \
    -Xcompiler -Wno-deprecated-enum-enum-conversion