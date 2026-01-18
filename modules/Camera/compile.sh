#!/bin/bash
g++ ./Camera.cpp    -o camera.o
g++ ./Generate.cpp  -o generate.o
g++ ./Video.cpp     -o video.o -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc