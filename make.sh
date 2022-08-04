#!/bin/bash

if [ $1 -gt 0 ]
then
	nvcc -shared -Xcompiler -fPIC $(python3 -m pybind11 --includes) bind.cpp cudaMatrix.cu -o cudaMatrix$(python3-config --extension-suffix)
	python3 test.py
else
	python3 test.py
fi