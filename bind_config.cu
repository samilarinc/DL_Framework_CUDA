#include <stdio.h>
#include "headers/Dense.cuh"
#include<pybind11/pybind11.h>
#include "headers/Optimizer.cuh"
#include <pybind11/stl.h>
#include <vector>

using namespace std;
namespace py = pybind11;

class DenseTrampoline : public Dense {
public:
    DenseTrampoline(int in_size, int out_size) : Dense(in_size, out_size) {}

    vector<double> forward(vector<double> input){
        double *device_data;
        cudaError_t err = cudaMalloc((void**)&device_data, input.size() * sizeof(double));
        if (err != cudaSuccess)printf("Error allocating memory on GPU");
        err = cudaMemcpy(device_data, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)printf("Error copying data to GPU");
        double *device_output;
        err = cudaMalloc((void**)&device_output, out_size * sizeof(double));
        if (err != cudaSuccess)printf("Error allocating memory on GPU");
        device_output = Dense::forward(device_data);
        vector<double> output(out_size);
        err = cudaMemcpy(output.data(), device_output, out_size * sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)printf("Error copying data to GPU");
        return output;
    }

    vector<double> backward(vector<double> error){
        double *device_data;
        cudaError_t err = cudaMalloc((void**)&device_data, error.size() * sizeof(double));
        if (err != cudaSuccess)printf("Error allocating memory on GPU");
        err = cudaMemcpy(device_data, error.data(), error.size() * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)printf("Error copying data to GPU");
        double *device_output;
        err = cudaMalloc((void**)&device_output, in_size * sizeof(double));
        if (err != cudaSuccess)printf("Error allocating memory on GPU");
        device_output = Dense::backward(device_data);
        vector<double> output(in_size);
        err = cudaMemcpy(output.data(), device_output, in_size * sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)printf("Error copying data to GPU");
        return output;
    }
};

PYBIND11_MODULE(FullyConnected, m) {
    py::class_<DenseTrampoline>(m, "Dense")
        .def(py::init<int, int>())
        .def("forward", &DenseTrampoline::forward)
        .def("backward", &DenseTrampoline::backward)
    ;

    py::class_<Dense>(m, "DenseBase")
        .def(py::init<int, int>())
        .def("forward", &Dense::forward)
        .def("backward", &Dense::backward)
        ;
}