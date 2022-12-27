#include <stdio.h>
#include<pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "headers/Dense.cuh"
#include "headers/Optimizer.cuh"
#include "Tensor.cuh"

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
    py::class_<DenseTrampoline>(m, "DenseTr")
        .def(py::init<int, int>())
        .def("forward", &DenseTrampoline::forward)
        .def("backward", &DenseTrampoline::backward)
    ;

    py::class_<Dense>(m, "Dense")
        .def(py::init<int, int>())
        .def("forward", py::overload_cast<Tensor>(&Dense::forward))
        .def("backward", &Dense::backward)
        .def_readwrite("weights", &Dense::weights)
        .def_readwrite("bias", &Dense::bias)
        .def_readwrite("in_size", &Dense::in_size)
        .def_readwrite("out_size", &Dense::out_size)
        .def_readwrite("weight_size", &Dense::weight_size)
        .def_readwrite("last_input", &Dense::last_input)
        .def_readwrite("output", &Dense::output)
        .def_readwrite("dx", &Dense::dx)
        .def_readwrite("dW", &Dense::dW)
        .def_readwrite("db", &Dense::db)
    ;

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<int, int>())
        .def(py::init<int, int, int>())
        .def(py::init<vector<double>>())
        .def(py::init<vector<vector<double>>>())
        .def(py::init<vector<vector<vector<double>>>>())
        .def("__repr__", &Tensor::print_tensor)
        // .def("__str__", &Tensor::print_tensor)
        .def("__len__", &Tensor::len)
        // .def("T", &Tensor::transpose_2d)
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("w", &Tensor::w)
        .def_readwrite("h", &Tensor::h)
        .def_readwrite("batch_size", &Tensor::batch_size)
        ;
}