#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "cudaMatrix.hpp"
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(cudaMatrix, m) {
    py::class_<cuda1DArray>(m, "cuda1DArray")
        .def(py::init< vector<vector<double>> >())
        .def("__add__", &cuda1DArray::addMatrix)
        .def("__sub__", &cuda1DArray::subMatrix)
        .def("__mul__", &cuda1DArray::mulMatrix)
        .def("__truediv__", &cuda1DArray::divMatrix)
        .def("__repr__", &cuda1DArray::printMatrix)
        .def("__add__", &cuda1DArray::addConstant)
        .def("__sub__", &cuda1DArray::subConstant)
        .def("__mul__", &cuda1DArray::mulConstant)
        .def("__truediv__", &cuda1DArray::divConstant)
        .def("T", &cuda1DArray::T)
        .def("transpose", &cuda1DArray::T)
        .def("det", &cuda1DArray::det)
        .def("inv", &cuda1DArray::inv)
        .def("adjoint", &cuda1DArray::adjoint)
        .def("cofactor", &cuda1DArray::cofactor)
        .def("copy", &cuda1DArray::copy)
        .def_readonly("height", &cuda1DArray::height)
        .def_readonly("width", &cuda1DArray::width)
        .def_readonly("shape", &cuda1DArray::shape)
        ;
}