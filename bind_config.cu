#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>

#include <vector>

#include "Tensor.cuh"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(pyflow, m) {
    py::class_<Tensor>(m, "Tensor")
         .def(py::init<>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int, int, double>())
        .def(py::init<int, int, int, double, double, string>())
        .def(py::init<const Tensor&>())
        .def(py::init<vector<vector<vector<double>>>>())
        .def(py::init<vector<vector<double>>>())
        .def(py::init<vector<double>>())
        .def("__len__", &Tensor::size)
        .def("__add__", py::overload_cast<const Tensor&>(&Tensor::operator+))
        .def("__add__", py::overload_cast<double>(&Tensor::operator+))
        .def("__sub__", py::overload_cast<const Tensor&>(&Tensor::operator-))
        .def("__sub__", py::overload_cast<double>(&Tensor::operator-))
        .def("__mul__", py::overload_cast<const Tensor&>(&Tensor::operator*))
        .def("__mul__", py::overload_cast<double>(&Tensor::operator*))
        .def("__truediv__", py::overload_cast<const Tensor&>(&Tensor::operator/))
        .def("__truediv__", py::overload_cast<double>(&Tensor::operator/))
        .def("__pow__", py::overload_cast<double>(&Tensor::power))
        .def("__iadd__", py::overload_cast<const Tensor&>(&Tensor::operator+=))
        .def("__iadd__", py::overload_cast<double>(&Tensor::operator+=))
        .def("__isub__", py::overload_cast<const Tensor&>(&Tensor::operator-=))
        .def("__isub__", py::overload_cast<double>(&Tensor::operator-=))
        .def("__imul__", py::overload_cast<const Tensor&>(&Tensor::operator*=))
        .def("__imul__", py::overload_cast<double>(&Tensor::operator*=))
        .def("__itruediv__", py::overload_cast<const Tensor&>(&Tensor::operator/=))
        .def("__itruediv__", py::overload_cast<double>(&Tensor::operator/=))
        .def("__pow__", py::overload_cast<double>(&Tensor::power))
        .def("power", py::overload_cast<double>(&Tensor::power))
        .def("abs", &Tensor::abs)
        .def("sign", &Tensor::sign)
        .def("setitem", py::overload_cast<int, int, int, double>(&Tensor::setitem))
        .def("setitem", py::overload_cast<int, int, const Tensor&>(&Tensor::setitem))
        .def("setitem", py::overload_cast<int, const Tensor&>(&Tensor::setitem))
        .def("rows", &Tensor::getRows)
        .def("cols", &Tensor::getCols)
        .def("batch_size", &Tensor::getBatchsize)
        .def("__repr__", &Tensor::toString)
        .def("transpose", &Tensor::transpose)
        .def("dot_product", &Tensor::dot_product)
        .def_readonly("shape", &Tensor::shape_)
        .def("copy", &Tensor::copy)
        .def("sum", &Tensor::sum)
        .def("tolist", &Tensor::tolist)
        .def("getitem", py::overload_cast<int>(&Tensor::getitem, py::const_))
        .def("getitem", py::overload_cast<int, int>(&Tensor::getitem, py::const_))
        .def("getitem", py::overload_cast<int, int, int>(&Tensor::getitem, py::const_))
        .def("__call__", [](Tensor& t, int batch_size, int row, int col) {
            return t(batch_size, row, col);
        })
        .def("__call__", [](Tensor& t, int batch_size, int row, int col) {
            return t(batch_size, row, col);
        })
        .def("__repr__", [](Tensor& t) {
            return t.toString();
        })
        ;
}