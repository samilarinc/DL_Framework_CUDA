#ifndef TENSOR_H
#define TENSOR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace std;

class Tensor {
   public:
    Tensor();
    Tensor(int, int, int);
    Tensor(int, int, int, double);
    Tensor(int, int, int, double, double, string);
    Tensor(const Tensor&);
    Tensor(vector<double>);
    Tensor(vector<vector<double>>);
    Tensor(vector<vector<vector<double>>>);
    ~Tensor();
    Tensor transpose();
    Tensor dot_product(const Tensor&) const;
    Tensor& operator=(const Tensor&);
    double& operator()(int, int, int);
    double operator()(int, int, int) const;
    Tensor operator+(const Tensor&);
    Tensor operator-(const Tensor&);
    Tensor operator*(const Tensor&);
    Tensor operator/(const Tensor&);
    Tensor operator+(double);
    Tensor operator-(double);
    Tensor operator*(double);
    Tensor operator/(double);
    Tensor& operator+=(const Tensor&);
    Tensor& operator-=(const Tensor&);
    Tensor& operator*=(const Tensor&);
    Tensor& operator/=(const Tensor&);
    Tensor& operator+=(double);
    Tensor& operator-=(double);
    Tensor& operator*=(double);
    Tensor& operator/=(double);
    Tensor power(double);
    Tensor sign() const;
    Tensor abs() const;
    Tensor copy();
    Tensor getitem(int) const;
    Tensor getitem(int, int) const;
    double getitem(int, int, int) const;
    void setitem(int, int, int, double);
    void setitem(int, int, double);
    void setitem(int, int, const Tensor&);
    void setitem(int, const Tensor&);
    double sum() const;
    int size() const;
    int getBatchsize() const;
    int getRows() const;
    int getCols() const;
    string toString() const;
    static void err_check(cudaError_t, string);
    vector<vector<vector<double>>> tolist() const;

    int rows_;
    int cols_;
    int batch_size_;
    tuple<int, int, int> shape_;
    int size_;
    double* data_;
};

#endif  // TENSOR_H
