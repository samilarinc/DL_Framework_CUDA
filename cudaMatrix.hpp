#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

class cuda1DArray{
    public:
    void control_cuda(cudaError_t error);
    cuda1DArray(vector<vector<double>> data_vec);
    cuda1DArray(int height, int width);
    cuda1DArray(int height, int width, double* arr, bool already_cuda);
    cuda1DArray addConstant(double constant);
    cuda1DArray subConstant(double constant);
    cuda1DArray mulConstant(double constant);
    cuda1DArray divConstant(double constant);
    cuda1DArray addMatrix(cuda1DArray other);
    cuda1DArray subMatrix(cuda1DArray other);
    cuda1DArray mulMatrix(cuda1DArray other);
    cuda1DArray divMatrix(cuda1DArray other);
    string printMatrix();
    cuda1DArray T();
    static void gaussianElimination(double* dev_a, int N);
    double det();
    cuda1DArray copy();
    double cofactor(int i, int j, bool internal_call = false);
    cuda1DArray adjoint();
    cuda1DArray inv();

    size_t height;
    size_t width;
    double *data = NULL;
    pair<size_t, size_t> shape;
    size_t size;
};