// #include <stdio.h>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <vector>
// #include <string>
#include "cudaMatrix.hpp"

using namespace std;

__global__ void constAdd(const double *first_vec, const double num, 
                                double *result_vec, size_t N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        result_vec[i] = first_vec[i] + num + 0.0f;
    }
}

__global__ void constSub(const double *first_vec, const double num, 
                                double *result_vec, size_t N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        result_vec[i] = first_vec[i] - num + 0.0f;
    }
}

__global__ void constMul(const double *first_vec, const double num, 
                                double *result_vec, size_t N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        result_vec[i] = first_vec[i] + num + 0.0f;
    }
}

__global__ void constDiv(const double *first_vec, const double num, 
                                double *result_vec, size_t N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        result_vec[i] = first_vec[i] + num + 0.0f;
    }
}

__global__ void vectorAdd(const double *first_vec, const double *second_vec, 
                                double *result_vec, size_t N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        result_vec[i] = first_vec[i] + second_vec[i] + 0.0f;
    }
}

__global__ void vectorSub(const double *first_vec, const double *second_vec, 
                                double *result_vec, size_t N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        result_vec[i] = first_vec[i] - second_vec[i] + 0.0f;
    }
}

__global__ void vectorMul(const double *first_vec, const double *second_vec, 
                                double *result_vec, size_t N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        result_vec[i] = first_vec[i] * second_vec[i] + 0.0f;
    }
}

__global__ void vectorDiv(const double *first_vec, const double *second_vec, 
                                double *result_vec, size_t N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        result_vec[i] = first_vec[i] / second_vec[i] + 0.0f;
    }
}

__global__ void transpose(double *odata, double* idata, size_t width, size_t height){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < width * height){
        size_t i_out = i/height + width*(i%height);
        odata[i_out] = idata[i];
    }
}

__global__ void scale(double *a, int size, int index) {
	int i;
	int start = (index*size + index);
	int end = (index*size + size);
	for (i = start + 1; i<end; i++) {
		a[i] = (a[i] / a[start]);
	}
}

__global__ void reduce(double *a, int size, int index) {
	int i;
	int tid = threadIdx.x;
	int start = ((index + tid + 1)*size + index);
	int end = ((index + tid + 1)*size + size);
	for (i = start + 1; i<end; i++) {
		a[i] = a[i] - (a[start] * a[(index*size) + (index + (i - start))]);
	}
}

__global__ void cof(double *a, double *b, int i, int j, int N){
    int ind1 = blockDim.x * blockIdx.x + threadIdx.x;
    int ind2 = blockDim.y * blockIdx.y + threadIdx.y;
    if(ind1 < N && ind2 < N){
        if(ind1 != i && ind2 != j){
            int sub_row = ind1 > i;
            int sub_col = ind2 > j;
            b[(ind1-sub_row)*(N-1) + (ind2-sub_col)] = a[ind1*N + ind2];
        }
    }
}

__global__ void gaussianElimination(double* dev_a, int N) {
        int i;
        for (i = 0; i<N; i++) {
            scale << <1, 1 >> >(dev_a, N, i);
            reduce << <1, (N - i - 1) >> >(dev_a, N, i);
        }
}

__global__ void get_det(double *a, size_t N, double *c_temp){
    int i, k;
	double* c = (double *) malloc(N*N*sizeof(double));
	double *dev_b, *dev_c;
	double l;
	// cudaMalloc((void**)&dev_a, N*N * sizeof(double));
	cudaMalloc((void**)&dev_b, N*N * sizeof(double));
	cudaMalloc((void**)&dev_c, N*N * sizeof(double));
	// cudaMemcpy(dev_a, a, N*N * sizeof(double), cudaMemcpyHostToDevice);
	gaussianElimination<<<1,1>>>(a, N);
	cudaMemcpyAsync(c, a, N*N * sizeof(double), cudaMemcpyDeviceToHost);
	double det = 1.0;
	for (i = 0; i<N; i++) {
		for (k = 0; k<N; k++) {
			if (i >= k) {
				l = c[i*N + k];
				if (i == k) {
					det *= l;
				}
			}
			else l = 0;
		}	
	}
	// cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	*c_temp = det;
}

__global__ void adj_cuda(double *a, double *c, int N){
    int ind1 = blockDim.x * blockIdx.x + threadIdx.x;
    int ind2 = blockDim.y * blockIdx.y + threadIdx.y;
    if(ind1 < N && ind2 < N){
        double *b = NULL;
        cudaError_t err = cudaMalloc((void**)&b, N*N*sizeof(double));
        int threadsPerBlock = 512;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        cof<<<blocksPerGrid, threadsPerBlock>>>(a, b, ind1, ind2, N);
        double *c_temp = NULL;
        get_det<<<blocksPerGrid, threadsPerBlock>>>(b, N, c_temp);
        c[ind1*N + ind2] = *c_temp;
    }
}

void cuda1DArray::control_cuda(cudaError_t error){
    if(error != cudaSuccess){
        cout << "Error in cuda" << cudaGetErrorString(error) << endl;
        exit(1);
    }
}

cuda1DArray::cuda1DArray(vector<vector<double>> data_vec){
    cudaError_t err = cudaSuccess;
    height = data_vec.size();
    width = data_vec[0].size();
    size = height*width*sizeof(double);
    shape.first = height;
    shape.second = width;
    double* temp = new double[height*width];
    
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            temp[i*width + j] = data_vec[i][j];
        }
    }
    err = cudaMalloc((double **)&data, size);
    control_cuda(err);
    err = cudaMemcpy(data, temp, size, cudaMemcpyHostToDevice);
    control_cuda(err);
    delete[] temp;
}

cuda1DArray::cuda1DArray(int height, int width){
    cudaError_t err = cudaSuccess;
    this->height = height;
    this->width = width;
    size = height*width*sizeof(double);
    shape.first = height;
    shape.second = width;
    err = cudaMalloc((double **)&data, size);
    control_cuda(err);
}

cuda1DArray::cuda1DArray(int height, int width, double* arr, bool already_cuda){
    cudaError_t err = cudaSuccess;
    this->height = height;
    this->width = width;
    shape.first = height;
    shape.second = width;
    if(already_cuda){
        size = height*width*sizeof(double);
        err = cudaMalloc((double **)&data, size);
        control_cuda(err);
        err = cudaMemcpy(data, arr, size, cudaMemcpyHostToDevice);
        control_cuda(err);
    }
    else{
        data = arr;
    }
}

cuda1DArray cuda1DArray::addConstant(double constant){
    cuda1DArray result(height, width);
    int threadsPerBlock = 512;
    int blocksPerGrid = (width*height + threadsPerBlock - 1) / threadsPerBlock;
    constAdd<<<blocksPerGrid, threadsPerBlock>>>(data, constant, result.data, width*height);
    return result;
}

cuda1DArray cuda1DArray::subConstant(double constant){
    cuda1DArray result(height, width);
    int threadsPerBlock = 512;
    int blocksPerGrid = (width*height + threadsPerBlock - 1) / threadsPerBlock;
    constSub<<<blocksPerGrid, threadsPerBlock>>>(data, constant, result.data, width*height);
    return result;
}

cuda1DArray cuda1DArray::mulConstant(double constant){
    cuda1DArray result(height, width);
    int threadsPerBlock = 512;
    int blocksPerGrid = (width*height + threadsPerBlock - 1) / threadsPerBlock;
    constMul<<<blocksPerGrid, threadsPerBlock>>>(data, constant, result.data, width*height);
    return result;
}

cuda1DArray cuda1DArray::divConstant(double constant){
    cuda1DArray result(height, width);
    int threadsPerBlock = 512;
    int blocksPerGrid = (width*height + threadsPerBlock - 1) / threadsPerBlock;
    constDiv<<<blocksPerGrid, threadsPerBlock>>>(data, constant, result.data, width*height);
    return result;
}

cuda1DArray cuda1DArray::addMatrix(cuda1DArray other){
    if(width != other.width || height != other.height){
        cout << "Error in addMatrix: Matrix dimensions do not match" << endl;
        return other;
    }
    cuda1DArray result(height, width);
    int threadsPerBlock = 512;
    int blocksPerGrid = (width*height + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(data, other.data, result.data, width*height);
    return result;
}

cuda1DArray cuda1DArray::subMatrix(cuda1DArray other){
    if(width != other.width || height != other.height){
        cout << "Error in subMatrix: Matrix dimensions do not match" << endl;
        return other;
    }
    cuda1DArray result(height, width);
    int threadsPerBlock = 512;
    int blocksPerGrid = (width*height + threadsPerBlock - 1) / threadsPerBlock;
    vectorSub<<<blocksPerGrid, threadsPerBlock>>>(data, other.data, result.data, width*height);
    return result;
}

cuda1DArray cuda1DArray::mulMatrix(cuda1DArray other){
    if(width != other.width || height != other.height){
        cout << "Error in mulMatrix: Matrix dimensions do not match" << endl;
        return other;
    }
    cuda1DArray result(height, width);
    int threadsPerBlock = 512;
    int blocksPerGrid = (width*height + threadsPerBlock - 1) / threadsPerBlock;
    vectorMul<<<blocksPerGrid, threadsPerBlock>>>(data, other.data, result.data, width*height);
    return result;
}

cuda1DArray cuda1DArray::divMatrix(cuda1DArray other){
    if(width != other.width || height != other.height){
        cout << "Error in divMatrix: Matrix dimensions do not match" << endl;
        return other;
    }
    cuda1DArray result(height, width);
    int threadsPerBlock = 512;
    int blocksPerGrid = (width*height + threadsPerBlock - 1) / threadsPerBlock;
    vectorDiv<<<blocksPerGrid, threadsPerBlock>>>(data, other.data, result.data, width*height);
    return result;
}

string cuda1DArray::printMatrix(){
    string str;
    double *temp_data = new double[height*width];
    cudaError_t err = cudaMemcpy(temp_data, data, size, cudaMemcpyDeviceToHost);
    control_cuda(err);
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            str += to_string(temp_data[i*width + j]) + " ";
        }
        str += '\n';
    }
    return str;
}

cuda1DArray cuda1DArray::T(){
    cuda1DArray result_cuda(height, width);
    int threadsPerBlock = 512;
    int blocksPerGrid = (width*height + threadsPerBlock - 1) / threadsPerBlock;
    transpose<<<blocksPerGrid, threadsPerBlock>>>(result_cuda.data, data, height, width);
    return result_cuda;
}

double cuda1DArray::det(){
    if(width != height){
        cout << "Error in det: Matrix is not square" << endl;
        return 0;
    }
    size_t N = width;
    double *temp_data = new double[N*N];
    cudaError_t err = cudaMemcpy(temp_data, data, size, cudaMemcpyDeviceToHost);
    control_cuda(err);
    int threadsPerBlock = 512;
    int blocksPerGrid = (N*N + threadsPerBlock - 1) / threadsPerBlock;
    double determinant;
    get_det<<<blocksPerGrid, threadsPerBlock>>>(temp_data, N, &determinant);
    return (determinant==0) ?0 :determinant;
}

cuda1DArray cuda1DArray::copy(){
    cuda1DArray result(height, width);
    cudaError_t err = cudaMemcpy(result.data, data, size, cudaMemcpyDeviceToDevice);
    control_cuda(err);
    return result;
}

double cuda1DArray::cofactor(int i, int j, bool internal_call){
    if(!internal_call){
        if(width != height){
            cout << "Error in cofactor: Matrix is not square" << endl;
        }
        if(i < 0 || i >= height || j < 0 || j >= width){
            cout << "Error in cofactor: Index out of bounds" << endl;
        }
    }
    cuda1DArray result(height-1, width-1);
    dim3 grid(height, width, 1);
    dim3 threads(height, width, 1);
    cof<<<grid, threads>>>(data, result.data, i, j, height);
    int sign = (i+j)%2 ? -1 : 1;
    return result.det()*sign;
}

cuda1DArray cuda1DArray::adjoint(){
    cuda1DArray result(height, width);
    double *temp_data = new double[height*width];
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            temp_data[i*height+j] = cofactor(i, j, true);
            // cout<<"cof"<< cofactor(i, j, true) << endl;
        }
    }
    cudaError_t err = cudaMemcpy(result.data, temp_data, size, cudaMemcpyHostToDevice);
    control_cuda(err);
    return result.T();
}

cuda1DArray cuda1DArray::inv(){
    int threadsPerBlock = 512;
    int blocksPerGrid = (width*height + threadsPerBlock - 1) / threadsPerBlock;
    double determinant;
    get_det<<<blocksPerGrid, threadsPerBlock>>>(data, height, &determinant);
    if(determinant == 0){
        cout << "Error in inverse: Determinant is 0" << endl;
        return *this;
    }
    double* adj = NULL;
    cudaError_t err = cudaMalloc((double **)&adj, size);
    control_cuda(err);
    adj_cuda<<<blocksPerGrid, threadsPerBlock>>>(data, adj, height);
    control_cuda(err);
    constDiv<<<blocksPerGrid, threadsPerBlock>>>(adj, determinant, adj, height*width);
    cuda1DArray result(height, width, adj, true);
    return result;
}

int main(){
    vector<vector<double>> data_vec = {
        {1,2,3},
        {4,5,6},
        {7,8,10}
    };
    cuda1DArray A(data_vec);
    vector<vector<double>> data_vec2 = {
        {1,2,3},
        {4,5,6},
        {7,8,9}
    };
    cuda1DArray B(data_vec2);
    vector<vector<double>> data_vec3 = {
        {1,2},
        {3,4}
    };
    cuda1DArray C(data_vec3);
    cuda1DArray D = A.inv();
    // double D = A.cofactor(0, 0, false);
    cout << D.printMatrix() << endl;
    // cout << D << endl;
    return 0;
}