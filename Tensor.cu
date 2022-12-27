#include "Tensor.cuh"

__global__ void AddKernel(double *a, double *b, double *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void SubKernel(double *a, double *b, double *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void MulKernel(double *a, double *b, double *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void DivKernel(double *a, double *b, double *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void AddScalarKernel(double *a, double b, double *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b;
    }
}

__global__ void SubScalarKernel(double *a, double b, double *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] - b;
    }
}

__global__ void MulScalarKernel(double *a, double b, double *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b;
    }
}

__global__ void DivScalarKernel(double *a, double b, double *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] / b;
    }
}

__global__ void PowerKernel(double *a, double b, double *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = pow(a[idx], b);
    }
}

__global__ void SignKernel(double *a, double *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = (a[idx] > 0) - (a[idx] < 0);
    }
}

__global__ void CopyKernel(double *a, double *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx];
    }
}

__global__ void TransposeKernel(double *a, double *c, int batch_size, int rows,
                                int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * rows * cols) {
        int batch = idx / (rows * cols);
        int row = (idx % (rows * cols)) / cols;
        int col = (idx % (rows * cols)) % cols;
        c[batch * rows * cols + col * rows + row] = a[idx];
    }
}

__global__ void MatrixMulKernel(double *a, double *b, double *c, int a_batch,
                                int a_rows, int a_cols, int b_rows,
                                int b_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a_batch * a_rows * b_cols) {
        int batch = idx / (a_rows * b_cols);
        int row = (idx % (a_rows * b_cols)) / b_cols;
        int col = (idx % (a_rows * b_cols)) % b_cols;
        double sum = 0;
        for (int i = 0; i < a_cols; i++) {
            sum += a[batch * a_rows * a_cols + row * a_cols + i] *
                   b[batch * b_rows * b_cols + i * b_cols + col];
        }
        c[batch * a_rows * b_cols + row * b_cols + col] = sum;
    }
}

void Tensor::err_check(cudaError_t err, string msg) {
    if (err != cudaSuccess) {
        throw runtime_error(msg + " failed with error: " +
                            string(cudaGetErrorString(err)) + "\n");
    }
}

Tensor::Tensor() {
    this->batch_size_ = 0;
    this->rows_ = 0;
    this->cols_ = 0;
    this->data_ = nullptr;
    this->shape_ = make_tuple(0, 0, 0);
    this->size_ = 0;
}

Tensor::Tensor(int batch_size, int rows, int cols) {
    this->batch_size_ = batch_size;
    this->rows_ = rows;
    this->cols_ = cols;
    this->shape_ = make_tuple(batch_size_, rows_, cols_);
    this->size_ = batch_size_ * rows_ * cols_;
    cudaError_t err = cudaMallocManaged((double **)&this->data_, this->size_ * sizeof(double));
    err_check(err, "Tensor::Tensor(int batch_size, int rows, int cols)");
}

Tensor::Tensor(int batch_size, int rows, int cols, double constant) {
    this->batch_size_ = batch_size;
    this->rows_ = rows;
    this->cols_ = cols;
    this->shape_ = make_tuple(batch_size_, rows_, cols_);
    this->size_ = batch_size_ * rows_ * cols_;
    cudaError_t err = cudaMallocManaged(&data_, this->size_ * sizeof(double));
    err_check(
        err,
        "Tensor::Tensor(int batch_size, int rows, int cols, double constant)");

    for (int i = 0; i < this->size_; i++) {
        this->data_[i] = constant;
    }
}

Tensor::Tensor(int batch_size, int rows, int cols, double min, double max,
               string initializer) {
    this->batch_size_ = batch_size;
    this->rows_ = rows;
    this->cols_ = cols;
    this->shape_ = make_tuple(batch_size_, rows_, cols_);
    this->size_ = batch_size_ * rows_ * cols_;
    cudaError_t err = cudaMallocManaged(&data_, this->size_ * sizeof(double));
    err_check(err,
              "Tensor::Tensor(int batch_size, int rows, int cols, double min, "
              "double max, string initializer)");
    mt19937 rng(0);
    uniform_real_distribution<double> dist(min, max);

    double scale;

    if (initializer == "uniform") {
        scale = 1.0;
    } else if (initializer == "xavier") {
        scale = sqrt(6.0 / (rows + cols));
    } else if (initializer == "he") {
        scale = sqrt(2.0 / rows);
    } else {
        throw invalid_argument("Invalid initializer");
    }

    for (int i = 0; i < this->size_; i++) {
        this->data_[i] = dist(rng) * scale;
    }
}

Tensor::Tensor(const Tensor &t) {
    this->batch_size_ = t.batch_size_;
    this->rows_ = t.rows_;
    this->cols_ = t.cols_;
    this->shape_ = t.shape_;
    this->size_ = t.size_;
    cudaError_t err =
        cudaMallocManaged(&this->data_, this->size_ * sizeof(double));
    err_check(err, "Tensor::Tensor(const Tensor &t)");

    for (int i = 0; i < this->size_; i++) {
        this->data_[i] = t.data_[i];
    }
}

Tensor::Tensor(vector<double> vec) {
    this->batch_size_ = 1;
    this->rows_ = vec.size();
    this->cols_ = 1;
    this->shape_ = make_tuple(batch_size_, rows_, cols_);
    this->size_ = batch_size_ * rows_ * cols_;
    cudaError_t err =
        cudaMallocManaged((double **)&this->data_, this->size_ * sizeof(double));
    err_check(err, "Tensor::Tensor(vector<double> vec)");

    for (int i = 0; i < this->rows_; i++) {
        this->data_[i] = vec[i];
    }
}

Tensor::Tensor(vector<vector<double>> vec) {
    this->batch_size_ = 1;
    this->rows_ = vec.size();
    this->cols_ = vec[0].size();
    this->shape_ = make_tuple(batch_size_, rows_, cols_);
    this->size_ = batch_size_ * rows_ * cols_;
    cudaError_t err =
        cudaMallocManaged(&this->data_, this->size_ * sizeof(double));
    err_check(err, "Tensor::Tensor(vector<vector<double>> vec)");

    for (int i = 0; i < this->rows_; i++) {
        for (int j = 0; j < this->cols_; j++) {
            this->data_[i * this->cols_ + j] = vec[i][j];
        }
    }
}

Tensor::Tensor(vector<vector<vector<double>>> vec) {
    this->batch_size_ = vec.size();
    this->rows_ = vec[0].size();
    this->cols_ = vec[0][0].size();
    this->shape_ = make_tuple(batch_size_, rows_, cols_);
    this->size_ = batch_size_ * rows_ * cols_;
    cudaError_t err =
        cudaMallocManaged(&this->data_, this->size_ * sizeof(double));
    err_check(err, "Tensor::Tensor(vector<vector<vector<double>>> vec)");

    for (int i = 0; i < this->batch_size_; i++) {
        for (int j = 0; j < this->rows_; j++) {
            for (int k = 0; k < this->cols_; k++) {
                this->data_[i * this->rows_ * this->cols_ + j * this->cols_ +
                            k] = vec[i][j][k];
            }
        }
    }
}

Tensor::~Tensor() {
    // printf("Tensor destructor called\n");
    cudaError_t err = cudaFree(this->data_);
    err_check(err, "Tensor::~Tensor()");
}

double &Tensor::operator()(int batch, int row, int col) {
    return this
        ->data_[batch * this->rows_ * this->cols_ + row * this->cols_ + col];
}

double Tensor::operator()(int batch, int row, int col) const {
    return this
        ->data_[batch * this->rows_ * this->cols_ + row * this->cols_ + col];
}

Tensor Tensor::operator+(const Tensor &t) {
    if (this->shape_ != t.shape_) {
        throw invalid_argument("Tensor shapes do not match for addition\n");
    }

    Tensor result(this->batch_size_, this->rows_, this->cols_);

    int grid_size = this->size_ / 1024 + 1;
    int block_size = 1024;
    dim3 grid(grid_size);
    dim3 block(block_size);
    AddKernel<<<grid, block>>>(this->data_, t.data_, result.data_, this->size_);
    cudaDeviceSynchronize();
    // printf("AddKernel launched with %d blocks of %d threads\n", grid.x,
    //        block.x);
    // printf("Result: %s\n", result.toString().c_str());
    return result;
}

Tensor Tensor::operator-(const Tensor &t) {
    if (this->shape_ != t.shape_) {
        throw invalid_argument("Tensor shapes do not match for subtraction\n");
    }

    Tensor result(this->batch_size_, this->rows_, this->cols_);

    dim3 grid(1);
    dim3 block(this->size_);
    SubKernel<<<grid, block>>>(this->data_, t.data_, result.data_, this->size_);
    cudaDeviceSynchronize();

    return result;
}

Tensor Tensor::operator*(const Tensor &t) {
    if (this->shape_ != t.shape_) {
        throw invalid_argument(
            "Tensor shapes do not match for multiplication\n");
    }

    Tensor result(this->batch_size_, this->rows_, this->cols_);

    dim3 grid(1);
    dim3 block(this->size_);
    MulKernel<<<grid, block>>>(this->data_, t.data_, result.data_, this->size_);
    cudaDeviceSynchronize();

    return result;
}

Tensor Tensor::operator/(const Tensor &t) {
    if (this->shape_ != t.shape_) {
        throw invalid_argument("Tensor shapes do not match for division\n");
    }

    Tensor result(this->batch_size_, this->rows_, this->cols_);

    dim3 grid(1);
    dim3 block(this->size_);
    DivKernel<<<grid, block>>>(this->data_, t.data_, result.data_, this->size_);
    cudaDeviceSynchronize();

    return result;
}

Tensor Tensor::operator+(double scalar) {
    Tensor result(this->batch_size_, this->rows_, this->cols_);
    int grid_size = this->size_ / 1024 + 1;
    int block_size = 1024;
    dim3 grid(grid_size);
    dim3 block(block_size);
    AddScalarKernel<<<grid, block>>>(this->data_, scalar, result.data_,
                                     this->size_);
    cudaDeviceSynchronize();

    return result;
}

Tensor Tensor::operator-(double scalar) {
    Tensor result(this->batch_size_, this->rows_, this->cols_);

    dim3 grid(1);
    dim3 block(this->size_);
    SubScalarKernel<<<grid, block>>>(this->data_, scalar, result.data_,
                                     this->size_);
    cudaDeviceSynchronize();

    return result;
}

Tensor Tensor::operator*(double scalar) {
    Tensor result(this->batch_size_, this->rows_, this->cols_);

    dim3 grid(1);
    dim3 block(this->size_);
    MulScalarKernel<<<grid, block>>>(this->data_, scalar, result.data_,
                                     this->size_);
    cudaDeviceSynchronize();

    return result;
}

Tensor Tensor::operator/(double scalar) {
    Tensor result(this->batch_size_, this->rows_, this->cols_);

    dim3 grid(1);
    dim3 block(this->size_);
    DivScalarKernel<<<grid, block>>>(this->data_, scalar, result.data_,
                                     this->size_);
    cudaDeviceSynchronize();

    return result;
}

Tensor &Tensor::operator+=(const Tensor &t) {
    if (this->shape_ != t.shape_) {
        throw invalid_argument("Tensor shapes do not match for addition\n");
    }

    dim3 grid(1);
    dim3 block(this->size_);
    AddKernel<<<grid, block>>>(this->data_, t.data_, this->data_, this->size_);
    cudaDeviceSynchronize();

    return *this;
}

Tensor &Tensor::operator-=(const Tensor &t) {
    if (this->shape_ != t.shape_) {
        throw invalid_argument("Tensor shapes do not match for subtraction\n");
    }

    dim3 grid(1);
    dim3 block(this->size_);
    SubKernel<<<grid, block>>>(this->data_, t.data_, this->data_, this->size_);
    cudaDeviceSynchronize();

    return *this;
}

Tensor &Tensor::operator*=(const Tensor &t) {
    if (this->shape_ != t.shape_) {
        throw invalid_argument(
            "Tensor shapes do not match for multiplication\n");
    }

    dim3 grid(1);
    dim3 block(this->size_);
    MulKernel<<<grid, block>>>(this->data_, t.data_, this->data_, this->size_);
    cudaDeviceSynchronize();

    return *this;
}

Tensor &Tensor::operator/=(const Tensor &t) {
    if (this->shape_ != t.shape_) {
        throw invalid_argument("Tensor shapes do not match for division\n");
    }

    dim3 grid(1);
    dim3 block(this->size_);
    DivKernel<<<grid, block>>>(this->data_, t.data_, this->data_, this->size_);
    cudaDeviceSynchronize();

    return *this;
}

Tensor &Tensor::operator+=(double scalar) {
    dim3 grid(1);
    dim3 block(this->size_);
    AddScalarKernel<<<grid, block>>>(this->data_, scalar, this->data_,
                                     this->size_);
    cudaDeviceSynchronize();

    return *this;
}

Tensor &Tensor::operator-=(double scalar) {
    dim3 grid(1);
    dim3 block(this->size_);
    SubScalarKernel<<<grid, block>>>(this->data_, scalar, this->data_,
                                     this->size_);
    cudaDeviceSynchronize();

    return *this;
}

Tensor &Tensor::operator*=(double scalar) {
    dim3 grid(1);
    dim3 block(this->size_);
    MulScalarKernel<<<grid, block>>>(this->data_, scalar, this->data_,
                                     this->size_);
    cudaDeviceSynchronize();

    return *this;
}

Tensor &Tensor::operator/=(double scalar) {
    dim3 grid(1);
    dim3 block(this->size_);
    DivScalarKernel<<<grid, block>>>(this->data_, scalar, this->data_,
                                     this->size_);
    cudaDeviceSynchronize();

    return *this;
}

Tensor Tensor::power(double exponent) {
    Tensor result(this->batch_size_, this->rows_, this->cols_);

    dim3 grid(1);
    dim3 block(this->size_);
    PowerKernel<<<grid, block>>>(this->data_, exponent, result.data_,
                                 this->size_);
    cudaDeviceSynchronize();

    return result;
}

Tensor Tensor::sign() const {
    Tensor result(this->batch_size_, this->rows_, this->cols_);

    dim3 grid(1);
    dim3 block(this->size_);
    SignKernel<<<grid, block>>>(this->data_, result.data_, this->size_);
    cudaDeviceSynchronize();

    return result;
}

Tensor Tensor::abs() const {
    Tensor result(this->batch_size_, this->rows_, this->cols_);
    result = this->sign() * (*this);

    return result;
}

Tensor Tensor::copy() {
    // printf("copying tensor\n");
    Tensor result(this->batch_size_, this->rows_, this->cols_);
    // printf("created tensor\n");
    dim3 grid(1);
    dim3 block(this->size_);
    CopyKernel<<<grid, block>>>(this->data_, result.data_, this->size_);
    cudaDeviceSynchronize();
    // printf("done\n");
    return result;
}

Tensor Tensor::getitem(int indice) const {
    if (batch_size_ == 1) {
        Tensor result(1, cols_, 1);
        for (int i = 0; i < cols_; i++) {
            result(0, i, 0) = (*this)(0, indice, i);
        }
        return result;
    }

    Tensor result(1, rows_, cols_);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            result(0, i, j) = (*this)(indice, i, j);
        }
    }
    return result;
}

Tensor Tensor::getitem(int indice1, int indice2) const {
    Tensor result(batch_size_, 1, 1);
    for (int i = 0; i < batch_size_; i++) {
        result(i, 0, 0) = (*this)(i, indice1, indice2);
    }
    return result;
}

double Tensor::getitem(int indice1, int indice2, int indice3) const {
    return (*this)(indice1, indice2, indice3);
}

void Tensor::setitem(int num, const Tensor &other) {
    if (this->batch_size_ == 1) {
        for (int i = 0; i < this->cols_; i++) {
            (*this)(0, num, i) = other(0, i, 0);
        }
        return;
    }
    for (int i = 0; i < this->rows_; i++) {
        for (int j = 0; j < this->cols_; j++) {
            (*this)(num, i, j) = other(0, i, j);
        }
    }
}

void Tensor::setitem(int num1, int num2, const Tensor &other) {
    for (int i = 0; i < this->batch_size_; i++) {
        (*this)(i, num1, num2) = other(i, 0, 0);
    }
}

void Tensor::setitem(int num1, int num2, double value) {
    for (int i = 0; i < this->batch_size_; i++) {
        (*this)(i, num1, num2) = value;
    }
}

void Tensor::setitem(int num1, int num2, int num3, double value) {
    (*this)(num1, num2, num3) = value;
}

double Tensor::sum() const {
    double result = 0;
    for (int i = 0; i < this->batch_size_; i++) {
        for (int j = 0; j < this->rows_; j++) {
            for (int k = 0; k < this->cols_; k++) {
                result += (*this)(i, j, k);
            }
        }
    }
    return result;
}

int Tensor::size() const { return this->size_; }

int Tensor::getBatchsize() const { return this->batch_size_; }

int Tensor::getRows() const { return this->rows_; }

int Tensor::getCols() const { return this->cols_; }

string Tensor::toString() const {
    string result = "";
    for (int i = 0; i < this->batch_size_; i++) {
        result += "[";
        for (int j = 0; j < this->rows_; j++) {
            result += "[";
            for (int k = 0; k < this->cols_; k++) {
                result += to_string((*this)(i, j, k));
                if (k != this->cols_ - 1) {
                    result += ", ";
                }
            }
            result += "]";
            if (j != this->rows_ - 1) {
                result += ", ";
            }
        }
        result += "]";
        if (i != this->batch_size_ - 1) {
            result += ", ";
        }
    }
    return result;
}

vector<vector<vector<double>>> Tensor::tolist() const {
    double *data = nullptr;
    data = (double *)malloc(this->size_ * sizeof(double));
    cudaError_t err = cudaMemcpy(data, this->data_, this->size_ * sizeof(double),
                     cudaMemcpyDeviceToHost);
    vector<vector<vector<double>>> result = vector<vector<vector<double>>>(
        this->batch_size_,
        vector<vector<double>>(this->rows_, vector<double>(this->cols_)));

    for (int i = 0; i < this->batch_size_; i++) {
        for (int j = 0; j < this->rows_; j++) {
            for (int k = 0; k < this->cols_; k++) {
                result[i][j][k] =
                    data[i * this->rows_ * this->cols_ + j * this->cols_ + k];
            }
        }
    }
    return result;
}

Tensor Tensor::transpose() {
    Tensor result(this->batch_size_, this->cols_, this->rows_);

    dim3 grid(1);
    dim3 block(this->size_);
    TransposeKernel<<<grid, block>>>(
        this->data_, result.data_, this->batch_size_, this->rows_, this->cols_);
    cudaDeviceSynchronize();

    return result;
}

Tensor Tensor::dot_product(const Tensor &other) const {
    Tensor result(this->batch_size_, this->rows_, other.cols_);

    dim3 grid(1);
    dim3 block(this->size_);
    MatrixMulKernel<<<grid, block>>>(this->data_, other.data_, result.data_,
                                     this->batch_size_, this->rows_,
                                     this->cols_, other.rows_, other.cols_);
    cudaDeviceSynchronize();

    return result;
}

Tensor &Tensor::operator=(const Tensor &other) {
    if (this->size_ != other.size_) {
        this->size_ = other.size_;
        cudaError_t err = cudaFree(this->data_);
        err_check(err, "cudaFree =");
        err = cudaMalloc(&this->data_, this->size_ * sizeof(double));
        err_check(err, "cudaMalloc =");
    }
    this->batch_size_ = other.batch_size_;
    this->rows_ = other.rows_;
    this->cols_ = other.cols_;
    this->size_ = other.size_;
    this->shape_ = other.shape_;

    dim3 grid(1);
    dim3 block(this->size_);
    CopyKernel<<<grid, block>>>(other.data_, this->data_, this->size_);
    cudaDeviceSynchronize();

    return *this;
}