#include <iostream>
#include <vector>



#ifndef KERNEL_H
#define KERNEL_H 
namespace Kernel {
    // Kernel functions
    __global__ void dot(float *a, float *b, float *c, int N, int M, int P);
    __global__ void add(float *a, float *b, float *c, int R, int C);
    __global__ void sub(float *a, float *b, float *c, int R, int C);
    __global__ void transpose(float *a, float *c, int R, int C);
    __global__ void map(float *a, float *c, int R, int C);
}
#endif

__device__ float activation_function(float x) {
    return 1.0f / (1.0f + expf(-x)); 
}

class Matrix {
    public:
        float* values;
        float* cuda_values;
        int rows;
        int cols;
        std::size_t bsize;
    
        Matrix(float* m_values, int m_rows, int m_cols) {
            cuda_values = 0;
            values = m_values;
            rows = m_rows;
            cols = m_cols;
            bsize = sizeof(float) * m_rows * m_cols;
        }
    
        // Utility functions
        void init();
        void initRandomi(int min = 0, int max = 10);
        void initRandomf(float min = -1, float max = 1);
    
        static void init_static(float* m, int R, int C);
        static void initRandomi_static(float* m, int R, int C, int min = 0, int max = 10);
        static void initRandomf_static(float* m, int R, int C, float min = -1, float max = 1);
        
        static void log_static(float* m, int R, int C, char name = 'M');
        static void logVector(std::vector<float> input);
        void log(char name = 'M');
};



void Matrix::init_static(float* m, int R, int C) {
    for (int i = 0; i < R*C; i++) {
        m[i] = 0;		
    }
}


void Matrix::init() {
    for (int i = 0; i < rows*cols; i++) {
        values[i] = 0;		
    }
}


void Matrix::initRandomi_static(float* m, int R, int C, int min, int max) {
    for (int i = 0; i < R*C; i++) {
        m[i] = (rand() % (max - min)) + min;		
    }
}

void Matrix::initRandomi(int min, int max) {
    for (int i = 0; i < rows*cols; i++) {
        values[i] = (rand() % (max - min)) + min;		
    }
}

void Matrix::initRandomf_static(float* m, int R, int C, float min, float max) {
    for (int i = 0; i < R*C; i++) {
        m[i] = ((float)rand()/(float)RAND_MAX) * (max-min) + min;	
    }
}

void Matrix::initRandomf(float min, float max) {
    for (int i = 0; i < rows*cols; i++) {
        values[i] = ((float)rand()/(float)RAND_MAX) * (max-min) + min;	
    }
}

// ---------------------- LOGS ---------------------- 
void Matrix::log_static(float* m, int R, int C, char name) {
    char delim = ',';
    std::cout << name << " [\n";
    for (int i = 0; i < R; i++) {
        std::cout << "\t";
        for (int j = 0; j < C; j++) {
            delim = (j < C-1) ? ',' : ' ';
            std::cout << m[i * C + j] << delim;
        }
        std::cout << "\n";
    }

    std::cout << "]\n" << std::endl;
}

void Matrix::log(char name) {
    char delim = ',';
    std::cout << name << " [\n";
    for (int i = 0; i < rows; i++) {
        std::cout << "\t";
        for (int j = 0; j < cols; j++) {
            delim = (j == 0) ? ' ' : ',';
            std::cout << delim << values[i * cols + j];
        }
        std::cout << "\n";
    }

    std::cout << "]\n" << std::endl;
}

void Matrix::logVector(std::vector<float> input) {
    std::cout << "Vector [\n";
    char delim = ',';
    for (int i = 0; i < input.size(); i++) {
		delim = (i == input.size()-1) ? ' ' : ',';
        std::cout << "\t" << input[i] << delim << "\n";
	}
    std::cout << "]" << std::endl;
}

// --------------- kernels ----------------------
__global__ void Kernel::add(
    float *a,
    float *b,
    float *c,
    int R,
    int C
) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;	

    // Abort if out of range
    if (row >= R || col >= C) return;

    c[row * C + col] = a[row * C + col] + b[row * C + col];

    return;
}

__global__ void Kernel::dot(
    float *a,
    float *b,
    float *c,
    int M,
    int N,
    int P
) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * P + col];
        }
        c[row * P + col] = sum;
    }
}
__global__ void Kernel::map(float *input, float *output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        output[idx] = activation_function(input[idx]);
    }
}	
__global__ void Kernel::sub(
    float *a,
    float *b,
    float *c,
    int R,
    int C
) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;	

    // Abort if out of range
    if (row >= R || col >= C) return;

    c[row * C + col] = a[row * C + col] - b[row * C + col];

    return;
}
__global__ void Kernel::transpose(
    float *a,
    float *c,
    int R,
    int C
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Abort if out of range
    if (row >= R || col >= C) return; 

    c[col * R + row] = a[row * C + col];
}