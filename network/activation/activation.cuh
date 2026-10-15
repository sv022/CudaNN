#pragma once
#include<string>


enum class ActivationType { Sigmoid, ReLU, Softmax, Linear };

__host__ __device__ inline float activation_function(float z, ActivationType act) {
    switch (act) {
        case ActivationType::Sigmoid:
            return 1.0f / (1.0f + expf(-z));
        case ActivationType::ReLU:
            return z > 0.0f ? z : 0.0f;
        case ActivationType::Linear:
            return z;
        case ActivationType::Softmax:
            return z;
        default:
            return z;
    }
}

__host__ __device__ inline float activation_derivative(float next_error, float output, ActivationType act) {
    switch (act) {
        case ActivationType::Sigmoid:
            return next_error * output * (1.0f - output);
        case ActivationType::ReLU:
            return next_error * (output > 0.0f ? 1.0f : 0.0f);
        case ActivationType::Linear:
            return next_error;
        case ActivationType::Softmax:
            return next_error;
        default:
            return next_error;
    }
}

__global__ void act_forward_kernel(const float* z, float* out, size_t n, ActivationType t) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = activation_function(z[i], t);
}

__global__ void act_backward_kernel(const float* next_err, const float* out, float* grad,
                                    size_t n, ActivationType t) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) grad[i] = activation_derivative(next_err[i], out[i], t);
}

__global__ void softmax_forward_kernel(const float* z, float* out, int size) {
    extern __shared__ float sh[];
    const float* rz = z   + (size_t)blockIdx.x * size;
    float*       ro = out + (size_t)blockIdx.x * size;
    int t = threadIdx.x;

    float m = -INFINITY;
    for (int i = t; i < size; i += blockDim.x) m = fmaxf(m, rz[i]);
    sh[t] = m; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] = fmaxf(sh[t], sh[t + s]);
        __syncthreads();
    }
    m = sh[0]; __syncthreads();

    float sum = 0.0f;
    for (int i = t; i < size; i += blockDim.x) { float e = expf(rz[i] - m); ro[i] = e; sum += e; }
    sh[t] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    sum = sh[0];
    for (int i = t; i < size; i += blockDim.x) ro[i] /= sum;
}

ActivationType map_activation_type(std::string activation_str) {
    if (activation_str == "sigmoid") return ActivationType::Sigmoid;
    if (activation_str == "relu") return ActivationType::ReLU;
    if (activation_str == "softmax") return ActivationType::Softmax;
    throw std::runtime_error("Unknown activation function: " + activation_str);
}

std::string activation_type_to_str(ActivationType act){
    if (act == ActivationType::Sigmoid) return "\"sigmoid\"";
    if (act == ActivationType::ReLU) return "\"relu\"";
    if (act == ActivationType::Softmax) return "\"softmax\"";
    return "null";
}