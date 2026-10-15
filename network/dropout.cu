#pragma once
#include"layer.cuh"
#include<cstdint>

__device__ __forceinline__ uint32_t dropout_hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__global__ void dropout_forward_kernel(const float* in, float* out, float* mask, size_t n,
                                       float p, float scale, uint32_t seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float r = (dropout_hash32((uint32_t)i ^ dropout_hash32(seed)) >> 8) * (1.0f / 16777216.0f);
    float m = (r > p) ? 1.0f : 0.0f;
    mask[i] = m;
    out[i]  = in[i] * m * scale;
}

__global__ void dropout_backward_kernel(const float* next_err, const float* mask, float* grad,
                                        size_t n, float scale) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) grad[i] = next_err[i] * mask[i] * scale;
}

class Dropout : public Layer 
{
private:
    float drop_prob;
    float* mask;

    float *d_mask = nullptr, *d_prev_errors = nullptr;
    uint32_t seed_base = 0, seed_counter = 0;
    void realloc_device();

public:
    Dropout(int size_, float p);

    void set_batch_size(int bs) override;

    void forward(float* inputs) override;

    float* backward(float* /*inputs*/, float* next_errors, bool /*raw_gradient*/) override;

    ~Dropout() { free(mask); cudaFree(d_outputs); cudaFree(d_mask); cudaFree(d_prev_errors); }
};

void Dropout::realloc_device() {
    size_t bytes = sizeof(float) * (size_t)batch_size * size;
    cudaFree(d_outputs); cudaFree(d_mask); cudaFree(d_prev_errors);
    cudaMalloc(&d_outputs, bytes);
    cudaMalloc(&d_mask, bytes);
    cudaMalloc(&d_prev_errors, bytes);
}

Dropout::Dropout(int size_, float p) {
    size = size_;
    output_size = size_;
    batch_size = 1;
    drop_prob = p;
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * size_);
    mask = (float*)malloc(sizeof(float) * (size_t)batch_size * size_);
    seed_base = (uint32_t)rand();
    realloc_device();
}

void Dropout::set_batch_size(int bs) {
    batch_size = bs;
    free(outputs);
    free(mask);
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * size);
    mask = (float*)malloc(sizeof(float) * (size_t)batch_size * size);
    realloc_device();
}

void Dropout::forward(float* d_in) {
    size_t total = (size_t)batch_size * size;
    if (!is_training) {
        cudaMemcpyAsync(d_outputs, d_in, sizeof(float) * total, cudaMemcpyDeviceToDevice);
        return;
    }
    uint32_t seed = seed_base + (seed_counter++) * 0x9E3779B9u;
    float scale = 1.0f / (1.0f - drop_prob);
    int threads = 256;
    dropout_forward_kernel<<<(unsigned)((total + threads - 1) / threads), threads>>>(
        d_in, d_outputs, d_mask, total, drop_prob, scale, seed);
}

float* Dropout::backward(float*, float* d_next_errors, bool) {
    if (!is_training) return d_next_errors;
    size_t total = (size_t)batch_size * size;
    float scale = 1.0f / (1.0f - drop_prob);
    int threads = 256;
    dropout_backward_kernel<<<(unsigned)((total + threads - 1) / threads), threads>>>(
        d_next_errors, d_mask, d_prev_errors, total, scale);
    return d_prev_errors;
}
