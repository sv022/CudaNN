#pragma once
#include"layer.cuh"
#include"activation/activation.cuh"
#include<cstdio>
#include<cassert>
#include<cstring>
#include<cmath>


class Activation : public Layer {
private:
    float* d_prev_errors = nullptr;
    void realloc_device();

public:
    Activation(int _size, ActivationType act_type);
    ~Activation() { free(outputs); cudaFree(d_outputs); cudaFree(d_prev_errors); }
    void forward(float* z) override;
    float* backward(float*, float* next_errors, bool raw_gradient) override;

    void set_batch_size(int bs) override;
};

void Activation::realloc_device() {
    size_t bytes = sizeof(float) * (size_t)batch_size * size;
    cudaFree(d_outputs); cudaFree(d_prev_errors);
    cudaMalloc(&d_outputs, bytes);
    cudaMalloc(&d_prev_errors, bytes);
}

Activation::Activation(int _size, ActivationType act_type) {
    size = _size;
    output_size = _size;
    batch_size = 1;
    activation_type = act_type;
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * _size);
    realloc_device();
}

void Activation::set_batch_size(int bs) {
    batch_size = bs;
    free(outputs);
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * size);
    realloc_device();
}

void Activation::forward(float* d_z) {
    size_t total = (size_t)batch_size * size;
    if (activation_type == ActivationType::Softmax) {
        softmax_forward_kernel<<<batch_size, 128, 128 * sizeof(float)>>>(d_z, d_outputs, size);
    } else {
        int threads = 256;
        act_forward_kernel<<<(unsigned)((total + threads - 1) / threads), threads>>>(
            d_z, d_outputs, total, activation_type);
    }
}

float* Activation::backward(float*, float* d_next_errors, bool raw_gradient) {
    if (raw_gradient) return d_next_errors;
    assert(activation_type != ActivationType::Softmax && "Softmax activation without raw_gradient is unsupported");
    size_t total = (size_t)batch_size * size;
    int threads = 256;
    act_backward_kernel<<<(unsigned)((total + threads - 1) / threads), threads>>>(
        d_next_errors, d_outputs, d_prev_errors, total, activation_type);
    return d_prev_errors;
}
