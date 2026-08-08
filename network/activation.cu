#pragma once
#include"layer.cuh"
#include"activation/activation.cuh"
#include<cstdio>
#include<cassert>
#include<cstring>
#include<cmath>

class Activation : public Layer {
public:
    Activation(int _size, ActivationType act_type);
    ~Activation() { free(outputs); }
    void forward(float* z) override;
    float* backward(float*, float* next_errors, bool raw_gradient) override;

    void set_batch_size(int bs) override;
};

Activation::Activation(int _size, ActivationType act_type) {
    size = _size;
    output_size = _size;
    batch_size = 1;
    activation_type = act_type;
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * _size);
}

void Activation::set_batch_size(int bs) {
    batch_size = bs;
    free(outputs);
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * size);
}

void Activation::forward(float* z) {
    size_t total = (size_t)batch_size * size;

    if (activation_type == ActivationType::Softmax) {
        for (int b = 0; b < batch_size; ++b) {
            float* row_z = z + (size_t)b * size;
            float* row_out = outputs + (size_t)b * size;

            float m = row_z[0];
            for (int i = 1; i < size; ++i) if (row_z[i] > m) m = row_z[i];

            float sum = 0.0f;
            for (int i = 0; i < size; ++i) { row_out[i] = expf(row_z[i] - m); sum += row_out[i]; }
            for (int i = 0; i < size; ++i) row_out[i] /= sum;
        }
    } else {
        for (size_t i = 0; i < total; ++i) {
            outputs[i] = activation_function(z[i], activation_type);
        }
    }
}

float* Activation::backward(float* /*inputs*/, float* next_errors, bool raw_gradient) {
    if (activation_type == ActivationType::Softmax && !raw_gradient) {
        assert(false && "Softmax activation without raw_gradient is unsupported");
    }

    size_t total = (size_t)batch_size * size;
    float* local_grad = (float*)malloc(sizeof(float) * total);

    if (raw_gradient) {
        memcpy(local_grad, next_errors, sizeof(float) * total);
    } else {
        for (size_t i = 0; i < total; ++i) {
            local_grad[i] = activation_derivative(next_errors[i], outputs[i], activation_type);
        }
    }

    return local_grad;
}
