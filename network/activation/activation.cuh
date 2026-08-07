#pragma once
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
