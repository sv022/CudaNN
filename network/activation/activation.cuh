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