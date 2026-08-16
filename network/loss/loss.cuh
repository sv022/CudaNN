#pragma once
#include<string>

enum class LossType { MSE, BinaryCrossEntropy, CategoricalCrossEntropy };

__host__ __device__ inline float loss_value(float output, float target, LossType loss_type) {
    const float eps = 1e-7f;
    switch (loss_type) {
        case LossType::MSE: {
            float diff = target - output;
            return 0.5f * diff * diff;
        }
        case LossType::BinaryCrossEntropy: {
            float o = output;
            if (o < eps) o = eps;
            else if (o > 1.0f - eps) o = 1.0f - eps;
            return -(target * logf(o) + (1.0f - target) * logf(1.0f - o));
        }
        case LossType::CategoricalCrossEntropy: {
            float o = output;
            if (o < eps) o = eps;
            return -(target * logf(o));
        }
        default:
            return 0.0f;
    }
}

LossType parse_loss_type(std::string loss_str) {
    if (loss_str == "mse") return LossType::MSE;
    if (loss_str == "bce") return LossType::BinaryCrossEntropy;
    if (loss_str == "cce") return LossType::CategoricalCrossEntropy;
    throw std::runtime_error("Unknown loss function: " + loss_str);
}

std::string loss_type_to_str(LossType loss) {
    if (loss == LossType::MSE) return "\"mse\"";
    if (loss == LossType::BinaryCrossEntropy) return "\"bce\"";
    if (loss == LossType::CategoricalCrossEntropy) return "\"cce\"";
    return "null";
}