#pragma once

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