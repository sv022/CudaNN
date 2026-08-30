#pragma once

__global__
void sgd_update_kernel(float* weights, const float* dWeights, float lr, int total, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    weights[idx] += lr * (dWeights[idx] / (float)batch_size);
}

__global__
void momentum_update_kernel(float* weights, const float* dWeights, float* velocity, float lr, float beta, int total, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float g = dWeights[idx] / (float)batch_size;
    velocity[idx] = beta * velocity[idx] + (1.0f - beta) * g;
    weights[idx] += lr * velocity[idx];
}

__global__
void adam_update_kernel(float* weights, const float* dWeights, float* m, float* v,
                        float lr, float beta1, float beta2, float eps,
                        float bias_correction1, float bias_correction2,
                        int total, int batch_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float g = dWeights[idx] / (float)batch_size;

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float m_hat = m[idx] / bias_correction1;
    float v_hat = v[idx] / bias_correction2;

    weights[idx] += lr * m_hat / (sqrtf(v_hat) + eps);
}