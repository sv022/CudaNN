#pragma once


__global__ void conv_forward_kernel(
    const float* __restrict__ inputs,
    const float* __restrict__ kernels,
    const float* __restrict__ biases,
    float* outputs,
    int C, int H, int W,
    int K, int stride, int pad,
    int N, int OH, int OW,
    int batch_size
){
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int n = blockIdx.z % N;
    int b = blockIdx.z / N;

    if (out_x >= OW || out_y >= OH || b >= batch_size) return;

    int in_y_origin = out_y * stride - pad;
    int in_x_origin = out_x * stride - pad;

    float sum = biases[n];

    const float* inputs_b = inputs + (size_t)b * C * H * W;

    for (int c = 0; c < C; ++c)
    {
        int input_off = c * (H * W);
        int kernel_off = n * (C * K * K) + c * (K * K);

        for (int u = 0; u < K; ++u)
        {
            int in_y = in_y_origin + u;
            if (in_y < 0 || in_y >= H) continue;

            for (int v = 0; v < K; ++v)
            {
                int in_x = in_x_origin + v;
                if (in_x < 0 || in_x >= W) continue;

                float val = inputs_b[input_off + in_y * W + in_x];
                float w   = kernels[kernel_off + u * K + v];
                sum += val * w;
            }
        }
    }
    outputs[(size_t)b * N * OH * OW + n * (OH * OW) + out_y * OW + out_x] = sum;
}


__global__ void dbias_kernel(const float* __restrict__ dOut, float* dBiases,
                              int OH, int OW, int N, int batch_size)
{
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int stride_t = blockDim.x;

    int size = OH * OW;
    float s = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
        const float* base = dOut + (size_t)b * N * size + (size_t)n * size;
        for (int idx = tid; idx < size; idx += stride_t) s += base[idx];
    }

    atomicAdd(&dBiases[n], s);
}

__global__
void dkernel_kernel(const float* __restrict__ inputs, const float* __restrict__ dOut,
                    float* dKernels,
                    int C, int H, int W,
                    int K, int S, int P,
                    int N, int OH, int OW,
                    int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * K * K;
    if (idx >= total) return;

    int kv = K * K;

    int n = idx / (C * kv);
    int rem = idx % (C * kv);
    int c = rem / kv;
    rem = rem % kv;
    int u = rem / K;
    int v = rem % K;

    float grad = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
        const float* inputs_b = inputs + (size_t)b * C * H * W;
        const float* dout_base = dOut + (size_t)b * N * OH * OW + (size_t)n * OH * OW;

        for (int i = 0; i < OH; ++i) {
            int in_y = i * S - P + u;
            if (in_y < 0 || in_y >= H) continue;

            for (int j = 0; j < OW; ++j) {
                int in_x = j * S - P + v;
                if (in_x < 0 || in_x >= W) continue;

                float x = inputs_b[c * (H * W) + in_y * W + in_x];
                float dout = dout_base[i * OW + j];
                grad += x * dout;
            }
        }
    }

    dKernels[idx] = grad;
}

__global__
void dinput_kernel(const float* __restrict__ kernels, const float* __restrict__ dOut,
                   float* dInputs,
                   int C, int H, int W,
                   int K, int S, int P,
                   int N, int OH, int OW,
                   int batch_size)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int n = blockIdx.z % N;
    int b = blockIdx.z / N;

    if (out_x >= OW || out_y >= OH || b >= batch_size) return;

    const float* dOut_b = dOut + (size_t)b * N * OH * OW;
    float* dInputs_b = dInputs + (size_t)b * C * H * W;

    int out_idx = n * (OH * OW) + out_y * OW + out_x;
    float dout = dOut_b[out_idx];

    for (int c = 0; c < C; ++c) {
        int kernel_off = n * (C * K * K) + c * (K * K);
        for (int u = 0; u < K; ++u) {
            int in_y = out_y * S - P + u;
            if (in_y < 0 || in_y >= H) continue;
            for (int v = 0; v < K; ++v) {
                int in_x = out_x * S - P + v;
                if (in_x < 0 || in_x >= W) continue;

                float w = kernels[kernel_off + u * K + v];
                atomicAdd(&dInputs_b[c * (H * W) + in_y * W + in_x], w * dout);
            }
        }
    }
}
