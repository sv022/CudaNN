#pragma once


__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = (lane < (blockDim.x >> 5)) ? sh[lane] : 0.0f;
    if (warp == 0) for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}


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
    int n = blockIdx.x, b = blockIdx.y;
    int size = OH * OW;
    const float* base = dOut + (size_t)b * N * size + (size_t)n * size;
    float s = 0.0f;
    for (int idx = threadIdx.x; idx < size; idx += blockDim.x) s += base[idx];
    s = block_reduce_sum(s);
    if (threadIdx.x == 0) atomicAdd(&dBiases[n], s);
}

__global__ void dkernel_kernel(const float* __restrict__ inputs, const float* __restrict__ dOut,
                               float* dKernels,
                               int C, int H, int W, int K, int S, int P,
                               int N, int OH, int OW, int batch_size)
{
    int widx = blockIdx.x;
    int kv = K * K;
    int n = widx / (C * kv);
    int rem = widx % (C * kv);
    int c = rem / kv; rem %= kv;
    int u = rem / K, v = rem % K;

    int OHW = OH * OW;
    long long total = (long long)batch_size * OHW;
    long long stride_t = (long long)gridDim.y * blockDim.x;

    float grad = 0.0f;
    for (long long t = (long long)blockIdx.y * blockDim.x + threadIdx.x; t < total; t += stride_t) {
        int b = (int)(t / OHW);
        int p = (int)(t % OHW);
        int in_y = (p / OW) * S - P + u;
        int in_x = (p % OW) * S - P + v;
        if (in_y < 0 || in_y >= H || in_x < 0 || in_x >= W) continue;
        grad += inputs[(size_t)b * C * H * W + c * (H * W) + in_y * W + in_x]
              * dOut[(size_t)b * N * OHW + (size_t)n * OHW + p];
    }
    grad = block_reduce_sum(grad);
    if (threadIdx.x == 0) atomicAdd(&dKernels[widx], grad);
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
