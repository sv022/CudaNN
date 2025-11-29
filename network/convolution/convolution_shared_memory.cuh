#define BLOCK_W 16
#define BLOCK_H 16

__global__
void conv_kernel_shared(
    const float* __restrict__ inputs,   // [channels * input_height * input_width]
    const float* __restrict__ kernels,  // [num_kernels * channels * kernel_size * kernel_size]
    const float* __restrict__ biases,   // [num_kernels]
    float* outputs,                     // [num_kernels * output_height * output_width]
    int C, int H, int W,
    int K, int stride, int pad,
    int N, int OH, int OW
){
    extern __shared__ float smem[];

    const int tileH = BLOCK_H + K - 1;
    const int tileW = BLOCK_W + K - 1;

    int n = blockIdx.z;
    if (n >= N) return;

    int out_x = blockIdx.x * BLOCK_W + threadIdx.x;
    int out_y = blockIdx.y * BLOCK_H + threadIdx.y;

    int in_x_origin = out_x * stride - pad;
    int in_y_origin = out_y * stride - pad;

    float* tile = smem; 

    for (int c = 0; c < C; ++c) {
        for (int dy = threadIdx.y; dy < tileH; dy += BLOCK_H) {
            int in_y = in_y_origin + dy;
            for (int dx = threadIdx.x; dx < tileW; dx += BLOCK_W) {
                int in_x = in_x_origin + dx;
                float value = 0.0f;
                if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H)
                    value = inputs[c * (H*W) + in_y * W + in_x];
                tile[c * (tileH * tileW) + dy * tileW + dx] = value;
            }
        }
    }

    __syncthreads();

    if (out_x >= OW || out_y >= OH) return;

    float sum = biases[n];
    
    for (int c = 0; c < C; ++c)
    {
        const float* ker = &kernels[n * (C*K*K) + c * (K*K)];
        const float* t   = &tile[c * (tileH*tileW)];

        for (int u = 0; u < K; ++u)
        {
            for (int v = 0; v < K; ++v)
            {
                float val = t[(threadIdx.y + u) * tileW + (threadIdx.x + v)];
                sum += val * ker[u * K + v];
            }
        }
    }
    outputs[n * (OH*OW) + out_y * OW + out_x] = sum;
}
