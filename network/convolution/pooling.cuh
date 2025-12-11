__global__ void maxpool_forward_kernel(
    const float* __restrict__ d_inputs,
    float* d_outputs,
    int* d_indices,
    int C, int H, int W,
    int pool, int stride,
    int OH, int OW
){
    int c = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    if (oh >= OH || ow >= OW) return;

    int out_index = c * OH * OW + oh * OW + ow;

    int h0 = oh * stride;
    int w0 = ow * stride;

    float max_val = -1e30f;
    int max_idx = -1;

    int base = c * H * W;

    for (int kh = 0; kh < pool; kh++) {
        for (int kw = 0; kw < pool; kw++) {
            
            int ih = h0 + kh;
            int iw = w0 + kw;

            int idx = base + ih * W + iw;
            float v = d_inputs[idx];

            if (v > max_val) {
                max_val = v;
                max_idx = idx;
            }
        }
    }

    d_outputs[out_index] = max_val;
    d_indices[out_index] = max_idx;
}

__global__ void maxpool_backward_kernel(
    const int* __restrict__ d_indices,
    const float* __restrict__ d_next,
    float* d_dInput,
    int total_outputs // C * OH * OW
)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= total_outputs) return;

    int idx = d_indices[o];
    float grad = d_next[o];
    
    atomicAdd(&d_dInput[idx], grad);
}

