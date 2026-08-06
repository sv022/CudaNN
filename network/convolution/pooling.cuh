__global__ void maxpool_forward_kernel(
    const float* __restrict__ d_inputs,   // [B*C*H*W]
    float* d_outputs,                     // [B*C*OH*OW]
    int* d_indices,                       // [B*C*OH*OW]
    int C, int H, int W,
    int pool, int stride,
    int OH, int OW,
    int batch_size
){
    int c = blockIdx.z % C;
    int b = blockIdx.z / C;

    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    if (oh >= OH || ow >= OW || b >= batch_size) return;

    int output_size_per_image = C * OH * OW;
    int input_size_per_image = C * H * W;

    const float* inputs_b = d_inputs + (size_t)b * input_size_per_image;
    float* outputs_b = d_outputs + (size_t)b * output_size_per_image;
    int* indices_b = d_indices + (size_t)b * output_size_per_image;

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
            float v = inputs_b[idx];

            if (v > max_val) {
                max_val = v;
                max_idx = idx;
            }
        }
    }

    outputs_b[out_index] = max_val;
    indices_b[out_index] = max_idx;
}

__global__ void maxpool_backward_kernel(
    const int* __restrict__ d_indices,    // [B*C*OH*OW]
    const float* __restrict__ d_next,     // [B*C*OH*OW]
    float* d_dInput,                      // [B*C*H*W]
    int output_size_per_image,            // C*OH*OW
    int input_size_per_image,             // C*H*W
    int total_outputs                     // batch_size*C*OH*OW
)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= total_outputs) return;

    int b = o / output_size_per_image;
    int local_idx = d_indices[o]; 

    float grad = d_next[o];

    atomicAdd(&d_dInput[(size_t)b * input_size_per_image + local_idx], grad);
}

