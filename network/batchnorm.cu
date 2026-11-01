#pragma once
#include"layer.cuh"
#include<vector>
#include<cmath>

#define BN_THREADS 256


__global__ void bn_stats_kernel(const float* __restrict__ x, float* mean, float* invstd,
                                float* run_mean, float* run_var,
                                int N, int C, int S, float eps, float momentum)
{
    const int c = blockIdx.x;
    const long long M = (long long)N * S;
    __shared__ float s_val;

    float sum = 0.0f;
    for (long long i = threadIdx.x; i < M; i += blockDim.x)
        sum += x[((size_t)(i / S) * C + c) * S + (i % S)];
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) s_val = sum / (float)M;
    __syncthreads();
    const float mu = s_val;

    float sq = 0.0f;
    for (long long i = threadIdx.x; i < M; i += blockDim.x) {
        float d = x[((size_t)(i / S) * C + c) * S + (i % S)] - mu;
        sq += d * d;
    }
    sq = block_reduce_sum(sq);
    if (threadIdx.x == 0) {
        const float var = sq / (float)M;
        mean[c] = mu;
        invstd[c] = rsqrtf(var + eps);
        if (M > 1) {
            run_mean[c] = (1.0f - momentum) * run_mean[c] + momentum * mu;
            run_var[c]  = (1.0f - momentum) * run_var[c]
                        + momentum * var * (float)M / (float)(M - 1);
        }
    }
}

__global__ void bn_eval_invstd_kernel(const float* run_var, float* invstd, int C, float eps)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) invstd[c] = rsqrtf(run_var[c] + eps);
}

__global__ void bn_forward_kernel(const float* __restrict__ x, float* y,
                                  const float* mean, const float* invstd,
                                  const float* gamma, const float* beta,
                                  int C, int S, size_t total)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (int)((i / S) % C);
    y[i] = gamma[c] * (x[i] - mean[c]) * invstd[c] + beta[c];
}

__global__ void bn_backward_reduce_kernel(const float* __restrict__ x, const float* __restrict__ e,
                                          const float* mean, const float* invstd,
                                          float* g_gamma, float* g_beta,
                                          int N, int C, int S)
{
    const int c = blockIdx.x;
    const long long M = (long long)N * S;
    const float mu = mean[c], is = invstd[c];

    float sb = 0.0f, sg = 0.0f;
    for (long long i = threadIdx.x; i < M; i += blockDim.x) {
        size_t idx = ((size_t)(i / S) * C + c) * S + (i % S);
        float ev = e[idx];
        sb += ev;
        sg += ev * (x[idx] - mu) * is;
    }
    sb = block_reduce_sum(sb);
    __syncthreads();
    sg = block_reduce_sum(sg);
    if (threadIdx.x == 0) { g_beta[c] = sb; g_gamma[c] = sg; }
}

__global__ void bn_dinput_kernel(const float* __restrict__ x, const float* __restrict__ e, float* dx,
                                 const float* mean, const float* invstd, const float* gamma,
                                 const float* g_gamma, const float* g_beta,
                                 int C, int S, float invM, size_t total)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (int)((i / S) % C);
    float xhat = (x[i] - mean[c]) * invstd[c];
    dx[i] = gamma[c] * invstd[c] * (e[i] - g_beta[c] * invM - xhat * g_gamma[c] * invM);
}


class BatchNorm : public Layer
{
private:
    int channels, spatial;
    const float eps = 1e-5f;
    const float momentum = 0.1f;

    float *d_params, *d_work;
    float *d_gamma, *d_beta, *d_run_mean, *d_run_var;
    float *d_g_gamma, *d_g_beta, *d_mean, *d_invstd;

    std::vector<float*> d_gamma_state;
    std::vector<float*> d_beta_state;

    bool buffers_allocated;
    bool optimizer_state_allocated;

    void allocate_batch_buffers(int bs);
    void free_batch_buffers();
    void allocate_optimizer_state();
    void free_optimizer_state();

public:
    BatchNorm(int layer_size, int channels_);
    ~BatchNorm();

    void set_batch_size(int bs) override;

    void forward(float* inputs) override;
    float* backward(float* inputs, float* next_errors, bool raw_gradient = false) override;

    void save_weights(std::string path) override;
    int load_weights(std::string path, int start) override;

    friend struct BatchNormTestAccess;
};

BatchNorm::BatchNorm(int layer_size, int channels_) {
    if (channels_ <= 0 || layer_size % channels_ != 0) {
        std::cerr << "BatchNorm: layer size " << layer_size << " is not divisible by channel count " << channels_ << std::endl;
        std::exit(1);
    }
    size = layer_size;
    output_size = layer_size;
    channels = channels_;
    spatial = layer_size / channels_;

    batch_size = 1;
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * output_size);

    std::vector<float> init(4 * (size_t)channels, 0.0f);
    for (int c = 0; c < channels; ++c) {
        init[c] = 1.0f;
        init[3 * channels + c] = 1.0f;
    }
    cudaMalloc(&d_params, init.size() * sizeof(float));
    cudaMalloc(&d_work, init.size() * sizeof(float));
    cudaMemcpy(d_params, init.data(), init.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_work, 0, init.size() * sizeof(float));

    d_gamma = d_params;
    d_beta = d_params + channels;
    d_run_mean = d_params + 2 * channels;
    d_run_var = d_params + 3 * channels;
    d_g_gamma = d_work;
    d_g_beta = d_work + channels;
    d_mean = d_work + 2 * channels;
    d_invstd = d_work + 3 * channels;

    buffers_allocated = false;
    allocate_batch_buffers(batch_size);

    optimizer_state_allocated = false;
}

BatchNorm::~BatchNorm() {
    free(outputs);
    cudaFree(d_params);
    cudaFree(d_work);
    free_batch_buffers();
    free_optimizer_state();
}

void BatchNorm::allocate_optimizer_state() {
    if (optimizer_state_allocated) return;
    if (!optimizer) return;

    int n = optimizer->state_buffers_needed();
    d_gamma_state.resize(n);
    d_beta_state.resize(n);
    for (int i = 0; i < n; i++) {
        cudaMalloc(&d_gamma_state[i], (size_t)channels * sizeof(float));
        cudaMemset(d_gamma_state[i], 0, (size_t)channels * sizeof(float));

        cudaMalloc(&d_beta_state[i], (size_t)channels * sizeof(float));
        cudaMemset(d_beta_state[i], 0, (size_t)channels * sizeof(float));
    }
    optimizer_state_allocated = true;
}

void BatchNorm::free_optimizer_state() {
    if (!optimizer_state_allocated) return;
    for (float* buf : d_gamma_state) cudaFree(buf);
    for (float* buf : d_beta_state) cudaFree(buf);
    d_gamma_state.clear();
    d_beta_state.clear();
    optimizer_state_allocated = false;
}

void BatchNorm::free_batch_buffers() {
    if (!buffers_allocated) return;
    free_io();
    buffers_allocated = false;
}

void BatchNorm::allocate_batch_buffers(int bs) {
    free_batch_buffers();
    alloc_io(bs);
    buffers_allocated = true;
}

void BatchNorm::set_batch_size(int bs) {
    if (bs == batch_size && buffers_allocated) return;
    batch_size = bs;
    free(outputs);
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * output_size);
    allocate_batch_buffers(bs);
}

void BatchNorm::forward(float* d_in) {
    size_t total = (size_t)batch_size * size;
    const float* use_mean;

    if (is_training) {
        bn_stats_kernel<<<channels, BN_THREADS>>>(d_in, d_mean, d_invstd, d_run_mean, d_run_var,
                                                  batch_size, channels, spatial, eps, momentum);
        use_mean = d_mean;
    } else {
        bn_eval_invstd_kernel<<<(channels + 127) / 128, 128>>>(d_run_var, d_invstd, channels, eps);
        use_mean = d_run_mean;
    }
    bn_forward_kernel<<<(unsigned)((total + 255) / 256), 256>>>(
        d_in, d_outputs, use_mean, d_invstd, d_gamma, d_beta, channels, spatial, total);
}

float* BatchNorm::backward(float* d_in, float* d_err, bool) {
    bn_backward_reduce_kernel<<<channels, BN_THREADS>>>(
        d_in, d_err, d_mean, d_invstd, d_g_gamma, d_g_beta, batch_size, channels, spatial);

    if (compute_input_grad) {
        size_t total = (size_t)batch_size * size;
        float invM = 1.0f / ((float)batch_size * (float)spatial);
        bn_dinput_kernel<<<(unsigned)((total + 255) / 256), 256>>>(
            d_in, d_err, d_prev_errors, d_mean, d_invstd, d_gamma,
            d_g_gamma, d_g_beta, channels, spatial, invM, total);
    }

    if (!optimizer_state_allocated) allocate_optimizer_state();
    optimizer->update(d_gamma, d_g_gamma,
                      d_gamma_state.empty() ? nullptr : d_gamma_state.data(), channels, batch_size);
    optimizer->update(d_beta, d_g_beta,
                      d_beta_state.empty() ? nullptr : d_beta_state.data(), channels, batch_size);
    return compute_input_grad ? d_prev_errors : nullptr;
}

void BatchNorm::save_weights(std::string path) {
    std::vector<float> h(4 * (size_t)channels);
    cudaMemcpy(h.data(), d_params, h.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream file(path, std::ios::binary | std::ios::app);
    file.write(reinterpret_cast<char*>(h.data()), sizeof(float) * h.size());
    file.close();
}

int BatchNorm::load_weights(std::string path, int start) {
    std::vector<float> h(4 * (size_t)channels);

    std::ifstream file(path, std::ios::binary);
    file.seekg(start, std::ios::beg);
    file.read(reinterpret_cast<char*>(h.data()), sizeof(float) * h.size());
    if (!file) {
        std::cerr << "BatchNorm: weights file is too short or does not match the network." << std::endl;
        std::exit(1);
    }
    file.close();

    cudaMemcpy(d_params, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return (int)h.size();
}