#include <vector>
#include "layer.cuh"

class Conv : public Layer
{
private:
    int input_width, input_height, channels;
    int kernel_size, num_kernels, stride, padding;
    int output_width, output_height;

    float *kernels, *biases;

    float *d_kernels, *d_biases, *d_dKernels, *d_dBiases;
    float *d_inputs, *d_local_grad, *d_dInputs;
    bool buffers_allocated;

    std::vector<float*> d_kernel_state;
    std::vector<float*> d_bias_state;
    bool optimizer_state_allocated;

    void allocate_batch_buffers(int bs);
    void free_batch_buffers();
    void allocate_optimizer_state();
    void free_optimizer_state();

public:
    Conv(int input_height, int input_width, int channels, int kernel_size,
         int n_kernels = 1, int stride = 1, int padding = 0);
    ~Conv();

    void set_batch_size(int bs) override;
    void sync_weights_to_device();
    void sync_weights_to_host();

    float* backward(float *inputs, float *targets, bool raw_grad = false) override;
    void forward(float *inputs) override;

    void save_weights(std::string path) override;
    int load_weights(std::string path, int start) override;
};

Conv::Conv(int input_h, int input_w, int c, int k, int n_kernels, int stride, int padding) {
    input_width = input_w;
    input_height = input_h;
    channels = c;

    size = input_w * input_h * channels;
    kernel_size = k;
    this->stride = stride;
    this->padding = padding;
    num_kernels = n_kernels;

    output_width = ((input_width + 2 * padding - kernel_size) / stride) + 1;
    output_height = ((input_height + 2 * padding - kernel_size) / stride) + 1;
    output_size = num_kernels * output_height * output_width;

    batch_size = 1;
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * output_size);

    int kernel_total = num_kernels * channels * kernel_size * kernel_size;
    kernels = (float*)malloc(sizeof(float) * kernel_total);
    biases = (float*)malloc(sizeof(float) * num_kernels);

    float init_range = 1.0f / sqrt(channels * kernel_size * kernel_size);
    Matrix::initRandomf_static(kernels, 1, kernel_total, -init_range, init_range);
    Matrix::initRandomf_static(biases, 1, num_kernels, -init_range, init_range);

    cudaMalloc(&d_kernels, kernel_total * sizeof(float));
    cudaMalloc(&d_biases, num_kernels * sizeof(float));
    cudaMalloc(&d_dKernels, kernel_total * sizeof(float));
    cudaMalloc(&d_dBiases, num_kernels * sizeof(float));

    sync_weights_to_device();

    d_inputs = d_outputs = d_local_grad = d_dInputs = nullptr;
    buffers_allocated = false;
    allocate_batch_buffers(batch_size);

    optimizer_state_allocated = false;
}

Conv::~Conv() {
    free(outputs);
    free(kernels);
    free(biases);
    cudaFree(d_kernels);
    cudaFree(d_biases);
    cudaFree(d_dKernels);
    cudaFree(d_dBiases);
    free_batch_buffers();
    free_optimizer_state();
}

void Conv::allocate_optimizer_state() {
    if (optimizer_state_allocated) return;
    if (!optimizer) return;

    int kernel_total = num_kernels * channels * kernel_size * kernel_size;
    int n = optimizer->state_buffers_needed();

    d_kernel_state.resize(n);
    d_bias_state.resize(n);
    for (int i = 0; i < n; i++) {
        cudaMalloc(&d_kernel_state[i], (size_t)kernel_total * sizeof(float));
        cudaMemset(d_kernel_state[i], 0, (size_t)kernel_total * sizeof(float));

        cudaMalloc(&d_bias_state[i], (size_t)num_kernels * sizeof(float));
        cudaMemset(d_bias_state[i], 0, (size_t)num_kernels * sizeof(float));
    }
    optimizer_state_allocated = true;
}

void Conv::free_optimizer_state() {
    if (!optimizer_state_allocated) return;
    for (float* buf : d_kernel_state) cudaFree(buf);
    for (float* buf : d_bias_state) cudaFree(buf);
    d_kernel_state.clear();
    d_bias_state.clear();
    optimizer_state_allocated = false;
}

void Conv::set_batch_size(int bs) {
    if (bs == batch_size && buffers_allocated) return;
    batch_size = bs;
    allocate_batch_buffers(bs);
}

void Conv::free_batch_buffers() {
    if (!buffers_allocated) return;
    free_io();
    buffers_allocated = false;
}

void Conv::allocate_batch_buffers(int bs) {
    free_batch_buffers();
    alloc_io(bs);
    buffers_allocated = true;
}

void Conv::sync_weights_to_device() {
    int kernel_total = num_kernels * channels * kernel_size * kernel_size;
    cudaMemcpy(d_kernels, kernels, kernel_total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, num_kernels * sizeof(float), cudaMemcpyHostToDevice);
}

void Conv::sync_weights_to_host() {
    int kernel_total = num_kernels * channels * kernel_size * kernel_size;
    cudaMemcpy(kernels, d_kernels, (size_t)kernel_total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases, d_biases, (size_t)num_kernels * sizeof(float), cudaMemcpyDeviceToHost);
}

void Conv::forward(float *d_in) {
    size_t total_output = (size_t)batch_size * output_size;
    cudaMemset(d_outputs, 0, total_output * sizeof(float));
    dim3 BLOCK(16, 16);
    dim3 GRID((output_width + BLOCK.x - 1) / BLOCK.x, (output_height + BLOCK.y - 1) / BLOCK.y,
              num_kernels * batch_size);
    conv_forward_kernel<<<GRID, BLOCK>>>(d_in, d_kernels, d_biases, d_outputs,
        channels, input_height, input_width, kernel_size, stride, padding,
        num_kernels, output_height, output_width, batch_size);
}

float* Conv::backward(float *d_in, float *d_err, bool) {
    int kernel_total = num_kernels * channels * kernel_size * kernel_size;
    size_t total_input = (size_t)batch_size * size;
    cudaMemset(d_dKernels, 0, kernel_total * sizeof(float));
    cudaMemset(d_dBiases, 0, num_kernels * sizeof(float));

    int threads1d = 256;
    dim3 grid1(num_kernels);
    dbias_kernel<<<grid1, threads1d>>>(d_err, d_dBiases, output_height, output_width, num_kernels, batch_size);
    int blocks = (kernel_total + threads1d - 1) / threads1d;
    dkernel_kernel<<<blocks, threads1d>>>(d_in, d_err, d_dKernels, channels, input_height, input_width,
        kernel_size, stride, padding, num_kernels, output_height, output_width, batch_size);

    if (compute_input_grad) {
        cudaMemset(d_prev_errors, 0, total_input * sizeof(float));
        dim3 THREADS(16, 16);
        dim3 grid((output_width + THREADS.x - 1) / THREADS.x, (output_height + THREADS.y - 1) / THREADS.y,
                  num_kernels * batch_size);
        dinput_kernel<<<grid, THREADS>>>(d_kernels, d_err, d_prev_errors, channels, input_height, input_width,
            kernel_size, stride, padding, num_kernels, output_height, output_width, batch_size);
    }

    if (!optimizer_state_allocated) allocate_optimizer_state();
    optimizer->update(d_kernels, d_dKernels, d_kernel_state.empty() ? nullptr : d_kernel_state.data(), kernel_total, batch_size);
    optimizer->update(d_biases, d_dBiases, d_bias_state.empty() ? nullptr : d_bias_state.data(), num_kernels, batch_size);
    return compute_input_grad ? d_prev_errors : nullptr;
}

void Conv::save_weights(std::string path) {
    sync_weights_to_host();
    std::ofstream file(path, std::ios::binary | std::ios::app);

    int kernel_total = num_kernels * channels * kernel_size * kernel_size;

    file.write(reinterpret_cast<char*>(kernels), sizeof(float) * kernel_total);
    file.write(reinterpret_cast<char*>(biases), sizeof(float) * num_kernels);

    file.close();
}

int Conv::load_weights(std::string path, int start) {
    std::ifstream file(path, std::ios::binary);

    file.seekg(start, std::ios::beg);

    int kernel_total = num_kernels * channels * kernel_size * kernel_size;

    free(kernels);
    kernels = (float*)malloc(sizeof(float) * kernel_total);
    file.read(reinterpret_cast<char*>(kernels), sizeof(float) * kernel_total);

    free(biases);
    biases = (float*)malloc(sizeof(float) * num_kernels);
    file.read(reinterpret_cast<char*>(biases), sizeof(float) * num_kernels);

    file.close();

    sync_weights_to_device();

    return kernel_total + num_kernels;
}