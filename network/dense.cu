#include <vector>
#include "layer.cuh"

class Dense : public Layer
{
private:
    float *weights, *biases;

    float *d_weights, *d_biases;
    float *d_weight_grad_sum, *d_bias_grad_sum;

    std::vector<float*> d_weight_state;
    std::vector<float*> d_bias_state;

    bool buffers_allocated;
    bool optimizer_state_allocated;

    void allocate_batch_buffers(int bs);
    void free_batch_buffers();
    void allocate_optimizer_state();
    void free_optimizer_state();

public:
    Dense(int layer_size, int next_size);
    ~Dense();

    void set_batch_size(int bs) override;
    void sync_weights_to_device();
    void sync_weights_to_host();

    float* backward(float *inputs, float *next_errors, bool raw_gradient = false) override;
    void forward(float *inputs) override;

    void save_weights(std::string path) override;
    int load_weights(std::string path, int start) override;
};

Dense::Dense(int layer_size, int next_size){
    size = layer_size;
    output_size = next_size;

    batch_size = 1;
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * output_size);

    weights = (float*)malloc(sizeof(float) * size * output_size);
    Matrix::initRandomf_static(weights, layer_size, next_size, -1 / sqrt((float)layer_size), 1 / sqrt((float)layer_size));

    biases = (float*)malloc(sizeof(float) * output_size);
    Matrix::initRandomf_static(biases, 1, output_size, -1 / sqrt((float)layer_size), 1 / sqrt((float)layer_size));

    int in_size = size, out_size = output_size;
    cudaMalloc(&d_weights, in_size * out_size * sizeof(float));
    cudaMalloc(&d_biases, out_size * sizeof(float));
    cudaMalloc(&d_weight_grad_sum, in_size * out_size * sizeof(float));
    cudaMalloc(&d_bias_grad_sum, out_size * sizeof(float));

    sync_weights_to_device();

    buffers_allocated = false;
    allocate_batch_buffers(batch_size);

    optimizer_state_allocated = false;
}

Dense::~Dense() {
    free(outputs); free(weights); free(biases);
    cudaFree(d_weights); cudaFree(d_biases);
    cudaFree(d_weight_grad_sum); cudaFree(d_bias_grad_sum);
    free_batch_buffers();
    free_optimizer_state();
}

void Dense::allocate_optimizer_state() {
    if (optimizer_state_allocated) return;
    if (!optimizer) return;

    int in_size = size, out_size = output_size;
    int n = optimizer->state_buffers_needed();

    d_weight_state.resize(n);
    d_bias_state.resize(n);
    for (int i = 0; i < n; i++) {
        cudaMalloc(&d_weight_state[i], (size_t)in_size * out_size * sizeof(float));
        cudaMemset(d_weight_state[i], 0, (size_t)in_size * out_size * sizeof(float));

        cudaMalloc(&d_bias_state[i], (size_t)out_size * sizeof(float));
        cudaMemset(d_bias_state[i], 0, (size_t)out_size * sizeof(float));
    }
    optimizer_state_allocated = true;
}

void Dense::free_optimizer_state() {
    if (!optimizer_state_allocated) return;
    for (float* buf : d_weight_state) cudaFree(buf);
    for (float* buf : d_bias_state) cudaFree(buf);
    d_weight_state.clear();
    d_bias_state.clear();
    optimizer_state_allocated = false;
}

void Dense::set_batch_size(int bs) {
    if (bs == batch_size && buffers_allocated) return;
    batch_size = bs;
    allocate_batch_buffers(bs);
}

void Dense::free_batch_buffers() {
    if (!buffers_allocated) return;
    free_io();
    buffers_allocated = false;
}
void Dense::allocate_batch_buffers(int bs) {
    free_batch_buffers();
    alloc_io(bs);
    buffers_allocated = true;
}

void Dense::sync_weights_to_device() {
    int in_size = size, out_size = output_size;
    cudaMemcpy(d_weights, weights, in_size * out_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, out_size * sizeof(float), cudaMemcpyHostToDevice);
}

void Dense::sync_weights_to_host() {
    cudaMemcpy(weights, d_weights, (size_t)size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases, d_biases, (size_t)output_size * sizeof(float), cudaMemcpyDeviceToHost);
}

void Dense::forward(float *d_in) {
    dim3 THREADS(32, 32);
    dim3 weightsBlocksPerGrid((output_size + THREADS.x - 1) / THREADS.x,
                              (batch_size  + THREADS.y - 1) / THREADS.y);
    Kernel::dot_bias<<<weightsBlocksPerGrid, THREADS>>>(d_in, d_weights, d_biases, d_outputs, batch_size, size, output_size);
}

float* Dense::backward(float *d_in, float *d_err, bool) {
    int in_size = size, out_size = output_size;

    dim3 wg_grid((in_size + 127) / 128, out_size);
    dense_dweights_kernel<<<wg_grid, 128>>>(d_in, d_err, d_weight_grad_sum, batch_size, in_size, out_size);

    dense_dbias_kernel<<<out_size, 256>>>(d_err, d_bias_grad_sum, batch_size, out_size);

    if (compute_input_grad) {
        dim3 dx_grid((in_size + 127) / 128, batch_size);
        dense_dinput_kernel<<<dx_grid, 128>>>(d_err, d_weights, d_prev_errors, batch_size, in_size, out_size);
    }

    if (!optimizer_state_allocated) allocate_optimizer_state();
    optimizer->update(d_weights, d_weight_grad_sum,
                      d_weight_state.empty() ? nullptr : d_weight_state.data(), in_size * out_size, batch_size);
    optimizer->update(d_biases, d_bias_grad_sum,
                      d_bias_state.empty() ? nullptr : d_bias_state.data(), out_size, batch_size);
    return compute_input_grad ? d_prev_errors : nullptr;
}

void Dense::save_weights(std::string path) {
    sync_weights_to_host();
    std::ofstream file(path, std::ios::binary | std::ios::app);

    file.write(reinterpret_cast<char*>(weights), sizeof(float) * size * output_size);
    file.write(reinterpret_cast<char*>(biases), sizeof(float) * output_size);

    file.close();
}

int Dense::load_weights(std::string path, int start) {
    std::ifstream file(path, std::ios::binary);

    file.seekg(start, std::ios::beg);

    free(weights);
    weights = (float*)malloc(sizeof(float) * size * output_size);
    file.read(reinterpret_cast<char*>(weights), sizeof(float) * size * output_size);

    free(biases);
    biases = (float*)malloc(sizeof(float) * output_size);
    file.read(reinterpret_cast<char*>(biases), sizeof(float) * output_size);

    file.close();

    sync_weights_to_device();

    return (size * output_size) + output_size;
}