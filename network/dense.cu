#include <utility>
#include"layer.cuh"


class Dense : public Layer
{
    private:
    float *weights, *biases;

    float *d_weights, *d_biases;
    float *d_weights_updated, *d_biases_updated;
    float *d_weight_grad_sum, *d_bias_grad_sum;
    float *d_weights_T;

    float *d_inputs, *d_outputs, *d_inputs_T, *d_local_grad, *d_prev_errors;
    bool buffers_allocated;

    void allocate_batch_buffers(int bs);
    void free_batch_buffers();

    public:
    Dense(int layer_size, int next_size);
    ~Dense();

    void set_batch_size(int bs) override;
    void sync_weights_to_device();

    float* backward(float *inputs, float *targets);
    void forward(float *inputs) override;

    void save_weights(std::string path) override;
    int load_weights(std::string path, int start) override;

    friend void test_dense_forward_batch_size_1_equivalence();
    friend void test_dense_forward_batch();
    friend void test_dense_backward_batch();

    friend void test_dense_save_load_roundtrip();
    friend void test_multi_layer_save_load_offsets();
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
    cudaMalloc(&d_weights_updated, in_size * out_size * sizeof(float));
    cudaMalloc(&d_biases_updated, out_size * sizeof(float));
    cudaMalloc(&d_weight_grad_sum, in_size * out_size * sizeof(float));
    cudaMalloc(&d_bias_grad_sum, out_size * sizeof(float));
    cudaMalloc(&d_weights_T, out_size * in_size * sizeof(float));

    sync_weights_to_device();

    d_inputs = d_outputs = d_inputs_T = d_local_grad = d_prev_errors = nullptr;
    buffers_allocated = false;
    allocate_batch_buffers(batch_size);
}

Dense::~Dense() {
    free(outputs); free(weights); free(biases);
    cudaFree(d_weights); cudaFree(d_biases);
    cudaFree(d_weights_updated); cudaFree(d_biases_updated);
    cudaFree(d_weight_grad_sum); cudaFree(d_bias_grad_sum);
    cudaFree(d_weights_T);
    free_batch_buffers();
}

void Dense::set_batch_size(int bs) {
    if (bs == batch_size && buffers_allocated) return;
    if (bs != batch_size) {
        free(outputs);
        outputs = (float*)malloc(sizeof(float) * (size_t)bs * output_size);
    }
    batch_size = bs;
    allocate_batch_buffers(bs);
}

void Dense::sync_weights_to_device() {
    int in_size = size, out_size = output_size;
    cudaMemcpy(d_weights, weights, in_size * out_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, out_size * sizeof(float), cudaMemcpyHostToDevice);
}

void Dense::free_batch_buffers() {
    if (!buffers_allocated) return;
    cudaFree(d_inputs); cudaFree(d_outputs);
    cudaFree(d_inputs_T); cudaFree(d_local_grad); cudaFree(d_prev_errors);
    buffers_allocated = false;
}

void Dense::allocate_batch_buffers(int bs) {
    free_batch_buffers();
    int in_size = size, out_size = output_size;
    size_t total_input = (size_t)bs * in_size;
    size_t total_output = (size_t)bs * out_size;

    cudaMalloc(&d_inputs, total_input * sizeof(float));
    cudaMalloc(&d_outputs, total_output * sizeof(float));
    cudaMalloc(&d_inputs_T, total_input * sizeof(float));
    cudaMalloc(&d_local_grad, total_output * sizeof(float));
    cudaMalloc(&d_prev_errors, total_input * sizeof(float));
    buffers_allocated = true;
}

void Dense::forward(float *inputs) {
    int in_size = size, out_size = output_size;
    size_t total_input_size = (size_t)batch_size * in_size;
    size_t total_output_size = (size_t)batch_size * out_size;

    cudaMemcpy(d_inputs, inputs, total_input_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 THREADS(32, 32);
    dim3 weightsBlocksPerGrid(
        (out_size + THREADS.x - 1) / THREADS.x,
        (batch_size + THREADS.y - 1) / THREADS.y
    );

    Kernel::dot_bias_softmax<<<weightsBlocksPerGrid, THREADS>>>(
        d_inputs, d_weights, d_biases, d_outputs,
        batch_size, in_size, out_size
    );

    cudaDeviceSynchronize();
    cudaMemcpy(outputs, d_outputs, total_output_size * sizeof(float), cudaMemcpyDeviceToHost);
}


float* Dense::backward(float *inputs, float *next_errors) {
    int in_size = size, out_size = output_size;
    size_t total_out = (size_t)batch_size * out_size;
    size_t total_in = (size_t)batch_size * in_size;

    float *local_grad = (float*)malloc(sizeof(float) * total_out);
    for (size_t i = 0; i < total_out; ++i) {
        local_grad[i] = next_errors[i] * outputs[i] * (1.0f - outputs[i]);
    }

    cudaMemcpy(d_inputs, inputs, total_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_local_grad, local_grad, total_out * sizeof(float), cudaMemcpyHostToDevice);

    dim3 THREADS(32, 32);

    dim3 IN_TRANS_BLOCKS(
        (in_size + THREADS.x - 1) / THREADS.x,
        (batch_size + THREADS.y - 1) / THREADS.y
    );
    Kernel::transpose<<<IN_TRANS_BLOCKS, THREADS>>>(d_inputs, d_inputs_T, batch_size, in_size);
    cudaDeviceSynchronize();

    dim3 WG_BLOCKS(
        (out_size + THREADS.x - 1) / THREADS.x,
        (in_size + THREADS.y - 1) / THREADS.y
    );
    Kernel::dot<<<WG_BLOCKS, THREADS>>>(d_inputs_T, d_local_grad, d_weight_grad_sum, in_size, batch_size, out_size);
    cudaDeviceSynchronize();

    int bias_threads = 256;
    int bias_blocks = (out_size + bias_threads - 1) / bias_threads;
    Kernel::sum_rows<<<bias_blocks, bias_threads>>>(d_local_grad, d_bias_grad_sum, batch_size, out_size);
    cudaDeviceSynchronize();

    float effective_lr = learning_rate / (float)batch_size;

    dim3 W_TRANS_BLOCKS(
        (out_size + THREADS.x - 1) / THREADS.x,
        (in_size + THREADS.y - 1) / THREADS.y
    );
    Kernel::transpose<<<W_TRANS_BLOCKS, THREADS>>>(d_weights, d_weights_T, in_size, out_size);
    cudaDeviceSynchronize();

    dim3 DIN_BLOCKS(
        (in_size + THREADS.x - 1) / THREADS.x,
        (batch_size + THREADS.y - 1) / THREADS.y
    );
    Kernel::dot<<<DIN_BLOCKS, THREADS>>>(d_local_grad, d_weights_T, d_prev_errors, batch_size, out_size, in_size);
    cudaDeviceSynchronize();

    dim3 W_UPD_BLOCKS(
        (out_size + THREADS.x - 1) / THREADS.x,
        (in_size + THREADS.y - 1) / THREADS.y
    );
    Kernel::multadd<<<W_UPD_BLOCKS, THREADS>>>(d_weights, d_weight_grad_sum, d_weights_updated, in_size, out_size, effective_lr);

    dim3 B_UPD_BLOCKS((out_size + THREADS.x - 1) / THREADS.x, 1);
    Kernel::multadd<<<B_UPD_BLOCKS, THREADS>>>(d_biases, d_bias_grad_sum, d_biases_updated, 1, out_size, effective_lr);

    cudaDeviceSynchronize();

    std::swap(d_weights, d_weights_updated);
    std::swap(d_biases, d_biases_updated);


    cudaMemcpy(weights, d_weights, in_size * out_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases, d_biases, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    float *prev_errors = (float*)malloc(sizeof(float) * total_in);
    cudaMemcpy(prev_errors, d_prev_errors, total_in * sizeof(float), cudaMemcpyDeviceToHost);

    free(local_grad);
    return prev_errors;
}


void Dense::save_weights(std::string path) {
    std::ofstream file(path, std::ios::binary | std::ios::app);

    file.write(reinterpret_cast<const char*>(weights), sizeof(float) * size * output_size);
    file.write(reinterpret_cast<const char*>(biases), sizeof(float) * output_size);

    file.close();
}

int Dense::load_weights(std::string path, int start) { // return loaded weights size in bytes
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