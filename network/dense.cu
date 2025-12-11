#include"layer.cu"


class Dense : public Layer
{
    private:
    float *weights;
    float *biases;
    float learning_rate = 0.1;
    
    public:
    Dense(int layer_size, int next_size);
    
    float* backward(float *inputs, float *targets);
    void forward(float *inputs) override;

    void save_weights(std::string path) override;
    void load_weights(std::string path, int start) override;
};


Dense::Dense(int layer_size, int next_size){
    size = layer_size;
    output_size = next_size;

    outputs = (float*)malloc(sizeof(float) * output_size);

    weights = (float*)malloc(sizeof(float) * size * output_size);
    Matrix::initRandomf_static(weights, layer_size, next_size, -1 / sqrt(layer_size), 1 / sqrt(layer_size));

    biases = (float*)malloc(sizeof(float) * output_size);
    Matrix::initRandomf_static(biases, 1, output_size, -1 / sqrt(layer_size), 1 / sqrt(layer_size));

    // Matrix::log_static(weights, layer_size, next_size, 'W');
    // save_weights(size);
}


void Dense::forward(float *inputs) {
    float *d_weights = 0;
    float *d_biases = 0;
    float *d_inputs = 0;
    float *d_outputs = 0;
    
	cudaMalloc(&d_weights, size * output_size * sizeof(float));
	cudaMalloc(&d_biases, 1 * output_size * sizeof(float));
	cudaMalloc(&d_inputs, 1 * size * sizeof(float));
    cudaMalloc(&d_outputs, 1 * output_size * sizeof(float));
    
    cudaMemcpy(
        d_weights,
        weights,
        size * output_size * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_biases,
        biases,
        1 * output_size * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_inputs,
        inputs,
        1 * size * sizeof(float),
        cudaMemcpyHostToDevice
    );

    dim3 THREADS(32, 32);

    dim3 weightsBlocksPerGrid(
        (output_size + THREADS.x - 1) / THREADS.x,
        (1 + THREADS.y - 1) / THREADS.y
    );

    Kernel::dot_bias_softmax<<<weightsBlocksPerGrid, THREADS>>>(d_inputs, d_weights, d_biases, d_outputs, 1, size, output_size);

    cudaDeviceSynchronize();

    cudaMemcpy(
        outputs,
        d_outputs,
        output_size * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // Matrix::log_static(outputs, 1, output_size, 'O');

    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_inputs);
    cudaFree(d_outputs);
}


float* Dense::backward(float *inputs, float *next_errors) {
    const int in_size = size;
    const int out_size = output_size;

    float *local_grad = (float*)malloc(sizeof(float) * out_size);
    for (int i = 0; i < out_size; ++i) {
        local_grad[i] = next_errors[i] * outputs[i] * (1.0f - outputs[i]);
    }

    // Matrix::log_static(local_grad, 1, out_size, 'G');

    float *d_local_grad = nullptr;
    float *d_inputs = nullptr;
    float *d_weights = nullptr;
    float *d_biases = nullptr;
    float *d_weight_grad = nullptr;
    float *d_updated_weights = nullptr;
    float *d_updated_biases = nullptr;
    float *d_weights_T = nullptr;
    float *d_prev_errors = nullptr;

    cudaMalloc(&d_local_grad, out_size * sizeof(float));
    cudaMalloc(&d_inputs, in_size * sizeof(float));
    cudaMalloc(&d_weights, in_size * out_size * sizeof(float));
    cudaMalloc(&d_biases, out_size * sizeof(float));
    cudaMalloc(&d_weight_grad, in_size * out_size * sizeof(float));
    cudaMalloc(&d_updated_weights, in_size * out_size * sizeof(float));
    cudaMalloc(&d_updated_biases, out_size * sizeof(float));
    cudaMalloc(&d_weights_T, out_size * in_size * sizeof(float));
    cudaMalloc(&d_prev_errors, in_size * sizeof(float));

    cudaMemcpy(
        d_local_grad, 
        local_grad, 
        out_size * sizeof(float), 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_inputs, 
        inputs, 
        in_size * sizeof(float), 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_weights, 
        weights, 
        in_size * out_size * sizeof(float), 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_biases, 
        biases, 
        out_size * sizeof(float), 
        cudaMemcpyHostToDevice
    );

    dim3 THREADS(32, 32);
    dim3 BLOCKS(
        (out_size + THREADS.x - 1) / THREADS.x,
        (in_size + THREADS.y - 1) / THREADS.y
    );

    coarse_1d_xgemm(d_inputs, d_local_grad, d_weight_grad, in_size, out_size, 1);
    cudaDeviceSynchronize();

    Kernel::multadd<<<BLOCKS, THREADS>>>(d_weights, d_weight_grad, d_updated_weights, in_size, out_size, learning_rate);
    cudaDeviceSynchronize();

    dim3 BIAS_BLOCKS(
        (out_size + 255) / 256, 
        1
    );
    Kernel::multadd<<<BIAS_BLOCKS, THREADS>>>(d_biases, d_local_grad, d_updated_biases, 1, out_size, learning_rate);
    cudaDeviceSynchronize();

    dim3 TRANS_BLOCKS(
        (out_size + THREADS.x - 1) / THREADS.x,
        (in_size + THREADS.y - 1) / THREADS.y
    );

    Kernel::transpose<<<TRANS_BLOCKS, THREADS>>>(d_weights, d_weights_T, in_size, out_size);
    cudaDeviceSynchronize();

    dim3 ERROR_BLOCKS(
        (in_size + THREADS.x - 1) / THREADS.x,
        1
    );
    coarse_1d_xgemm(d_local_grad, d_weights_T, d_prev_errors, 1, in_size, out_size);
    cudaDeviceSynchronize();

    free(weights);
    weights = (float*)malloc(sizeof(float) * in_size * out_size);
    cudaMemcpy(
        weights, 
        d_updated_weights, 
        in_size * out_size * sizeof(float), 
        cudaMemcpyDeviceToHost
    );

    free(biases);
    biases = (float*)malloc(sizeof(float) * out_size);
    cudaMemcpy(
        biases, 
        d_updated_biases, 
        out_size * sizeof(float), 
        cudaMemcpyDeviceToHost
    );

    float *prev_errors = (float*)malloc(sizeof(float) * in_size);
    cudaMemcpy(
        prev_errors, 
        d_prev_errors, 
        in_size * sizeof(float), 
        cudaMemcpyDeviceToHost
    );

    // Matrix::log_static(weights, in_size, out_size, 'W');
    // Matrix::log_static(biases, 1, out_size, 'B');

    cudaFree(d_local_grad);
    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_weight_grad);
    cudaFree(d_updated_weights);
    cudaFree(d_updated_biases);
    cudaFree(d_weights_T);
    cudaFree(d_prev_errors);
    free(local_grad);

    return prev_errors;
}


void Dense::save_weights(std::string path) {
    std::ofstream file(path, std::ios::binary | std::ios::app);

    file.write(reinterpret_cast<const char*>(weights), sizeof(float) * size * output_size);
    file.write(reinterpret_cast<const char*>(biases), sizeof(float) * output_size);

    file.close();
}

void Dense::load_weights(std::string path, int start) {
    std::ifstream file(path, std::ios::binary);

    file.seekg(start, std::ios::beg);

    free(weights);
    weights = (float*)malloc(sizeof(float) * size * output_size);
    
    file.read(reinterpret_cast<char*>(weights), sizeof(float) * size * output_size);

    free(biases);
    biases = (float*)malloc(sizeof(float) * output_size);
    
    file.read(reinterpret_cast<char*>(biases), sizeof(float) * output_size);

    file.close();
}