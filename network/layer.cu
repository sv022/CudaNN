#pragma once
#include "matrix/matrix.cuh"


class Layer
{
    public:
    int size;
    int output_size;

    float *outputs;

    virtual void forward(float *inputs) {};
};


class Dense : public Layer
{
    private:
    float *weights;
    float learning_rate = 0.1;
    
    public:
    Dense(int layer_size, int next_size);
    void set_learning_rate(float lr) { learning_rate = lr; };
    

    float* backward(float *inputs, float *targets);
    void forward(float *inputs) override;
};


Dense::Dense(int layer_size, int next_size){
    size = layer_size;
    output_size = next_size;

    outputs = (float*)malloc(sizeof(float) * layer_size);

    weights = (float*)malloc(sizeof(float) * layer_size * next_size);
    Matrix::initRandomf_static(weights, layer_size, next_size, -1 / sqrt(layer_size), 1 / sqrt(layer_size));

    // Matrix::log_static(weights, layer_size, next_size, 'W');
}


void Dense::forward(float *inputs) {
    float *d_weights = 0;
    float *d_inputs = 0;
    float *d_outputs = 0;
    
	cudaMalloc(&d_weights, size * output_size * sizeof(float));
	cudaMalloc(&d_inputs, 1 * size * sizeof(float));
    cudaMalloc(&d_outputs, 1 * output_size * sizeof(float));
    
    cudaMemcpy(
        d_weights,
        weights,
        size * output_size * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_inputs,
        inputs,
        1 * size * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_outputs,
        outputs,
        output_size * sizeof(float),
        cudaMemcpyHostToDevice
    );

    dim3 THREADS(32, 32);

    dim3 weightsBlocksPerGrid(
		((1 + THREADS.x - 1) / THREADS.x),
        ((output_size + THREADS.y - 1) / THREADS.y)
	);
    dim3 activationsBlocksPerGrid(
        (output_size + THREADS.x - 1) / THREADS.x,
        (output_size + THREADS.x - 1) / THREADS.x
    );

    Kernel::dot_softmax<<<weightsBlocksPerGrid, THREADS>>>(d_inputs, d_weights, d_outputs, 1, size, output_size);

    cudaDeviceSynchronize();

    cudaMemcpy(
        outputs,
        d_outputs,
        output_size * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_weights);
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

    float *d_local_grad = nullptr;
    float *d_inputs = nullptr;
    float *d_weights = nullptr;
    float *d_weight_grad = nullptr;
    float *d_updated_weights = nullptr;
    float *d_weights_T = nullptr;
    float *d_prev_errors = nullptr;

    cudaMalloc(&d_local_grad, out_size * sizeof(float));
    cudaMalloc(&d_inputs, in_size * sizeof(float));
    cudaMalloc(&d_weights, in_size * out_size * sizeof(float));
    cudaMalloc(&d_weight_grad, in_size * out_size * sizeof(float));
    cudaMalloc(&d_updated_weights, in_size * out_size * sizeof(float));
    cudaMalloc(&d_weights_T, out_size * in_size * sizeof(float));
    cudaMalloc(&d_prev_errors, in_size * sizeof(float));

    cudaMemcpy(d_local_grad, local_grad, out_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, inputs, in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, in_size * out_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 THREADS(32, 32);
    dim3 BLOCKS(
        (in_size + THREADS.x - 1) / THREADS.x,
        (out_size + THREADS.y - 1) / THREADS.y
    );

    Kernel::dot<<<BLOCKS, THREADS>>>(d_inputs, d_local_grad, d_weight_grad, in_size, 1, out_size);
    cudaDeviceSynchronize();

    Kernel::multadd<<<BLOCKS, THREADS>>>(d_weights, d_weight_grad, d_updated_weights,
                                         in_size, out_size, learning_rate);
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
    Kernel::dot<<<ERROR_BLOCKS, THREADS>>>(d_local_grad, d_weights_T, d_prev_errors, 1, out_size, in_size);
    cudaDeviceSynchronize();

    free(weights);
    weights = (float*)malloc(sizeof(float) * in_size * out_size);
    cudaMemcpy(weights, d_updated_weights, in_size * out_size * sizeof(float), cudaMemcpyDeviceToHost);

    float *prev_errors = (float*)malloc(sizeof(float) * in_size);
    cudaMemcpy(prev_errors, d_prev_errors, in_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Matrix::log_static(weights, in_size, out_size, 'W');

    cudaFree(d_local_grad);
    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_weight_grad);
    cudaFree(d_updated_weights);
    cudaFree(d_weights_T);
    cudaFree(d_prev_errors);
    free(local_grad);

    return prev_errors;
}
