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
    
    public:
    Dense(int layer_size, int next_size);
    

    void backward(float *inputs, float *targets);
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


void Dense::backward(float *inputs, float *targets){

}