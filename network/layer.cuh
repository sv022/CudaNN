#pragma once
#include"matrix/matrix.cuh"
#include"matrix/coarse_1d.cuh"
#include"convolution/convolution.cuh"
#include"convolution/pooling.cuh"
#include"optimizer/optimizer.cuh"
#include<fstream>


class Layer
{
    public:
    int size;
    int output_size;
    int batch_size;

    bool is_training = true;

    std::shared_ptr<Optimizer> optimizer;
    ActivationType activation_type = ActivationType::Linear;

    float *outputs;
    float* d_outputs = nullptr;
    float* d_prev_errors = nullptr;

    bool compute_input_grad = true;

    virtual void forward(float *inputs) {};
    virtual float* backward(float *inputs, float *next_errors, bool raw_gradient = false) = 0;

    virtual void set_batch_size(int bs) { batch_size = bs; };
    virtual void set_is_training(bool training) { is_training = training; };
    virtual void save_weights(std::string path) {};
    virtual int load_weights(std::string path, int start) { return 0; };

    void download_outputs() {
        size_t n = (size_t)batch_size * output_size;
        outputs = (float*)realloc(outputs, n * sizeof(float));
        cudaMemcpy(outputs, d_outputs, n * sizeof(float), cudaMemcpyDeviceToHost);
    }

    virtual ~Layer() {}

    protected:
    void alloc_io(int bs) {
        free_io();
        cudaMalloc(&d_outputs, (size_t)bs * output_size * sizeof(float));
        cudaMalloc(&d_prev_errors, (size_t)bs * size * sizeof(float));
    }
    void free_io() {
        cudaFree(d_outputs); cudaFree(d_prev_errors);
        d_outputs = d_prev_errors = nullptr;
    }
};
