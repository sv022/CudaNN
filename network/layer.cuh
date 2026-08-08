#pragma once
#include"matrix/matrix.cuh"
#include"matrix/coarse_1d.cuh"
#include"convolution/convolution.cuh"
#include"convolution/pooling.cuh"
#include<fstream>


class Layer
{
    public:
    int size;
    int output_size;
    int batch_size;

    float learning_rate = 0.1;

    ActivationType activation_type = ActivationType::Linear;

    float *outputs;

    virtual void forward(float *inputs) {};
    virtual float* backward(float *inputs, float *next_errors, bool raw_gradient = false) = 0;

    virtual void set_batch_size(int bs) { batch_size = bs; };
    virtual void set_learning_rate(float lr) { learning_rate = lr; };
    virtual void save_weights(std::string path) {};
    virtual int load_weights(std::string path, int start) { return 0; };
};
