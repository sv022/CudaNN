#pragma once
#include"matrix/matrix.cuh"
#include"matrix/coarse_1d.cuh"
#include<fstream>


class Layer
{
    public:
    int size;
    int output_size;

    float *outputs;

    virtual void forward(float *inputs) {};
    virtual void save_weights(std::string path) {};
    virtual void load_weights(std::string path, int start) {};
};
