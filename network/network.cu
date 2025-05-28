#pragma once
#include<iostream>
#include"matrix/matrix.cuh"
#include<string>
#include<sstream>
#include<fstream>
#include<vector>
#include"utils/progressbar.h"
#include"utils/utils.h"
#include"utils/file.h"
#include"layer.cu"


class NeuralNetwork 
{
private:
    float learning_rate;
    std::vector<Layer*> layers;

    int input_nodes;
    int output_nodes;

    // int getMaxActivationIndex(float *target);

    public:
    // bool train(float *inputs, float *targets);

    NeuralNetwork(float lr);
    void add_layer(Dense* Layer);

    void forward(float *inputs);
    void train(std::string data, int data_size, int epochs, float* accuracy_by_epoch);
    bool train(float *inputs, float *targets);
    // int predict(float *input);
    float test(std::string filePath, int data_size, int* test_targets, int* test_guesses);

    void save_weights(std::string filename) {};
    void load_weights(std::string filename) {};
};

NeuralNetwork::NeuralNetwork(float lr) {
    learning_rate = lr;
    input_nodes = 0;
    output_nodes = 0;
}

void NeuralNetwork::add_layer(Dense* layer){
    if (layers.size() == 0) {
        layers.push_back(layer);
        input_nodes = layer->size;
        output_nodes = layer->output_size;
        return;
    }
    if (layer->size != layers.back()->output_size) {
        std::cout << "Unmatched layer " << layers.size() + 1 << " dim: " << layer->size << ' ' << "(expected " << layers.back()->output_size << ")" << '\n';
        exit(1);
    }

    layers.push_back(layer);
    output_nodes = layer->output_size;
}


void NeuralNetwork::forward(float *inputs){
    float *layer_inputs;
    layer_inputs = (float*)malloc(sizeof(float) * layers[0]->size);
    for (int i = 0; i < layers[0]->size; i++) layer_inputs[i] = inputs[i];

    for (auto &layer : layers) {
        layer->forward(layer_inputs);
        free(layer_inputs);
        layer_inputs = (float*)malloc(sizeof(float) * layer->output_size);
        for (int i = 0; i < layer->output_size; i++) layer_inputs[i] = layer->outputs[i];
    }
    
    Layer* out_layer = layers.back();
    // Matrix::log_static(out_layer->outputs, 1, output_nodes, 'O');
}


void NeuralNetwork::train(std::string data, int data_size, int epochs, float* accuracy_by_epoch){

}


float NeuralNetwork::test(std::string filePath, int data_size, int* test_targets, int* test_guesses) {
    
}
