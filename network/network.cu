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

    int getMaxActivationIndex(float *target);

    public:
    // bool train(float *inputs, float *targets);

    NeuralNetwork(float lr);
    void add_layer(Dense* Layer);

    void forward(float *inputs);
    bool backward(float *inputs, float *targets);
    int predict(float *input);
    void train(std::string data, int data_size, int epochs, float* accuracy_by_epoch);
    float test(std::string filePath, int data_size, int* test_targets, int* test_guesses);

    // void save_weights(std::string filename);
    // void load_weights(std::string filename);
};

NeuralNetwork::NeuralNetwork(float lr) {
    learning_rate = lr;
    input_nodes = 0;
    output_nodes = 0;
}

void NeuralNetwork::add_layer(Dense* layer){
    layer->set_learning_rate(learning_rate);
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


bool NeuralNetwork::backward(float *inputs, float *targets){
    int _guess = getMaxActivationIndex(layers.back()->outputs);
    int _target = getMaxActivationIndex(targets);

    int num_layers = layers.size();
    
    float *output = layers.back()->outputs; 
    
    float *output_errors = (float*)malloc(sizeof(float) * output_nodes);
    for (int i = 0; i < output_nodes; ++i) {
        output_errors[i] = targets[i] - output[i];
    }

    // Matrix::log_static(output_errors, 1, output_nodes, 'E');

    float *current_errors = output_errors;

    for (int i = num_layers - 1; i >= 0; --i) {
        Dense *dense_layer = dynamic_cast<Dense*>(layers[i]);

        float *layer_input = nullptr;
        if (i == 0) {
            layer_input = inputs;
        } else {
            layer_input = layers[i - 1]->outputs;
        }

        float *prev_errors = dense_layer->backward(layer_input, current_errors);
        // Matrix::log_static(prev_errors, 1, dense_layer->size);

        if (i != num_layers - 1) {
            free(current_errors);
        }

        current_errors = prev_errors;
    }

    // Matrix::log_static(output, 1, output_nodes, 'O');

    free(current_errors);
    free(output_errors);

    return _guess == _target; 
}


void NeuralNetwork::train(std::string data, int data_size, int epochs, float* accuracy_by_epoch){
    DatasetFile train_file(data, data_size, input_nodes, output_nodes);
    
    std::cout << "Data " << data << " loaded. Starting training for " << epochs << " epochs..." << '\n';

    int progress_tick = data_size / 10;
    
    for (int epoch = 1; epoch <= epochs; epoch++) {

        ProgressBar data_progress('.', '#', 30);
        data_progress.done = 0;
        data_progress.todo = data_size;

        std::cout << "Epoch " << epoch << " / " << epochs << '\n';
        int totalCorrect = 0;

        for (int i = 0; i < data_size; i++){
            forward(train_file.image);
            bool isCorrent = backward(train_file.image, train_file.target);
            if (isCorrent) totalCorrect++;
            
            if (i % progress_tick == 0 && i != 0){		
                data_progress.fillUp();
                data_progress.fillUp();
                data_progress.fillUp();
                data_progress.displayPercentage();
                std::cout << " | ";
                data_progress.displayTasksDone();
                std::cout << " | ";
                data_progress.displayTimeElapsed();	
            }
            data_progress.done++;
            train_file.next();
        }

        float epoch_accuracy = totalCorrect / (float)data_size;
        accuracy_by_epoch[epoch - 1] = epoch_accuracy;

        data_progress.fillUp();
        data_progress.fillUp();
        data_progress.fillUp();
        data_progress.displayPercentage();
        std::cout << " | ";
        data_progress.displayTasksDone();
        std::cout << " | ";
        data_progress.displayTimeElapsed();
        data_progress.end();

        train_file.reset();
    }
}


float NeuralNetwork::test(std::string filePath, int data_size, int* test_targets, int* test_guesses) {
    DatasetFile test_file(filePath, data_size, input_nodes, output_nodes);

	int correctGuesses = 0;

    std::cout << "Data " << filePath << " loaded. Starting testing..." << '\n';

	for (int i = 0; i < data_size; i++) {
		int result = predict(test_file.image);
        int test_target = getMaxActivationIndex(test_file.target);

		if (result == test_target) correctGuesses++; 

        test_targets[i] = test_target;
        test_guesses[i] = result;

        test_file.next();
	}

	return (float)correctGuesses / data_size;
}


int NeuralNetwork::getMaxActivationIndex(float *target){
	int maxIndex = -1;
	float maxVal = -1000000;
	for (unsigned i = 0; i < output_nodes; i++){
		if (target[i] > maxVal) {
			maxVal = target[i];
			maxIndex = i;
		}
	}
	if (maxIndex == -1) throw std::runtime_error("Incorrent output values.");
	return maxIndex;
}

int NeuralNetwork::predict(float *input){
    forward(input);
    int prediction = getMaxActivationIndex(layers.back()->outputs);
    return prediction;
}
