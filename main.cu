#pragma once
#include"network/train.cu"
#include"network/predict.cu"
#include <iostream>
#include <cstdlib>
#include <stdexcept>


#define TRAIN 1
#define TEST 0


bool isCudaDeviceAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';
        return false;
    }
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found" << '\n';
        return false;
    }
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    if (deviceProp.major < 1) {
        std::cerr << "Device does not support CUDA" << '\n';
        return false;
    }
    
    return true;
}


void log_arg_error(std::string arg){
    std::cerr << "Usage for mode TRAIN = 1: " << arg << " "
        << "<mode> <layer_count> <*layers> "
        << "<learning_rate> <epochs> <train_data_size> <test_data_size> "
        << "<train_data_path> <test_data_path> <save_weights> <weights_path> <current_dir>\n";
    std::cerr << "Usage for mode TEST = 0: " << arg << " "
        << "<mode> <layer_count> <*layers> "
        << "<test_data_size> <test_data_path> "
        << "<weights_path> <current_dir>\n";
}


int main(int argc, char* argv[]) {
    if (!isCudaDeviceAvailable()) {
        std::cerr << "CUDA device not available, exiting..." << '\n';
        return 1;
    }

    if (argc < 3) {
        log_arg_error(argv[0]);
        return 1;
    }

    const int mode = std::stoi(argv[1]);
    if ((mode != TRAIN) && (mode != TEST)) {
        log_arg_error(argv[0]);
        return 1;
    }

    const int layer_count = std::stoi(argv[2]);
    if (layer_count < 2) {
        std::cerr << "Invalid layer number: " << layer_count << ". Add at least 2 layers to compile model." << '\n';
        return 1;
    }
    int layer = 0;
    int* layers = (int*)malloc(sizeof(int) * layer_count);
    while (layer < layer_count) {
        layers[layer] = std::stoi(argv[layer + 3]);
        layer++;
    }

    
    if (mode == TRAIN) {
        float learning_rate = std::stof(argv[3 + layer_count]);
        int epochs = std::stoi(argv[4 + layer_count]);
        int train_data_size = std::stoi(argv[5 + layer_count]);
        int test_data_size = std::stoi(argv[6 + layer_count]);
        
        std::string train_data = argv[7 + layer_count];
        std::string test_data = argv[8 + layer_count];
        bool save_weights = std::stoi(argv[9 + layer_count]);

        std::string weights_path = "";
        std::string current_directory = "";

        if (argc == 11 + layer_count){
            weights_path = "";
            current_directory = argv[10 + layer_count];
        } else {
            current_directory = argv[11 + layer_count];
            weights_path = argv[10 + layer_count];
        }

        train(
            layer_count,
            layers,
            learning_rate,
            epochs,
            train_data_size,
            test_data_size,
            train_data,
            test_data,
            save_weights,
            weights_path,
            current_directory
        );

    } else {
        int test_data_size = std::stoi(argv[3 + layer_count]);
        std::string test_data = argv[4 + layer_count];
        
        std::string weights_path = argv[5 + layer_count];

        std::string current_directory = argv[6 + layer_count];

        predict(
            layer_count,
            layers,
            test_data_size,
            test_data,
            weights_path,
            current_directory
        );
    }

    free(layers);
    return 0;
}