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
        << "<mode> <layers_conf_string> "
        << "<learning_rate> <epochs> <train_data_size> <test_data_size> "
        << "<train_data_path> <test_data_path> <save_weights> <weights_path> <current_dir>\n";
    std::cerr << "Usage for mode TEST = 0: " << arg << " "
        << "<mode> <layers_conf_string> "
        << "<test_data_size> <test_data_path> "
        << "<weights_path> <current_dir>\n";
}


void parse_network_structure(const std::string net_str, float learning_rate, NetworkStructure& net) {
    auto layers = split(net_str, '-');

    net.learning_rate = learning_rate;

    for (const auto& layer : layers) {
        if (layer.rfind("conv_", 0) == 0) {
            auto parts = split(layer.substr(5), '_');
            auto wh = split(parts[0], 'x');
            net.add_conv(std::stoi(wh[0]), std::stoi(wh[1]), std::stoi(parts[1]), std::stoi(parts[2]), std::stoi(parts[3]), std::stoi(parts[4]), std::stoi(parts[5]));
        } else if (layer.rfind("pool_", 0) == 0) {
            auto parts = split(layer.substr(5), '_');
            auto wh = split(parts[0], 'x');
            net.add_pool(std::stoi(wh[0]), std::stoi(wh[1]), std::stoi(parts[1]), std::stoi(parts[2]), std::stoi(parts[3]));
        } else if (layer.rfind("dense_", 0) == 0) {
            auto parts = split(layer.substr(6), '_');
            net.add_dense(std::stoi(parts[0]), std::stoi(parts[1]));
        } else {
            throw std::runtime_error("Unknown layer type: " + layer);
        }
    }
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

    NetworkStructure structure;
    float learning_rate = std::stof(argv[3]);
    parse_network_structure(argv[2], learning_rate, structure);

    if (mode == TRAIN) {
        int epochs = std::stoi(argv[4]);
        int train_data_size = std::stoi(argv[5]);
        int test_data_size = std::stoi(argv[6]);
        
        std::string train_data = argv[7];
        std::string test_data = argv[8];
        bool save_weights = std::stoi(argv[9]);
        
        std::string weights_path = argv[10];
        std::string current_directory = argv[11];

        train(
            structure,
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
        int test_data_size = std::stoi(argv[3]);
        std::string test_data = argv[4];
        std::string weights_path = argv[5];
        std::string current_directory = argv[6];

        predict(
            structure,
            test_data_size,
            test_data,
            weights_path,
            current_directory
        );
    }

    return 0;
}