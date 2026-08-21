#pragma once
#include"network/train.cu"
#include"network/predict.cu"
#include<iostream>
#include<cstdlib>
#include<stdexcept>


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
        << "<learning_rate> <epochs> <batch_size> <train_data_size> <test_data_size> "
        << "<train_data_path> <test_data_path> <save_weights> <weights_path> <current_dir>\n";
    std::cerr << "Usage for mode TEST = 0: " << arg << " "
        << "<mode> <layers_conf_string> "
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

    std::cout.setf(std::ios::unitbuf);

    NetworkStructure* structure = new NetworkStructure();
    
    if (mode == TRAIN) {
        float learning_rate = std::stof(argv[3]);
        LossType loss_function = parse_loss_type(argv[4]);

        parse_network_structure(argv[2], learning_rate, loss_function, structure);

        int epochs = std::stoi(argv[5]);
        int batch_size = std::stoi(argv[6]);
        int train_data_size = std::stoi(argv[7]);
        int test_data_size = std::stoi(argv[8]);
        
        std::string train_data = argv[9];
        std::string test_data = argv[10];
        bool save_weights = std::stoi(argv[11]);
        
        std::string weights_path = argv[12];
        std::string current_directory = argv[13];
        std::string report_file_name = argv[14];

        train(
            structure,
            epochs,
            batch_size,
            train_data_size,
            test_data_size,
            train_data,
            test_data,
            save_weights,
            weights_path,
            current_directory,
            report_file_name
        );

    } else {
        parse_network_structure(argv[2], 0.0f, LossType::MSE, structure);
        
        int test_data_size = std::stoi(argv[3]);
        std::string test_data = argv[4];
        std::string weights_path = argv[5];
        std::string current_directory = argv[6];
        std::string report_file_name = argv[7];

        predict(
            structure,
            test_data_size,
            test_data,
            weights_path,
            current_directory,
            report_file_name
        );
    }

    free(structure);

    return 0;
}