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
        << "<optimizer> <learning_rate> <epochs> <batch_size> "
        << "<train_data_size> <test_data_size> <train_data_path> <test_data_path> " 
        << "<save_weights> <weights_path> <current_dir> <report_file_name>\n";
    std::cerr << "Usage for mode TEST = 0: " << arg << " "
        << "<mode> <layers_conf_string> "
        << "<test_data_size> <test_data_path> "
        << "<weights_path> <current_dir> <report_file_name>\n";
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

    int arg_counter = 3;
    
    if (mode == TRAIN) {
        if (argc != 16) {
            log_arg_error(argv[0]);
            return 1;
        }

        OptimizerType optimizer = parse_opt_type(argv[arg_counter++]);
        float learning_rate = std::stof(argv[arg_counter++]);
        LossType loss_function = parse_loss_type(argv[arg_counter++]);

        parse_network_structure(argv[2], optimizer, learning_rate, loss_function, structure);

        int epochs = std::stoi(argv[arg_counter++]);
        int batch_size = std::stoi(argv[arg_counter++]);
        int train_data_size = std::stoi(argv[arg_counter++]);
        int test_data_size = std::stoi(argv[arg_counter++]);
        
        std::string train_data = argv[arg_counter++];
        std::string test_data = argv[arg_counter++];
        bool save_weights = std::stoi(argv[arg_counter++]);
        
        std::string weights_path = argv[arg_counter++];
        std::string current_directory = argv[arg_counter++];
        std::string report_file_name = argv[arg_counter++];

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
        if (argc != 8) {
            log_arg_error(argv[0]);
            return 1;
        }
        parse_network_structure(argv[2], OptimizerType::SGD, 0.0f, LossType::MSE, structure);
        
        int test_data_size = std::stoi(argv[arg_counter++]);
        std::string test_data = argv[arg_counter++];
        std::string weights_path = argv[arg_counter++];
        std::string current_directory = argv[arg_counter++];
        std::string report_file_name = argv[arg_counter++];

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