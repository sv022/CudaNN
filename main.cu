#pragma once
#include"network/train.cu"
#include"network/predict.cu"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc == 12) {
        const int input_nodes = std::stoi(argv[1]);
        int hidden_nodes = std::stoi(argv[2]);
        const int output_nodes = std::stoi(argv[3]);
        float learning_rate = std::stof(argv[4]);
        int epochs = std::stoi(argv[5]);
        int train_data_size = std::stoi(argv[6]);
        int test_data_size = std::stoi(argv[7]);
    
        std::string train_data = argv[8];
        std::string test_data = argv[9];
        bool save_weights = std::stoi(argv[10]);
    
        std::string current_directory = argv[11];

        train(
            input_nodes,
            hidden_nodes,
            output_nodes,
            learning_rate,
            epochs,
            train_data_size,
            test_data_size,
            train_data,
            test_data,
            save_weights,
            current_directory
        );

    } else if (argc == 8) {
        const int input_nodes = std::stoi(argv[1]);
        int hidden_nodes = std::stoi(argv[2]);
        const int output_nodes = std::stoi(argv[3]);

        int test_data_size = std::stoi(argv[4]);
        std::string test_data = argv[5];
        
        std::string weights_path = argv[6];

        std::string current_directory = argv[7];

        predict(
            input_nodes,
            hidden_nodes,
            output_nodes,
            test_data_size,
            test_data,
            weights_path,
            current_directory
        );

    } else {
        std::cerr << "Usage for train: " << argv[0] << " "
                  << "<input_nodes> <hidden_nodes> <output_nodes> <learning_rate> "
                  << "<epochs> <train_data_size> <test_data_size> "
                  << "<train_data_path> <test_data_path> <save_weights> <current_dir>\n";
        std::cerr << "Usage for predict: " << argv[0] << " "
                  << "<input_nodes> <hidden_nodes> <output_nodes> "
                  << "<test_data_size> <test_data_path> "
                  << "<weights_path> <current_dir>\n";
        return 1;
    }
}