#include"network.cu"
#include"utils/report.h"
#include<string>


void predict(
    int input_nodes,
    int hidden_nodes,
    int output_nodes,
    int test_data_size,
    std::string test_data,
    std::string weights_path,
    std::string current_directory
){
    NeuralNetwork n(0.01);

    n.load_weights(weights_path);

    int* test_targets = (int*)malloc(sizeof(int) * test_data_size);
    int* test_guesses = (int*)malloc(sizeof(int) * test_data_size);

    
    float accuracy = n.test(test_data, test_data_size, test_targets, test_guesses);

    save_report(
        current_directory,
        input_nodes,
        hidden_nodes,
        output_nodes,
        test_data_size,
        accuracy,
        test_targets,
        test_guesses
    );

    free(test_targets);
    free(test_guesses);
}