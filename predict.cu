#include"network.cu"
#include"utils/report.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]){
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << " "
                  << "<input_nodes> <hidden_nodes> <output_nodes> "
                  << "<test_data_size> <test_data_path> "
                  << "<weights_path> <current_dir>\n";
        return 1;
    }

    const int input_nodes = std::stoi(argv[1]);
    int hidden_nodes = std::stoi(argv[2]);
    const int output_nodes = std::stoi(argv[3]);

    NeuralNetwork n(input_nodes, hidden_nodes, output_nodes, 0.01);

    int test_data_size = std::stoi(argv[4]);
    std::string test_data = argv[5];
    
    std::string weights_path = argv[6];

    std::string current_directory = argv[7];


    
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