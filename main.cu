#include"network.cu"
#include"utils/report.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 11) {
        std::cerr << "Usage: " << argv[0] << " "
                  << "<input_nodes> <hidden_nodes> <output_nodes> <learning_rate> "
                  << "<epochs> <train_data_size> <test_data_size> "
                  << "<train_data_path> <test_data_path> <save_weights>\n";
        return 1;
    }

    const int input_nodes = std::stoi(argv[1]);
    int hidden_nodes = std::stoi(argv[2]);
    const int output_nodes = std::stoi(argv[3]);
    float learning_rate = std::stof(argv[4]);

    NeuralNetwork n(input_nodes, hidden_nodes, output_nodes, learning_rate);

    int epochs = std::stoi(argv[5]);
    int train_data_size = std::stoi(argv[6]);
    int test_data_size = std::stoi(argv[7]);

    std::string train_data = argv[8];
    std::string test_data = argv[9];
    bool save_weights = std::stoi(argv[10]);

    n.train(train_data, train_data_size, epochs);

    std::string saved_weights_filename = "";

    if (save_weights) {
        saved_weights_filename = get_current_datetime_simple() + "_neuron.weights.bin";
        n.save_weights(saved_weights_filename);
    }

    int* test_targets = (int*)malloc(sizeof(int) * test_data_size);
    int* test_guesses = (int*)malloc(sizeof(int) * test_data_size);

    float accuracy = n.test(test_data, test_data_size, test_targets, test_guesses);
    
    free(test_targets);
    free(test_guesses);

    save_report(
        input_nodes,
        hidden_nodes,
        output_nodes,
        learning_rate,
        epochs,
        train_data_size,
        test_data_size,
        accuracy,
        test_targets,
        test_guesses,
        saved_weights_filename
    );

    return 0;
}