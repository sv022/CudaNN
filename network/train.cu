#include<string>
#include"network_test.cu"
#include"utils/report.h"

void train(
    int input_nodes,
    int hidden_nodes,
    int output_nodes,
    float learning_rate,
    int epochs,
    int train_data_size,
    int test_data_size,
    std::string train_data,
    std::string test_data,
    bool save_weights,
    std::string current_directory
) {
    NeuralNetwork n(learning_rate);
    n.add_layer(new Dense(input_nodes, hidden_nodes));
    n.add_layer(new Dense(hidden_nodes, output_nodes));

    float* accuracy_history = (float*)malloc(sizeof(float) * epochs);

    n.train(train_data, train_data_size, epochs, accuracy_history);

    std::string saved_weights_filename = "";

    // if (save_weights) {
    //     saved_weights_filename = current_directory + "/runs/" + get_current_datetime_simple() + "_neuron.weights.bin";
    //     n.save_weights(saved_weights_filename);
    // }

    int* test_targets = (int*)malloc(sizeof(int) * test_data_size);
    int* test_guesses = (int*)malloc(sizeof(int) * test_data_size);

    float accuracy = n.test(test_data, test_data_size, test_targets, test_guesses);
    
    save_report(
        current_directory,
        input_nodes,
        hidden_nodes,
        output_nodes,
        learning_rate,
        epochs,
        train_data_size,
        test_data_size,
        accuracy,
        accuracy_history,
        test_targets,
        test_guesses,
        saved_weights_filename
    );

    free(test_targets);
    free(test_guesses);
}