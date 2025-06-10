#include"network.cu"
#include"utils/report.h"
#include<string>


void predict(
    int layer_count,
    int* layers,
    int test_data_size,
    std::string test_data,
    std::string weights_path,
    std::string current_directory
){
    NeuralNetwork model(0.01);
    for (int i = 0; i < layer_count - 1; i++) {
        model.add_layer(new Dense(layers[i], layers[i + 1]));
    }

    model.load_weights(weights_path);

    int* test_targets = (int*)malloc(sizeof(int) * test_data_size);
    int* test_guesses = (int*)malloc(sizeof(int) * test_data_size);

    float accuracy = model.test(test_data, test_data_size, test_targets, test_guesses);

    // TODO
    save_report( 
        current_directory,
        layer_count,
        layers,
        test_data_size,
        accuracy,
        test_targets,
        test_guesses
    );
    
    free(test_targets);
    free(test_guesses);
}