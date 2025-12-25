#include<string>
#include"network.cu"
#include"utils/report.h"


void train(
    NetworkStructure structure,
    int epochs,
    int train_data_size,
    int test_data_size,
    std::string train_data,
    std::string test_data,
    bool save_weights,
    std::string weights_path,
    std::string current_directory
) {
    NeuralNetwork model(structure);

    float* accuracy_history = (float*)malloc(sizeof(float) * epochs);

    if (weights_path != "-"){
        model.load_weights(weights_path);
    }

    model.train(train_data, train_data_size, epochs, accuracy_history);
    
    int* test_targets = (int*)malloc(sizeof(int) * test_data_size);
    int* test_guesses = (int*)malloc(sizeof(int) * test_data_size);
    
    float accuracy = model.test(test_data, test_data_size, test_targets, test_guesses);
    
    std::string saved_weights_filename = "";
    if (save_weights) {
        // TODO
    }

    // TODO
    // save_report( 
    //     current_directory,
    //     layer_count,
    //     layers,
    //     learning_rate,
    //     epochs,
    //     train_data_size,
    //     test_data_size,
    //     accuracy,
    //     accuracy_history,
    //     test_targets,
    //     test_guesses,
    //     saved_weights_filename
    // );

    free(test_targets);
    free(test_guesses);
}