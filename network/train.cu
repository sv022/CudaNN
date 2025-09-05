#include<string>
#include"network.cu"
#include"utils/report.h"


void train(
    int layer_count,
    int* layers,
    float learning_rate,
    int epochs,
    int train_data_size,
    int test_data_size,
    std::string train_data,
    std::string test_data,
    bool save_weights,
    std::string weights_path,
    std::string current_directory
) {
    NeuralNetwork model(learning_rate);
    for (int i = 0; i < layer_count - 1; i++) {
        model.add_layer(new Dense(layers[i], layers[i + 1]));
    }

    float* accuracy_history = (float*)malloc(sizeof(float) * epochs);

    if (weights_path != ""){
        model.load_weights(weights_path);
    }

    model.train(train_data, train_data_size, epochs, accuracy_history);

    std::string saved_weights_filename = "";
    if (save_weights) {
        std::string layer_conf_string = "_";
        int i = 0;
        while (i < layer_count - 1) {
            layer_conf_string += std::to_string(layers[i]);
            layer_conf_string += "-";
            i++;
        }
        layer_conf_string += std::to_string(layers[i]);

        saved_weights_filename = current_directory + "/runs/" + get_current_datetime_simple() + layer_conf_string + ".weights.bin";
        model.save_weights(saved_weights_filename);
    }

    int* test_targets = (int*)malloc(sizeof(int) * test_data_size);
    int* test_guesses = (int*)malloc(sizeof(int) * test_data_size);

    float accuracy = model.test(test_data, test_data_size, test_targets, test_guesses);
    
    // TODO
    save_report( 
        current_directory,
        layer_count,
        layers,
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