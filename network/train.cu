#include<string>
#include"network.cu"
#include"utils/report.h"


void train(
    NetworkStructure* structure,
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
    std::cout << "Final Test Accuracy: " << accuracy * 100.0f << "%" << std::endl;
    
    std::string saved_weights_filename = "";
    if (save_weights) {
        std::string structure_str = structure->get_structure_string();
        saved_weights_filename = current_directory + "/runs/" + structure_str + "-Z" + get_current_datetime_simple() + ".bin";
        
        model.save_weights(saved_weights_filename);
        std::cout << "Weights saved to " << saved_weights_filename << std::endl;
    }

    std::string json_network_structure = structure->structure_to_json();
    
    save_report( 
        current_directory,
        json_network_structure,
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