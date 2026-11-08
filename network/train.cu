#include<string>
#include"network.cu"
#include"utils/report.h"

enum class WeightsMode;

void train(
    NetworkStructure* structure,
    int epochs,
    int batch_size,
    int train_data_size,
    float val_fraction,
    int test_data_size,
    std::string train_data,
    std::string test_data,
    WeightsMode weights_mode,
    std::string weights_path,
    std::string current_directory,
    std::string report_file_name
) {
    NeuralNetwork model(structure);

    float* train_loss_history = (float*)malloc(sizeof(float) * epochs);
    float* val_loss_history = (float*)malloc(sizeof(float) * epochs);

    if (weights_path != "-"){
        model.load_weights(weights_path);
    }

    std::string saved_weights_filename = current_directory + "/runs/" + 
        (report_file_name == "-" ? ( "Z-" + get_current_datetime_simple() + (weights_mode == WeightsMode::Best ? "-best" : "-final")) : (report_file_name)) + ".bin";
    
    std::string checkpoint_filename = (weights_mode == WeightsMode::Best ? saved_weights_filename : "");

    model.train(train_data, train_data_size, epochs, batch_size, train_loss_history, val_loss_history, val_fraction, checkpoint_filename);
    
    int* test_targets = (int*)malloc(sizeof(int) * test_data_size);
    int* test_guesses = (int*)malloc(sizeof(int) * test_data_size);

    if (weights_mode == WeightsMode::Best) model.load_weights(checkpoint_filename);
    float accuracy = model.test(test_data, test_data_size, test_targets, test_guesses);
    std::cout << (weights_mode == WeightsMode::Best ? "Best" : "Final") << " test accuracy: " << accuracy * 100.0f << "%" << std::endl;

    if (weights_mode == WeightsMode::Final) {
        model.save_weights(saved_weights_filename);
        std::cout << "Final weights saved to " << saved_weights_filename << std::endl;
    }

    std::string json_network_structure = structure->structure_to_json();
    
    save_report( 
        current_directory,
        json_network_structure,
        epochs,
        batch_size,
        train_data_size,
        val_fraction,
        test_data_size,
        accuracy,
        train_loss_history,
        val_loss_history,
        test_targets,
        test_guesses,
        saved_weights_filename,
        report_file_name
    );

    free(train_loss_history);
    free(val_loss_history);
    free(test_targets);
    free(test_guesses);
}