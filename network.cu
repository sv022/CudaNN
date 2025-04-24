#pragma once
#include<iostream>
#include"matrix/matrix.cuh"
#include<string>
#include <sstream>
#include<fstream>
#include<vector>
#include"utils/progressbar.h"
#include"utils/utils.h"


std::vector<std::string> split(std::string str, char c) {
	std::vector<std::string> array;
	std::string element = "";

	for (unsigned i = 0; i < str.length(); i++) {
		if (str[i] != c)
			element += str[i];
		else if (str[i] == c && element != "") {
			array.push_back(element);
			element = "";
		}
	} if (element != "")
		array.push_back(element);

	return array;
}



class NeuralNetwork 
{
private:
    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    float learning_rate;

    float *wih;
    float *who;

    float *hidden_inputs;
    float *hidden_outputs;
    float *final_inputs;
    float *output;

    int getMaxActivationIndex(float *target);
    void LoadData(std::string filePath, int data_size, float *images, float *labels);
    void LoadDataCVS(std::string filePath, int data_size, float* images, float* labels);
    public:
    bool train(float *inputs, float *targets);
    NeuralNetwork(int i_nodes, int h_nodes, int o_nodes, double lr);
    void forward(float *inputs);
    void train(std::string data, int data_size, int epochs, float* accuracy_by_epoch);
    int predict(float *input);
    float test(std::string filePath, int data_size, int* test_targets, int* test_guesses);

    void save_weights(std::string filename);
    void load_weights(std::string filename);
};

NeuralNetwork::NeuralNetwork(int i_nodes, int h_nodes, int o_nodes, double lr) {
    input_nodes = i_nodes;
    hidden_nodes = h_nodes;
    output_nodes = o_nodes;
    learning_rate = lr;

    wih = (float*)malloc(sizeof(float) * i_nodes * h_nodes);
    who = (float*)malloc(sizeof(float) * h_nodes * o_nodes);

    Matrix::initRandomf_static(wih, i_nodes, h_nodes, -1 / sqrt(i_nodes), 1 / sqrt(i_nodes));
    Matrix::initRandomf_static(who, h_nodes, o_nodes, -1 / sqrt(h_nodes), 1 / sqrt(h_nodes));

    hidden_inputs = (float*)malloc(sizeof(float) * h_nodes);
    hidden_outputs = (float*)malloc(sizeof(float) * h_nodes);
    final_inputs = (float*)malloc(sizeof(float) * o_nodes);
    output = (float*)malloc(sizeof(float) * o_nodes);

    // Matrix::log_static(wih, h_nodes, i_nodes);
    // Matrix::log_static(wih, o_nodes, h_nodes);

    learning_rate = lr;
}

void NeuralNetwork::LoadData(std::string filePath, int data_size, float *images, float *labels) {
	std::string line;
	std::vector<std::string> part;
	std::ifstream input_file(filePath);

	std::vector<float> inputs;
	std::vector<float> targets;

	// чтение параметров из файла
	int index = 0;
    int image_count = 0;
	if (input_file.is_open()) {
		while (std::getline(input_file, line) && image_count < data_size) {
			if (index % 2 == 0) { // входные данные
				std::vector<double> input;
				part = split(line, ' ');
				for (unsigned p = 0; p < part.size(); p++){
                    inputs.push_back(atof(part[p].c_str()));
                }
                
			}
			else { // разметка входных данных
				std::vector<double> target;
				part = split(line, ' ');
				for (unsigned p = 0; p < part.size(); p++) {
					targets.push_back(atof(part[p].c_str()));
                }
                image_count++;
			}
			index++;
		}
	}

    for (int i = 0; i < inputs.size(); i++){
        images[i] = inputs[i];
    }
    for (int i = 0; i < targets.size(); i++){
        labels[i] = targets[i];
    }

	input_file.close();
}

void NeuralNetwork::LoadDataCVS(std::string filePath, int data_size, float* images, float* labels) {
    std::ifstream input_file(filePath);
    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return;
    }

    std::string line;
    int image_count = 0;
    bool skip_header = true;

    while (std::getline(input_file, line) && image_count < data_size) {
        if (skip_header) {
            skip_header = false;
            continue;
        }

        std::vector<std::string> parts;
        std::stringstream ss(line);
        std::string item;

        while (std::getline(ss, item, ',')) {
            parts.push_back(item);
        }

        if (parts.size() < 2) {
            continue; 
        }

        float label = std::stof(parts[0]);
        for (int i = 0; i < output_nodes; i++){
            if (i == label)
                labels[image_count * output_nodes + i] = 1;
            else 
                labels[image_count * output_nodes + i] = 0;
        }

        for (size_t i = 1; i < parts.size(); i++) {
            images[image_count * (parts.size() - 1) + (i - 1)] = std::stof(parts[i]);
        }

        image_count++;
    }

    input_file.close();
}

void NeuralNetwork::forward(float *inputs){
    // ----- step 1 -----
    float *d_wih = 0;
    float *d_inputs = 0;
	float *d_hidden_inputs = 0;

    // Matrix::log_static(wih, hidden_nodes, input_nodes);
    // Matrix::log_static(inputs, input_nodes, 1);
    
	cudaMalloc(&d_wih, input_nodes * hidden_nodes * sizeof(float));
	cudaMalloc(&d_inputs, 1 * input_nodes * sizeof(float));
    cudaMalloc(&d_hidden_inputs, 1 * hidden_nodes * sizeof(float));
    
    cudaMemcpy(
        d_wih,
        wih,
        input_nodes * hidden_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_inputs,
        inputs,
        1 * input_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_hidden_inputs,
        hidden_inputs,
        1 * hidden_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    
    dim3 THREADS(32, 32);

	dim3 weightsInputBlocksPerGrid(
		((1 + THREADS.x - 1) / THREADS.x),
        ((hidden_nodes + THREADS.y - 1) / THREADS.y)
	);
    
    Kernel::dot<<<weightsInputBlocksPerGrid, THREADS>>>(d_inputs, d_wih, d_hidden_inputs, 1, input_nodes, hidden_nodes);
    
    cudaMemcpy(
        hidden_inputs,
        d_hidden_inputs,
        1 * hidden_nodes * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    
    // Matrix::log_static(hidden_inputs, hidden_nodes, 1);
    
    cudaFree(d_wih);
    cudaFree(d_inputs);
    cudaFree(d_hidden_inputs);
    
    cudaDeviceSynchronize();

    // ----- step 2 -----
    
	// *d_hidden_inputs = 0;
	float *d_hidden_outputs;
    
    cudaMalloc(&d_hidden_inputs, hidden_nodes * sizeof(float));
    cudaMalloc(&d_hidden_outputs, hidden_nodes * sizeof(float));
    
    cudaMemcpy(
        d_hidden_inputs,
        hidden_inputs,
        hidden_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_hidden_outputs,
        hidden_outputs,
        hidden_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    
    dim3 activationsHiddenBlocksPerGrid(
        (hidden_nodes + THREADS.x - 1) / THREADS.x,
        (hidden_nodes + THREADS.x - 1) / THREADS.x
	);
    
    Kernel::map<<<activationsHiddenBlocksPerGrid, THREADS>>>(d_hidden_inputs, d_hidden_outputs, 1, hidden_nodes);
    
    cudaMemcpy(
        hidden_outputs,
        d_hidden_outputs,
        hidden_nodes * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    
    
    cudaFree(d_hidden_inputs);
    cudaFree(d_hidden_outputs);
    
    cudaDeviceSynchronize();
    
    // ----- step 3 -----
    
    // d_hidden_inputs = 0;
    float *d_who = 0;
    float *d_final_inputs = 0;
    
    cudaMalloc(&d_hidden_outputs, 1 * hidden_nodes * sizeof(float));
    cudaMalloc(&d_who, hidden_nodes * output_nodes * sizeof(float));
    cudaMalloc(&d_final_inputs, 1 * output_nodes * sizeof(float));
    
    cudaMemcpy(
        d_who,
        who,
        hidden_nodes * output_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_hidden_outputs,
        hidden_outputs,
        1 * hidden_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_final_inputs,
        final_inputs,
        1 * output_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    
    dim3 weightsHiddenBlocksPerGrid(
        ((1 + THREADS.x - 1) / THREADS.x),
        ((hidden_nodes + THREADS.y - 1) / THREADS.y)
    );

    Kernel::dot<<<weightsHiddenBlocksPerGrid, THREADS>>>(d_hidden_outputs, d_who, d_final_inputs, 1, hidden_nodes, output_nodes);
    
    cudaMemcpy(
        final_inputs,
        d_final_inputs,
        1 * output_nodes * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // Matrix::log_static(final_inputs, output_nodes, 1);

    cudaFree(d_who);
    cudaFree(d_hidden_outputs);
    cudaFree(d_final_inputs);

    // ----- step 4 -----

    // *d_final_inputs = 0
    float *d_output = 0;

    cudaMalloc(&d_final_inputs, output_nodes * sizeof(float));
    cudaMalloc(&d_output, output_nodes * sizeof(float));

    cudaMemcpy(
        d_final_inputs,
        final_inputs,
        output_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_output,
        output,
        output_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );

    dim3 activationsOutputBlocksPerGrid(
		(output_nodes + THREADS.x - 1) / THREADS.x,
        (output_nodes + THREADS.x - 1) / THREADS.x
	);
    
    Kernel::map<<<activationsOutputBlocksPerGrid, THREADS>>>(d_final_inputs, d_output, 1, output_nodes);

    cudaMemcpy(
        output,
        d_output,
        output_nodes * sizeof(float),
        cudaMemcpyDeviceToHost
    );


    cudaFree(d_final_inputs);
    cudaFree(d_output);

    cudaDeviceSynchronize();

    // Matrix::log_static(output, output_nodes, 1);
}

bool NeuralNetwork::train(float *inputs, float *targets){
    forward(inputs);

    int _guess = getMaxActivationIndex(output);
    int _target = getMaxActivationIndex(targets);

    // ---------- step 1 ----------

    float *output_errors = (float*)malloc(sizeof(float) * output_nodes);
    for (int i = 0; i < output_nodes; i++){
        output_errors[i] = targets[i] - output[i];
    }

    // ---------- step 2 ----------

    float *hidden_errors = (float*)malloc(sizeof(float) * hidden_nodes);
    float *who_T = (float*)malloc(sizeof(float) * hidden_nodes * output_nodes);


    float *d_who = 0;
    float *d_who_T = 0;
    float *d_output_errors = 0;
    float *d_hidden_errors = 0;

    cudaMalloc(&d_who, hidden_nodes * output_nodes * sizeof(float));
    cudaMalloc(&d_who_T, output_nodes * hidden_nodes * sizeof(float));
    cudaMalloc(&d_output_errors, 1 * output_nodes * sizeof(float));
    cudaMalloc(&d_hidden_errors, 1 * hidden_nodes * sizeof(float));

    cudaMemcpy(
        d_who,
        who,
        hidden_nodes * output_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_who_T,
        who_T,
        output_nodes * hidden_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_output_errors,
        output_errors,
        output_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_hidden_errors,
        hidden_errors,
        hidden_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );

    dim3 THREADS(32, 32);
    dim3 d_whoTransposeBlocksPerGrid(
        (output_nodes + THREADS.x - 1) / THREADS.x, 
        (output_nodes + THREADS.x - 1) / THREADS.x
    );

    Kernel::transpose<<<d_whoTransposeBlocksPerGrid, THREADS>>>(d_who, d_who_T, output_nodes, hidden_nodes);

    dim3 hidden_errorsBlocksPerGrid(
        (hidden_nodes + THREADS.x - 1) / THREADS.x, 
        (hidden_nodes + THREADS.x - 1) / THREADS.x
    );

    Kernel::dot<<<hidden_errorsBlocksPerGrid, THREADS>>>(d_output_errors, d_who_T, d_hidden_errors, 1, output_nodes, hidden_nodes);

    cudaMemcpy(
        hidden_errors,
        d_hidden_errors,
        hidden_nodes * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // Matrix::log_static(hidden_errors, hidden_nodes, 1);

    
    cudaFree(d_who);
    cudaFree(d_who_T);
    cudaFree(d_output_errors);
    cudaFree(d_hidden_errors);

    // ---------- step 3 ----------

    float *output_errors_sum = (float*)malloc(sizeof(float) * output_nodes);
    for (int i = 0; i < output_nodes; i++){
        output_errors_sum[i] = output_errors[i] * output[i] * (1 - output[i]);
    }

    // ---------- step 4 ----------

    float *who_grad = (float*)malloc(sizeof(float) * hidden_nodes * output_nodes);
    for (int i = 0; i < hidden_nodes; i++){
        for (int j = 0; j < output_nodes; j++){
            who_grad[i * output_nodes + j] = learning_rate * output_errors_sum[j] * hidden_outputs[i];
        }
    }
    // Matrix::log_static(who_grad, output_nodes, hidden_nodes);

    // ---------- step 5 ----------
    
    float *who_grad_res = (float*)malloc(sizeof(float) * hidden_nodes * output_nodes);


    // *d_who = 0;
    float *d_who_grad = 0;
    float *d_who_grad_res = 0;

    cudaMalloc(&d_who, hidden_nodes * output_nodes * sizeof(float));
    cudaMalloc(&d_who_grad, hidden_nodes * output_nodes * sizeof(float));
    cudaMalloc(&d_who_grad_res, hidden_nodes * output_nodes * sizeof(float));

    cudaMemcpy(
        d_who,
        who,
        hidden_nodes * output_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_who_grad,
        who_grad,
        hidden_nodes * output_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_who_grad_res,
        who_grad_res,
        hidden_nodes * output_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );

    dim3 d_who_gradBlocksPerGrid(
        (hidden_nodes + THREADS.x - 1) / THREADS.x, 
        (hidden_nodes + THREADS.x - 1) / THREADS.x
    );

    Kernel::add<<<d_who_gradBlocksPerGrid, THREADS>>>(d_who, d_who_grad, d_who_grad_res, hidden_nodes, output_nodes);

    cudaMemcpy(
        who_grad_res,
        d_who_grad_res,
        hidden_nodes * output_nodes * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < hidden_nodes * output_nodes; i++){
        who[i] = who_grad_res[i];
    }

    // Matrix::log_static(who_grad, output_nodes, hidden_nodes);
    // Matrix::log_static(who, output_nodes, hidden_nodes);


    cudaFree(d_who);
    cudaFree(d_who_grad);
    cudaFree(d_who_grad_res);

    // ---------- step 6 ----------

    float *hidden_errors_sum = (float*)malloc(sizeof(float) * hidden_nodes);
    for (int i = 0; i < hidden_nodes; i++){
        hidden_errors_sum[i] = hidden_errors[i] * hidden_outputs[i] * (1 - hidden_outputs[i]);
    }

    // ---------- step 7 ----------

    float *wih_grad = (float*)malloc(sizeof(float) * input_nodes * hidden_nodes);
    for (int i = 0; i < input_nodes; i++){
        for (int j = 0; j < hidden_nodes; j++){
            wih_grad[i * hidden_nodes + j] = learning_rate * hidden_errors_sum[j] * inputs[i];
        }
    }
    // Matrix::log_static(wih_grad, hidden_nodes, input_nodes);


    // ---------- step 8 ----------

    float *wih_grad_res = (float*)malloc(sizeof(float) * hidden_nodes * input_nodes);

    float *d_wih = 0;
    float *d_wih_grad = 0;
    float *d_wih_grad_res = 0;

    cudaMalloc(&d_wih, hidden_nodes * input_nodes * sizeof(float));
    cudaMalloc(&d_wih_grad, hidden_nodes * input_nodes * sizeof(float));
    cudaMalloc(&d_wih_grad_res, hidden_nodes * input_nodes * sizeof(float));

    cudaMemcpy(
        d_wih,
        wih,
        input_nodes * hidden_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_wih_grad,
        wih_grad,
        input_nodes * hidden_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_wih_grad_res,
        wih_grad_res,
        input_nodes * hidden_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );

    dim3 d_wih_gradBlocksPerGrid(
        (input_nodes + THREADS.x - 1) / THREADS.x, 
        (input_nodes + THREADS.x - 1) / THREADS.x
    );

    Kernel::add<<<d_wih_gradBlocksPerGrid, THREADS>>>(d_wih, d_wih_grad, d_wih_grad_res, input_nodes, hidden_nodes);

    cudaMemcpy(
        wih_grad_res,
        d_wih_grad_res,
        input_nodes * hidden_nodes * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < input_nodes * hidden_nodes; i++){
        wih[i] = wih_grad_res[i];
    }

    // Matrix::log_static(wih_grad, output_nodes, hidden_nodes);
    // Matrix::log_static(wih_grad_res, output_nodes, hidden_nodes);

    cudaFree(d_wih);
    cudaFree(d_wih_grad);
    cudaFree(d_wih_grad_res);

    free(output_errors);
    free(hidden_errors);
    free(who_T);
    free(output_errors_sum);
    free(who_grad);
    free(who_grad_res);
    free(hidden_errors_sum);
    free(wih_grad);
    free(wih_grad_res);

    return _guess == _target;
}

void NeuralNetwork::train(std::string data, int data_size, int epochs, float* accuracy_by_epoch){
    float *images = (float*)malloc(input_nodes * sizeof(float) * data_size);
    float *targets = (float*)malloc(output_nodes * sizeof(float) * data_size);

    if (data.substr(data.find_last_of(".") + 1) == "csv") 
        LoadDataCVS(data, data_size, images, targets);
    else 
        LoadData(data, data_size, images, targets);
    
    std::cout << "Data " << data << " loaded. Starting training for " << epochs << " epochs..." << '\n';

    int progress_tick = data_size / 10;
    
    for (int epoch = 1; epoch <= epochs; epoch++) {

        ProgressBar data_progress('.', '#', 30);
        data_progress.done = 0;
        data_progress.todo = data_size;

        std::cout << "Epoch " << epoch << " / " << epochs << '\n';
        int totalCorrect = 0;

        for (int i = 0; i < data_size; i++){
            float *image = (float*)malloc(input_nodes * sizeof(float));
            float *target = (float*)malloc(output_nodes * sizeof(float));

            for (int p = 0; p < input_nodes; p++) image[p] = images[i * input_nodes + p];
            for (int p = 0; p < output_nodes; p++) target[p] = targets[i * output_nodes + p];

            bool isCorrent = train(image, target);
            if (isCorrent) totalCorrect++;
            
            if (i % progress_tick == 0 && i != 0){		
                data_progress.fillUp();
                data_progress.fillUp();
                data_progress.fillUp();
                data_progress.displayPercentage();
                std::cout << " | ";
                data_progress.displayTasksDone();	
            }
            data_progress.done++;

            free(image);
            free(target);
        }

        float epoch_accuracy = totalCorrect / (float)data_size;
        accuracy_by_epoch[epoch - 1] = epoch_accuracy;

        data_progress.fillUp();
        data_progress.fillUp();
        data_progress.fillUp();
        data_progress.displayPercentage();
        std::cout << " | ";
        data_progress.displayTasksDone();	
        data_progress.end();
    }
    free(images);
    free(targets);
}

int NeuralNetwork::getMaxActivationIndex(float *target){
	int maxIndex = -1;
	float maxVal = -1000000;
	for (unsigned i = 0; i < output_nodes; i++){
		if (target[i] > maxVal) {
			maxVal = target[i];
			maxIndex = i;
		}
	}
	if (maxIndex == -1) throw std::runtime_error("Incorrent output values.");
	return maxIndex;
}

int NeuralNetwork::predict(float *input){
    forward(input);
    int prediction = getMaxActivationIndex(output);
    return prediction;
}

float NeuralNetwork::test(std::string filePath, int data_size, int* test_targets, int* test_guesses) {
	// bool debug = true;
	int correctGuesses = 0;

    float *images = (float*)malloc(input_nodes * sizeof(float) * data_size);
    float *targets = (float*)malloc(output_nodes * sizeof(float) * data_size);
    
    if (filePath.substr(filePath.find_last_of(".") + 1) == "csv") 
        LoadDataCVS(filePath, data_size, images, targets);
    else 
        LoadData(filePath, data_size, images, targets);

    std::cout << "Data " << filePath << " loaded. Starting testing..." << '\n';

	for (int i = 0; i < data_size; i++) {
		float *image = (float*)malloc(input_nodes * sizeof(float));
        float *target = (float*)malloc(output_nodes * sizeof(float));

        for (int p = 0; p < input_nodes; p++) image[p] = images[i * input_nodes + p];
        for (int p = 0; p < output_nodes; p++) target[p] = targets[i * output_nodes + p];

		int result = predict(image);
        int test_target = getMaxActivationIndex(target);

		if(result == test_target) correctGuesses++; 

		// if (debug) std::cout << test_target << " guess: " << result << '\n';

        test_targets[i] = test_target;
        test_guesses[i] = result;
        
        free(image);
        free(target);
	}

    free(images);
    free(targets);

	return (float)correctGuesses / data_size;
}

void NeuralNetwork::save_weights(std::string filename){
    std::ofstream file(filename, std::ios::binary);
    
    file.write(reinterpret_cast<const char*>(wih), sizeof(float) * hidden_nodes * input_nodes);
    file.write(reinterpret_cast<const char*>(who), sizeof(float) * output_nodes * hidden_nodes);
}

void NeuralNetwork::load_weights(std::string filename){
    if (!check_file_exists(filename)) {
        std::cout << "Failed to load weights " << filename << ". Press any key to continue...";
        char press_to_continue;
        std::cin >> press_to_continue;
        exit(1);
    }
    std::ifstream file(filename, std::ios::binary);

    free(wih);
    free(who);

    wih = (float*)malloc(sizeof(float) * hidden_nodes * input_nodes);
    who = (float*)malloc(sizeof(float) * output_nodes * hidden_nodes);
    
    file.read(reinterpret_cast<char*>(wih), sizeof(float) * hidden_nodes * input_nodes);
    file.read(reinterpret_cast<char*>(who), sizeof(float) * output_nodes * hidden_nodes);
}