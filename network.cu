#pragma once
#include<iostream>
#include"matrix/matrix.cuh"
#include<string>
#include<sstream>
#include<fstream>
#include<vector>
#include"utils/progressbar.h"
#include"utils/utils.h"
#include"utils/file.h"



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


void NeuralNetwork::forward(float *inputs){
    // ----- step 1 -----
    float *d_wih = 0;
    float *d_inputs = 0;
	float *d_hidden_inputs = 0;
    float *d_hidden_outputs;

    // Matrix::log_static(wih, hidden_nodes, input_nodes);
    // Matrix::log_static(inputs, input_nodes, 1);
    
	cudaMalloc(&d_wih, input_nodes * hidden_nodes * sizeof(float));
	cudaMalloc(&d_inputs, 1 * input_nodes * sizeof(float));
    cudaMalloc(&d_hidden_inputs, 1 * hidden_nodes * sizeof(float));
    cudaMalloc(&d_hidden_outputs, hidden_nodes * sizeof(float));
    
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
    cudaMemcpy(
        d_hidden_outputs,
        hidden_outputs,
        hidden_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    
    dim3 THREADS(32, 32);

	dim3 weightsInputBlocksPerGrid(
		((1 + THREADS.x - 1) / THREADS.x),
        ((hidden_nodes + THREADS.y - 1) / THREADS.y)
	);
    dim3 activationsHiddenBlocksPerGrid(
        (hidden_nodes + THREADS.x - 1) / THREADS.x,
        (hidden_nodes + THREADS.x - 1) / THREADS.x
    );
    
    Kernel::dot<<<weightsInputBlocksPerGrid, THREADS>>>(d_inputs, d_wih, d_hidden_inputs, 1, input_nodes, hidden_nodes);

    cudaDeviceSynchronize();

    Kernel::map<<<activationsHiddenBlocksPerGrid, THREADS>>>(d_hidden_inputs, d_hidden_outputs, 1, hidden_nodes);

    // Matrix::log_static(hidden_inputs, hidden_nodes, 1);
    
    cudaMemcpy(
        hidden_outputs,
        d_hidden_outputs,
        hidden_nodes * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    
    cudaFree(d_wih);
    cudaFree(d_inputs);
    cudaFree(d_hidden_inputs);
    cudaFree(d_hidden_outputs);
    
    // ----- step 2 -----
    
    // d_hidden_inputs = 0;
    float *d_who = 0;
    float *d_final_inputs = 0;
    float *d_output = 0;
    
    cudaMalloc(&d_hidden_outputs, 1 * hidden_nodes * sizeof(float));
    cudaMalloc(&d_who, hidden_nodes * output_nodes * sizeof(float));
    cudaMalloc(&d_final_inputs, 1 * output_nodes * sizeof(float));
    cudaMalloc(&d_output, output_nodes * sizeof(float));
    
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
    cudaMemcpy(
        d_output,
        output,
        output_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    
    dim3 weightsHiddenBlocksPerGrid(
        ((1 + THREADS.x - 1) / THREADS.x),
        ((hidden_nodes + THREADS.y - 1) / THREADS.y)
    );
    dim3 activationsOutputBlocksPerGrid(
		(output_nodes + THREADS.x - 1) / THREADS.x,
        (output_nodes + THREADS.x - 1) / THREADS.x
	);

    Kernel::dot<<<weightsHiddenBlocksPerGrid, THREADS>>>(d_hidden_outputs, d_who, d_final_inputs, 1, hidden_nodes, output_nodes);

    cudaDeviceSynchronize();

    Kernel::map<<<activationsOutputBlocksPerGrid, THREADS>>>(d_final_inputs, d_output, 1, output_nodes);

    // Matrix::log_static(final_inputs, output_nodes, 1);
    
    cudaMemcpy(
        output,
        d_output,
        output_nodes * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_who);
    cudaFree(d_hidden_outputs);
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
    dim3 hidden_errorsBlocksPerGrid(
        (hidden_nodes + THREADS.x - 1) / THREADS.x, 
        (hidden_nodes + THREADS.x - 1) / THREADS.x
    );

    Kernel::transpose<<<d_whoTransposeBlocksPerGrid, THREADS>>>(d_who, d_who_T, output_nodes, hidden_nodes);

    cudaDeviceSynchronize();

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

    // ---------- step 4-5 ----------

    float *who_grad = (float*)malloc(sizeof(float) * hidden_nodes * output_nodes);
    float *who_grad_res = (float*)malloc(sizeof(float) * hidden_nodes * output_nodes);

    float *d_output_errors_sum = 0;
    float *d_hidden_outputs = 0;
    float *d_who_grad = 0;
    float *d_who_grad_res = 0;
    
    cudaMalloc(&d_output_errors_sum, 1 * output_nodes * sizeof(float));
    cudaMalloc(&d_hidden_outputs, 1 * hidden_nodes * sizeof(float));
    cudaMalloc(&d_who_grad, hidden_nodes * output_nodes * sizeof(float));
    cudaMalloc(&d_who_grad_res, hidden_nodes * output_nodes * sizeof(float));
    cudaMalloc(&d_who, hidden_nodes * output_nodes * sizeof(float));
    
    cudaMemcpy(
        d_output_errors_sum,
        output_errors_sum,
        output_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_hidden_outputs,
        hidden_outputs,
        hidden_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_who_grad,
        who_grad,
        hidden_nodes *  output_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_who,
        who,
        hidden_nodes * output_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        who_grad_res,
        d_who_grad_res,
        hidden_nodes * output_nodes * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    dim3 who_gradBlocksPerGrid(
        (hidden_nodes + THREADS.x - 1) / THREADS.x, 
        (hidden_nodes + THREADS.x - 1) / THREADS.x
    );

    Kernel::dot<<<who_gradBlocksPerGrid, THREADS>>>(d_hidden_outputs, d_output_errors_sum, d_who_grad, hidden_nodes, 1, output_nodes);

    cudaDeviceSynchronize();

    Kernel::multadd<<<who_gradBlocksPerGrid, THREADS>>>(d_who, d_who_grad, d_who_grad_res, hidden_nodes, output_nodes, learning_rate);

    free(who);
    who = (float*)malloc(sizeof(float) * hidden_nodes * output_nodes);

    cudaMemcpy(
        who,
        d_who_grad_res,
        hidden_nodes * output_nodes * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // Matrix::log_static(who, hidden_nodes, output_nodes, 'G');
    // Matrix::log_static(who, output_nodes, hidden_nodes);
                
    cudaFree(d_hidden_outputs);
    cudaFree(d_output_errors_sum);
    cudaFree(d_who_grad);
    cudaFree(d_who);
    cudaFree(d_who_grad_res);

    // ---------- step 6 ----------

    float *hidden_errors_sum = (float*)malloc(sizeof(float) * hidden_nodes);
    for (int i = 0; i < hidden_nodes; i++){
        hidden_errors_sum[i] = hidden_errors[i] * hidden_outputs[i] * (1 - hidden_outputs[i]);
    }      
            
    // ---------- step 7-8 ----------
            
    float *wih_grad_res = (float*)malloc(sizeof(float) * hidden_nodes * input_nodes);
    float *wih_grad = (float*)malloc(sizeof(float) * input_nodes * hidden_nodes);
    
    float *d_wih = 0;
    float *d_wih_grad = 0;
    float *d_wih_grad_res = 0;
    float *d_hidden_errors_sum = 0;
    float *d_inputs = 0;

    cudaMalloc(&d_wih, hidden_nodes * input_nodes * sizeof(float));
    cudaMalloc(&d_wih_grad, hidden_nodes * input_nodes * sizeof(float));
    cudaMalloc(&d_wih_grad_res, hidden_nodes * input_nodes * sizeof(float));
    cudaMalloc(&d_hidden_errors_sum, hidden_nodes  * sizeof(float));
    cudaMalloc(&d_inputs, input_nodes * sizeof(float));

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
    cudaMemcpy(
        d_hidden_errors_sum,
        hidden_errors_sum,
        hidden_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_inputs,
        inputs,
        input_nodes * sizeof(float),
        cudaMemcpyHostToDevice
    );

    dim3 wih_gradBlocksPerGrid(
        (input_nodes + THREADS.x - 1) / THREADS.x, 
        (input_nodes + THREADS.x - 1) / THREADS.x
    );

    Kernel::dot<<<wih_gradBlocksPerGrid, THREADS>>>(d_inputs, d_hidden_errors_sum, d_wih_grad, input_nodes, 1, hidden_nodes);

    cudaDeviceSynchronize();

    Kernel::multadd<<<wih_gradBlocksPerGrid, THREADS>>>(d_wih, d_wih_grad, d_wih_grad_res, input_nodes, hidden_nodes, learning_rate);

    free(wih);
    wih = (float*)malloc(sizeof(float) * input_nodes * hidden_nodes);

    cudaMemcpy(
        wih,
        d_wih_grad_res,
        input_nodes * hidden_nodes * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // Matrix::log_static(wih_grad, input_nodes, hidden_nodes);
    // Matrix::log_static(wih, input_nodes, hidden_nodes);

    cudaFree(d_wih);
    cudaFree(d_wih_grad);
    cudaFree(d_wih_grad_res);
    cudaFree(d_inputs);
    cudaFree(d_hidden_errors_sum);

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
    DatasetFile train_file(data, data_size, input_nodes, output_nodes);
    
    std::cout << "Data " << data << " loaded. Starting training for " << epochs << " epochs..." << '\n';

    int progress_tick = data_size / 10;
    
    for (int epoch = 1; epoch <= epochs; epoch++) {

        ProgressBar data_progress('.', '#', 30);
        data_progress.done = 0;
        data_progress.todo = data_size;

        std::cout << "Epoch " << epoch << " / " << epochs << '\n';
        int totalCorrect = 0;

        for (int i = 0; i < data_size; i++){
            bool isCorrent = train(train_file.image, train_file.target);
            if (isCorrent) totalCorrect++;
            
            if (i % progress_tick == 0 && i != 0){		
                data_progress.fillUp();
                data_progress.fillUp();
                data_progress.fillUp();
                data_progress.displayPercentage();
                std::cout << " | ";
                data_progress.displayTasksDone();
                std::cout << " | ";
                data_progress.displayTimeElapsed();	
            }
            data_progress.done++;
            train_file.next();
        }

        float epoch_accuracy = totalCorrect / (float)data_size;
        accuracy_by_epoch[epoch - 1] = epoch_accuracy;

        data_progress.fillUp();
        data_progress.fillUp();
        data_progress.fillUp();
        data_progress.displayPercentage();
        std::cout << " | ";
        data_progress.displayTasksDone();
        std::cout << " | ";
        data_progress.displayTimeElapsed();
        data_progress.end();

        train_file.reset();
    }
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
    DatasetFile test_file(filePath, data_size, input_nodes, output_nodes);

	// bool debug = true;
	int correctGuesses = 0;

    std::cout << "Data " << filePath << " loaded. Starting testing..." << '\n';

	for (int i = 0; i < data_size; i++) {
		int result = predict(test_file.image);
        int test_target = getMaxActivationIndex(test_file.target);

		if(result == test_target) correctGuesses++; 

		// if (debug) std::cout << test_target << " guess: " << result << '\n';

        test_targets[i] = test_target;
        test_guesses[i] = result;

        test_file.next();
	}

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