#pragma once
#include<iostream>
#define LOG 0

void log_train_process(int epochs, int current_epoch, int batches, int current_batch, float epoch_loss){
    if (!LOG) return;
    std::cout << "Epoch-" << current_epoch << ": batch " << current_batch << "/" << batches << " AvgLoss: " << epoch_loss << '\n';
}

void log_test_process(int images, int current_image){
    if (!LOG) return;
    std::cout << "Testing: " << current_image << " / " << images << '\n';
}

void log_data_loading_process(int images, int current_image){
    if (!LOG) return;
    std::cout << "Loading data: " << current_image << " / " << images << '\n';
}