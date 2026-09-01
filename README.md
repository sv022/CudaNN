# CUDA Neural Network

Simple convolutional neural network in CUDA C++. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

Download to your project directory, make sure you have the latest version of `nvcc` installed:

```sh
nvcc .\main.cu -o network.exe
```

## Usage

You can use command line to run the program with given parameters.

To configure network structure, a `layer_config_string` is used. It contains information about all layers in the network, as well as layers' properties.

Example of a config string with a visualisation:

```sh
conv_28x28x1_3_16_1_0-activation_relu-pool_26x26x16_2_2-conv_13x13x16_3_32_1_0-activation_relu-pool_11x11x32_2_2-dense_800_256-activation_sigmoid-dense_256_10-activation_softmax
```

![Network structure visualisation](data/img/conv_28x28x1_3_16_1_0-pool_26x26x16_2_2-conv_13x13x16_3_32_1_0-pool_11x11x32_2_2-dense_800_256-dense_256_10.png)

Args list for training:
```sh
<mode (TRAIN=1)> <layer_config_string> <optimizer> <learning_rate> <loss_function> <epochs> <batch_size> <train_data_size> <test_data_size> <train_data_path> <test_data_path> <save_weights> <current_dir> <report_file_name>
```

Args list for testing:
```sh
<mode (TEST=0)> <layer_config_string> <test_data_size> <test_data_path> <weights_path> <current_dir> <report_file_name>
```

`<current_dir>` is where results will be saved in `./runs` subdirectory.

`<weights_path>` is the path to `.bin` file that contains weights that will be loaded and used in the model.

`<current_dir>` is where results will be saved in `./predict` subdirectory.

`<report_file_name>` will be used to generate the report file after the run is completed. Using '-' will automatically generate report name using datetime. 

Example:

```sh
.\network.exe 1 "conv_28x28x1_3_16_1_0-activation_relu-pool_26x26x16_2_2-conv_13x13x16_3_32_1_0-activation_relu-pool_11x11x32_2_2-dense_800_256-activation_sigmoid-dropout_0.5-dense_256_10-activation_softmax" sgd 0.05 cce 5 128 60000 10000 D:/path/to/digits-mnist_train.csv D:/path/to/digits-mnist_test.csv 1 - D:/results/are/saved/here -
```
