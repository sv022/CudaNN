# CUDA Neural Network

Simple fully-connected neural network in CUDA C++. 

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

Args for training:
```sh
<mode (TRAIN=1)> <layer_count> <*layers> <learning_rate> <epochs> <train_data_size> <test_data_size> <train_data_path> <test_data_path> <save_weights> <current_dir>
```

`<current_dir>` is where results will be saved in `./runs` folder.

Args for training:
```sh
<mode (TEST=0)> <layer_count> <*layers> <test_data_size> <test_data_path> <weights_path> <current_dir>
```

`<weights_path>` is the path to `.bin` file that contains weights that will be loaded and used in the model.

`<current_dir>` is where results will be saved in `./predict` folder.

Example:

```sh
.\network.exe 1 3 784 256 10 0.2 10 60000 10000 C:/Users/YourUserName/network/mnist_train_60000.csv C:/Users/YourUserName/network/mnist_test_10000.csv 1 C:/Users/YourUserName/network
```
