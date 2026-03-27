#pragma once
#include<string>
#include<vector>
#include<fstream>
#include<sstream>
#include<iostream>
#include<stdexcept>
#include<cstdlib>

#include"server.h"


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


class DatasetFile 
{
private:
    int size;
    int current_index;
    int input_nodes;
    int output_nodes;
    std::string filepath;

    bool isopen;

    
    float *images;
    float *labels;
    
    void LoadData();
    void LoadDataCSV();
public:
    float *image;
    float *target;

    DatasetFile(std::string filepath, int data_size, int inodes, int onodes);
    ~DatasetFile();
    void next();
    void get(int index);
    void reset();
};

DatasetFile::DatasetFile(std::string path, int data_size, int inodes, int onodes){
    filepath = path;
    size = data_size;
    input_nodes = inodes;
    output_nodes = onodes;
    current_index = 0;

    isopen = false;

    image = (float*)malloc(input_nodes * sizeof(float));
    target = (float*)malloc(output_nodes * sizeof(float));

    images = (float*)malloc(input_nodes * (size + 1) *  sizeof(float));
    labels = (float*)malloc(output_nodes * (size + 1) *  sizeof(float));

    if (path.substr(path.find_last_of(".") + 1) == "csv") 
        LoadDataCSV();
    else 
        LoadData();
}

DatasetFile::~DatasetFile()
{
    free(image);
    free(target);
    free(images);
    free(labels);
}

void DatasetFile::LoadData() {
	std::string line;
	std::vector<std::string> part;
	std::ifstream input_file(filepath);

    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        throw std::runtime_error("Error opening file.");
    }

    isopen = true;

	std::vector<float> inputs;
	std::vector<float> targets;

	int index = 0;
    int image_count = 0;
	if (input_file.is_open()) {
		while (std::getline(input_file, line) && image_count < size) {
			if (index % 2 == 0) {
				std::vector<double> input;
				part = split(line, ' ');
				for (unsigned p = 0; p < part.size(); p++){
                    inputs.push_back(atof(part[p].c_str()));
                }
                
			} else {
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

    for (int p = 0; p < input_nodes; p++) image[p] = images[current_index * input_nodes + p];
    for (int p = 0; p < output_nodes; p++) target[p] = labels[current_index * output_nodes + p];

    reset();

	input_file.close();
}


void DatasetFile::LoadDataCSV() {
    std::ifstream input_file(filepath, std::ios::in);
    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        throw std::runtime_error("Error opening file.");
    }

    isopen = true;

    std::string line;
    int image_count = 0;

    int progress_tick = size / 100;

    if (!std::getline(input_file, line)) {
        throw std::runtime_error("CSV file is empty");
    }

    while (image_count < size && std::getline(input_file, line)) {
        const char* ptr = line.c_str();
        const char* end = ptr + line.size();

        char* next_ptr;
        int label = static_cast<int>(std::strtol(ptr, &next_ptr, 10));
        if (ptr == next_ptr || label < 0 || label >= output_nodes) {
            continue;
        }

        for (int i = 0; i < output_nodes; ++i) {
            labels[image_count * output_nodes + i] = (i == label) ? 1.0f : 0.0f;
        }

        int feature_index = 0;
        ptr = next_ptr;
        while (ptr < end && feature_index < input_nodes) {
            if (*ptr == ',') ++ptr; 
            float value = std::strtof(ptr, &next_ptr);
            if (ptr == next_ptr) break;
            images[image_count * input_nodes + feature_index] = value;
            ++feature_index;
            ptr = next_ptr;
        }

        if (feature_index != input_nodes) {
            std::cerr << "Warning: line " << image_count + 2 << " has wrong number of features\n";
            continue;
        }

        if ((image_count % progress_tick == 0) && SERVER_LOGGING) {
            std::cout << "Loading data: " << image_count << '\n';
        }
        ++image_count;
    }

    reset();
    input_file.close();
}

void DatasetFile::next() {
    current_index = (current_index + 1) % size;
    const size_t offset = current_index * input_nodes;

    memcpy(image, images + offset, input_nodes * sizeof(float));
    memcpy(target, labels + offset * output_nodes / input_nodes, output_nodes * sizeof(float));
}


void DatasetFile::get(int index){
    for (int p = 0; p < input_nodes; p++) image[p] = images[index * input_nodes + p];
    for (int p = 0; p < output_nodes; p++) target[p] = labels[index * output_nodes + p];
}


void DatasetFile::reset(){
    current_index = 0;

    for (int p = 0; p < input_nodes; p++) image[p] = images[current_index * input_nodes + p];
    for (int p = 0; p < output_nodes; p++) target[p] = labels[current_index * output_nodes + p];
}

