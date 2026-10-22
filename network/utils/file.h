#pragma once
#include<string>
#include<vector>
#include<fstream>
#include<sstream>
#include<iostream>
#include<stdexcept>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<random>
#include<cuda_runtime.h>
#include"logging.h"


struct ImageShape {
    int c, h, w;
    ImageShape(int flat) : c(flat), h(0), w(0) {}
    ImageShape(int c_, int h_, int w_) : c(c_), h(h_), w(w_) {}
    int numel() const { return c * (h > 0 ? h : 1) * (w > 0 ? w : 1); }
    bool is_image() const { return h > 0 && w > 0; }
};

inline std::vector<std::string> split(std::string str, char c) {
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

inline std::pair<int, float> parse_val_fraction(const std::string str) {
    std::vector<std::string> parts = split(str, '/');

    if (parts.empty()) {
        return {0, 0.0f};
    }

    int val_base = std::stoi(parts[0]);
    float val_fraction = 0.0;
    
    if (parts.size() > 1) {
        val_fraction = std::stof(parts[1]);
    }
    
    if (val_fraction >= 1.0) {
        throw std::out_of_range("DatasetFile: val_fraction can not be greater than 1");
    }
    if (val_fraction < 0.0) {
        throw std::out_of_range("DatasetFile: val_fraction can not be smaller than 0");
    }
    
    return {val_base, val_fraction};
}

class DatasetFile
{
private:
    int size;
    int input_nodes;
    int output_nodes;
    int batch_size;
    int current_batch;
    std::string filepath;

    float *images;
    float *labels;

    std::vector<int> indices;
    std::mt19937 rng;

    int train_size;
    int val_size;
    int train_num_batches;
    int val_num_batches;
    bool val_mode;

    void LoadDataCSV();
    void gather_batch(int start_pos, int count);
    void allocate_batch_buffers();
    void build_split(float val_fraction);

    int range_offset() const { return val_mode ? train_size : 0; }
    int range_size() const { return val_mode ? val_size : train_size; }

    void release() noexcept;

public:
    float *image_batch;
    float *target_batch;

    int current_batch_size;

    DatasetFile(std::string filepath, int data_size, ImageShape input_shape, int onodes, int batch_size_, float val_fraction = 0.0f, unsigned int seed = 12345);
    ~DatasetFile();

    DatasetFile(const DatasetFile&) = delete;
    DatasetFile& operator=(const DatasetFile&) = delete;

    void next_batch();
    void get_batch(int index);
    void reset();
    void shuffle();

    void use_train();
    void use_val();
    bool has_val() const { return val_size > 0; }

    int get_num_batches() const { return val_mode ? val_num_batches : train_num_batches; }
    int get_train_num_batches() const { return train_num_batches; }
    int get_val_num_batches() const { return val_num_batches; }
    int get_train_size() const { return train_size; }
    int get_val_size() const { return val_size; }

    int get_batch_size() const { return batch_size; }
    int get_input_nodes() const { return input_nodes; }
    int get_output_nodes() const { return output_nodes; }

    friend void register_isolated_cases();
    friend void test_batches_content();
    friend void test_shuffle_and_seed();
    friend void test_train_val_modes();
};


DatasetFile::DatasetFile(std::string path, int data_size, ImageShape input_shape, int onodes, int batch_size_, float val_fraction, unsigned int seed) {
    images = nullptr;
    labels = nullptr;
    image_batch = nullptr;
    target_batch = nullptr;

    try {
        if (batch_size_ <= 0) throw std::invalid_argument("DatasetFile: batch_size must be positive.");
        if (data_size < 0)    throw std::invalid_argument("DatasetFile: size can not be negative.");

        filepath = path;
        size = data_size;
        input_nodes = input_shape.numel();
        output_nodes = onodes;
        batch_size = batch_size_;
        current_batch = 0;
        val_mode = false;

        images = (float*)malloc((size_t)input_nodes * size * sizeof(float));
        labels = (float*)malloc((size_t)output_nodes * size * sizeof(float));

        if (!images || !labels) {
            throw std::runtime_error("DatasetFile: failed to allocate host memory for dataset.");
        }

        current_batch_size = 0;
        rng.seed(seed);

        allocate_batch_buffers();

        if (path.substr(path.find_last_of(".") + 1) == "csv") LoadDataCSV();
        else throw std::runtime_error("DatasetFile: Unsupported file format.");

        indices.resize(size);
        for (int i = 0; i < size; i++) indices[i] = i;

        if (val_fraction > 0.0f) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }

        build_split(val_fraction);
        reset();
    } catch (...) {
        release();
        throw;
    }
}


void DatasetFile::allocate_batch_buffers() {
    cudaError_t err1 = cudaHostAlloc((void**)&image_batch,
                                      (size_t)batch_size * input_nodes * sizeof(float),
                                      cudaHostAllocDefault);
    cudaError_t err2 = cudaHostAlloc((void**)&target_batch,
                                      (size_t)batch_size * output_nodes * sizeof(float),
                                      cudaHostAllocDefault);

    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        throw std::runtime_error("DatasetFile: cudaHostAlloc failed for batch buffers.");
    }
}

void DatasetFile::LoadDataCSV() {
    std::ifstream input_file(filepath, std::ios::in);
    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        throw std::runtime_error("Error opening file.");
    }

    std::string line;
    int image_count = 0;

    int progress_tick = size / 100;
    if (size < 100) progress_tick = 1;

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

        if ((image_count % progress_tick == 0)) {
            log_data_loading_process(size, image_count);
        }
        ++image_count;
    }
    
    input_file.close();

    if (image_count < size) throw std::runtime_error("DatasetFile: CSV contains " + std::to_string(image_count) + " valid rows, expected " + std::to_string(size) + ".");
}


void DatasetFile::gather_batch(int start_pos, int count) {
    for (int b = 0; b < count; b++) {
        int src_idx = indices[start_pos + b];

        memcpy(image_batch + (size_t)b * input_nodes,
               images + (size_t)src_idx * input_nodes,
               input_nodes * sizeof(float));

        memcpy(target_batch + (size_t)b * output_nodes,
               labels + (size_t)src_idx * output_nodes,
               output_nodes * sizeof(float));
    }
    current_batch_size = count;
}


void DatasetFile::get_batch(int batch_index) {
    int local_start = batch_index * batch_size;
    int start_pos = range_offset() + local_start;
    int count = std::min(batch_size, range_size() - local_start);
    gather_batch(start_pos, count);
    current_batch = batch_index;
}


void DatasetFile::next_batch() {
    const int n = get_num_batches();
    if (n <= 0) {
        throw std::runtime_error("DatasetFile::next_batch: no batches available.");
    }
    get_batch((current_batch + 1) % n);
}


void DatasetFile::reset() {
    current_batch = 0;
    get_batch(0);
}


void DatasetFile::shuffle() {
    int offset = range_offset();
    int rsize = range_size();

    std::shuffle(indices.begin() + offset, indices.begin() + offset + rsize, rng);
    current_batch = 0;
    get_batch(0);
}

void DatasetFile::build_split(float val_fraction) {
    if (val_fraction <= 0.0f) {
        train_size = size;
        val_size = 0;
    } else {
        val_size = static_cast<int>(size * val_fraction);
        train_size = size - val_size;
    }

    train_num_batches = (train_size + batch_size - 1) / batch_size;
    val_num_batches = val_size > 0 ? (val_size + batch_size - 1) / batch_size : 0;
}

void DatasetFile::use_train() {
    val_mode = false;
    reset();
}

void DatasetFile::use_val() {
    if (val_size == 0) throw std::runtime_error("DatasetFile::use_val: dataset was constructed without a validation split (val_fraction=0).");
    
    val_mode = true;
    reset();
}

void DatasetFile::release() noexcept {
    if (image_batch)  { cudaFreeHost(image_batch);  image_batch = nullptr; }
    if (target_batch) { cudaFreeHost(target_batch); target_batch = nullptr; }
    free(images);  images = nullptr;
    free(labels);  labels = nullptr;
}

DatasetFile::~DatasetFile()
{
    release();
}