#pragma once
#include"include.h"

#define STB_IMAGE_IMPLEMENTATION
#include"stb_image.h"
#include"image.h"
#include"../utils/utils.h"
#include"../utils/logging.h"


const float inv255 = 1.0f / 255.0f;


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

    int img_c, img_h, img_w;
    bool lazy_images;
    std::vector<std::string> paths;

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
    void LoadDataBin();
    void LoadDataImages();

    void load_image_into(const std::string& path, float* dst) const;
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
    friend void test_bin_basic_loading();
};


DatasetFile::DatasetFile(std::string path, int data_size, ImageShape input_shape, int onodes, int batch_size_, float val_fraction, unsigned int seed) {
    images = nullptr;
    labels = nullptr;
    image_batch = nullptr;
    target_batch = nullptr;

    img_c = img_h = img_w = 0;
    lazy_images = false;

    try {
        if (batch_size_ <= 0) throw std::invalid_argument("DatasetFile: batch_size must be positive.");
        if (data_size < 0) throw std::invalid_argument("DatasetFile: size can not be negative.");

        filepath = path;
        size = data_size;
        input_nodes = input_shape.numel();
        output_nodes = onodes;
        batch_size = batch_size_;
        current_batch = 0;
        val_mode = false;

        img_c = input_shape.c;
        img_h = input_shape.h;
        img_w = input_shape.w;

        std::error_code ec;
        lazy_images = check_folder_exists(path);
        if (lazy_images) {
            if (!input_shape.is_image())
                throw std::invalid_argument("DatasetFile: directory mode requires ImageShape{c, h, w}.");
            if (img_c < 1 || img_c > 4)
                throw std::invalid_argument("DatasetFile: channels must be in [1, 4].");
        }

        if (!lazy_images) images = (float*)malloc((size_t)input_nodes * size * sizeof(float));
        labels = (float*)malloc((size_t)output_nodes * size * sizeof(float));

        if ((!lazy_images && !images) || !labels) {
            throw std::runtime_error("DatasetFile: failed to allocate host memory for dataset.");
        }

        current_batch_size = 0;
        rng.seed(seed);

        allocate_batch_buffers();

        std::string ext = path.substr(path.find_last_of(".") + 1);

        if (lazy_images) LoadDataImages();                           
        else if (ext == "csv") LoadDataCSV();
        else if (ext == "bin") LoadDataBin();
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

void DatasetFile::LoadDataBin() {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        throw std::runtime_error("Error opening file.");
    }

    char sig[4];
    uint8_t version, channels;
    uint16_t w, h, n_labels;
    f.read(sig, 4);
    f.read((char*)&version, 1);
    f.read((char*)&w, 2);
    f.read((char*)&h, 2);
    f.read((char*)&channels, 1);
    f.read((char*)&n_labels, 2);
    if (!f || memcmp(sig, "IMGS", 4) != 0)
        throw std::runtime_error("DatasetFile: bad .bin header");
    if (version != 2)
        throw std::runtime_error("DatasetFile: unsupported .bin version " + std::to_string((int)version) +
                                 " (expected 2: CHW, RGB); re-export the dataset");
    for (int i = 0; i < n_labels; ++i) {
        uint16_t len;
        f.read((char*)&len, 2);
        f.seekg(len, std::ios::cur);
    }
    uint32_t count;
    f.read((char*)&count, 4);
    if (!f) throw std::runtime_error("DatasetFile: truncated .bin header");

    size_t pix = (size_t)w * h * channels;
    if ((int)pix != input_nodes || (int)n_labels != output_nodes)
        throw std::runtime_error("DatasetFile: .bin shape (w*h*c=" + std::to_string(pix) + ", labels=" + std::to_string(n_labels) + ") does not match the network");

    std::vector<uint8_t> rec(2 + pix);
    int n = 0;
    while (n < size && f.read((char*)rec.data(), rec.size())) {
        uint16_t lab;
        memcpy(&lab, rec.data(), 2);
        if (lab >= output_nodes) continue;
        const uint8_t *p = rec.data() + 2;
        float *img = images + (size_t)n * input_nodes;
        for (size_t k = 0; k < pix; ++k) img[k] = p[k] * inv255;
        float *lb = labels + (size_t)n * output_nodes;
        for (int i = 0; i < output_nodes; ++i) lb[i] = (i == lab) ? 1.0f : 0.0f;
        ++n;
    }
    if (n < size)
        throw std::runtime_error("DatasetFile: file contains only " + std::to_string(n) + " valid rows, expected " + std::to_string(size));
}

void DatasetFile::LoadDataImages() {
    std::string root = filepath;
    while (!root.empty() && (root.back() == '/' || root.back() == '\\')) root.pop_back();
    const std::string list_path = root + "/labels.txt";

    std::ifstream list(list_path, std::ios::in);
    if (!list.is_open()) {
        std::cerr << "Error: Could not open file " << list_path << std::endl;
        throw std::runtime_error("Error opening file.");
    }

    int progress_tick = size / 100;
    if (size < 100) progress_tick = 1;

    paths.reserve(size);
    std::string line;
    int image_count = 0;
    long line_no = 0;

    while (image_count < size && std::getline(list, line)) {
        ++line_no;

        if (line_no == 1 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        if (line.empty()) continue;

        size_t sep = line.find_last_of(" \t");
        if (sep == std::string::npos) continue;

        const char* num = line.c_str() + sep + 1;
        char* num_end;
        long label = std::strtol(num, &num_end, 10);
        if (num_end == num || *num_end != '\0' || label < 0 || label >= output_nodes) continue;

        std::string rel = line.substr(0, sep);
        while (!rel.empty() && (rel.back() == ' ' || rel.back() == '\t')) rel.pop_back();
        std::replace(rel.begin(), rel.end(), '\\', '/');
        const std::string full = root + "/" + rel;

        int w = 0, h = 0, n = 0;
        if (!stbi_info(full.c_str(), &w, &h, &n)) {
            std::cerr << "Warning: line " << line_no << " image can not be read: " << full << "\n";
            continue;
        }

        for (int i = 0; i < output_nodes; ++i)
            labels[(size_t)image_count * output_nodes + i] = (i == (int)label) ? 1.0f : 0.0f;
        paths.push_back(full);

        if (image_count % progress_tick == 0) {
            log_data_loading_process(size, image_count);
        }
        ++image_count;
    }

    if (image_count < size)
        throw std::runtime_error("DatasetFile: image list contains " + std::to_string(image_count) + " valid rows, expected " + std::to_string(size) + ".");
}

void DatasetFile::load_image_into(const std::string& path, float* dst) const
{
    int w = 0, h = 0, n = 0;
    std::unique_ptr<unsigned char, void(*)(void*)> data(
        stbi_load(path.c_str(), &w, &h, &n, img_c), stbi_image_free);
    if (!data) {
        throw std::runtime_error("DatasetFile: failed to decode image " + path + ": " + stbi_failure_reason());
    }
    resize_to_chw(data.get(), w, h, img_c, dst, img_w, img_h);
}

void DatasetFile::gather_batch(int start_pos, int count) {
    for (int b = 0; b < count; b++) {
        int src_idx = indices[start_pos + b];

        if (lazy_images) {
            load_image_into(paths[src_idx], image_batch + (size_t)b * input_nodes);
        } else {
            memcpy(image_batch + (size_t)b * input_nodes,
                   images + (size_t)src_idx * input_nodes,
                   input_nodes * sizeof(float));
        }

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