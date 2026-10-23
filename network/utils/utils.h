#pragma once
#include<string>
#include<vector>
#include<ctime>
#include<sys/stat.h>

static void ensure_device(float **p, size_t *cap, size_t n) {
    if (n > *cap) { cudaFree(*p); cudaMalloc(p, n * sizeof(float)); *cap = n; }
}

std::string get_current_datetime_simple() {
    time_t now = time(nullptr);
    char buf[80];
    strftime(buf, sizeof(buf), "%Y_%m_%d_%H%M%S", localtime(&now));
    return std::string(buf);
}

bool check_file_exists(std::string name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

bool check_folder_exists(const std::string path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
        return false;
#ifdef _WIN32
    return (info.st_mode & _S_IFDIR);
#else 
    return (info.st_mode & S_IFDIR);
#endif
}

void make_folder(std::string dir) {
    #if defined(_WIN32)
        std::system(("mkdir \"" + dir + "\" 2>nul").c_str());
    #else
        std::system(("mkdir -p \"" + dir + "\"").c_str());
    #endif
}

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
