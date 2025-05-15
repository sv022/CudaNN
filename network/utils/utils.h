#pragma once
#include<string>
#include<ctime>
#include <sys/stat.h>

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