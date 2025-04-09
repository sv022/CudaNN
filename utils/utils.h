#include <string>
#include <ctime>

std::string get_current_datetime_simple() {
    time_t now = time(nullptr);
    char buf[80];
    strftime(buf, sizeof(buf), "%Y_%m_%d_%H%M%S", localtime(&now));
    return std::string(buf);
}