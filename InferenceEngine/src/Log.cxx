#include "../include/Log.h"

namespace baojiayi
{
    std::ostream& Log(const std::string& level, const std::string& fileName, int lineNum)
    {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        char timeStr[26] = {0};
        std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", std::localtime(&now_c));

        std::ostringstream oss;
        oss << "[" << level << "]";
        oss << "[" << fileName << "-" << std::to_string(lineNum) << "]";
        oss << "[" << std::string(timeStr) << "]";
        std::string message = oss.str();

        std::cout << message; // 存入缓冲区，不刷新待填充报错信息
        return std::cout;
    }
}