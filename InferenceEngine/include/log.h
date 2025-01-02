#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cstdarg>
#include <chrono>
#include <ctime>

namespace baojiayi
{
    #define LOG(level) Log(#level, __FILE__, __LINE__)
    //日志级别
    enum {
        DEBUG, NORMAL,
        WARNING, ERROR,
        FATAL
    };
	std::ostream& Log(const std::string& level, const std::string& fileName, int lineNum);
}