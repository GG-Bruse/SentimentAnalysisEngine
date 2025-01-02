#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <boost/lockfree/queue.hpp>
#include "../include/iniparser.h"
#include "../include/CoreProcessor.h"
#include "../include/log.h"

namespace baojiayi
{
    class InferenceEngine
    {
    public:
        static InferenceEngine* GetInstance();
        void AddCoreProcessor(const std::string& modelName, const std::string& configPath);

    public:
        void Handle(std::string text);

    private:
        InferenceEngine() {}
        InferenceEngine(const InferenceEngine&) = delete;
        InferenceEngine& operator=(const InferenceEngine&) = delete;
    private:
        boost::lockfree::queue<std::string*> _queue; 
        std::unordered_map<std::string, std::vector<CoreProcessor*>>  _allCores;

    private:
        static InferenceEngine* _instance;
        static std::mutex _mutex;
    };
    InferenceEngine* InferenceEngine::_instance = nullptr;
    std::mutex InferenceEngine::_mutex;
}
