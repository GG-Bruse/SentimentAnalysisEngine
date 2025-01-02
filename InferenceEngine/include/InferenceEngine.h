#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <boost/lockfree/queue.hpp>
#include "../include/CoreProcessor.h"

namespace baojiayi
{
    class InferenceEngine
    {
    public:
        static InferenceEngine* GetInstance();
        static void AddCoreProcessor(const std::string& modelName, const std::string& configPath);

    public:
        void Handle(std::string text);

    private:
        InferenceEngine() {}
        InferenceEngine(const InferenceEngine&) = delete;
        InferenceEngine& operator=(const InferenceEngine&) = delete;
    private:
        boost::lockfree::queue<std::string*> _queue; 
        std::vector<CoreProcessor*>  _allCores;

    private:
        static InferenceEngine* _instance;
        static std::mutex _mutex;
    };
    InferenceEngine* InferenceEngine::_instance = nullptr;
    std::mutex InferenceEngine::_mutex;
}
