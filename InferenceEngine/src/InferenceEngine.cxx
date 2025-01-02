#include "../include/InferenceEngine.h"

namespace baojiayi
{
    InferenceEngine* InferenceEngine::GetInstance() 
    {
        if(nullptr == _instance) {
            std::unique_lock<std::mutex> lock(_mutex);
            if(nullptr == _instance)
                _instance = new InferenceEngine;
        }
        return _instance;
    }

    void InferenceEngine::AddCoreProcessor(const std::string& modelName, const std::string& configPath)
    {
        
    }

    void InferenceEngine::Handle(std::string text)
    {

    }
}

