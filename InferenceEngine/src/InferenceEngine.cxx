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
        dictionary* dict = iniparser_load(const_cast<char*>(configPath.c_str()));
        if(nullptr == dict) {
            LOG(ERROR) << "Unable open model:" << modelName << " config file" << std::endl;
            exit(1);
        }

        std::vector<CoreProcessor*> cores;
        int modelNumber = iniparser_getnsec(dict);
        for(int index = 0; index < modelNumber; ++index)
        {
            std::string secName = iniparser_getsecname(dict, index);
            CoreProcessor* coreProcessor = new CoreProcessor();
            if(coreProcessor->Init(dict, secName)) 
                cores.push_back(coreProcessor);
        }
        iniparser_freedict(dict);
        _allCores[modelName] = cores;
    }

    void InferenceEngine::Handle(const std::string& text)
    {
        LOG(NORMAL) << "input text : " << text << std::endl;
    }

    void InferenceEngine::GetResult()
    {

    }
}

