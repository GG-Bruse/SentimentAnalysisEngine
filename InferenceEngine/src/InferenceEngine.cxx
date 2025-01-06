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

    void InferenceEngine::Init(const size_t threadNumber)
    {
        _queue = new boost::lockfree::queue<InferenceRequest*>(1024 * 3);
        _threadWorks.resize(threadNumber);
        for(size_t index = 0; index < threadNumber; ++index)
            _threadWorks[index] = new std::thread(&InferenceEngine::ThreadWorkFunction, this);
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

    void InferenceEngine::Handle(const std::string& text, const std::string model, CallBack* callBack, bool over)
    {
        // LOG(DEBUG) << "input text : " << text << std::endl;
        json* request = new json();
        (*request)["text"] = text;
        
        for(int index = 0; index < _allCores[model].size(); ++index)
        {
            InferenceRequest* inferenceRequest = new InferenceRequest;
            inferenceRequest->_request = request;
            inferenceRequest->_coreProcessor = _allCores[model][index];
            inferenceRequest->_callBack = callBack;
            inferenceRequest->_over = over;
            while(_queue->push(inferenceRequest) == false) {
                LOG(NORMAL) << "InferenceEngine queue is full, retrying..." << std::endl;
            }
        }
    }

    void InferenceEngine::ThreadWorkFunction()
    {
        while(true)
        {
            InferenceRequest* request = nullptr; 
            while(false == _queue->pop(request)) {
                // LOG(DEBUG) << "InferenceEngine queue is empty, retrying..." << std::endl;
            }
            if(request->_over) break;
            request->_coreProcessor->Handle(request);
        }
    }

    void InferenceEngine::GetResult()
    {

    }
}

