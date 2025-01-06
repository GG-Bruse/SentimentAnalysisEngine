#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <boost/lockfree/queue.hpp>
#include "../include/iniparser.h"
#include "../include/json.hpp"
#include "../include/Log.h"
#include "../include/CallBack.h"

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

using namespace nvinfer1;
using json = nlohmann::json;

namespace baojiayi
{
    class CoreProcessor;
    struct InferenceRequest
    {
        json* _request;
        CoreProcessor* _coreProcessor;
        CallBack* _callBack;
        bool _over = false;
    };

    class CoreWorker
    {
    public:
        CoreWorker(boost::lockfree::queue<std::string*>* queue, dictionary* dict, const std::string& secName);

    public:
        ICudaEngine* DeserializeEngine(int& selectedProfile);
        IExecutionContext* BuildContext(const int& selectedProfile);

        void ThreadWorkFunction();
        void Start();

    private:
        inline std::vector<char> readModelFromFile();

    private:
        boost::lockfree::queue<std::string*>* _fixed_length_queue; // TODO
        std::thread _thread;

        int _batchSize;
        int _sequenceLength;
        std::string _modelPath;
        int _outputNumber;

        const int _inputNumber = 3;
        int _bindingNumberPerProfile; // 每个配置文件的bind数量
        
        cudaStream_t _cudaStream;
        ICudaEngine* _engine;
        IExecutionContext* _context;
        void* _inputDevices[3] = {nullptr};
        void* _outputDevice = nullptr;
    };

    class CoreProcessor
    {
    public:
        CoreProcessor();
        bool Init(dictionary* dict, const std::string& secName);
        void Handle(InferenceRequest* request);
        
    private:
        boost::lockfree::queue<std::string*>* _fixed_length_queue; // TODO
        CoreWorker* _fixed_length_core;
    };
}