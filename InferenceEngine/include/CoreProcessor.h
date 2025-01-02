#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <boost/lockfree/queue.hpp>
#include "../include/iniparser.h"
#include "../include/log.h"

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

using namespace nvinfer1;

namespace baojiayi
{
    class CoreWorker
    {
    public:
        CoreWorker(boost::lockfree::queue<std::string*>* queue, dictionary* dict, const std::string& secName);

    public:
        ICudaEngine* DeserializeEngine();

    private:
        inline std::vector<char> readModelFromFile();

    private:
        boost::lockfree::queue<std::string*>* _fixed_length_queue; // TODO

        int _batchSize;
        int _sequenceLength;
        std::string _modelPath;
    };

    class CoreProcessor
    {
    public:
        CoreProcessor();
        bool Init(dictionary* dict, const std::string& secName);
        
    private:
        boost::lockfree::queue<std::string*>* _fixed_length_queue; // TODO
        CoreWorker* _fixed_length_core;
    };
}