#include "../include/CoreProcessor.h"

namespace baojiayi
{
    class Logger : public nvinfer1::ILogger           
    {
        void log(Severity severity, const char* msg) noexcept override
        {
            if (severity != Severity::kINFO)
                std::cout << msg << std::endl;
        }
    } logger;

    CoreWorker::CoreWorker(boost::lockfree::queue<std::string*>* queue, dictionary* dict, const std::string& secName)
    {
        LOG(NORMAL) << "start init model engine" << std::endl;
        _fixed_length_queue = queue;

        _batchSize = iniparser_getint(dict, const_cast<char*>((secName + ":batch-size").c_str()), 5);
        _sequenceLength = iniparser_getint(dict, const_cast<char*>((secName + ":sequence-length").c_str()), 512);
        _modelPath = iniparser_getstring(dict, const_cast<char*>((secName + ":model-path").c_str()), const_cast<char*>(""));
        _outputNumber = iniparser_getint(dict, const_cast<char*>((secName + ":output-number").c_str()), 2);

        cudaStreamCreate(&_cudaStream);

        int selectedProfile = 0;
        _engine = DeserializeEngine(selectedProfile);
        _context = BuildContext(selectedProfile);

        // 分配内存
        int inMemAllocationSize = _batchSize * _sequenceLength * sizeof(int32_t);
        for(int i = 0; i < _inputNumber; ++i)
            cudaMallocHost(&_inputDevices[i], inMemAllocationSize);
        int outMemAllocationSize = _batchSize * _sequenceLength * _outputNumber * sizeof(float);
        cudaMallocHost(&_outputDevice, outMemAllocationSize);
    }

    ICudaEngine* CoreWorker::DeserializeEngine(int& selectedProfile)
    {
        std::vector<char> modelData = readModelFromFile();
        if(0 == modelData.size()) return nullptr;

        IRuntime* runtime = createInferRuntime(logger);
        ICudaEngine* engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());

        _bindingNumberPerProfile = engine->getNbIOTensors() / engine->getNbOptimizationProfiles();
        LOG(NORMAL) << "bindingNumberPerProfile:" << _bindingNumberPerProfile << std::endl;

        // 查看输入输出各个张量的维度
        for (int index = 0; index < engine->getNbIOTensors(); ++index) 
        {
            std::string tensorName = engine->getIOTensorName(index);
            auto dim = engine->getTensorShape(tensorName.c_str());
            std::string dimString = "[ ";
            for (int j = 0; j < dim.nbDims; j++) {
                dimString += std::to_string(dim.d[j]);
                dimString += " ";
            }
            dimString += "]";
            LOG(NORMAL) << "Binding index: " << index << ", Name: " << tensorName << ", Shape: " << dimString << "\n";
        }
        
        selectedProfile = -1;
        std::string tensorName = "input_ids";
        for (int index = 0; index < engine->getNbOptimizationProfiles(); ++index) 
        {
            nvinfer1::Dims minDims, optDims, maxDims;
            minDims = engine->getProfileDimensions(index, index * _bindingNumberPerProfile, nvinfer1::OptProfileSelector::kMIN);
            optDims = engine->getProfileDimensions(index, index * _bindingNumberPerProfile, nvinfer1::OptProfileSelector::kOPT);
            maxDims = engine->getProfileDimensions(index, index * _bindingNumberPerProfile, nvinfer1::OptProfileSelector::kMAX);
            
            std::ostringstream oss;
            oss << "min profile shape: ";
            for (int i = 0; i < minDims.nbDims; ++i) 
                oss << minDims.d[i] << " ";
            std::string message = oss.str();
            oss.clear();
            oss.str("");
            LOG(NORMAL) << message << std::endl;
            
            oss << "opt profile shape: ";
            for (int i = 0; i < optDims.nbDims; ++i)
                oss << optDims.d[i] << " ";
            message = oss.str();
            oss.clear();
            oss.str("");
            LOG(NORMAL) << message << std::endl;

            oss << "max profile shape: ";
            for (int i = 0; i < maxDims.nbDims; ++i)
                oss << maxDims.d[i] << " ";
            message = oss.str();
            oss.clear();
            oss.str("");
            LOG(NORMAL) << message << std::endl;

            if(minDims.d[0] <= _batchSize && maxDims.d[0] >= _batchSize && \
                minDims.d[1] <= _sequenceLength && maxDims.d[1] >= _sequenceLength) {
                selectedProfile = index;
                break;
            }
            LOG(MORMAL) << "Selected Profile index : " << selectedProfile << std::endl;
        }
        return engine;
    }

    inline std::vector<char> CoreWorker::readModelFromFile()
    {
        std::ifstream stream(_modelPath.c_str(), std::ios::binary);
        if(!stream) {
            LOG(ERROR) << "open model file : " << _modelPath << "fail" << std::endl;
            return std::vector<char>();
        }
        // 获取文件大小
        stream.seekg(0, std::ios::end);
        std::streampos fileSize = stream.tellg();
        stream.seekg(0, std::ios::beg);

        std::vector<char> buffer(fileSize); 
        stream.read(buffer.data(), fileSize);
        if (!stream) {
            LOG(ERROR) << "Failed to read model file completely : " << _modelPath << std::endl;
            return std::vector<char>();
        }
        return buffer;
    }

    IExecutionContext* CoreWorker::BuildContext(const int& selectedProfile)
    {
        IExecutionContext* context = _engine->createExecutionContext();
        context->setOptimizationProfileAsync(selectedProfile, _cudaStream);
        
        // 设置输入bind的shape
        int bindingIdxOffset = selectedProfile * _bindingNumberPerProfile;
        for(int input = 0; input < _inputNumber; ++input)
            context->setBindingDimensions(input + bindingIdxOffset, Dims2(_batchSize, _sequenceLength));
        if(context->allInputDimensionsSpecified() == false) {
            LOG(ERROR) << "Not all input dimensions are specified" << std::endl;
            exit(1);
        }
        return context;
    }

    void CoreWorker::ThreadWorkFunction()
    {
        while(1) {
            // LOG(DEBUG) << "1" << std::endl;
        }
    }

    void CoreWorker::Start()
    {
        _thread = std::thread(&CoreWorker::ThreadWorkFunction, this);
    }





    /*****************************************************************************************************************************************/
    CoreProcessor::CoreProcessor() {}

    bool CoreProcessor::Init(dictionary* dict, const std::string& secName)
    {
        std::string modelPath = iniparser_getstring(dict, const_cast<char*>((secName + ":model-path").c_str()), (char*)"");
        if("" != modelPath) {
            _fixed_length_queue = new boost::lockfree::queue<std::string*>(1024 * 3); // TODO
            _fixed_length_core = new CoreWorker(_fixed_length_queue, dict, secName);
            _fixed_length_core->Start();
        }
        else {
            LOG(EROOR) << "The model path: " << modelPath << "does not exist" << std::endl;
            exit(1);
        }
        return true;
    }
    static int count = 0;
    void CoreProcessor::Handle(InferenceRequest* request)
    {
        std::string text = (*(request->_request))["text"];
        ++count;
        LOG(DEBUG) << "count:" << count << std::endl;
        LOG(NORMAL) << text << std::endl;
    }
}