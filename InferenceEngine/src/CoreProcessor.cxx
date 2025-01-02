#include "../include/CoreProcessor.h"

namespace baojiayi
{
    CoreWorker::CoreWorker(boost::lockfree::queue<std::string*>* queue, dictionary* dict, const std::string& secName)
    {
        LOG(NORMAL) << "start init model engine" << std::endl;
        _fixed_length_queue = queue;

        int _batchSize = iniparser_getint(dict, const_cast<char*>((secName + ":batch-size").c_str()), 5);
        int _sequenceLength = iniparser_getint(dict, const_cast<char*>((secName + ":sequence-length").c_str()), 512);
        std::string _modelPath = iniparser_getstring(dict, const_cast<char*>((secName + ":model-path").c_str()), const_cast<char*>(""));

        int selected_profile = 0;
        ICudaEngine* engine = DeserializeEngine();
    }

    ICudaEngine* CoreWorker::DeserializeEngine()
    {
        std::vector<char> modelData = readModelFromFile();
        if(0 == modelData.size()) return nullptr;

        IRuntime* runtime = createInferRuntime(logger);
        ICudaEngine* engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());

        bindingNumberPerProfile = engine->getNbIOTensors() / engine->getNbOptimizationProfiles()
        
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


    /*****************************************************************************************************************************************/
    CoreProcessor::CoreProcessor()
    {

    }

    bool CoreProcessor::Init(dictionary* dict, const std::string& secName)
    {
        std::string modelPath = iniparser_getstring(dict, const_cast<char*>((secName + ":model_path").c_str()), (char*)"");

        if("" != modelPath) {
            _fixed_length_queue = new boost::lockfree::queue<std::string*>(); // TODO
            _fixed_length_core = new CoreWorker(_fixed_length_queue, dict, secName);
        }
        else {
            LOG(EROOR) << "The model path: " << modelPath << "does not exist" << std::endl;
            exit(1);
        }
        return true;
    }
}