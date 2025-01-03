#include <iostream>
#include <fstream>
#include "../include/InferenceEngine.h"
#include "../include/Log.h"
using namespace std;
using namespace baojiayi;

string splitFirstTab(const string& str) {
    istringstream iss(str);
    string part;
    getline(iss, part, '\t');
    return part;
}

int main()
{
    InferenceEngine*  inferenceEngine = InferenceEngine::GetInstance();
    inferenceEngine->AddCoreProcessor("emotion", "/data/project/bjy/EmotionClassificationEngine/InferenceEngine/TestDemo/models/ModelInfo.conf");

    ifstream file("/data/project/bjy/EmotionClassificationEngine/InferenceEngine/TestDemo/models/test.txt");
    if(file.is_open() == false) {
        LOG(ERROR) << "open test inference file fail" << endl; 
        exit(1);
    }

    string line;
    while(getline(file, line))
    {
        string input = splitFirstTab(line);
        inferenceEngine->Handle(input);
    }
    

    return 0;
}