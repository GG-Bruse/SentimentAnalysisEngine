import time
import torch
import numpy as np

import dataloader
import config
from bert_model import Model
from train import train
from test import test



if __name__ == '__main__':
    config_inf = config.Config()
    np.random.seed(1)
    # 在 PyTorch 框架内所有依赖这个默认随机数生成器的随机操作，只要代码逻辑和其他条件不变，都会生成相同的结果
    # 如: 对同一个神经网络模型进行多次初始化，每次得到的初始权重和偏置等参数都会是一样的，有助于复现模型训练的起始状态，便于对比不同阶段训练效果或者调试模型相关的代码
    torch.manual_seed(1)
    if torch.cuda.is_available():  # 保证每次结果一样
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Loading data...")
    start_time = time.time()
    train_data_loader, dev_data_loader, test_data_loader = dataloader.get_data_loader()
    end_time = time.time()
    print("Time usage:", end_time - start_time)

    # train
    bert_model = Model(config_inf.bert_path, config_inf.hidden_size, config_inf.num_classes).to(config_inf.device)
    # bert_model.load_state_dict(torch.load('/data/project/bjy/SentimentAnalysisEngine/BertModel/saved/1736395113.ckpt', weights_only=True))
    train(bert_model, train_data_loader, dev_data_loader)
    # test(bert_model, test_data_loader, "./saved/1736413398.ckpt")
