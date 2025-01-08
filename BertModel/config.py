import time
import torch
from transformers import BertTokenizer

class Config(object):
    """ 配置参数 """
    def __init__(self):
        self.data_path = './dataset/OCEMOTION.csv'
        self.class_list = ['sadness', 'happiness', 'disgust', 'anger', 'like', 'surprise', 'fear']     # 类别名单
        self.save_path = './saved/' + str(int(time.time())) + '.ckpt'          # 模型训练结果
        self.output_path = './saved/python_inference_result.jsonl'               # 推理训练数据的输出结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过 1000batch 效果还没提升, 则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 16
        self.max_sequence_length = 512                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path + '/')
        self.hidden_size = 768
