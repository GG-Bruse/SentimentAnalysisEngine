import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

class Dataset(Dataset):
    def __init__(self, contents, labels, transform=None):  # 加载所有文件到数组
        self.config_inf = config.Config()

        self.contents = contents
        # 构建标签到下标的映射
        label_to_index = {label: idx for idx, label in enumerate(self.config_inf.class_list)}
        # print('label_to_index', label_to_index)
        # 转换 labels 字典中的键为下标
        self.labels = [label_to_index[label] for label in labels]
        # for index in range(len(self.contents)):
        #     print(self.contents[index])
        #     print(self.labels[index])
        print("contentSize = ", len(self.contents))
        print("labelsSize = ", len(self.labels))
        

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):  # 获取单条信息
        tokens = self.config_inf.tokenizer.tokenize(self.contents[idx])  # 分词
        encode_dict = self.config_inf.tokenizer.encode_plus(
            text=tokens,
            add_special_tokens=True,
            max_length=self.config_inf.max_sequence_length,
            truncation=True,
            padding='max_length' if self.config_inf.max_sequence_length is not None else 'longest',  # 传参则填充至指定长度, 未传则填充到批次最长
            return_token_type_ids=True,
            return_attention_mask=True
        )
        input_ids = torch.LongTensor(encode_dict['input_ids']).to(self.config_inf.device)
        attention_mask = torch.LongTensor(encode_dict['attention_mask']).to(self.config_inf.device)
        segment_ids = torch.LongTensor(encode_dict['token_type_ids']).to(self.config_inf.device)
        label = self.labels[idx]
        label_tensor = torch.tensor(label).to(self.config_inf.device)
        return (input_ids, segment_ids, attention_mask), label_tensor, self.contents[idx]


def get_data_loader():  # return trainloader and testloader
    config_inf = config.Config()
    
    data = pd.read_csv(config_inf.data_path, sep='\t', header=None)
    contents = data[data.columns[1]].to_numpy()
    labels = data[data.columns[2]].to_numpy()
    
    contents_train, contents_test, labels_train, labels_test = train_test_split(contents, labels, test_size = 0.15, stratify = labels, random_state = 42)
    contents_test, contents_dev, labels_test, labels_dev = train_test_split(contents_test, labels_test, test_size = 0.4, stratify = labels_test, random_state = 42)

    # for index in range(len(contents_train)):
    #     print(contents_train[index])
    #     print(labels_train[index])

    data_train = Dataset(contents_train, labels_train)
    data_test = Dataset(contents_test, labels_test)
    data_dev = Dataset(contents_dev, labels_dev)

    train_loader = DataLoader(data_train, batch_size=config_inf.batch_size, shuffle=True)  # , collate_fn=collate
    test_loader = DataLoader(data_test, batch_size=config_inf.batch_size) 
    dev_loader = DataLoader(data_dev, batch_size=config_inf.batch_size, shuffle=True)
    return train_loader, dev_loader, test_loader
