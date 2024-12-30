import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import config

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

class Dataset(Dataset):
    def __init__(self, file_path, transform=None):  # 加载所有文件到数组
        self.contents = []
        self.labels = []
        with open(file_path, 'r', encoding='UTF-8') as file:
            for line in tqdm(file):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')  # 切分语句和标签
                self.contents.append(content)
                self.labels.append(label)
        print(file_path)
        print("contentSize = ", len(self.contents))
        print("labelsSize = ", len(self.labels))
        self.config_inf = config.Config()

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
        label_tensor = torch.tensor(int(label)).to(self.config_inf.device)
        return (input_ids, segment_ids, attention_mask), label_tensor, self.contents[idx]


def get_data_loader():  # return trainloader and testloader
    config_inf = config.Config()
    data_train = Dataset(config_inf.train_path)
    data_test = Dataset(config_inf.test_path)
    data_dev = Dataset(config_inf.dev_path)
    train_loader = DataLoader(data_train, batch_size=config_inf.batch_size, shuffle=True)  # , collate_fn=collate
    test_loader = DataLoader(data_test, batch_size=config_inf.batch_size) 
    dev_loader = DataLoader(data_dev, batch_size=config_inf.batch_size, shuffle=True)
    return train_loader, dev_loader, test_loader
