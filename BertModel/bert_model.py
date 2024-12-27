from transformers import BertModel
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, bert_path, hidden_size, num_classes):
        super(Model, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_path)
        # 模型参数参与梯度更新
        for param in self.bert_model.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        input_ids = input[0]
        segment_ids = input[1]
        attention_mask = input[2]
        all_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        output = self.fc(all_output[1])
        return torch.sigmoid(output)
