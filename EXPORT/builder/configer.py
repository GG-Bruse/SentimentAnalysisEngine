import json

class BertConfig:
    def __init__(self, model_config_path):
        with open(model_config_path, "r") as f:
            data = json.load(f)
            self.num_attention_heads = data["num_attention_heads"]
            print('num_attention_heads', self.num_attention_heads)
            self.hidden_size = data["hidden_size"]
            print('hidden_size', self.hidden_size)
            self.intermediate_size = data["intermediate_size"]
            print('intermediate_size', self.intermediate_size)
            self.num_hidden_layers = data["num_hidden_layers"]
            print('num_hidden_layers', self.num_hidden_layers)
            self.head_size = self.hidden_size // self.num_attention_heads
            print('head_size', self.head_size)
            self.use_fp16 = data["use_fp16"]
            print('use_fp16', self.use_fp16)

