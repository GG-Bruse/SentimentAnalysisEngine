import torch
import tensorrt as trt
import numpy as np

from configer import BertConfig 

class BertLoader:
    def __init__(self, config):
         self.weights_dict = dict()
         self.config = config

    def load_weight(self, model_path):
        self.tensor_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        # self.tensor_dict = torch.load(model_path)
        # for key in self.tensor_dict:
        #    print(key)
        # embeddings
        self.weights_dict['bert_embeddings_word_embeddings'] = trt.Weights(self.tensor_dict['bert_model.embeddings.word_embeddings.weight'].cpu().numpy().flatten())
        self.weights_dict['bert_embeddings_position_embeddings'] = trt.Weights(self.tensor_dict['bert_model.embeddings.position_embeddings.weight'].cpu().numpy().flatten())
        self.weights_dict['bert_embeddings_token_type_embeddings'] = trt.Weights(self.tensor_dict['bert_model.embeddings.token_type_embeddings.weight'].cpu().numpy().flatten())
        self.weights_dict['bert_embeddings_layernorm_gamma'] = trt.Weights(self.tensor_dict['bert_model.embeddings.LayerNorm.weight'].cpu().numpy().flatten())
        self.weights_dict['bert_embeddings_layernorm_beta'] = trt.Weights(self.tensor_dict['bert_model.embeddings.LayerNorm.bias'].cpu().numpy().flatten())
        for layer in range(self.config.num_hidden_layers):
            # encoder-attention-self
            self.weights_dict["l"+str(layer)+"_attention_self_query_kernel"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.attention.self.query.weight'].cpu().numpy().flatten())
            self.weights_dict["l"+str(layer)+"_attention_self_query_bias"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.attention.self.query.bias'].cpu().numpy().flatten())
            self.weights_dict["l"+str(layer)+"_attention_self_key_kernel"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.attention.self.key.weight'].cpu().numpy().flatten())
            self.weights_dict["l"+str(layer)+"_attention_self_key_bias"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.attention.self.key.bias'].cpu().numpy().flatten())
            self.weights_dict["l"+str(layer)+"_attention_self_value_kernel"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.attention.self.value.weight'].cpu().numpy().flatten())
            self.weights_dict["l"+str(layer)+"_attention_self_value_bias"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.attention.self.value.bias'].cpu().numpy().flatten())
            # encoder-attention-output
            self.weights_dict["l"+str(layer)+"_attention_output_dense_kernel"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.attention.output.dense.weight'].cpu().numpy().flatten())
            self.weights_dict["l"+str(layer)+"_attention_output_dense_bias"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.attention.output.dense.bias'].cpu().numpy().flatten())
            self.weights_dict["l"+str(layer)+"_attention_output_layernorm_gamma"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.attention.output.LayerNorm.weight'].cpu().numpy().flatten())
            self.weights_dict["l"+str(layer)+"_attention_output_layernorm_beta"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.attention.output.LayerNorm.bias'].cpu().numpy().flatten())
            # encoder-intermediate-dense
            self.weights_dict["l"+str(layer)+"_intermediate_dense_kernel"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.intermediate.dense.weight'].cpu().numpy().flatten())
            self.weights_dict["l"+str(layer)+"_intermediate_dense_bias"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.intermediate.dense.bias'].cpu().numpy().flatten())
            # encoder-output
            self.weights_dict["l"+str(layer)+"_output_dense_kernel"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.output.dense.weight'].cpu().numpy().flatten())
            self.weights_dict["l"+str(layer)+"_output_dense_bias"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.output.dense.bias'].cpu().numpy().flatten())
            self.weights_dict["l"+str(layer)+"_output_layernorm_gamma"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.output.LayerNorm.weight'].cpu().numpy().flatten())
            self.weights_dict["l"+str(layer)+"_output_layernorm_beta"] = trt.Weights(self.tensor_dict['bert_model.encoder.layer.'+ str(layer) +'.output.LayerNorm.bias'].cpu().numpy().flatten())
        self.weights_dict["bert_pooler_dense_kernel"] = trt.Weights(self.tensor_dict['bert_model.pooler.dense.weight'].cpu().numpy().flatten())
        self.weights_dict["bert_pooler_dense_bias"] = trt.Weights(self.tensor_dict['bert_model.pooler.dense.bias'].cpu().numpy().flatten())
        self.weights_dict["output_weights"] = trt.Weights(self.tensor_dict['fc.weight'].cpu().numpy().flatten())
        self.weights_dict["output_bias"] = trt.Weights(self.tensor_dict['fc.bias'].cpu().numpy().flatten())

    def load_weight_bias_dict(self): # 三矩阵\向量合一
        N = self.config.num_attention_heads # 12
        H = self.config.head_size # 64
        WQ = "self_query_kernel"
        BQ = "self_query_bias"
        WK = "self_key_kernel"
        BK = "self_key_bias"
        WV = "self_value_kernel"
        BV = "self_value_bias"
        WQKV = "self_qkv_kernel"
        BQKV = "self_qkv_bias"

        sum_dict = dict()
        for key,value in self.weights_dict.items():
            position = key.find(BQ)
            if(position != -1):
                hidden_size = value.size # 768
                prefix = key[:position]
                Wq = self.weights_dict[prefix + WQ]
                Wk = self.weights_dict[prefix + WK]
                Wv = self.weights_dict[prefix + WV]
                Bq = value
                Bk = self.weights_dict[prefix + BK]
                Bv = self.weights_dict[prefix + BV]

                matrix_size = hidden_size * hidden_size
                w_all_count = 3 * matrix_size
                w_all = np.zeros(w_all_count, np.float32)
                b_all_count = 3 * hidden_size
                b_all = np.zeros(b_all_count, np.float32)

                w_all[0:matrix_size] = Wq.numpy()[0:matrix_size]
                w_all[matrix_size:2 * matrix_size] = Wk.numpy()[0:matrix_size]
                w_all[2 * matrix_size:3 * matrix_size] = Wv.numpy()[0:matrix_size]
                b_all[0:hidden_size] = Bq.numpy()[0:hidden_size]
                b_all[hidden_size:2 * hidden_size] = Bk.numpy()[0:hidden_size]
                b_all[2 * hidden_size:3 * hidden_size] = Bv.numpy()[0:hidden_size]

                w_all = np.ascontiguousarray(w_all.reshape((3, N, H, N, H)).transpose(1, 0, 2, 3, 4), dtype=np.float32)
                b_all = np.ascontiguousarray(b_all.reshape((3, N, H)).transpose(1, 0, 2), dtype=np.float32)
                sum_dict[prefix + WQKV] = trt.Weights(w_all.flatten())
                sum_dict[prefix + BQKV] = trt.Weights(b_all.flatten())
        self.weights_dict.update(sum_dict)



if __name__ == "__main__":
    config = BertConfig('/data/project/bjy/emotion-classification-engine/EXPORT/model/config.json')
    loader = BertLoader(config)
    loader.load_weight('/data/project/bjy/emotion-classification-engine/EXPORT/model/1733990412.ckpt')
    loader.load_weight_bias_dict()

