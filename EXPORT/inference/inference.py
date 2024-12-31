import tensorrt as trt
import argparse
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class BertEngine(object):
    def __init__(self, engine_path, batch_size, max_seq_length):
        self.engine_path = engine_path
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.Time = 0

    def ready_inference(self):
        with open(self.engine_path, 'rb') as file:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(file.read())
            self.context = engine.create_execution_context()

            selected_profile = -1
            num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles # 4 // 1 = 4
            # print(engine.num_bindings)
            tensor_name = 'input_ids' # or 'segment_ids' or 'input_mask'
            for idx in range(engine.num_optimization_profiles):
                profile_shape = engine.get_tensor_profile_shape(name = tensor_name, profile_index = idx)
                print("profile shape:", profile_shape)
                if profile_shape[0][0] <= self.batch_size and profile_shape[2][0] >= self.batch_size and profile_shape[0][1] <= self.max_seq_length and profile_shape[2][1] >= self.max_seq_length:
                    selected_profile = idx
                    break
            if selected_profile == -1:
                raise RuntimeError("Could not find any profile that can run batch size {}.".format(args.batch_size))

            self.context.set_optimization_profile_async(selected_profile, cuda.Stream().handle)
            print("selected_profile", selected_profile)

            binding_idx_offset = selected_profile * num_binding_per_profile # 0
            input_shape = (self.batch_size, self.max_seq_length)
            input_nbytes = trt.volume(input_shape) * trt.int32.itemsize
            for binding in range(3):
                self.context.set_binding_shape(binding_idx_offset + binding, input_shape)
            assert self.context.all_binding_shapes_specified

            self.stream = cuda.Stream()

            # Allocate device memory for inputs.
            self.d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]
            # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
            # print("this1", tuple(self.context.get_binding_shape(binding_idx_offset + 3))) # (512, 5, 2, 1, 1)
            # print("this2", tuple(self.context.get_tensor_shape("squad_logits_out"))) # (512, -1, 2, 1, 1)
            self.h_output = cuda.pagelocked_empty(tuple(self.context.get_binding_shape(binding_idx_offset + 3)), dtype=np.float32)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)

            def inference(feature):
                start_time = time.time()
                cuda.memcpy_htod_async(self.d_inputs[0], feature['input_ids'], self.stream)
                cuda.memcpy_htod_async(self.d_inputs[1], feature['segment_ids'], self.stream)
                cuda.memcpy_htod_async(self.d_inputs[2], feature['input_mask'], self.stream)
                self.context.execute_async_v2(bindings=[0 for i in range(binding_idx_offset)] + [int(d_inp) for d_inp in self.d_inputs] + [int(self.d_output)], stream_handle=self.stream.handle)
                self.stream.synchronize()
                cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
                self.stream.synchronize()
                end_time = time.time()
                self.Time = end_time - start_time

                return self.h_output
        return inference





def main(args):
    batch_size = 5
    max_seq_length = 512

    bert_engine = BertEngine(args.engine_path, batch_size, max_seq_length)
    inference_engine = bert_engine.ready_inference()

    tokenizer = BertTokenizer.from_pretrained(args.vocab_file.rstrip('vocab.txt'))
    def text2feature(text):
        tokens = tokenizer.tokenize(text)  # 分词
        encode_dict = tokenizer.encode_plus(
            text=tokens,
            add_special_tokens=True,
            max_length=max_seq_length,
            truncation=True,
            padding='max_length' if max_seq_length is not None else 'longest',
            return_token_type_ids=True,
            return_attention_mask=True
        )
        return encode_dict

    cost_time = 0
    infer_count = 0
    batch_count = 0
    batch_dic = {}
    texts_list = []
    labels_list = []
    input_ids_list = []
    segment_ids_list = []
    input_mask_list = []

    
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    texts_all = np.array([], dtype=str)
    input_ids_all = np.array([[]], dtype=int).reshape(0, max_seq_length)
    attention_mask_all = np.array([[]], dtype=int).reshape(0, max_seq_length)
    segment_ids_all = np.array([[]], dtype=int).reshape(0, max_seq_length)
    outputs_all = np.array([[]], dtype=np.float64).reshape(0, int(os.environ["output_numbers"]))

    with open(args.input_path, 'r') as input_file, open('./inference/engine_result_' + str(args.inference_count) + '.jsonl', 'w') as output_file:
        for line in tqdm(input_file):
            if infer_count == int(args.inference_count):
                break

            line = line.strip()
            content, label = line.split('\t')
            texts_list.append(content)
            labels_list.append(label)
            feature = text2feature(content)
            input_ids_list.append(feature['input_ids'])
            input_mask_list.append(feature['attention_mask'])
            segment_ids_list.append(feature['token_type_ids'])
            infer_count += 1
            batch_count += 1

            if batch_count % batch_size == 0 and batch_count != 0:
                batch_dic['input_ids'] = np.array(input_ids_list, dtype = np.int32)
                batch_dic['segment_ids'] = np.array(segment_ids_list, dtype = np.int32)
                batch_dic['input_mask'] = np.array(input_mask_list, dtype = np.int32)
                # print("batch_dic:", batch_dic)

                infer_start = time.time()
                result = np.squeeze(inference_engine(batch_dic)[0])
                # print("result shape:", np.squeeze(inference_engine(batch_dic)).shape)
                print("result", np.squeeze(inference_engine(batch_dic)[0]))
                infer_end = time.time()
                cost_time += (infer_end - infer_start)

                outputs_all = np.append(outputs_all, result, axis = 0)
                np.set_printoptions(threshold=np.inf)

                batch_dic = {}
                texts_list = []
                labels_list = []
                input_ids_list = []
                segment_ids_list = []
                input_mask_list = []



            




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--engine_path', default='./output/model.engine')
    argparser.add_argument('--vocab_file', default='./model/vocab.txt')
    argparser.add_argument('--input_path', default='./inference/test.txt')
    argparser.add_argument('--inference_count', default=10000)
    args = argparser.parse_args()
    main(args)

    